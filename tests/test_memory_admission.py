import struct
import threading
import time
import json
from pathlib import Path

from soyrootbio.batch import BatchScheduler, BatchJobState
from soyrootbio.hardware import GIB, HardwareInfo, allocate_resources
from soyrootbio.memory_budget import (
    MemoryAdmission, MemorySnapshot, estimate_decoded_memory, estimate_input_memory,
    observed_memory_budget,
)


def test_binary_stl_budget_uses_facet_count_and_scales_with_geometry(tmp_path):
    small = tmp_path / 'small.stl'
    small.write_bytes(bytes(80) + struct.pack('<I', 1) + bytes(50))
    large = tmp_path / 'large.stl'
    with large.open('wb') as stream:
        stream.write(bytes(80) + struct.pack('<I', 1_000_000))
        stream.seek(84 + 50 * 1_000_000 - 1)
        stream.write(b'\0')
    assert estimate_input_memory(small) == 1_932_735_284  # 1.8 GiB, rounded up
    assert estimate_input_memory(large) > 3.6 * GIB


def test_initial_budgets_are_cut_40_percent_but_observed_peaks_are_not_discounted():
    path = Path(__file__).parents[1] / 'docs/benchmarks/memory_admission/recorded_peaks.json'
    records = json.loads(path.read_text())['records']
    assert len(records) == 25
    assert len({row['sample'] for row in records}) == 6
    assert any(row['noise_reduction'] for row in records)
    # Freeze the previous policy's budgets independently of the new estimator.
    previous_budgets = {
        'BaxiNo2_4-2_20260525': 3_221_225_472,
        'Kaixinlv_3-2_20260525': 5_125_324_208,
        'SN14_6-2_20260405': 6_686_209_216,
        'w5168-3_m4-2_20260415': 3_623_159_936,
        'W82_9cm_water_1-2_20260522': 3_567_491_376,
        'W82_MS4-2_20260617': 3_674_813_808,
    }
    below_recorded_peak = set()
    for row in records:
        budget = estimate_decoded_memory(
            row['full_vertex_count'], row['triangle_count'], row['original_vertex_count'])
        target = previous_budgets[row['sample']] * 0.6
        assert target <= budget < target + 1, row['source_log']
        assert observed_memory_budget(row['peak_private_bytes']) >= 1.25 * row['peak_private_bytes']
        if budget < row['peak_private_bytes']:
            below_recorded_peak.add(row['sample'])
    # Lower initial estimates no longer claim to cover all historical peaks.
    assert len(below_recorded_peak) == 5


def test_unknown_inputs_keep_fallback_and_stl_raw_records_still_cost_memory(tmp_path):
    assert estimate_input_memory(tmp_path / 'missing.ply') == 2_576_980_378  # 2.4 GiB
    indexed = estimate_decoded_memory(500_000, 1_000_000, 500_000)
    facets = estimate_decoded_memory(500_000, 1_000_000, 3_000_000)
    assert facets > indexed
    assert estimate_decoded_memory(1_000_000, 2_000_000, 6_000_000) > facets


def test_live_admission_uses_cpu_ceiling_and_honors_manual_limits():
    hardware = HardwareInfo(16, 8, 32 * GIB, available_memory_bytes=6 * GIB)
    static = allocate_resources(hardware, sample_count=6)
    live = allocate_resources(hardware, sample_count=6, live_memory_admission=True)
    assert static.max_concurrent_samples == 1
    assert live.max_concurrent_samples == 6
    assert live.threads_per_sample == 2
    # This diagnostic still describes startup RAM; it is not the live ceiling.
    assert live.memory_limited_samples == static.memory_limited_samples == 1
    manual = allocate_resources(hardware, sample_count=6, max_concurrent_samples=1,
                                threads_per_sample=4, live_memory_admission=True)
    assert (manual.max_concurrent_samples, manual.threads_per_sample) == (1, 4)
    single = allocate_resources(hardware, sample_count=1, live_memory_admission=True)
    assert (single.max_concurrent_samples, single.threads_per_sample) == (1, 2)


def test_automatic_batch_admits_more_workers_when_free_memory_recovers(tmp_path):
    state = MemorySnapshot(4 * GIB, 30 * GIB, 32 * GIB, 64 * GIB)
    hardware = HardwareInfo(16, 8, 32 * GIB, available_memory_bytes=4 * GIB)
    release = threading.Event()
    started = [threading.Event() for _ in range(3)]
    lock = threading.Lock()
    active = peak = 0

    def runner(job, control, progress):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active <= 3:
                started[active - 1].set()
        try:
            assert release.wait(10)
        finally:
            with lock:
                active -= 1

    scheduler, allocation = BatchScheduler.automatic(
        runner, sample_count=4, hardware=hardware, live_memory_admission=True)
    assert isinstance(scheduler.memory_admission, MemoryAdmission)
    scheduler.memory_admission.snapshot = lambda: state
    assert allocation.max_concurrent_samples == 4
    with scheduler:
        jobs = []
        for i in range(4):
            path = tmp_path / f'{i}.ply'
            path.write_text('ply\nformat ascii 1.0\nelement vertex 101102\n'
                            'element face 202202\nend_header\n')
            jobs.append(scheduler.submit(path, tmp_path / f'out-{i}'))
        scheduler.start()
        try:
            assert started[0].wait(3)
            assert not started[1].wait(.35)
            state = MemorySnapshot(6 * GIB, 30 * GIB, 32 * GIB, 64 * GIB)
            assert started[1].wait(3)
            assert not started[2].wait(.35)
            state = MemorySnapshot(8 * GIB, 30 * GIB, 32 * GIB, 64 * GIB)
            assert started[2].wait(3)
            assert sum(job.started_at is not None for job in jobs) == 3
            assert sum(job.state == BatchJobState.QUEUED for job in jobs) == 1
        finally:
            release.set()
        assert scheduler.wait(5)
        assert all(job.state == BatchJobState.COMPLETED for job in jobs)
        assert peak == 3


def test_nine_gib_available_admits_two_smaller_samples_beside_kaixinlv():
    # A running Kaixinlv worker occupies 1.5 GiB; the snapshot already excludes
    # it. The previous budgets admit one w5168-sized neighbor; the cut admits two.
    state = MemorySnapshot(16 * GIB, 30 * GIB, 32 * GIB, 64 * GIB)
    gate = MemoryAdmission(lambda: state)
    kai = estimate_decoded_memory(345297, 690590, 345297)
    small = estimate_decoded_memory(171120, 341914, 171120)
    assert gate.try_acquire('kaixinlv', kai)
    state = MemorySnapshot(9 * GIB, 30 * GIB, 32 * GIB, 64 * GIB)
    gate.update('kaixinlv', int(1.5 * GIB), kai)
    assert gate.try_acquire('w5168', small)
    assert gate.try_acquire('third', small)
    assert not gate.try_acquire('fourth', small)


def test_observed_growth_stops_further_admission():
    gate = MemoryAdmission(lambda: MemorySnapshot(12 * GIB, 12 * GIB, 32 * GIB, 64 * GIB))
    assert gate.try_acquire('a', 4 * GIB)
    gate.update('a', GIB, 10 * GIB)
    assert not gate.try_acquire('b', 4 * GIB)
    gate.release('a')
    assert gate.try_acquire('b', 4 * GIB)


def test_commit_headroom_and_reserved_growth_both_limit_admission():
    state = MemorySnapshot(30 * GIB, 9 * GIB, 32 * GIB, 64 * GIB)
    gate = MemoryAdmission(lambda: state)
    assert gate.try_acquire('a', 4 * GIB)
    assert not gate.try_acquire('b', 4 * GIB)
    # Allocated memory is already subtracted in the live snapshot, not twice.
    state = MemorySnapshot(28 * GIB, 8 * GIB, 32 * GIB, 64 * GIB)
    gate.update('a', 3 * GIB, 4 * GIB)
    assert gate.try_acquire('b', 4 * GIB)
    gate.release('a')
    gate.release('b')
    assert gate.try_acquire('c', 4 * GIB)


def test_unknown_capacity_admits_one_sample_and_keeps_paused_reservation():
    gate = MemoryAdmission(lambda: MemorySnapshot())
    assert gate.try_acquire('a', 4 * GIB)
    assert not gate.try_acquire('b', 4 * GIB)
    gate.release('a')
    assert gate.try_acquire('b', 4 * GIB)


def test_waiting_for_memory_stays_queued_and_can_be_cancelled(tmp_path):
    invoked = []
    gate = MemoryAdmission(lambda: MemorySnapshot(3 * GIB, 3 * GIB, 32 * GIB, 64 * GIB))
    with BatchScheduler(lambda *args: invoked.append(True), memory_admission=gate) as scheduler:
        job = scheduler.submit(tmp_path / 'sample.stl', tmp_path / 'out')
        scheduler.start()
        deadline = time.monotonic() + 3
        while job.step != 'Waiting for available memory' and time.monotonic() < deadline:
            time.sleep(.01)
        assert job.state == BatchJobState.QUEUED
        assert job.started_at is None
        assert scheduler.pause(job.job_id)
        scheduler.drain_events()
        assert scheduler.resume(job.job_id)
        deadline = time.monotonic() + 3
        refreshed = False
        while not refreshed and time.monotonic() < deadline:
            refreshed = any(event.job.step == 'Waiting for available memory' for event in scheduler.drain_events())
            time.sleep(.01)
        assert refreshed
        assert scheduler.cancel(job.job_id)
        assert scheduler.wait(3)
        assert job.state == BatchJobState.CANCELLED
        assert not invoked and not job.output_dir.exists()


def test_atomic_reservations_prevent_simultaneous_overadmission(tmp_path):
    state = MemorySnapshot(10 * GIB, 10 * GIB, 32 * GIB, 64 * GIB)
    gate = MemoryAdmission(lambda: state)
    active = peak = 0
    lock = threading.Lock()
    def runner(job, control, progress):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(active, peak)
        time.sleep(.1)
        with lock:
            active -= 1
    with BatchScheduler(runner, max_concurrent_samples=4, memory_admission=gate) as scheduler:
        jobs = [scheduler.submit(tmp_path / f'{i}.ply', tmp_path / str(i)) for i in range(4)]
        scheduler.start()
        assert scheduler.wait(5)
        assert all(job.state == BatchJobState.COMPLETED for job in jobs)
    assert peak == 3


def test_oversized_sample_fails_visibly_without_launch(tmp_path):
    gate = MemoryAdmission(lambda: MemorySnapshot(3 * GIB, 8 * GIB, 3 * GIB, 8 * GIB))
    with BatchScheduler(lambda *args: None, memory_admission=gate) as scheduler:
        job = scheduler.submit(tmp_path / 'sample.stl', tmp_path / 'out')
        scheduler.start()
        assert scheduler.wait(3)
        assert job.state == BatchJobState.FAILED
        assert job.started_at is None
        assert 'exceeds machine capacity' in job.error_log_path.read_text()
