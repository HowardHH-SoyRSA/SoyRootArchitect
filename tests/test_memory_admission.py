import struct
import threading
import time

from soyrootbio.batch import BatchScheduler, BatchJobState
from soyrootbio.hardware import GIB
from soyrootbio.memory_budget import MemoryAdmission, MemorySnapshot, estimate_input_memory


def test_binary_stl_budget_uses_facet_count_and_scales_with_geometry(tmp_path):
    small = tmp_path / 'small.stl'
    small.write_bytes(bytes(80) + struct.pack('<I', 1) + bytes(50))
    large = tmp_path / 'large.stl'
    with large.open('wb') as stream:
        stream.write(bytes(80) + struct.pack('<I', 1_000_000))
        stream.seek(84 + 50 * 1_000_000 - 1)
        stream.write(b'\0')
    assert estimate_input_memory(small) == 4 * GIB
    assert estimate_input_memory(large) > 9 * GIB


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
    assert peak == 2


def test_oversized_sample_fails_visibly_without_launch(tmp_path):
    gate = MemoryAdmission(lambda: MemorySnapshot(3 * GIB, 8 * GIB, 3 * GIB, 8 * GIB))
    with BatchScheduler(lambda *args: None, memory_admission=gate) as scheduler:
        job = scheduler.submit(tmp_path / 'sample.stl', tmp_path / 'out')
        scheduler.start()
        assert scheduler.wait(3)
        assert job.state == BatchJobState.FAILED
        assert job.started_at is None
        assert 'exceeds machine capacity' in job.error_log_path.read_text()
