import hashlib

from soyrootbio.provenance import file_sha256, processing_provenance


def test_provenance_identifies_actual_source_and_input(tmp_path):
    source = tmp_path / "root.stl"
    source.write_bytes(b"native-root-geometry")
    evidence = processing_provenance(source, {"gravity": [0, 0, -1]})
    assert evidence["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert evidence["source_files_sha256"]["provenance.py"]
    assert len(evidence["source_tree_sha256"]) == 64
    assert evidence["input_size_bytes"] == source.stat().st_size
    assert file_sha256(source) == evidence["input_sha256"]
    changed = processing_provenance(source, {"gravity": [0, -1, 0]})
    assert changed["config_sha256"] != evidence["config_sha256"]
