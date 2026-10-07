import hashlib
from pathlib import Path

import pytest

from triwave.data import download
from triwave.data.download import ParallelDownload, checksum_map, file_digest, object_url, safe_path


def test_plus_filename_and_checksum_are_preserved(tmp_path: Path):
    payload = b"exact official object"
    path = tmp_path / "chb02_16+.edf"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert checksum_map(f"{expected}  ./chb02/chb02_16+.edf\n") == {
        "chb02/chb02_16+.edf": file_digest(path)}
    assert object_url("chb02/chb02_16+.edf").endswith("chb02/chb02_16%2B.edf")


@pytest.mark.parametrize("name", ["../other", "/tmp/file", "folder/../../other", ""])
def test_inventory_cannot_escape_dataset_root(name):
    with pytest.raises(ValueError, match="unsafe"):
        safe_path(name)


def test_parallel_ranges_finalize_only_after_whole_sha256(tmp_path, monkeypatch):
    payload = b"complete verified EEG bytes" * 20
    expected = hashlib.sha256(payload).hexdigest()
    entries = [{"path": "chb01/test.edf", "size": len(payload)}]
    monkeypatch.setattr(download, "inventory", lambda metadata: entries)
    monkeypatch.setattr(ParallelDownload, "fetch_metadata", lambda self, entries: {
        "chb01/test.edf": expected})

    class Response:
        status = 206

        def __init__(self, start, end):
            self.headers = {"Content-Range": f"bytes {start}-{end}/{len(payload)}"}
            self.value = payload[start:end+1]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, count):
            return self.value

    def fetch(request, timeout):
        span = request.headers["Range"].removeprefix("bytes=")
        start, end = map(int, span.split("-"))
        return Response(start, end)

    monkeypatch.setattr(download.urllib.request, "urlopen", fetch)
    report = ParallelDownload(tmp_path, connections=3, chunk_bytes=64).run()
    assert report["status"] == "complete" and report["completed_edfs"] == 1
    path = tmp_path / "raw/chb01/test.edf"
    assert path.read_bytes() == payload
    assert not list(path.parent.glob("*.part"))
    # The repeat call verifies completed files and makes no range requests.
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: pytest.fail("redownload"))
    assert ParallelDownload(tmp_path, chunk_bytes=64).run()["network_bytes_this_run"] == 0
