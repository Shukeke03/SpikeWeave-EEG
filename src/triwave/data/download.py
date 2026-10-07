"""Verified parallel Range downloads from the official PhysioNet open S3 mirror."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

PREFIX = "chbmit/1.0.0/"
MIRRORS = ("https://physionet-open.s3.amazonaws.com/",
           "https://physionet-open.s3.us-east-1.amazonaws.com/")


def safe_path(value: str) -> str:
    value = value.removeprefix("./")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError("unsafe inventory path")
    return str(path)


def object_url(path: str, mirror: str = MIRRORS[0]) -> str:
    # S3 treats a raw '+' as a space in object lookup. Keep filenames exact.
    return mirror + urllib.parse.quote(PREFIX + safe_path(path), safe="/")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(part)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def inventory(metadata: Path) -> list[dict]:
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    entries = []
    continuation = None
    while True:
        query = {"list-type": "2", "prefix": PREFIX}
        if continuation:
            query["continuation-token"] = continuation
        url = MIRRORS[0] + "?" + urllib.parse.urlencode(query)
        with urllib.request.urlopen(url, timeout=30) as response:
            root = ET.fromstring(response.read())
        for entry in root.findall("s:Contents", ns):
            key = entry.find("s:Key", ns).text
            if key.endswith("/"):
                continue
            entries.append({"path": safe_path(key.removeprefix(PREFIX)),
                            "size": int(entry.find("s:Size", ns).text)})
        token = root.find("s:NextContinuationToken", ns)
        continuation = token.text if token is not None else None
        if not continuation:
            break
    metadata.mkdir(parents=True, exist_ok=True)
    _atomic_json(metadata / "official-inventory.json", {"source": MIRRORS[0],
                  "version": "1.0.0", "items": entries})
    return entries


def checksum_map(text: str) -> dict[str, str]:
    result = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
            raise ValueError("invalid SHA256 inventory")
        result[safe_path(name.lstrip("*"))] = digest.lower()
    return result


class ParallelDownload:
    def __init__(self, base: str | Path, connections: int = 24, chunk_bytes: int = 8 * 1024**2):
        if connections < 1 or chunk_bytes < 1:
            raise ValueError("connections and chunk size must be positive")
        self.base = Path(base)
        self.raw, self.metadata = self.base / "raw", self.base / "metadata"
        self.raw.mkdir(parents=True, exist_ok=True)
        self.metadata.mkdir(parents=True, exist_ok=True)
        self.connections, self.chunk_bytes = connections, chunk_bytes
        self.started, self.network_bytes, self.completed = time.monotonic(), 0, 0
        self.lock = threading.Lock()

    def fetch_metadata(self, entries: list[dict]) -> dict[str, str]:
        checksum_file = self.raw / "SHA256SUMS.txt"
        with urllib.request.urlopen(object_url("SHA256SUMS.txt"), timeout=30) as response:
            content = response.read()
        checksum_file.write_bytes(content)
        hashes = checksum_map(content.decode())

        def fetch(entry: dict) -> None:
            path = self.raw / entry["path"]
            expected = hashes.get(entry["path"])
            if path.is_file() and path.stat().st_size == entry["size"]:
                if expected is None or file_digest(path) == expected:
                    return
            path.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(object_url(entry["path"]), timeout=20) as response:
                        payload = response.read()
                    if len(payload) != entry["size"] or (expected and hashlib.sha256(payload).hexdigest() != expected):
                        raise ValueError(f"metadata integrity error: {entry['path']}")
                    temporary = path.with_name(path.name + ".part")
                    temporary.write_bytes(payload)
                    temporary.replace(path)
                    return
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(min(2**attempt, 8))
        with ThreadPoolExecutor(min(16, self.connections)) as pool:
            list(pool.map(fetch, (e for e in entries if not e["path"].endswith(".edf"))))
        return hashes

    def run(self, limit: int | None = None) -> dict:
        entries = inventory(self.metadata)
        hashes = self.fetch_metadata(entries)
        edfs = [e for e in entries if e["path"].endswith(".edf")]
        if limit is not None:
            edfs = edfs[:limit]
        total = sum(e["size"] for e in edfs)
        verified = {}
        state_path = self.metadata / "download-verified.json"
        if state_path.exists():
            verified = json.loads(state_path.read_text()).get("files", {})
        jobs = []
        for entry in edfs:
            name, size = entry["path"], entry["size"]
            if name not in hashes:
                raise ValueError(f"official checksum missing: {name}")
            path = self.raw / name
            if path.is_file() and path.stat().st_size == size:
                stamp = verified.get(name, {})
                if (stamp.get("sha256") == hashes[name]
                        and stamp.get("mtime_ns") == path.stat().st_mtime_ns) or file_digest(path) == hashes[name]:
                    verified[name] = {"sha256": hashes[name], "size": size,
                                      "mtime_ns": path.stat().st_mtime_ns}
                    self.completed += 1
                    continue
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + ".part")
            statefile = path.with_name(path.name + ".chunks.json")
            chunks = (size + self.chunk_bytes - 1) // self.chunk_bytes
            state = {"sha256": hashes[name], "size": size, "chunk_bytes": self.chunk_bytes,
                     "done": [False] * chunks}
            if partial.exists() and statefile.exists():
                prior = json.loads(statefile.read_text())
                if all(prior.get(k) == state[k] for k in ("sha256", "size", "chunk_bytes")):
                    state = prior
            descriptor = os.open(partial, os.O_CREAT | os.O_RDWR, 0o600)
            os.ftruncate(descriptor, size)
            os.close(descriptor)
            _atomic_json(statefile, state)
            item = {"entry": entry, "partial": partial, "statefile": statefile,
                    "state": state, "lock": threading.Lock()}
            for index in range(chunks):
                if not state["done"][index]:
                    jobs.append((item, index))
            if all(state["done"]):
                jobs.append((item, -1))

        def finalize(item: dict) -> None:
            entry = item["entry"]
            name = entry["path"]
            if file_digest(item["partial"]) != hashes[name]:
                item["state"]["done"] = [False] * len(item["state"]["done"])
                _atomic_json(item["statefile"], item["state"])
                raise ValueError(f"SHA256 mismatch, partial retained for retry: {name}")
            destination = self.raw / name
            item["partial"].replace(destination)
            item["statefile"].unlink(missing_ok=True)
            with self.lock:
                verified[name] = {"sha256": hashes[name], "size": entry["size"],
                                  "mtime_ns": destination.stat().st_mtime_ns}
                self.completed += 1
                _atomic_json(state_path, {"files": verified})

        def download(job: tuple) -> None:
            item, index = job
            entry = item["entry"]
            if index < 0:
                finalize(item)
                return
            start = index * self.chunk_bytes
            end = min(start + self.chunk_bytes, entry["size"]) - 1
            for attempt in range(8):
                try:
                    request = urllib.request.Request(object_url(entry["path"], MIRRORS[attempt % len(MIRRORS)]),
                                                       headers={"Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(request, timeout=15) as response:
                        expected_range = f"bytes {start}-{end}/{entry['size']}"
                        if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                            raise ValueError("mirror did not honor the exact byte range")
                        payload = response.read(end - start + 2)
                    if len(payload) != end - start + 1:
                        raise ValueError("truncated range response")
                    descriptor = os.open(item["partial"], os.O_WRONLY)
                    try:
                        cursor = 0
                        while cursor < len(payload):
                            cursor += os.pwrite(descriptor, payload[cursor:], start + cursor)
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                    with self.lock:
                        self.network_bytes += len(payload)
                    with item["lock"]:
                        item["state"]["done"][index] = True
                        _atomic_json(item["statefile"], item["state"])
                        if all(item["state"]["done"]):
                            finalize(item)
                    return
                except Exception as error:
                    if isinstance(error, ValueError) and str(error).startswith("SHA256 mismatch"):
                        raise
                    if attempt == 7:
                        raise
                    time.sleep(min(2**attempt, 10))

        errors = []
        stop = threading.Event()

        def progress() -> None:
            previous, last = 0, time.monotonic()
            while not stop.wait(10):
                now = time.monotonic()
                with self.lock:
                    current = self.network_bytes
                    status = {"completed_edfs": self.completed, "total_edfs": len(edfs),
                              "network_bytes_this_run": current, "total_edf_bytes": total,
                              "recent_MiB_s": round((current - previous) / (now - last) / 1024**2, 2),
                              "elapsed_seconds": round(now - self.started), "status": "downloading"}
                _atomic_json(self.metadata / "download-progress.json", status)
                print(json.dumps(status), flush=True)
                previous, last = current, now

        reporter = threading.Thread(target=progress, daemon=True)
        reporter.start()
        try:
            with ThreadPoolExecutor(self.connections) as pool:
                futures = [pool.submit(download, job) for job in jobs]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as error:
                        errors.append(str(error))
        finally:
            stop.set()
            reporter.join()
        if self.completed != len(edfs) and not errors:
            errors.append(f"only {self.completed}/{len(edfs)} files were checksum verified")
        report = {"status": "complete" if not errors else "incomplete", "completed_edfs": self.completed,
                  "total_edfs": len(edfs), "edf_bytes": total,
                  "official_total_bytes": sum(e["size"] for e in entries),
                  "network_bytes_this_run": self.network_bytes, "connections": self.connections,
                  "elapsed_seconds": round(time.monotonic() - self.started, 2), "errors": errors}
        _atomic_json(self.metadata / "download-report.json", report)
        _atomic_json(state_path, {"files": verified})
        print(json.dumps(report), flush=True)
        if errors:
            raise RuntimeError(f"{len(errors)} ranges failed; rerun resumes completed ranges")
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--connections", type=int, default=24)
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    ParallelDownload(args.base, args.connections, args.chunk_mib * 1024**2).run(args.limit)


if __name__ == "__main__":
    main()
