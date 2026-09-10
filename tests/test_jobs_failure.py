"""Job failure handling."""

from __future__ import annotations

import time

from avilistener.server.jobs import JobRegistry


def _wait_done(job, timeout=5.0):
    deadline = time.time() + timeout
    while job.status == "running" and time.time() < deadline:
        time.sleep(0.01)
    return job


def test_systemexit_in_a_job_is_a_failure_not_a_zombie(tmp_path):
    registry = JobRegistry()

    def work(log):
        raise SystemExit("pyannote Python not found")

    job = _wait_done(registry.start("diarize", "p/m", work, log_dir=tmp_path))
    assert job.status == "error"
    assert "pyannote Python not found" in (job.error or "")


def test_plain_exceptions_still_reported(tmp_path):
    registry = JobRegistry()

    def work(log):
        raise ValueError("boom")

    job = _wait_done(registry.start("transcribe", "p/m", work, log_dir=tmp_path))
    assert job.status == "error"
    assert "boom" in (job.error or "")


def test_simultaneous_requests_only_start_one_job():
    from concurrent.futures import ThreadPoolExecutor
    import threading

    registry = JobRegistry()
    barrier = threading.Barrier(8)
    release = threading.Event()
    def request(_):
        barrier.wait()
        try:
            return registry.start("transcribe", "p/m", lambda log: release.wait(5))
        except RuntimeError:
            return None
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(request, range(8)))
        assert sum(result is not None for result in results) == 1
    finally:
        release.set()
