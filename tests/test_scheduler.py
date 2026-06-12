from regintel.monitoring.scheduler import make_poll_job


class _OkMonitor:
    def __init__(self):
        self.calls = 0
    def poll(self, query, *, forms, limit):
        self.calls += 1
        return ["entry"]


class _RaisingMonitor:
    def poll(self, query, *, forms, limit):
        raise RuntimeError("poll boom")


def test_poll_job_calls_monitor():
    m = _OkMonitor()
    job = make_poll_job(m, query="q", forms=["8-K"], limit=5)
    job()
    assert m.calls == 1


def test_poll_job_swallows_exceptions():
    job = make_poll_job(_RaisingMonitor(), query="q", forms=["8-K"], limit=5)
    job()  # must not raise
