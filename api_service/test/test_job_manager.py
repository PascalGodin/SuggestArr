"""
Tests for JobManager semaphore queuing behaviour.

Covers:
- Jobs run sequentially when triggered concurrently (semaphore enforced)
- A job is skipped cleanly if it cannot acquire the slot within the timeout
- Semaphore is always released even when the executor raises
- Jobs with missing data or no registered executor are rejected before queuing
"""

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

# Explicit import ensures api_service.jobs.job_manager is registered in
# sys.modules, which is required for patch() path resolution.
from api_service.jobs.job_manager import JobManager


def _make_manager():
    """Return a fresh, non-singleton JobManager with mocked dependencies."""
    JobManager._instance = None
    manager = JobManager()
    manager.repository = MagicMock()
    manager.scheduler = MagicMock()
    return manager


class TestJobManagerSemaphore(unittest.TestCase):

    def setUp(self):
        patcher = patch("api_service.jobs.job_manager.close_event_loop")
        self.mock_close = patcher.start()
        self.addCleanup(patcher.stop)

    def _fake_job_data(self, job_id=1, name="test-job", job_type="discover"):
        return {"id": job_id, "name": name, "job_type": job_type}

    # ------------------------------------------------------------------
    # Sequential execution
    # ------------------------------------------------------------------

    def test_concurrent_jobs_run_sequentially(self):
        """Two jobs fired at the same time must not overlap."""
        manager = _make_manager()

        execution_log = []
        lock = threading.Lock()

        async def slow_executor(jid):
            import asyncio
            with lock:
                execution_log.append(("start", jid))
            await asyncio.sleep(0.05)
            with lock:
                execution_log.append(("end", jid))

        manager.set_job_executor(slow_executor, "discover")
        manager.repository.get_job.side_effect = lambda jid: self._fake_job_data(jid)

        threads = [
            threading.Thread(target=manager._execute_job, args=(1,)),
            threading.Thread(target=manager._execute_job, args=(2,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(len(execution_log), 4, "All start/end events must be logged")

        # No overlap: every start must be paired with its end before the next start.
        for i in range(0, len(execution_log) - 1, 2):
            event, jid = execution_log[i]
            next_event, next_jid = execution_log[i + 1]
            self.assertEqual(event, "start")
            self.assertEqual(next_event, "end")
            self.assertEqual(jid, next_jid, "start/end must belong to the same job")

    # ------------------------------------------------------------------
    # Semaphore always released
    # ------------------------------------------------------------------

    def test_semaphore_released_after_executor_exception(self):
        """If the executor raises, the semaphore must still be released."""
        manager = _make_manager()

        async def boom(_jid):
            raise RuntimeError("executor blew up")

        manager.set_job_executor(boom, "discover")
        manager.repository.get_job.return_value = self._fake_job_data()

        manager._execute_job(1)

        self.assertEqual(manager._job_semaphore._value, 1)

    def test_semaphore_released_after_success(self):
        """After a normal run the semaphore slot is returned."""
        manager = _make_manager()

        async def noop(_jid):
            pass

        manager.set_job_executor(noop, "discover")
        manager.repository.get_job.return_value = self._fake_job_data()

        manager._execute_job(1)

        self.assertEqual(manager._job_semaphore._value, 1)

    # ------------------------------------------------------------------
    # Fast-fail before queuing
    # ------------------------------------------------------------------

    def test_missing_job_data_does_not_acquire_semaphore(self):
        """Unknown job_id should be rejected before touching the semaphore."""
        manager = _make_manager()
        manager.repository.get_job.return_value = None

        initial_value = manager._job_semaphore._value
        manager._execute_job(99)
        self.assertEqual(manager._job_semaphore._value, initial_value)

    def test_missing_executor_does_not_acquire_semaphore(self):
        """Unregistered job type should be rejected before touching the semaphore."""
        manager = _make_manager()
        manager.repository.get_job.return_value = self._fake_job_data(job_type="unknown")

        initial_value = manager._job_semaphore._value
        manager._execute_job(1)
        self.assertEqual(manager._job_semaphore._value, initial_value)

    # ------------------------------------------------------------------
    # Timeout / skip
    # ------------------------------------------------------------------

    def test_job_skipped_when_semaphore_not_available(self):
        """A job that cannot get the slot skips execution without calling the executor."""
        manager = _make_manager()

        called_executor = []

        async def tracking_executor(jid):
            called_executor.append(jid)

        manager.set_job_executor(tracking_executor, "discover")
        manager.repository.get_job.return_value = self._fake_job_data()

        with patch.object(manager._job_semaphore, "acquire", return_value=False):
            manager._execute_job(1)

        self.assertEqual(called_executor, [], "Executor must not run when slot is unavailable")


if __name__ == "__main__":
    unittest.main()
