import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from runtime_lifecycle import Lifecycle, shutdown_setting, write_state


class LifecycleTests(unittest.TestCase):
    def test_unclean_restart_keeps_last_seen_and_unknown_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            events = []
            old = Lifecycle(directory, lambda **row: events.append(row),
                            dict(boot_id='a', container_id='a', notebook_id='n'))
            last_seen = old.state['last_seen']
            new = Lifecycle(directory, lambda **row: events.append(row),
                            dict(boot_id='b', container_id='b', notebook_id='n'))
            previous = next(row for row in events if row['event'] == 'previous_collector_session')
            self.assertEqual(previous['previous']['last_seen'], last_seen)
            self.assertEqual(previous['observation'], 'no_exit_record')
            self.assertTrue(previous['environment_changed'])
            self.assertEqual(previous['shutdown_reason'], 'unknown')
            new.close('once_complete')

    def test_manual_observation_does_not_leak_fields_or_cross_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'shutdown-observation.json'
            who = dict(notebook_id='n', container_id='a')
            write_state(path, dict(who, status='disabled', secret='never-log-this'))
            self.assertNotIn('secret', shutdown_setting(path, who))
            self.assertEqual(shutdown_setting(path, dict(who, container_id='b'))['status'], 'unknown')

    def test_collector_sigterm_and_restart(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/collect_runtime_logs.py'
        with tempfile.TemporaryDirectory() as directory:
            process = subprocess.Popen([sys.executable, str(script), '--log-dir', directory,
                                        '--comfyui-dir', directory, '--interval', '1'])
            state_path = Path(directory) / 'lifecycle-state.json'
            try:
                deadline = time.monotonic() + 15
                while not state_path.exists() and time.monotonic() < deadline:
                    time.sleep(.1)
                self.assertTrue(state_path.exists())
                # Handler installation follows the initial state write.
                time.sleep(.3)
                process.send_signal(signal.SIGTERM)
                self.assertEqual(process.wait(timeout=20), 0)
                state = json.loads(state_path.read_text())
                self.assertTrue(state['clean_exit'])
                self.assertEqual(state['signal'], 'SIGTERM')
                subprocess.run([sys.executable, str(script), '--log-dir', directory,
                                '--comfyui-dir', directory, '--once'], check=True, timeout=20,
                               stdout=subprocess.DEVNULL)
                rows = [json.loads(line) for line in (Path(directory)/'lifecycle.jsonl').read_text().splitlines()]
                previous = next(r for r in rows if r['event'] == 'previous_collector_session')
                self.assertEqual(previous['observation'], 'clean_exit')
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()


if __name__ == '__main__':
    unittest.main()
