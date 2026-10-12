import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from runtime_lifecycle import runtime_log_root
from watch_remote_runtime import observe
from unittest.mock import patch


class CrashEvidenceTests(unittest.TestCase):
    def run_child(self, code):
        with tempfile.TemporaryDirectory() as root:
            result = subprocess.run([sys.executable, str(SCRIPTS/'supervise_comfyui.py'),
                '--cwd', root, '--log-dir', root, '--', sys.executable, '-c', code],
                capture_output=True, timeout=10)
            events = [json.loads(x) for x in (Path(root)/'process.jsonl').read_text().splitlines()]
            return result, events[-1]

    def test_exit_code(self):
        result, event = self.run_child('raise SystemExit(23)')
        self.assertEqual(result.returncode, 23)
        self.assertEqual(event['returncode'], 23)
        self.assertIsNone(event['signal'])

    def test_kill_not_mislabelled_oom(self):
        result, event = self.run_child('import os,signal;os.kill(os.getpid(),signal.SIGKILL)')
        self.assertEqual(result.returncode, 137)
        self.assertEqual(event['signal'], 'SIGKILL')
        self.assertEqual(event['returncode'], -9)
        self.assertNotIn('oom_reason', event)

    def test_fatal_trace(self):
        result, event = self.run_child('import os,signal;os.kill(os.getpid(),signal.SIGABRT)')
        self.assertEqual(event['signal'], 'SIGABRT')
        self.assertIn(b'Fatal Python error', result.stderr)

    def test_forward_term(self):
        with tempfile.TemporaryDirectory() as root:
            process = subprocess.Popen([sys.executable, str(SCRIPTS/'supervise_comfyui.py'),
                '--cwd', root, '--log-dir', root, '--', sys.executable, '-c', 'import time;time.sleep(60)'])
            try:
                state = Path(root)/'supervisor-state.json'
                deadline=time.monotonic()+5
                while not state.exists() and time.monotonic()<deadline:time.sleep(.05)
                self.assertTrue(state.exists())
                process.send_signal(signal.SIGTERM)
                self.assertEqual(process.wait(timeout=5),143)
                last=json.loads((Path(root)/'process.jsonl').read_text().splitlines()[-1])
                self.assertEqual(last['forwarded_signals'],['SIGTERM'])
            finally:
                if process.poll() is None:
                    process.kill();process.wait()

    def test_fast_memory(self):
        with tempfile.TemporaryDirectory() as root:
            subprocess.run([sys.executable,str(SCRIPTS/'collect_fast_runtime.py'),'--log-dir',root,'--once'],
                           check=True,timeout=5)
            rows=[json.loads(x) for x in (Path(root)/'fast_memory.jsonl').read_text().splitlines()]
            sample=next(x for x in rows if x['event']=='memory_sample')
            self.assertTrue(sample['cgroups'])
            self.assertIn('memory.events',sample['cgroups'][0])

    def test_notebook_isolation(self):
        with patch.dict(os.environ,{'PAPERSPACE_NOTEBOOK_ID':'n_one'}):a=runtime_log_root()
        with patch.dict(os.environ,{'PAPERSPACE_NOTEBOOK_ID':'n_two'}):b=runtime_log_root()
        self.assertNotEqual(a,b)

    def test_external_observer_stale_and_network_failure(self):
        state = json.dumps({'last_seen':'2000-01-01T00:00:00Z'})
        with patch('watch_remote_runtime.subprocess.run', return_value=
                   subprocess.CompletedProcess([], 0, state, '')):
            self.assertEqual(observe('host', 'notebook')['status'], 'heartbeat_stale')
        with patch('watch_remote_runtime.subprocess.run', side_effect=subprocess.TimeoutExpired('ssh',8)):
            result=observe('host','notebook')
            self.assertEqual(result['status'],'ssh_unavailable')
            self.assertNotIn('shutdown_reason',result)


if __name__ == '__main__':unittest.main()
