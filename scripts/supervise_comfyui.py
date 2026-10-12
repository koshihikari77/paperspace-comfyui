#!/usr/bin/env python3
"""Record child exit status and fatal Python traces without restarting generation."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time

from collect_runtime_logs import cgroup_path, emit, logger, read
from runtime_lifecycle import identity, runtime_log_root, write_state


def counters(group):
    return {name: read(group / name) for name in ('memory.current', 'memory.max', 'memory.events')}


def supervise(command, cwd, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    log = logger(root, 'process')
    group = cgroup_path()
    initial = counters(group)
    child = None
    signals = []
    def record(**fields):
        emit(log, **fields)
        for handler in log.handlers:
            handler.flush()
            os.fsync(handler.stream.fileno())
    def forward(signum, frame):
        signals.append(signal.Signals(signum).name)
        record(event='supervisor_signal_received', signal=signals[-1],
               child_pid=child.pid if child is not None else None)
        if child is not None and child.poll() is None:
            child.send_signal(signum)
    old = {s: signal.signal(s, forward) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    try:
        child = subprocess.Popen(command, cwd=cwd, env={**os.environ,
            'PYTHONFAULTHANDLER': '1', 'PYTHONUNBUFFERED': '1',
            'KOSHI_RUNTIME_LOG_DIR': str(root)}, start_new_session=True)
        record(event='child_started', pid=child.pid, supervisor_pid=os.getpid(),
               identity=identity(), memory=initial, fatal_trace='comfyui.log')
        write_state(root / 'supervisor-state.json', dict(pid=child.pid, supervisor_pid=os.getpid(),
                    identity=identity(), status='running'))
        code = child.wait()
        record(event='child_exited', pid=child.pid, returncode=code,
               signal=signal.Signals(-code).name if code < 0 else None,
               forwarded_signals=signals, memory_before=initial, memory_after=counters(group),
               note='SIGKILL alone does not prove OOM')
        write_state(root / 'supervisor-state.json', dict(pid=child.pid, supervisor_pid=os.getpid(),
                    identity=identity(), status='exited', returncode=code))
        return code if code >= 0 else 128 - code
    except OSError as error:
        record(event='supervisor_error', error_type=type(error).__name__, errno=error.errno)
        return 1
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cwd', required=True)
    parser.add_argument('--log-dir', default=str(runtime_log_root()))
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('child command required')
    return supervise(command, args.cwd, args.log_dir)


if __name__ == '__main__':
    raise SystemExit(main())
