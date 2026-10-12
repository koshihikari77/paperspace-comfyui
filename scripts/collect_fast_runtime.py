#!/usr/bin/env python3
"""One-second memory evidence, independent of GPU commands and HTTP requests."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import time

from collect_runtime_logs import cgroup_path, emit, logger, read
from runtime_lifecycle import collector_heartbeat, identity, runtime_log_root


def memory_chain(group):
    rows = []
    while str(group).startswith('/sys/fs/cgroup'):
        data = {key: read(group / key) for key in
                ('memory.current', 'memory.max', 'memory.peak', 'memory.events', 'memory.events.local', 'memory.pressure')}
        if data['memory.current'] is not None:
            rows.append(dict(path=str(group), **data))
        if group == Path('/sys/fs/cgroup'):
            break
        group = group.parent
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-dir', default=str(runtime_log_root()))
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    root = Path(args.log_dir)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'fast.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    log = logger(root, 'fast_memory')
    stopping = []
    def stop(signum, frame):
        stopping.append(signal.Signals(signum).name)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stop)
    group = cgroup_path()
    emit(log, event='fast_collector_start', pid=os.getpid(), identity=identity())
    while not stopping:
        started = time.monotonic()
        memory = {line.split(':')[0]: line.split(':')[1].strip()
                  for line in (read('/proc/meminfo') or '').splitlines()
                  if line.startswith(('MemAvailable:', 'MemFree:', 'SwapFree:'))}
        emit(log, event='memory_sample', monotonic=started, cgroups=memory_chain(group),
             host=memory, host_pressure=read('/proc/pressure/memory'))
        for handler in log.handlers:
            handler.flush()
            os.fsync(handler.stream.fileno())
        collector_heartbeat(root, 'fast', active=True)
        if args.once:
            break
        while not stopping and time.monotonic()-started < 1:
            time.sleep(.05)
    emit(log, event='fast_collector_exit', signals=stopping)
    collector_heartbeat(root, 'fast', active=False)
    for handler in log.handlers:
        handler.flush()
        os.fsync(handler.stream.fileno())


if __name__ == '__main__':
    main()
