#!/usr/bin/env python3
"""Local, bounded crash diagnostics running independently of ComfyUI."""
import argparse
import fcntl
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
from runtime_lifecycle import Lifecycle, identity, runtime_log_root


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def command(args):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=5)
        return dict(returncode=r.returncode, stdout=r.stdout[-65536:], stderr=r.stderr[-2048:])
    except (OSError, subprocess.TimeoutExpired) as e:
        return dict(error=type(e).__name__)


def logger(root, name):
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    handler = RotatingFileHandler(root / (name + '.jsonl'), maxBytes=10*1024**2, backupCount=4, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(message)s'))
    log.addHandler(handler)
    return log


def emit(log, **fields):
    log.info(json.dumps(dict(time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields), ensure_ascii=False))


def cgroup_path():
    for line in (read('/proc/self/cgroup') or '').splitlines():
        if line.startswith('0::'):
            path = Path('/sys/fs/cgroup') / line[3:].lstrip('/')
            if (path / 'memory.current').exists():
                return path
    return Path('/sys/fs/cgroup')


def processes():
    result = []
    for path in Path('/proc').glob('[0-9]*'):
        status = read(path / 'status')
        if not status:
            continue
        fields = dict(line.split(':', 1) for line in status.splitlines() if ':' in line)
        rss = int(fields.get('VmRSS', '0 kB').split()[0])
        name = fields.get('Name', '').strip()
        if rss < 100*1024 and not any(s in name.lower() for s in ('python', 'ffmpeg', 'jupyter')):
            continue
        result.append(dict(pid=int(path.name), name=name, rss_kib=rss,
                           state=fields.get('State', '').strip(), threads=fields.get('Threads', '').strip(),
                           oom_score=read(path / 'oom_score'), oom_score_adj=read(path / 'oom_score_adj')))
    return sorted(result, key=lambda row: row['rss_kib'], reverse=True)[:40]


def queue_status(port):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/queue', timeout=2) as r:
            data = json.load(r)
        return dict(reachable=True, running=len(data['queue_running']), pending=len(data['queue_pending']))
    except (OSError, ValueError, KeyError) as e:
        return dict(reachable=False, error=type(e).__name__)


def collect(args):
    root = Path(args.log_dir)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'collector.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    metrics, events, console, lifecycle_log = [logger(root, n) for n in ('metrics', 'events', 'comfyui', 'lifecycle')]
    from diagnose_runtime import recover, recover_previous_notebooks
    try:
        recover(root)
        recovered = recover_previous_notebooks(root)
        if recovered:
            emit(events, event='previous_notebook_evidence_saved', paths=recovered)
    except (OSError, ValueError) as error:
        emit(events, event='recovery_report_error', error_type=type(error).__name__)
    def lifecycle_emit(**fields):
        emit(lifecycle_log, **fields)
        for handler in lifecycle_log.handlers:
            handler.flush()
            os.fsync(handler.stream.fileno())
    lifecycle = Lifecycle(root, lifecycle_emit, identity())
    lifecycle.install_handlers()
    group = cgroup_path()
    emit(events, event='collector_start', pid=os.getpid(), boot_id=read('/proc/sys/kernel/random/boot_id'),
         cgroup=str(group), interval=args.interval,
         gpu=command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader']))
    offsets = {}
    previous = kernel_previous = health_previous = None
    next_kernel = 0
    while True:
        started = time.monotonic()
        try:
            memory = {k: read(group / k) for k in ('memory.current', 'memory.max', 'memory.peak', 'memory.events',
                       'memory.swap.current', 'memory.swap.max', 'memory.stat', 'memory.pressure', 'cpu.stat', 'io.stat')}
            queue = queue_status(args.port)
            state = (memory['memory.events'], queue['reachable'])
            if state != previous:
                emit(events, event='oom_or_liveness_change', memory_events=state[0], comfyui=queue)
                previous = state
            disks = {p: shutil.disk_usage(p)._asdict() for p in ('/notebooks', '/storage', '/app') if Path(p).exists()}
            emit(metrics, uptime=read('/proc/uptime'), meminfo=read('/proc/meminfo'), loadavg=read('/proc/loadavg'),
                 cgroup=memory, disks=disks, processes=processes(), comfyui=queue,
                 gpu=command(['nvidia-smi', '--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw', '--format=csv,noheader,nounits']),
                 gpu_processes=command(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits']))
            for name in ('comfyui.log', 'bootstrap.log'):
                source = Path(args.comfyui_dir) / 'user/logs' / name
                if not source.exists():
                    continue
                with source.open('rb') as stream:
                    stat = os.fstat(stream.fileno())
                    inode, offset = offsets.get(name, (stat.st_ino, max(0, stat.st_size - 1024**2)))
                    if inode != stat.st_ino or stat.st_size < offset:
                        offset = 0
                    stream.seek(offset)
                    chunk = stream.read(1024**2)
                    offsets[name] = (stat.st_ino, stream.tell())
                if chunk:
                    emit(console, source=name, text=chunk.decode('utf-8', errors='replace'))
            if started >= next_kernel:
                from platform_evidence import save as save_platform
                emit(events, event='platform_observation', observation=save_platform(root))
                emit(events, event='gpu_health', result=command(
                    ['nvidia-smi', '-q', '-d', 'TEMPERATURE,PERFORMANCE,ECC']))
                kernel = command(['dmesg', '--ctime'])
                if kernel != kernel_previous:
                    emit(events, event='kernel_log', result=kernel)
                    kernel_previous = kernel
                next_kernel = started + 60
        except Exception as e:
            emit(events, event='collector_error', error=type(e).__name__, message=str(e))
        lifecycle.heartbeat()
        from check_crash_diagnostics import coverage
        from runtime_lifecycle import write_state
        health = coverage(root, args.comfyui_dir)
        write_state(root/'coverage.json', health)
        missing = health['local_missing']
        if missing != health_previous:
            emit(events, event='diagnostic_coverage_changed', local_ready=health['local_ready'],
                 local_missing=missing, complete_cause_coverage=False)
            health_previous = missing
        for log in (metrics, events, console, lifecycle_log):
            for handler in log.handlers:
                handler.flush()
                try:
                    os.fsync(handler.stream.fileno())
                except OSError:
                    pass
        if args.once:
            lifecycle.close('once_complete')
            return
        deadline = started + args.interval
        while lifecycle.stop_signal is None and time.monotonic() < deadline:
            time.sleep(min(0.2, max(0, deadline - time.monotonic())))
        if lifecycle.stop_signal is not None:
            lifecycle.close('signal_received')
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-dir', default=str(runtime_log_root()))
    parser.add_argument('--comfyui-dir', default='/storage/ComfyUI')
    parser.add_argument('--port', type=int, default=6006)
    parser.add_argument('--interval', type=float, default=10)
    parser.add_argument('--start', action='store_true')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('--interval must be at least 1 second')
    if args.start:
        root = Path(args.log_dir)
        root.mkdir(parents=True, exist_ok=True)
        # Keep worker tracebacks instead of silently discarding failed collectors.
        launch_log = (root/'collector-launch.log').open('a')
        subprocess.Popen([sys.executable, str(Path(__file__).with_name('collect_file_memory.py')),
                          '--log-dir', args.log_dir, '--comfyui-dir', args.comfyui_dir],
                         stdin=subprocess.DEVNULL, stdout=launch_log,
                         stderr=subprocess.STDOUT, start_new_session=True)
        gpu_python = Path(args.comfyui_dir) / '.venv/bin/python'
        if gpu_python.exists():
            gpu_log = Path(args.log_dir) / 'gpu-monitor-launch.log'
            gpu_log.parent.mkdir(parents=True, exist_ok=True)
            with gpu_log.open('a') as stream:
                subprocess.Popen([str(gpu_python), str(Path(__file__).with_name('collect_gpu_events.py')),
                                  '--log-dir', args.log_dir], stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        subprocess.Popen([sys.executable, str(Path(__file__).with_name('collect_fast_runtime.py')),
                          '--log-dir', args.log_dir], stdin=subprocess.DEVNULL,
                         stdout=launch_log, stderr=subprocess.STDOUT, start_new_session=True)
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--log-dir', args.log_dir,
                          '--comfyui-dir', args.comfyui_dir, '--port', str(args.port), '--interval', str(args.interval)],
                         stdin=subprocess.DEVNULL, stdout=launch_log, stderr=subprocess.STDOUT,
                         start_new_session=True)
        launch_log.close()
        print(f'Runtime diagnostics launch requested: {args.log_dir}; readiness is verified separately')
    else:
        collect(args)


if __name__ == '__main__':
    main()
