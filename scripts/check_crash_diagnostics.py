#!/usr/bin/env python3
"""Verify running diagnostics, not installed files or historical markers."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from runtime_lifecycle import identity, load_state, process_start_ticks, runtime_log_root, utc_now, write_state

LOCAL_REQUIREMENTS = ('collector_active', 'one_second_memory', 'file_memory_collector',
                      'gpu_xid_subscription', 'supervisor_active', 'node_hooks_loaded',
                      'node_debug_enabled', 'tensor_retention_hooks_loaded', 'loaded_sources_current')


def process_matches(pid, script):
    try:
        args = Path('/proc', str(pid), 'cmdline').read_bytes().split(b'\0')
        return any(arg.endswith(('/'+script).encode()) for arg in args[:3])
    except OSError:
        return False


def fresh(path, seconds=30):
    try:
        return 0 <= time.time()-path.stat().st_mtime < seconds
    except OSError:
        return False


def live_collector(root, name, script):
    path = root/(name+'-state.json')
    state = load_state(path) or {}
    ticks = process_start_ticks(state.get('pid'))
    return (state.get('active') is True and state.get('identity') == identity()
            and ticks is not None and ticks == state.get('process_start_ticks')
            and process_matches(state.get('pid'), script) and fresh(path))


def coverage(root, comfyui_dir):
    root = Path(root)
    current = identity()
    state = load_state(root/'lifecycle-state.json') or {}
    supervisor = load_state(root/'supervisor-state.json') or {}
    gpu = load_state(root/'gpu-events-state.json') or {}
    hooks = load_state(root/'hooks-state.json') or {}
    active = (supervisor.get('identity') == current and supervisor.get('status') == 'running'
              and process_matches(supervisor.get('supervisor_pid'), 'supervise_comfyui.py')
              and process_matches(supervisor.get('pid'), 'main.py'))
    ticks = process_start_ticks(supervisor.get('pid'))
    hooks_live = (active and hooks.get('pid') == supervisor.get('pid')
                  and hooks.get('boot_id') == current['boot_id'] and ticks is not None
                  and ticks == hooks.get('process_start_ticks'))
    expected = {}
    try:
        node = Path(comfyui_dir)/'custom_nodes/runtime_diagnostics'
        expected = {name: hashlib.sha256((node/name).read_bytes()).hexdigest()
                    for name in ('__init__.py', 'memory_debug.py', 'tensor_retention.py')}
    except OSError:
        pass
    file_state = load_state(root/'file-memory-state.json') or {}
    platform = load_state(root/'platform-state.json') or {}
    report = dict(time=utc_now(), identity=current, comfyui_pid=supervisor.get('pid'),
        collector_active=(all(state.get(k) == current[k] for k in current)
                          and process_matches(state.get('pid'), 'collect_runtime_logs.py')
                          and fresh(root/'lifecycle-state.json')),
        one_second_memory=live_collector(root, 'fast', 'collect_fast_runtime.py'),
        file_memory_collector=(live_collector(root, 'file-memory', 'collect_file_memory.py')
                               and supervisor.get('pid') in file_state.get('sampled_pids', [])),
        supervisor_active=active,
        gpu_xid_subscription=(gpu.get('active') is True and gpu.get('identity') == current
                              and process_matches(gpu.get('pid'), 'collect_gpu_events.py')
                              and fresh(root/'gpu-events-state.json')),
        node_hooks_loaded=bool(hooks_live and hooks.get('node_hooks_loaded')),
        tensor_retention_hooks_loaded=bool(hooks_live and hooks.get('tensor_retention_hooks_loaded')),
        loaded_sources_current=bool(hooks_live and expected and hooks.get('source_hashes') == expected),
        node_debug_enabled=(load_state(root/'mode.json') or {}).get('mode') == 'debug',
        auto_shutdown_observation=state.get('auto_shutdown', {'status': 'unknown'}),
        platform_observation=platform if platform.get('identity') == current else None,
        complete_cause_coverage=False)
    report['local_missing'] = [name for name in LOCAL_REQUIREMENTS if not report[name]]
    report['local_ready'] = not report['local_missing']
    report['uncovered_causes'] = ['Provider shutdown decision without provider history',
                                'Host failure or storage outage stopping all local observers']
    return report


def verify(root, comfyui_dir, wait=0):
    deadline = time.monotonic()+wait
    while True:
        report = coverage(root, comfyui_dir)
        write_state(Path(root)/'coverage.json', report)
        if report['local_ready'] or time.monotonic() >= deadline:
            return report
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-dir', default=str(runtime_log_root()))
    parser.add_argument('--comfyui-dir', default='/storage/ComfyUI')
    parser.add_argument('--wait', type=float, default=0)
    args = parser.parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    report = verify(args.log_dir, args.comfyui_dir, args.wait)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['local_ready'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
