#!/usr/bin/env python3
"""Summarize recorded evidence near the last heartbeat, without guessing causes."""
import argparse
import datetime as dt
import json
from pathlib import Path

from runtime_lifecycle import load_state, runtime_log_root, write_state


def rows(root, name, since, until):
    for path in root.glob(name+'.jsonl*'):
        with path.open() as stream:
            for line in stream:
                try:
                    row=json.loads(line)
                    if since <= row.get('time','') <= until:yield row
                except ValueError:
                    continue


def classify(process_events, gpu_events, memory_events):
    evidence=[]
    for event in process_events:
        if event.get('event')=='child_exited':
            evidence.append(dict(kind='process_exit',returncode=event['returncode'],
                                 signal=event.get('signal'),forwarded_signals=event.get('forwarded_signals',[])))
    for event in gpu_events:
        if event.get('event')=='gpu_driver_event':
            evidence.append(dict(kind='gpu_driver_error',xid=event.get('xid'),time=event['time'],
                                 note='An Xid event is evidence of a driver-reported error, not proof it stopped the environment'))
        elif event.get('event')=='gpu_monitor_error':
            evidence.append(dict(kind='gpu_monitor_error',error_type=event.get('error_type')))
    counts={}
    for event in memory_events:
        groups=event.get('cgroups') or [dict(path='container',**event.get('cgroup',{}))]
        for group in groups:
            values=dict(line.split() for line in (group.get('memory.events') or '').splitlines())
            path=group.get('path','container')
            count=int(values.get('oom_kill',0))
            counts.setdefault(path,[]).append(count)
    for path,values in counts.items():
        if max(values)>min(values):
            evidence.append(dict(kind='oom_kill_counter_increased',cgroup=path,delta=max(values)-min(values),
                                 note='The counter does not identify the killed process'))
    return evidence


def diagnose(root):
    root=Path(root)
    state=load_state(root/'lifecycle-state.json')
    if not state or not state.get('last_seen'):return None
    end=dt.datetime.fromisoformat(state['last_seen'].replace('Z','+00:00'))
    since=(end-dt.timedelta(minutes=2)).strftime('%Y-%m-%dT%H:%M:%SZ')
    # Include exit flushes following the collector heartbeat.
    until=(end+dt.timedelta(seconds=15)).strftime('%Y-%m-%dT%H:%M:%SZ')
    processes=list(rows(root,'process',since,until))
    gpu=list(rows(root,'gpu_events',since,until))
    memory=sorted(list(rows(root,'fast_memory',since,until)) or list(rows(root,'metrics',since,until)),key=lambda r:r['time'])
    nodes=sorted(rows(root,'nodes',since,until),key=lambda r:r['time'])
    file_memory=sorted(rows(root,'file_memory',since,until),key=lambda r:r['time'])
    retention=sorted((r for r in rows(root,'memory_debug',since,until)
                      if r.get('event','').startswith('tensor_retention')),key=lambda r:r['time'])
    metrics=sorted(rows(root,'metrics',since,until),key=lambda r:r['time'])
    events=sorted(rows(root,'events',since,until),key=lambda r:r['time'])
    evidence=classify(processes,gpu,memory)
    report=dict(last_seen=state['last_seen'],notebook_id=state.get('notebook_id'),
                collector_clean_exit=state.get('clean_exit'),evidence=evidence,
                last_node={k:nodes[-1].get(k) for k in ('time','event','node_id','node_class','error_type')} if nodes else None,
                last_memory_sample=memory[-1] if memory else None,
                last_file_memory_sample=file_memory[-1] if file_memory else None,
                recent_tensor_retention=retention[-12:],
                recent_metrics=metrics[-6:], process_events=processes, gpu_events=gpu,
                recent_events=events[-12:],
                hook_activation=load_state(root/'hooks-state.json'),
                diagnostic_coverage=load_state(root/'coverage.json'),
                platform_observation=load_state(root/'platform-state.json'),
                supervisor_state=load_state(root/'supervisor-state.json'),
                auto_shutdown=state.get('auto_shutdown'),
                interpretation='Evidence summary only; absence of an exit record does not prove platform shutdown',
                provider_stop_reason='not_observable_with_current_permissions')
    return report


def recover(root):
    report=diagnose(root)
    if report is None:return
    path=Path(root)/'incidents'/report['last_seen'].replace(':','')
    path.mkdir(parents=True,exist_ok=True)
    if not (path/'evidence.json').exists():write_state(path/'evidence.json',report)
    return path/'evidence.json'


def recover_previous_notebooks(root):
    """Notebook IDs change across launches; recover the last stopped sessions too."""
    root = Path(root)
    if root.parent != runtime_log_root().parent:
        return []
    from runtime_lifecycle import identity
    current = identity()
    candidates = sorted(root.parent.glob('*/lifecycle-state.json'), key=lambda p:p.stat().st_mtime, reverse=True)
    recovered = []
    for path in candidates[:20]:
        state = load_state(path) or {}
        if path.parent == root or state.get('clean_exit') or not state.get('last_seen'):
            continue
        if all(state.get(k) == current[k] for k in current):
            continue
        last = dt.datetime.fromisoformat(state['last_seen'].replace('Z', '+00:00'))
        if (dt.datetime.now(dt.timezone.utc)-last).total_seconds() < 60:
            continue
        incident = recover(path.parent)
        if incident:
            recovered.append(str(incident))
    return recovered


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-dir',default=str(runtime_log_root()))
    args=parser.parse_args()
    print(json.dumps(diagnose(args.log_dir),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
