#!/usr/bin/env python3
"""Subscribe to NVML Xid events without requiring kernel log privileges."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import time

from collect_runtime_logs import emit, logger
from runtime_lifecycle import identity, runtime_log_root, write_state


def event_fields(event):
    return dict(event_type=int(event.eventType), event_data=int(event.eventData),
                xid=int(event.eventData) if int(event.eventType) & 8 else None,
                gpu_instance_id=int(event.gpuInstanceId), compute_instance_id=int(event.computeInstanceId))


def monitor(nvml, root, once=False):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'gpu-events.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        return
    log=logger(root,'gpu_events')
    stop=[]
    def record(**fields):
        emit(log,**fields)
        for handler in log.handlers:
            handler.flush();os.fsync(handler.stream.fileno())
    def halt(signum,frame):stop.append(signal.Signals(signum).name)
    old={s:signal.signal(s,halt) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP)}
    event_set=None
    initialized=False
    try:
        nvml.nvmlInit();initialized=True
        event_set=nvml.nvmlEventSetCreate()
        subscribed=[]
        for index in range(nvml.nvmlDeviceGetCount()):
            device=nvml.nvmlDeviceGetHandleByIndex(index)
            supported=nvml.nvmlDeviceGetSupportedEventTypes(device)
            mask=supported & nvml.nvmlEventTypeXidCriticalError
            if mask:
                nvml.nvmlDeviceRegisterEvents(device,mask,event_set)
                subscribed.append(index)
        record(event='gpu_subscription',devices=subscribed,identity=identity(),pid=os.getpid())
        if not subscribed:
            write_state(root/'gpu-events-state.json',dict(active=False,reason='Xid_not_supported',identity=identity()))
            return
        next_status=0
        while not stop:
            if time.monotonic()>=next_status:
                write_state(root/'gpu-events-state.json',dict(active=True,pid=os.getpid(),
                            identity=identity(),last_seen=time.time(),devices=subscribed))
                next_status=time.monotonic()+10
            try:
                event=nvml.nvmlEventSetWait(event_set,1000)
                record(event='gpu_driver_event',**event_fields(event))
            except nvml.NVMLError_Timeout:
                pass
            if once:break
    except nvml.NVMLError as error:
        record(event='gpu_monitor_error',error_type=type(error).__name__,nvml_code=error.value)
        write_state(root/'gpu-events-state.json',dict(active=False,reason=type(error).__name__,identity=identity()))
    finally:
        if event_set is not None:
            try:nvml.nvmlEventSetFree(event_set)
            except nvml.NVMLError:pass
        if initialized:
            try:nvml.nvmlShutdown()
            except nvml.NVMLError:pass
        for sig,handler in old.items():signal.signal(sig,handler)
        record(event='gpu_monitor_exit',signals=stop)
        write_state(root/'gpu-events-state.json',dict(active=False,identity=identity(),last_seen=time.time()))


def main():
    import pynvml
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-dir',default=str(runtime_log_root()))
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    monitor(pynvml,args.log_dir,args.once)


if __name__=='__main__':main()
