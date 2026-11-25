"""Replay actual raw capture timing with latest-frame drops during computation.

Example for the user's 14:27:13 recording:
python scripts/check_reentry_live.py runtime/reentry_20261113_142713.avi \
    --selection-frame 487 --box 161 59 352 443 --end 880
"""
import argparse
import csv
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from source_camera_tracker import SourceCameraTracker,ROOT
from hybrid_tracker import HybridTracker


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('--selection-frame',type=int,required=True)
    p.add_argument('--box',type=float,nargs=4,required=True)
    p.add_argument('--end',type=int)
    p.add_argument('--output',type=Path,default=ROOT/'runtime/latest_review/live_replay')
    p.add_argument('--expect-return',nargs=2,type=int,help='require recovery between these source frames')
    p.add_argument('--absent',nargs=2,type=int,help='assert no valid target in this fully absent interval')
    args=p.parse_args()
    cv2.setNumThreads(4)
    with args.source.with_suffix('.timestamps.csv').open(newline='') as handle:
        times=np.array([float(r['capture_monotonic_s']) for r in csv.DictReader(handle)])
    end=min(args.end or len(times),len(times))
    if not 0<=args.selection_frame<end:p.error('selection frame outside recording')
    capture=cv2.VideoCapture(str(args.source))
    if not capture.isOpened():raise RuntimeError(f'Cannot open {args.source}')
    base=SourceCameraTracker();base.warmup()
    tracker=HybridTracker(base,ROOT/'configs/small_target.json',ROOT/'configs/hybrid.json')
    stream=torch.cuda.Stream(priority=-1)
    args.output.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(args.output/'overlay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(640,480))
    if not writer.isOpened():raise RuntimeError('Cannot open output video')
    rows=[];index=args.selection_frame;clock=float(times[index]);identity=None
    try:
        while index<end:
            capture.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=capture.read()
            if not ok:raise RuntimeError(f'Cannot read frame {index}')
            tick=time.perf_counter()
            with torch.cuda.stream(stream):
                result=tracker.step(frame,float(times[index]),args.box if not rows else None)
            elapsed=time.perf_counter()-tick
            if not rows:identity=result.get('target_id')
            assert result.get('target_id')==identity,'Target ID changed during recovery'
            row={k:result.get(k) for k in ('status','transition_reason','valid','box','reacquired','proposal_search_ms','is_confirmed')}
            row.update(frame=index,time_s=float(times[index]-times[0]),step_ms=elapsed*1000,
                       discovery_class=tracker.discovery_class)
            rows.append(row)
            mask=result.get('mask');view=frame.copy()
            if mask is not None:view[mask]=(view[mask]*.6+[0,90,0]).astype('uint8')
            cv2.putText(view,f'{index} | {row["time_s"]:.2f}s | {result["status"]}',(10,25),0,.55,(0,220,255),2)
            writer.write(view)
            # Captures continue while the worker computes. Select the newest
            # available packet rather than replaying every skipped frame.
            clock=max(clock,float(times[index]))+elapsed
            index=max(index+1,int(np.searchsorted(times,clock,side='right')-1))
    finally:
        capture.release();writer.release();tracker.close()
    recovered=[r['frame'] for r in rows if r['status']=='RECOVERED']
    search=[r['proposal_search_ms'] for r in rows if r['proposal_search_ms'] is not None]
    report=dict(source=str(args.source),latest_frame_simulation=True,recovered=recovered,
                search_mean_ms=float(np.mean(search)) if search else None,
                search_p95_ms=float(np.percentile(search,95)) if search else None,rows=rows)
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print({k:v for k,v in report.items() if k!='rows'},flush=True)
    if args.expect_return:
        a,b=args.expect_return
        assert any(a<=i<=b for i in recovered),'No recovery in expected return window'
    if args.absent:
        a,b=args.absent;absent=[r for r in rows if a<=r['frame']<=b]
        assert absent and all(not r['valid'] for r in absent),'False target while absent'


if __name__=='__main__':main()
