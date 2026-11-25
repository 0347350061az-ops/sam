"""Derived bottle exit/return fixture from the cropped user recording.

The recording has an exit but no return. Replay its earlier visible frames,
translated horizontally, after an absent interval. This is a constructed
regression fixture, not a natural recorded return or an accuracy benchmark.
"""
import argparse
import json
import cv2
import numpy as np
from source_camera_tracker import SourceCameraTracker, ROOT
from hybrid_tracker import HybridTracker


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',help='640x480 LIVE PREVIEW crop of the supplied recording')
    args=p.parse_args()
    cv2.setNumThreads(4)
    base=SourceCameraTracker();base.warmup()
    tracker=HybridTracker(base,ROOT/'configs/small_target.json',ROOT/'configs/hybrid.json')
    capture=cv2.VideoCapture(args.source)
    if not capture.isOpened():raise RuntimeError(args.source)
    out=ROOT/'runtime/recording_review/return_verified';out.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(out/'overlay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(640,480))
    if not writer.isOpened():raise RuntimeError('Cannot create overlay')
    indices=list(range(210,443,3))+[440]*20+list(range(330,409,3))*2
    return_start=98;rows=[];target_id=None
    try:
        for n,i in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES,i);ok,frame=capture.read()
            if not ok:raise RuntimeError(f'Cannot read frame {i}')
            if n>=return_start:
                frame=cv2.warpAffine(frame,np.float32([[1,0,180],[0,1,0]]),(640,480))
            result=tracker.step(frame,n*.1,[120,131,275,448] if n==0 else None)
            if n==0:target_id=result['target_id']
            assert result['target_id']==target_id
            row={k:result.get(k) for k in ('status','transition_reason','valid','box','reacquired','proposal_top1','proposal_top2')}
            row.update(sample=n,source_frame=i);rows.append(row)
            mask=result.get('mask')
            if mask is not None:frame[mask]=(frame[mask]*.6+[0,90,0]).astype('uint8')
            cv2.putText(frame,f'{n*.1:.1f}s | {result["status"]}',(10,25),0,.6,(0,220,255),2)
            writer.write(frame)
    finally:
        writer.release();capture.release();tracker.close()
    recovered=[r['sample'] for r in rows[return_start:] if r['status']=='RECOVERED']
    report=dict(constructed_return=True,return_start=return_start,recovered=recovered,rows=rows)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    assert all(r['valid'] for r in rows[:67]),'Lost bottle before exit'
    assert all(not r['valid'] for r in rows[78:return_start]),'Tracked while absent'
    assert recovered,'Returning bottle was never recovered'
    assert recovered[0]-return_start<20,'Recovery exceeded two seconds in the derived fixture'
    print(dict(recovered=recovered,delay_seconds=(recovered[0]-return_start)*.1,output=str(out)))


if __name__=='__main__':main()
