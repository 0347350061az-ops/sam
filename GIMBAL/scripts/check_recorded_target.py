"""Replay a selected target from a recording through the real GPU tracker.

Screen crops are x,y,width,height; boxes are x1,y1,x2,y2 after resizing to 640x480.
Visibility assertions check tracker state, not segmentation IoU or identity accuracy.
"""
import argparse
import json
from pathlib import Path
import cv2
from source_camera_tracker import SourceCameraTracker, ROOT
from hybrid_tracker import HybridTracker


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source')
    p.add_argument('--crop', nargs=4, type=int)
    p.add_argument('--box', nargs=4, type=float, required=True)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int)
    p.add_argument('--stride', type=int, default=3)
    p.add_argument('--visible-through', type=int)
    p.add_argument('--absent-from', type=int)
    p.add_argument('--grayscale-search',action='store_true',help='A/B diagnostic: disable chromatic proposal matching')
    p.add_argument('--output', type=Path, default=ROOT/'runtime/recording_review/verified')
    args = p.parse_args()
    if args.stride < 1:
        p.error('--stride must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(args.source)
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open {args.source}')
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.
    end = args.end or int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2.setNumThreads(4)
    base = SourceCameraTracker()
    base.warmup()
    tracker = HybridTracker(base, ROOT/'configs/small_target.json', ROOT/'configs/hybrid.json')
    if args.grayscale_search:tracker.hybrid.recovery_color=False
    writer = cv2.VideoWriter(str(args.output/'overlay.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                             fps/args.stride, (640, 480))
    if not writer.isOpened():
        capture.release()
        tracker.close()
        raise RuntimeError('Cannot create review video')
    rows = []
    try:
        for i in range(args.start, end, args.stride):
            capture.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f'Cannot read frame {i}')
            if args.crop:
                x,y,w,h = args.crop
                frame = frame[y:y+h, x:x+w]
            frame = cv2.resize(frame, (640,480))
            result = tracker.step(frame, (i-args.start)/fps, args.box if not rows else None)
            row = {k:result.get(k) for k in ('status','transition_reason','valid','box','target_id','reacquired')}
            row['frame'] = i
            rows.append(row)
            view = frame.copy()
            mask = result.get('mask')
            if mask is not None:
                view[mask] = (view[mask]*.6+[0,90,0]).astype('uint8')
            cv2.putText(view, f'{i/fps:.1f}s | {result["status"]}', (10,25), 0,.6,(0,220,255),2)
            writer.write(view)
    finally:
        capture.release()
        writer.release()
        tracker.close()
    report = dict(source=args.source, kalman=tracker.hybrid.kalman,recovery_color=tracker.hybrid.recovery_color, rows=rows)
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    visible = [r for r in rows if args.visible_through is not None and r['frame'] <= args.visible_through]
    absent = [r for r in rows if args.absent_from is not None and r['frame'] >= args.absent_from]
    if args.visible_through is not None:
        assert visible and all(r['valid'] for r in visible), 'Lost target while annotated visible'
    if args.absent_from is not None:
        assert absent and all(not r['valid'] for r in absent), 'Continued tracking after target exit'
    print(dict(sampled_frames=len(rows), visible_confirmed=sum(r['valid'] for r in visible),
               visible_samples=len(visible), absent_samples=len(absent), output=str(args.output)))


if __name__ == '__main__':
    main()
