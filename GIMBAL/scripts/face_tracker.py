"""Explicit face-only tracking with spatial association and bounded SAM masks."""
from pathlib import Path
import cv2
import numpy as np


def overlap(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right-left)*max(0, bottom-top)
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/max(1, union)


def face_feature(frame, box):
    """Normalized central face texture, excluding the surrounding hands/background."""
    h, w = frame.shape[:2]
    x, y, r, b = box
    dx, dy = (r-x)*.15, (b-y)*.12
    crop = frame[max(0, int(y+dy)):min(h, int(b-dy)),
                 max(0, int(x+dx)):min(w, int(r-dx))]
    if not crop.size:
        return None
    gray = cv2.cvtColor(cv2.resize(crop, (32, 32)), cv2.COLOR_BGR2GRAY)
    feature = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32).ravel()
    feature -= feature.mean()
    norm = np.linalg.norm(feature)
    return feature/norm if norm > 1e-6 else None


class FaceTracker:
    target_mode = 'FACE'

    def __init__(self, base):
        self.base = base
        path = Path(__file__).resolve().parents[1]/'data/models/face_detection_yunet_2023mar.onnx'
        if not path.is_file():
            raise RuntimeError(f'Face detector missing: {path}')
        self.detector = cv2.FaceDetectorYN.create(str(path), '', (320, 320), .85, .3, 5000)
        self.reset()

    def reset(self):
        self.base.reset()
        self.anchor = None
        self.misses = 0
        self.references = []
        self.pending_box = None
        self.pending_hits = 0
        self.reacquired = 0
        self.shape = None

    def similarity(self, feature):
        if feature is None or not self.references:
            return -1.
        return max(float(np.dot(feature, ref)) for ref in self.references)

    def step(self, frame, stamp, box=None, mask_prompt=None):
        if box is not None:
            self.reset()
            self.anchor = list(box)
            self.shape = frame.shape[:2]
        elif self.shape is not None and self.shape != frame.shape[:2]:
            self.reset()
        self.detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = self.detector.detect(frame)
        candidates = [[int(x), int(y), int(x+w), int(y+h)]
                      for x,y,w,h in (faces[:, :4] if faces is not None else [])]
        chosen = None
        recovering = self.misses > 0 and bool(self.references)
        if recovering:
            # Search every detected face, even after a long occlusion or movement.
            ranked = sorted(((self.similarity(face_feature(frame, b)), b) for b in candidates),
                            key=lambda item: item[0], reverse=True)
            if ranked and ranked[0][0] >= .65 and (len(ranked) == 1 or ranked[0][0]-ranked[1][0] >= .08):
                proposal = ranked[0][1]
                self.pending_hits = self.pending_hits+1 if self.pending_box is not None and overlap(self.pending_box, proposal) >= .3 else 1
                self.pending_box = proposal
                if self.pending_hits >= 3:
                    chosen = proposal
            else:
                self.pending_box = None
                self.pending_hits = 0
        elif self.anchor is not None:
            a = self.anchor
            if a[0] == a[2] and a[1] == a[3]:
                candidates = [b for b in candidates if b[0] <= a[0] <= b[2] and b[1] <= a[1] <= b[3]]
                chosen = min(candidates, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), default=None)
            else:
                chosen = max(candidates, key=lambda b: overlap(a, b), default=None)
                if chosen is not None and overlap(a, chosen) < .3:
                    chosen = None
                # Once initialized, geometry alone must not transfer the target
                # to a different face passing through the old location.
                if chosen is not None and self.references and self.similarity(face_feature(frame, chosen)) < .5:
                    chosen = None
        if chosen is None:
            self.misses += 1
            self.base.reset()
            return dict(status='FACE_NOT_FOUND', valid=False, mask=None, box=None,
                        dx=None, dy=None, control_gain=0., captured=stamp,
                        inference_ms=0., predictor_called=False, target_mode=self.target_mode,
                        transition_reason='face_recovery_confirming' if self.pending_hits else 'face_not_detected_searching')
        if recovering:
            self.reacquired += 1
        self.anchor = chosen
        self.misses = 0
        self.pending_box = None
        self.pending_hits = 0
        feature = face_feature(frame, chosen)
        if feature is not None:
            if not self.references:
                self.references.append(feature.copy())
            elif not recovering and self.similarity(feature) >= .8:
                # Keep the original anchor plus a recent verified view.
                self.references = [self.references[0], feature.copy()]
        # Re-prompt on verified face bounds; contaminated hand memory is not reused.
        result = self.base.step(frame, stamp, chosen)
        mask = result.get('mask')
        if mask is not None:
            x,y,r,b = chosen
            support = np.zeros(frame.shape[:2], np.uint8)
            cv2.ellipse(support, ((x+r)//2, (y+b)//2),
                        (max(1,(r-x)//2), max(1,(b-y)//2)), 0, 0, 360, 1, -1)
            mask = mask & support.astype(bool)
        valid = mask is not None and bool(mask.any())
        result.update(mask=mask if valid else None, box=chosen if valid else None,
                      valid=valid, target_mode=self.target_mode,
                      reacquired=self.reacquired,
                      transition_reason='face_reacquired' if recovering else 'face_verified',
                      dx=(chosen[0]+chosen[2]-frame.shape[1])/2 if valid else None,
                      dy=(chosen[1]+chosen[3]-frame.shape[0])/2 if valid else None,
                      control_gain=1. if valid else 0.)
        return result

    def label(self, *args, **kwargs):
        return self.base.label(*args, **kwargs)

    def close(self):
        self.base.close()
