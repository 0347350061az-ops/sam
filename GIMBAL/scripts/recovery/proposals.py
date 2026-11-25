"""Multiscale structural proposals for arbitrary user-selected targets."""
import cv2
import numpy as np


def overlap(a, b):
    a, b = np.asarray(a), np.asarray(b)
    intersection = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    return float(intersection/max(1., np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection))


def same_location(a, b):
    """Nested scale proposals at one center are one physical competitor."""
    if overlap(a, b) >= .25:
        return True
    a, b = np.asarray(a), np.asarray(b)
    area = min(np.prod(a[2:]-a[:2]), np.prod(b[2:]-b[:2]))
    intersection = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    distance = np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)
    extent = max(np.linalg.norm(a[2:]-a[:2]), np.linalg.norm(b[2:]-b[:2]))
    return intersection/max(1., area) >= .8 and distance <= .2*extent


class CandidateGenerator:
    def __init__(self):
        self.fallback_turn=0

    def dark_region_proposals(self,memory,frame):
        """Class-free discovery for smooth dark targets without texture peaks."""
        if getattr(memory,'original_hue',None) is not None or getattr(memory,'original_value',255)>=80:
            return []
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        binary=(gray<max(55.,memory.original_value*1.8)).astype(np.uint8)
        binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        count,_,stats,_=cv2.connectedComponentsWithStats(binary,8)
        original_area=int(memory.mask.sum());candidates=[]
        for x,y,w,h,area in stats[1:]:
            if area<max(32,original_area*.05) or area>original_area*6 or min(w,h)<6:continue
            if max(w,h)>min(w,h)*4:continue
            candidates.append(dict(box=[int(x),int(y),int(x+w),int(y+h)],score=1.,rotation=0,dark_region=True))
        return candidates

    def partial_proposals(self,memory,frame,cfg,rotation):
        """Search a substantial target part; extrapolate the original full box.

        Alternate parts and small tilt angles to bound work on absent frames.
        A match is discovery evidence only, never a confirmed target mask.
        """
        turn=self.fallback_turn;self.fallback_turn+=1
        angle=(0.,-15.,15.)[(turn//3)%3]
        part=turn%3
        search_width=frame.shape[1] if min(memory.template.shape)<cfg.min_template*4 else cfg.search_width
        factor=min(1.,search_width/frame.shape[1])
        size=(max(1,round(frame.shape[1]*factor)),max(1,round(frame.shape[0]*factor)))
        scene=cv2.resize(frame,size)
        lab=cv2.cvtColor(scene,cv2.COLOR_BGR2LAB).astype(np.float32)*np.float32([.35,2.,2.])
        gray=cv2.cvtColor(scene,cv2.COLOR_BGR2GRAY).astype(np.float32)
        recent=max(0,getattr(cfg,'search_views',2)-1)
        colors=[memory.crop,*[v[2] for v in (memory.views[-recent:] if recent else []) if len(v)>2]]
        scales=sorted(set([.25,.35,*np.geomspace(.5,2.,13).tolist()]))
        candidates=[]
        for color in colors:
            color=np.rot90(color,rotation).copy();h,w=color.shape[:2]
            if min(h,w)<cfg.min_template:continue
            # Retain 65% of the target along its long axis, not a tiny patch.
            if h>=w:
                ph=max(1,round(h*.65));pw=w;px=0;py=round((h-ph)*part/2)
            else:
                pw=max(1,round(w*.65));ph=h;py=0;px=round((w-pw)*part/2)
            patch=color[py:py+ph,px:px+pw]
            chromatic=cv2.cvtColor(patch,cv2.COLOR_BGR2LAB).astype(np.float32)
            use_color=getattr(cfg,'recovery_color',True) and getattr(memory,'original_hue',None) is not None
            template=chromatic*np.float32([.35,2.,2.]) if use_color else cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY).astype(np.float32)
            image=lab if use_color else gray
            matrix=cv2.getRotationMatrix2D((pw/2,ph/2),angle,1.)
            rw=round(abs(matrix[0,0])*pw+abs(matrix[0,1])*ph)
            rh=round(abs(matrix[0,1])*pw+abs(matrix[0,0])*ph)
            matrix[:,2]+=np.array([rw/2-pw/2,rh/2-ph/2])
            template=cv2.warpAffine(template,matrix,(rw,rh))
            support=cv2.warpAffine(np.ones((ph,pw),np.uint8),matrix,(rw,rh),flags=cv2.INTER_NEAREST)
            corners=(np.float32([[0,0],[w,0],[w,h],[0,h]])-[px,py])@matrix[:,:2].T+matrix[:,2]
            for scale in scales:
                tw,th=round(rw*scale*factor),round(rh*scale*factor)
                if min(tw,th)<cfg.min_template or tw>size[0] or th>size[1]:continue
                resized=cv2.resize(template,(tw,th));mask=cv2.resize(support,(tw,th),interpolation=cv2.INTER_NEAREST)
                if resized[mask>0].std()<1.:continue
                if angle==0.:
                    scores=cv2.matchTemplate(image,resized,cv2.TM_CCOEFF_NORMED)
                else:
                    scores=cv2.matchTemplate(image,resized,cv2.TM_CCOEFF_NORMED,mask=mask)
                scores[~np.isfinite(scores)]=-1.
                for _ in range(3):
                    _,score,_,(x,y)=cv2.minMaxLoc(scores)
                    if score<cfg.proposal_threshold:break
                    mapped=corners*np.array([tw/rw,th/rh])+[x,y]
                    box=np.r_[mapped.min(0),mapped.max(0)]/factor
                    box[[0,2]]=np.clip(box[[0,2]],0,frame.shape[1])
                    box[[1,3]]=np.clip(box[[1,3]],0,frame.shape[0])
                    if min(box[2:]-box[:2])>=cfg.min_template:
                        candidates.append(dict(score=float(min(1.,score)),box=box.tolist(),rotation=rotation,
                                               partial_template=True,template_angle=angle))
                    scores[max(0,y-th//2):min(scores.shape[0],y+max(1,th//2)),max(0,x-tw//2):min(scores.shape[1],x+max(1,tw//2))]=-1
        return candidates

    def generate(self, memory, frame, cfg, predicted=None, local=False, radius_px=None, rotation=0):
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = 0, 0, w, h
        if local and predicted is not None:
            center = (np.asarray(predicted[:2])+predicted[2:])/2
            radius = radius_px or max(memory.template.shape)*2
            x0, y0 = max(0, int(center[0]-radius)), max(0, int(center[1]-radius))
            x1, y1 = min(w, int(center[0]+radius)), min(h, int(center[1]+radius))
        if x1 <= x0 or y1 <= y0:
            return []
        # Downsampling a 12-pixel target below min_template makes it impossible
        # to rediscover. Preserve source detail for small selections.
        search_width=w if min(memory.template.shape)<cfg.min_template*4 else cfg.search_width
        factor = min(1., search_width/(x1-x0))
        gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        sw, sh = max(1, round(gray.shape[1]*factor)), max(1, round(gray.shape[0]*factor))
        gray = cv2.resize(gray, (sw, sh))
        lab = cv2.cvtColor(cv2.resize(frame[y0:y1,x0:x1],(sw,sh)),cv2.COLOR_BGR2LAB).astype(np.float32)
        lab *= np.float32([.35,2.,2.])
        sx, sy = sw/(x1-x0), sh/(y1-y0)
        candidates = []
        recent=max(0,getattr(cfg,'search_views',2)-1)
        views=[(memory.template,memory.crop),*[(v[0],v[2] if len(v)>2 else None) for v in (memory.views[-recent:] if recent else [])]]
        for view,color in views:
            image=gray
            if color is not None and getattr(cfg,'recovery_color',True) and getattr(memory,'original_hue',None) is not None:
                chromatic=cv2.cvtColor(color,cv2.COLOR_BGR2LAB).astype(np.float32)
                if np.mean(np.linalg.norm(chromatic[:,:,1:]-128,axis=2)>12)>.05:
                    view=chromatic*np.float32([.35,2.,2.]);image=lab
            template = np.rot90(view, rotation).copy()
            for scale in cfg.scales:
                tw, th = round(template.shape[1]*scale*sx), round(template.shape[0]*scale*sy)
                if min(tw, th) < cfg.min_template or tw > sw or th > sh:
                    continue
                resized = cv2.resize(template, (tw, th))
                if resized.std() < 1.:
                    continue
                scores = cv2.matchTemplate(image, resized, cv2.TM_CCOEFF_NORMED)
                # Preserve multiple physical candidates for ambiguity checks.
                for _ in range(8):
                    _, score, _, (x, y) = cv2.minMaxLoc(scores)
                    if not np.isfinite(score) or score < cfg.proposal_threshold:
                        break
                    candidates.append(dict(score=float(score),
                        box=[x0+x/sx, y0+y/sy, x0+(x+tw)/sx, y0+(y+th)/sy],
                        rotation=rotation))
                    scores[max(0,y-th//2):min(scores.shape[0],y+max(1,th//2)),
                           max(0,x-tw//2):min(scores.shape[1],x+max(1,tw//2))] = -1
        if not candidates:
            candidates=self.partial_proposals(memory,frame,cfg,rotation)
        ranked = []
        for candidate in sorted(candidates, key=lambda c: c['score'], reverse=True):
            if all(not same_location(candidate['box'], old['box']) for old in ranked):
                ranked.append(candidate)
        memory.search_diagnostics = dict(candidates=len(ranked), rotation=rotation, local=local)
        return ranked[:20]
