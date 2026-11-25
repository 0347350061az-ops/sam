"""Immutable target identity and verified appearance gallery, outside SAM state."""
import uuid
import cv2
import numpy as np
from small_target import bounds,appearance,structural_match
from recovery.proposals import CandidateGenerator,overlap,same_location

def shape_elongation(mask):
    moments=cv2.moments(mask.astype('uint8'),binaryImage=True)
    area=moments['m00']
    if area<16:return None
    covariance=np.array([[moments['mu20'],moments['mu11']],
                         [moments['mu11'],moments['mu02']]],float)/area
    eigenvalues=np.linalg.eigvalsh(covariance)
    return float(np.sqrt(max(eigenvalues[-1],1e-6)/max(eigenvalues[0],1e-6)))


def hue_signature(frame,mask):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    colored=mask & (hsv[:,:,1]>=50) & (hsv[:,:,2]>=40)
    count=int(colored.sum())
    if count<16 or count/max(1,int(mask.sum()))<.08:return None
    hist=np.bincount((hsv[:,:,0][colored]//10).astype(int),minlength=18).astype(float)
    hist=.5*hist+.25*np.roll(hist,1)+.25*np.roll(hist,-1)
    return hist/max(1.,hist.sum())


def tone_signature(frame,mask):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    value=float(np.median(gray[mask]))
    box=bounds(mask)
    radius=max(3,min(15,round(min(box[2:]-box[:2])*.12)))
    ring=(cv2.dilate(mask.astype('uint8'),np.ones((radius*2+1,radius*2+1),np.uint8))>0)&~mask
    surrounding=float(np.median(gray[ring])) if ring.any() else value
    return value,value/max(10.,surrounding)

class IdentityMemory:
    def __init__(self,frame,mask):
        self.target_id=uuid.uuid4().hex[:12]
        self.mask=mask.copy();self.box=bounds(mask)
        x,y,r,b=map(int,self.box)
        self.crop=frame[y:b,x:r].copy()
        self.template=cv2.cvtColor(self.crop,cv2.COLOR_BGR2GRAY)
        self.feature,self.texture=appearance(frame,mask)
        self.views=[]
        self.template_cache={}
        self.generator=CandidateGenerator()
        self.search_diagnostics={}
        self.original_elongation=shape_elongation(mask)
        self.original_hue=hue_signature(frame,mask)
        self.original_value,self.original_contrast=tone_signature(frame,mask)

    def verify_support(self,frame,mask,min_shape_ratio=.55,max_shape_ratio=1.8,min_hue=.35):
        """Immutable gates, invariant to scale/rotation, before learning any view."""
        if mask is None or int(mask.sum())<32:return False,'identity_insufficient_pixels'
        elongation=shape_elongation(mask)
        if self.original_elongation is not None and elongation is not None:
            ratio=elongation/self.original_elongation
            if not min_shape_ratio<=ratio<=max_shape_ratio:
                return False,'identity_shape_mismatch'
        if self.original_hue is not None:
            hue=hue_signature(frame,mask)
            if hue is None or float(np.minimum(hue,self.original_hue).sum())<min_hue:
                return False,'identity_color_mismatch'
        else:
            hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            colored=float(((hsv[:,:,1]>=50)&(hsv[:,:,2]>=40)&mask).sum())/int(mask.sum())
            if colored>.35:return False,'identity_neutral_color_mismatch'
            if self.original_value<80 and self.original_contrast<.75:
                value,contrast=tone_signature(frame,mask)
                if value>max(90.,self.original_value*2.) and contrast>max(.55,self.original_contrast*2.5):
                    return False,'identity_dark_contrast_mismatch'
        return True,'identity_support_verified'

    def update(self,frame,mask,cfg,support_verified=False):
        # The controller may already have verified a person pose against fresh
        # detector evidence. Structural similarity remains mandatory below.
        if not support_verified and not self.verify_support(frame,mask,getattr(cfg,'identity_shape_min',.55),
            getattr(cfg,'identity_shape_max',1.8),getattr(cfg,'identity_hue_min',.35))[0]:return
        feature,texture=appearance(frame,mask)
        if texture<6 or structural_match(feature,self.feature)<cfg.gallery_update_similarity:return
        x,y,r,b=map(int,bounds(mask))
        view=(cv2.cvtColor(frame[y:b,x:r],cv2.COLOR_BGR2GRAY).copy(),feature.copy(),frame[y:b,x:r].copy())
        if self.views and max(structural_match(feature,ref) for ref in [self.feature,*[v[1] for v in self.views]])>=.97:
            self.views[-1]=view
            self.template_cache.clear()
            return
        self.views.append(view)
        self.views=self.views[-max(0,cfg.gallery_size-1):] if cfg.gallery_size>1 else []
        self.template_cache.clear()

    def similarity(self,feature):
        return max(structural_match(np.rot90(feature.reshape(24,24),k).ravel(),ref)
                   for k in range(4) for ref in [self.feature,*[v[1] for v in self.views]])

    def proposals(self,frame,cfg,predicted=None,local=False,radius_px=None,rotation=0):
        return self.generator.generate(self,frame,cfg,predicted,local,radius_px,rotation)

    def competing_score(self,frame,cfg,candidate):
        # Before accepting, compare different physical locations across rotations.
        others=[]
        for rotation in range(4):
            if rotation==candidate.get('rotation',0):continue
            others.extend(c['score'] for c in self.proposals(frame,cfg,rotation=rotation)
                          if not same_location(c['box'],candidate['box']))
        return max(others) if others else None

