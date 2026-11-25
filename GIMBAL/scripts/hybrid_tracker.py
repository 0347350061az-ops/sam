"""External hybrid controller: source SAM, covariance motion gate, ranked recovery."""
from dataclasses import dataclass
from pathlib import Path
import json,time,uuid
import cv2
import numpy as np
from small_target import SmallTargetTracker,bounds,appearance,structural_match
from tracking_control import BoxKalman,SimulatedGimbal

@dataclass
class HybridConfig:
    kalman: bool=False
    motion_flow: bool=True
    camera_compensation: bool=False
    min_mask_stability: float=.7
    reid: bool=True
    global_search: bool=True
    recovery_color: bool=True
    object_discovery: bool=False
    generic_embedding_min: float=.90
    generic_tracking_embedding_min: float=.86
    generic_embedding_margin: float=.03
    identity_shape_min: float=.55
    identity_shape_max: float=1.8
    identity_hue_min: float=.35
    motion_gate: float=25.
    process_acceleration: float=100.
    measurement_noise: float=5.
    relative_measurement_noise: float=.03
    search_interval: float=.25
    search_burst_interval: float=.05
    search_burst_seconds: float=2.
    proposal_threshold: float=.75
    proposal_margin: float=.10
    verification_threshold: float=.72
    confirmation_frames: int=3
    search_width: int=320
    search_views: int=2
    min_template: int=8
    scales: tuple=(.25,.4,.6,.8,1.,1.3,1.7,2.)
    max_rate: float=.6
    max_acceleration: float=2.
    dead_zone: float=.02
    gallery_size: int=6
    gallery_update_similarity: float=.85
    partial_similarity: float=.90
    partial_timeout: float=1.2

    def __post_init__(self):
        if min(self.confirmation_frames,self.search_width,self.search_views,self.min_template,self.gallery_size)<1:
            raise ValueError('Hybrid counts and sizes must be positive')
        if min(self.search_interval,self.search_burst_interval,self.search_burst_seconds)<=0 or self.motion_gate<=0 or min(self.max_rate,self.max_acceleration)<=0:
            raise ValueError('Hybrid intervals/gates/limits must be positive')
        if self.relative_measurement_noise<0 or min(self.measurement_noise,self.process_acceleration)<=0:
            raise ValueError('Invalid Kalman noise')
        if self.partial_timeout<=0:raise ValueError('partial_timeout must be positive')
        if not 0<self.identity_shape_min<=1<=self.identity_shape_max or not 0<=self.identity_hue_min<=1:
            raise ValueError('Invalid identity support thresholds')
        for name in ('min_mask_stability','proposal_threshold','proposal_margin','verification_threshold','gallery_update_similarity','partial_similarity','generic_embedding_min','generic_tracking_embedding_min','generic_embedding_margin'):
            if not 0<=getattr(self,name)<=1:raise ValueError(f'Invalid {name}')


def overlap(a,b):
    a=np.asarray(a);b=np.asarray(b)
    inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return float(inter/max(1,np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter))

from target_identity import IdentityMemory

class HybridTracker(SmallTargetTracker):
    def __init__(self,base,small_config=None,hybrid_config=None):
        data=json.loads(Path(hybrid_config).read_text(encoding='utf-8-sig')) if hybrid_config else {}
        self.hybrid=HybridConfig(**data)
        self.discovery=None
        if self.hybrid.object_discovery:
            from object_discovery import ObjectDiscovery
            self.discovery=ObjectDiscovery()
            print('Recovery: SSDlite object proposals + SAM identity verification.',flush=True)
        self.identity=None
        super().__init__(base,small_config)
    def toggle_feature(self,name):
        if name not in ('kalman','reid','motion_flow'):raise ValueError(name)
        setattr(self.hybrid,name,not getattr(self.hybrid,name))
        if name=='kalman' and self.last_box is not None:self.motion.reset(self.last_box,self.last_stamp)
        print(f'Hybrid {name}={getattr(self.hybrid,name)}',flush=True)
    def reset(self):
        super().reset()
        c=self.hybrid
        self.motion=BoxKalman(c.process_acceleration,c.measurement_noise,c.relative_measurement_noise)
        self.control=SimulatedGimbal(dead_zone=c.dead_zone,max_rate=c.max_rate,max_accel=c.max_acceleration)
        self.identity=None;self.frame_id=0;self.proposal_hits=0;self.proposal_box=None;self.last_probe=-float('inf')
        self.proposal_state=None
        self.discovery_class=None
        self.reference_embedding=None
        self.predicted=None;self.motion_distance=None;self.top1=self.top2=None
        self.previous_frame=None;self.previous_mask=None;self.flow_evidence=None
        self.partial_mask=None;self.partial_evidence=None;self.probe_inference_ms=0.
        self.probe_rejection_reason=None
        self.search_rotation=0
        self.motion_recovery_box=None;self.motion_recovery_hits=0
    def verify_identity_support(self,frame,mask,recovering=False):
        if self.identity is None or not hasattr(self.identity,'verify_support'):
            return True,'identity_support_unavailable'
        supported,reason=self.identity.verify_support(frame,mask,self.hybrid.identity_shape_min,
            self.hybrid.identity_shape_max,self.hybrid.identity_hue_min)
        if supported:
            # Smooth dark neutral objects otherwise resemble hands/hair in
            # grayscale. Verify their crop with an immutable learned descriptor.
            if (self.reference_embedding is not None and self.discovery is not None
                and getattr(self.identity,'original_hue',None) is None
                and getattr(self.identity,'original_value',255)<80 and self.discovery_class!=1):
                feature=self.discovery.embeddings(frame,[bounds(mask)])[0]
                minimum=self.hybrid.generic_embedding_min if recovering else self.hybrid.generic_tracking_embedding_min
                if float(feature@self.reference_embedding)<minimum:
                    return False,'identity_embedding_mismatch'
            return supported,reason
        if reason!='identity_shape_mismatch':return supported,reason
        # COCO category 1 is a person. A changing pose needs fresh whole-person
        # evidence; a face-sized mask inside a body is not sufficient.
        if self.discovery is None or self.discovery_class!=1:return supported,reason
        feature,_=appearance(frame,mask)
        if feature is None or self.identity.similarity(feature)<self.hybrid.gallery_update_similarity:
            return False,'person_pose_identity_mismatch'
        box=bounds(mask)
        people=self.discovery.proposals(frame,1)
        if not any(overlap(box,p['box'])>=.6 for p in people):
            return False,'person_pose_not_verified'
        return self.identity.verify_support(frame,mask,.25,4.,self.hybrid.identity_hue_min)
    def assess(self,frame,mask,stamp):
        verdict=super().assess(frame,mask,stamp)
        if getattr(self.base,'last_mask_stability',1.)<self.hybrid.min_mask_stability:
            return False,'unstable_sam_mask',bounds(mask),None
        good,reason,box,similarity=verdict
        # A persistent, strongly matching observation may outlive a stale motion
        # prediction. Verify against accepted image evidence before resetting it.
        strong_continuity=False
        if good and box is not None and similarity is not None and similarity>=.9:
            descriptor,_=appearance(frame,mask)
            recent,_=appearance(self.seed_frame,self.seed_mask)
            strong_continuity=structural_match(descriptor,recent)>=.9 and overlap(box,self.last_box)>=.3
        if not good:
            self.motion_recovery_box=None;self.motion_recovery_hits=0
        if box is not None and self.seed_mask is not None:
            ratio=float(mask.sum()/max(1,self.seed_mask.sum()))
            if .25<=ratio<.85:
                from motion_flow import partial_support
                evidence=partial_support(self.seed_frame,self.seed_mask,frame,mask,self.predicted)
                if evidence and evidence['texture']>=self.cfg.texture_min and evidence['containment']>=.90 and evidence['similarity']>=self.hybrid.partial_similarity:
                    if self.bad or stamp-self.last_stamp>self.hybrid.partial_timeout:
                        return False,'partial_timeout_requires_recovery',box,similarity
                    self.partial_mask=mask.copy();self.partial_evidence=evidence
                    # Parent quarantines this observation; retain only its visible mask.
                    return False,'partial_observation_frozen_memory',box,similarity
        if good and self.identity is not None and hasattr(self.identity,'verify_support'):
            supported,support_reason=self.verify_identity_support(frame,mask)
            if not supported:
                self.motion_recovery_box=None;self.motion_recovery_hits=0
                return False,support_reason,box,similarity
        if self.bad>=self.cfg.lost_after and not self.hits and self.hybrid.reid:
            return False,'requires_ranked_recovery',box,similarity
        if self.bad and self.hybrid.reid and good:
            # Short uncertainty can recover through the established continuous track;
            # an unrelated mask must use ranked search after LOST.
            if not strong_continuity and (similarity is None or similarity<self.cfg.gallery_update_similarity or overlap(box,self.predicted or self.last_box)<.5):
                return False,'uncertain_continuity_gate',box,similarity
        # Controller gating runs before parent's accepted-memory update.
        if good and self.hybrid.kalman and self.motion.x is not None:
            self.motion_distance=self.motion.innovation(box)
            if self.motion_distance>self.hybrid.motion_gate:
                flow_ok=False
                descriptor,_=appearance(frame,mask)
                recent,_=appearance(self.seed_frame,self.seed_mask)
                recent_similarity=structural_match(descriptor,recent)
                if self.flow_evidence and self.previous_mask is not None and recent_similarity>=.8:
                    old=bounds(self.previous_mask);corners=np.array([[old[0],old[1]],[old[2],old[1]],[old[2],old[3]],[old[0],old[3]]])
                    shifted=corners@self.flow_evidence['matrix'][:,:2].T+self.flow_evidence['matrix'][:,2]
                    flowbox=np.r_[shifted.min(0),shifted.max(0)]
                    flow_ok=overlap(box,flowbox)>=.5
                if not flow_ok:
                    if strong_continuity:
                        self.motion_recovery_hits=self.motion_recovery_hits+1 if self.motion_recovery_box is not None and overlap(box,self.motion_recovery_box)>=.5 else 1
                        self.motion_recovery_box=box.copy()
                        if self.motion_recovery_hits>=2:
                            self.motion.reset(box,stamp)
                            self.motion_distance=0.
                            self.motion_recovery_box=None;self.motion_recovery_hits=0
                            return True,'verified_motion_reset',box,similarity
                    else:
                        self.motion_recovery_box=None;self.motion_recovery_hits=0
                    return False,'kalman_gate',box,similarity
                self.motion.P+=np.diag([25.]*4+[400.]*4)
            else:
                self.motion_recovery_box=None;self.motion_recovery_hits=0
        return verdict
    def verify_proposal(self,frame,stamp,candidate):
        # Source initializes a new state dictionary; preserve prior state by reference.
        base=self.base;p=base.predictor
        snapshot=(p.condition_state,p.frame_idx,base.active,base.state,base.shape)
        accepted=False;valid_proposal=False
        self.probe_rejection_reason=None
        try:
            candidate_box=candidate['box']
            continuing=self.proposal_state is not None and self.proposal_hits>0
            result=base.step(frame,stamp,candidate_box)
            self.probe_inference_ms=result.get('inference_ms',0.)
            for attempt in range(2 if continuing else 1):
                support_reason=None
                mask=result.get('mask')
                bb=bounds(mask) if mask is not None else None
                feature,texture=appearance(frame,mask) if bb is not None else (None,0.)
                similarity=self.identity.similarity(feature) if feature is not None else -1.
                verified=(bb is not None and result.get('mask_stability',0)>=self.hybrid.min_mask_stability
                          and texture>=6 and similarity>=self.hybrid.verification_threshold
                          and overlap(bb,candidate_box)>=.4)
                if verified and hasattr(self.identity,'verify_support'):
                    verified,support_reason=self.verify_identity_support(frame,mask,recovering=True)
                if verified:break
                if continuing and attempt==0:
                    # A tight box may omit a small returning target. Fall back
                    # to its provisional temporal memory, with identical gates.
                    p.condition_state,p.frame_idx,base.active,base.state,base.shape=self.proposal_state
                    result=base.step(frame,stamp)
                    self.probe_inference_ms+=result.get('inference_ms',0.)
            if not verified:
                self.probe_rejection_reason=support_reason or 'candidate_quality_gate'
                return None
            result['inference_ms']=self.probe_inference_ms
            valid_proposal=True
            if self.proposal_box is not None and overlap(bb,self.proposal_box)>.3:self.proposal_hits+=1
            else:self.proposal_hits=1
            self.proposal_box=bb
            if self.proposal_hits<self.hybrid.confirmation_frames:return None
            self.seed_frame=frame.copy();self.seed_mask=mask.copy();self.last_box=bb;self.last_stamp=stamp
            self.roi=(0,0,frame.shape[1],frame.shape[0]);self.state='TRACKING';self.bad=self.hits=0
            self.velocity=np.zeros(2);self.reacquired+=1
            self.motion.reset(bb,stamp);accepted=True
            result.update(status='RECOVERED',transition_reason='ranked_sam_verified',appearance_score=similarity,
                          recovery_policy='hybrid',reacquired=self.reacquired)
            return result
        finally:
            if not valid_proposal:
                self.proposal_hits=0;self.proposal_box=None;self.proposal_state=None
            elif not accepted:
                self.proposal_state=(p.condition_state,p.frame_idx,base.active,base.state,base.shape)
            else:
                self.proposal_state=None
            if not accepted:
                p.condition_state,p.frame_idx,base.active,base.state,base.shape=snapshot
    def step(self,frame,stamp,box=None):
        tick=time.perf_counter();events=[];recovered=False;before=self.state
        if box is None and self.seed_frame is not None and frame.shape!=self.seed_frame.shape:
            self.reset();events.append('RESOLUTION_CHANGED_RESELECT')
        self.flow_evidence=None
        self.partial_mask=None;self.partial_evidence=None
        if box is None and self.motion.x is not None:
            self.predicted=self.motion.predict(stamp) if self.hybrid.kalman else self.last_box.tolist()
            if self.previous_frame is not None and self.previous_frame.shape==frame.shape and self.previous_mask is not None:
                from motion_flow import estimate_motion
                if self.hybrid.camera_compensation and self.hybrid.kalman:
                    camera=estimate_motion(self.previous_frame,frame,self.previous_mask,False)
                    if camera is not None:
                        shift=camera['matrix'][:,2];self.motion.x[:2]+=shift
                        self.predicted=self.motion.box()
                if self.hybrid.motion_flow:
                    self.flow_evidence=estimate_motion(self.previous_frame,frame,self.previous_mask)

        if box is not None:events.append('INITIALIZING')
        if box is None and self.identity is not None and self.bad>=self.cfg.lost_after and not self.hits and self.hybrid.reid:
            # Once LOST, do not run doomed propagation plus a second SAM prompt.
            # The frozen bank is left intact until a ranked proposal needs SAM.
            result=dict(status='REDETECTING',mask=None,box=None,dx=None,dy=None,valid=False,
                        inference_ms=0.,predictor_called=False,captured=stamp,
                        transition_reason='search_interval_wait',target_mode=self.target_mode,
                        source_width=frame.shape[1],source_height=frame.shape[0])
        else:
            result=super().step(frame,stamp,box)
        partial=self.partial_mask is not None
        if partial:
            bb=bounds(self.partial_mask)
            self.bad=self.hits=0;self.state='UNCERTAIN'
            result.update(status='UNCERTAIN',mask=self.partial_mask,box=[int(bb[0]),int(bb[1]),int(bb[2]-1),int(bb[3]-1)],
                          valid=True,dx=None,dy=None,control_gain=0.,partial_observation=True,memory_frozen=True,
                          partial_evidence=self.partial_evidence)
        if box is not None and result.get('valid'):
            self.identity=IdentityMemory(frame,result['mask']);self.motion.reset(bounds(result['mask']),stamp)
            if self.discovery is not None:
                self.discovery_class=self.discovery.selected_class(frame,bounds(result['mask']))
                if self.discovery_class!=1 and hasattr(self.discovery,'embeddings'):
                    self.reference_embedding=self.discovery.embeddings(frame,[bounds(result['mask'])])[0]
        if result.get('valid') and not partial and self.motion.x is not None and box is None:
            if self.hybrid.kalman:self.motion.update(bounds(result['mask']))
            else:self.motion.reset(bounds(result['mask']),stamp)
            if self.identity is not None and self.hybrid.reid:
                self.identity.update(frame,result['mask'],self.hybrid,support_verified=True)
        if not result.get('valid') and self.identity is not None and self.bad>=self.cfg.lost_after:
            elapsed=stamp-self.last_stamp
            # Four orientation passes share the old full-search interval.
            interval=self.hybrid.search_burst_interval if elapsed<=self.hybrid.search_burst_seconds else self.hybrid.search_interval/4
            if self.hybrid.reid and self.hybrid.global_search and (self.proposal_hits>0 or stamp-self.last_probe>=interval):
                self.last_probe=stamp;events.append('REDETECTING')
                elapsed=stamp-self.last_stamp
                uncertainty=3*float(np.sqrt(np.max(np.diag(self.motion.P)[:2]))) if self.hybrid.kalman else 0.
                radius=max(32.,float(np.linalg.norm(self.last_box[2:]-self.last_box[:2]))*2+uncertainty)
                result['search_stage']='LOCAL' if elapsed<.5 else ('EXPANDED' if elapsed<1. else 'GLOBAL')
                probe_start=time.perf_counter()
                confirming=self.proposal_hits>0 and self.proposal_box is not None
                if confirming:
                    ranked=[dict(box=self.proposal_box.tolist(),score=self.top1 or 0.,rotation=self.search_rotation)]
                elif self.discovery is not None and self.discovery_class is not None:
                    ranked=self.discovery.proposals(frame,self.discovery_class)
                else:
                    ranked=[]
                    if self.reference_embedding is not None and hasattr(self.identity,'generator'):
                        ranked=self.identity.generator.dark_region_proposals(self.identity,frame)
                    if not ranked:
                        ranked=self.identity.proposals(frame,self.hybrid,self.predicted,local=elapsed<1.,radius_px=radius,rotation=self.search_rotation)
                    if ranked and self.reference_embedding is not None and self.discovery is not None:
                        candidates=ranked[:12]
                        features=self.discovery.embeddings(frame,[c['box'] for c in candidates])
                        ranked=sorted([dict(c,template_score=c['score'],score=float(feature@self.reference_embedding),embedding=True)
                            for c,feature in zip(candidates,features) if float(feature@self.reference_embedding)>=self.hybrid.generic_embedding_min],
                            key=lambda c:c['score'],reverse=True)
                result.update(proposal_search_ms=(time.perf_counter()-probe_start)*1000,search_rotation=self.search_rotation*90)
                if elapsed<1. and self.predicted is not None:
                    c=(np.array(self.predicted[:2])+self.predicted[2:])/2
                    result['search_box']=[max(0,c[0]-radius),max(0,c[1]-radius),min(frame.shape[1],c[0]+radius),min(frame.shape[0],c[1]+radius)]
                # Global matching discovers a location; SAM then confirms its
                # continuity. Requiring a fresh global template peak on every
                # confirmation frame breaks recovery as a returning object moves.
                self.top1=ranked[0]['score'] if ranked else None
                self.top2=ranked[1]['score'] if len(ranked)>1 else None
                margin=self.hybrid.generic_embedding_margin if ranked and ranked[0].get('embedding') else self.hybrid.proposal_margin
                if ranked and not confirming and not ranked[0].get('detector') and not ranked[0].get('embedding') and (self.top2 is None or self.top1-self.top2>=margin):
                    competitor=self.identity.competing_score(frame,self.hybrid,ranked[0])
                    if competitor is not None:self.top2=max(self.top2 if self.top2 is not None else -1.,competitor)
                    result['proposal_search_ms']=(time.perf_counter()-probe_start)*1000
                if ranked and (self.top2 is None or self.top1-self.top2>=margin):
                    result['candidate_box']=ranked[0]['box']
                    found=self.verify_proposal(frame,stamp,ranked[0])
                    if found is not None:result=found;recovered=True
                    else:
                        result.update(inference_ms=result.get('inference_ms',0.)+self.probe_inference_ms,predictor_called=True,
                                      transition_reason='candidate_probation' if self.proposal_hits else (self.probe_rejection_reason or 'candidate_rejected'))
                else:
                    self.proposal_hits=0;self.proposal_box=None;self.proposal_state=None
                    result['transition_reason']='ambiguous_candidates' if ranked else 'no_structural_candidate'
                if not recovered:result['status']='REDETECTING'
                if not self.proposal_hits:self.search_rotation=(self.search_rotation+1)%4
        if self.identity is None and box is None:result['status']='IDLE'
        if 'RESOLUTION_CHANGED_RESELECT' in events:result['transition_reason']='resolution_changed_reselect'
        confirmed=bool(result.get('valid')) and not partial;h,w=frame.shape[:2]
        error=[result['dx']/(w/2),result['dy']/(h/2)] if confirmed else None
        self.previous_frame=frame.copy()
        self.previous_mask=result.get('mask').copy() if result.get('valid') else None
        self.frame_id+=1
        import torch
        result.update(target_id=self.identity.target_id if self.identity else None,frame_id=self.frame_id,
                      transition=f'{before}->{result["status"]}',engine='enhanced',
                      timestamp=stamp,state=result['status'],bbox=result.get('box'),
                      target_center=[result['dx']+w/2,result['dy']+h/2] if confirmed else None,
                      normalized_error_x=error[0] if error else None,normalized_error_y=error[1] if error else None,
                      target_velocity=self.motion.x[4:6].tolist() if self.hybrid.kalman and self.motion.x is not None else None,
                      identity_score=result.get('appearance_score'),tracking_quality=result.get('mask_stability'),
                      flow_inlier_ratio=self.flow_evidence['inlier_ratio'] if self.flow_evidence else None,
                      is_visible=bool(result.get('valid')),is_confirmed=confirmed,recommended_control_mode='TRACK' if confirmed else 'HOLD',
                      gimbal_command=self.control.step(error,stamp,confirmed),kalman_box=self.predicted if self.hybrid.kalman else None,
                      kalman_innovation=self.motion_distance,proposal_top1=self.top1,proposal_top2=self.top2,
                      identity_gallery=1+len(self.identity.views) if self.identity else 0,
                      features=dict(kalman=self.hybrid.kalman,reid=self.hybrid.reid,flow=self.hybrid.motion_flow),
                      gpu_allocated_mb=torch.cuda.memory_allocated()/1048576 if torch.cuda.is_available() else None,
                      state_events=events+(['RECOVERED'] if recovered else []),pipeline_ms=(time.perf_counter()-tick)*1000)
        return result
