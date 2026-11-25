import unittest
import numpy as np
import cv2
from tracking_control import BoxKalman,SimulatedGimbal,safe_output
from hybrid_tracker import IdentityMemory,HybridConfig
from motion_flow import estimate_motion,partial_support

class HybridTests(unittest.TestCase):
    def test_neutral_dark_target_rejects_skin_and_bright_region(self):
        frame=np.full((140,180,3),180,np.uint8)
        mask=np.zeros(frame.shape[:2],bool);mask[40:100,60:120]=True
        frame[mask]=30
        memory=IdentityMemory(frame,mask)
        brighter=np.full_like(frame,240);brighter[mask]=60
        self.assertTrue(memory.verify_support(brighter,mask)[0])
        hand=frame.copy();hand[mask]=(80,130,190)
        self.assertEqual(memory.verify_support(hand,mask),(False,'identity_neutral_color_mismatch'))
        bright=frame.copy();bright[mask]=130
        self.assertEqual(memory.verify_support(bright,mask),(False,'identity_dark_contrast_mismatch'))

    def test_dark_generic_target_requires_its_learned_descriptor(self):
        from types import SimpleNamespace
        from hybrid_tracker import HybridTracker
        tracker=HybridTracker(SimpleNamespace(reset=lambda:None))
        tracker.identity=SimpleNamespace(verify_support=lambda *a:(True,'identity_support_verified'),original_hue=None,original_value=30)
        tracker.reference_embedding=np.array([1.,0.])
        tracker.discovery=SimpleNamespace(embeddings=lambda *a:[np.array([0.,1.])])
        frame=np.zeros((50,50,3),np.uint8);mask=np.ones((50,50),bool)
        self.assertEqual(tracker.verify_identity_support(frame,mask),(False,'identity_embedding_mismatch'))
        tracker.discovery.embeddings=lambda *a:[np.array([1.,0.])]
        self.assertTrue(tracker.verify_identity_support(frame,mask)[0])

    def test_automatic_person_pose_requires_identity_and_whole_person_evidence(self):
        from types import SimpleNamespace
        from hybrid_tracker import HybridTracker
        frame=np.full((260,300,3),70,np.uint8)
        original=np.zeros(frame.shape[:2],bool);original[30:210,60:110]=True
        frame[original]=(30,180,30)
        tracker=HybridTracker(SimpleNamespace(reset=lambda:None))
        tracker.identity=IdentityMemory(frame,original)
        tracker.identity.similarity=lambda feature:.95
        sitting=np.zeros_like(original);sitting[110:190,180:260]=True
        frame[sitting]=(30,180,30)
        tracker.discovery=SimpleNamespace(proposals=lambda *a:[dict(box=[180,110,260,190])])
        tracker.discovery_class=1
        self.assertTrue(tracker.verify_identity_support(frame,sitting)[0])
        tracker.discovery_class=44
        self.assertEqual(tracker.verify_identity_support(frame,sitting),(False,'identity_shape_mismatch'))
        tracker.discovery_class=1
        tracker.discovery=SimpleNamespace(proposals=lambda *a:[dict(box=[100,10,290,250])])
        self.assertEqual(tracker.verify_identity_support(frame,sitting),(False,'person_pose_not_verified'))
        tracker.discovery=SimpleNamespace(proposals=lambda *a:[dict(box=[180,110,260,190])])
        tracker.identity.similarity=lambda feature:.5
        self.assertEqual(tracker.verify_identity_support(frame,sitting),(False,'person_pose_identity_mismatch'))

    def test_high_sam_similarity_cannot_override_original_shape(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from hybrid_tracker import HybridTracker
        frame=np.full((260,300,3),70,np.uint8)
        mask=np.zeros(frame.shape[:2],bool);mask[30:210,60:110]=True
        frame[mask]=(30,180,30)
        base=SimpleNamespace(reset=lambda:None,last_mask_stability=1.,
            predictor=SimpleNamespace(condition_state={'trusted':True},frame_idx=5),
            active=True,state='TRACKING',shape=frame.shape[:2])
        tracker=HybridTracker(base);tracker.identity=IdentityMemory(frame,mask)
        tracker.seed_frame=frame.copy();tracker.seed_mask=mask.copy()
        tracker.last_box=np.array([60,30,110,210],float);tracker.last_stamp=0.
        head=np.zeros(mask.shape,np.uint8);cv2.ellipse(head,(180,130),(30,35),0,0,360,1,-1)
        wrong=head>0;frame[wrong]=(30,180,30)
        bb=np.array([150,95,211,166],float)
        with patch('small_target.SmallTargetTracker.assess',return_value=(True,'continuity',bb,.99)):
            verdict=tracker.assess(frame,wrong,.1)
        self.assertFalse(verdict[0]);self.assertEqual(verdict[1],'identity_shape_mismatch')
        base.step=lambda *a:dict(mask=wrong,mask_stability=1.)
        tracker.identity.similarity=lambda feature:1.
        self.assertIsNone(tracker.verify_proposal(frame,.2,dict(box=bb.tolist())))
        self.assertEqual(tracker.probe_rejection_reason,'identity_shape_mismatch')
        self.assertEqual(base.predictor.condition_state,{'trusted':True})
        self.assertEqual(tracker.proposal_hits,0)

    def test_identity_survives_distance_and_rotation_but_rejects_other_shapes(self):
        frame=np.full((260,300,3),70,np.uint8)
        mask=np.zeros(frame.shape[:2],bool);mask[30:210,60:110]=True
        frame[mask]=210;frame[40:80,60:110]=(30,180,30)
        memory=IdentityMemory(frame,mask)
        distant=np.full_like(frame,70);small=np.zeros_like(mask)
        distant[80:152,200:220]=cv2.resize(frame[30:210,60:110],(20,72))
        small[80:152,200:220]=True
        self.assertTrue(memory.verify_support(distant,small)[0])
        matrix=cv2.getRotationMatrix2D((85,120),45,1.)
        rotated=cv2.warpAffine(frame,matrix,(300,260))
        rotated_mask=cv2.warpAffine(mask.astype('uint8'),matrix,(300,260),flags=cv2.INTER_NEAREST)>0
        self.assertTrue(memory.verify_support(rotated,rotated_mask)[0])
        head=np.zeros(mask.shape,np.uint8);cv2.ellipse(head,(180,130),(30,35),0,0,360,1,-1)
        other=frame.copy();other[head>0]=(30,180,30)
        self.assertEqual(memory.verify_support(other,head>0),(False,'identity_shape_mismatch'))
        other=frame.copy();other[mask]=(80,140,200)
        self.assertEqual(memory.verify_support(other,mask),(False,'identity_color_mismatch'))
        memory.update(other,mask,HybridConfig())
        self.assertEqual(memory.views,[])

    def test_small_target_keeps_detail_during_generic_global_search(self):
        frame=np.zeros((480,640,3),np.uint8)
        frame[220:232,200:212]=np.random.default_rng(111).integers(0,255,(12,12,3),dtype=np.uint8)
        mask=np.zeros(frame.shape[:2],bool);mask[220:232,200:212]=True
        memory=IdentityMemory(frame,mask)
        ranked=memory.proposals(np.roll(frame,180,axis=1),HybridConfig(scales=(1.,)))
        self.assertTrue(ranked)
        np.testing.assert_allclose(ranked[0]['box'],[380,220,392,232],atol=1)

    def test_discovery_associates_selected_region_not_highest_detection_score(self):
        from object_discovery import ObjectDiscovery
        detector=object.__new__(ObjectDiscovery)
        detector.detect=lambda frame:[dict(box=[0,0,100,100],score=.99,label=1),
                                      dict(box=[30,20,50,80],score=.4,label=44)]
        self.assertEqual(detector.selected_class(None,[30,20,50,80]),44)
        self.assertIsNone(detector.selected_class(None,[200,200,210,210]))
        self.assertEqual(len(detector.proposals(None,44)),1)

    def test_discovery_cannot_confirm_target_without_sam_evidence(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from hybrid_tracker import HybridTracker
        frame=np.zeros((100,100,3),np.uint8)
        base=SimpleNamespace(reset=lambda:None,predictor=SimpleNamespace(condition_state={},frame_idx=0),
                             active=True,state='TRACKING',shape=(100,100))
        base.step=Mock(return_value=dict(mask=None,mask_stability=0.,valid=False))
        tracker=HybridTracker(base)
        tracker.seed_frame=frame;tracker.seed_mask=np.ones((100,100),bool)
        tracker.last_box=np.array([20,20,70,70],float);tracker.last_stamp=0.
        tracker.motion.reset(tracker.last_box,0.);tracker.bad=3;tracker.state='LOST'
        tracker.identity=SimpleNamespace(target_id='same-target',views=[],proposals=Mock(side_effect=AssertionError('Slow search must not run')))
        tracker.discovery=SimpleNamespace(proposals=lambda *a:[dict(box=[20,20,70,70],score=.99,detector=True)])
        tracker.discovery_class=44
        for stamp in (2.,2.2,2.4):
            result=tracker.step(frame,stamp)
            self.assertFalse(result['valid']);self.assertEqual(result['gimbal_command']['mode'],'HOLD')
            self.assertEqual(result['target_id'],'same-target')
        tracker.identity.proposals.assert_not_called()

    def test_partial_return_extrapolates_and_clips_full_target_box(self):
        rng=np.random.default_rng(97)
        patch=rng.integers(0,255,(120,50,3),dtype=np.uint8)
        first=np.full((240,400,3),40,np.uint8);first[30:150,30:80]=patch
        mask=np.zeros(first.shape[:2],bool);mask[30:150,30:80]=True
        memory=IdentityMemory(first,mask);cfg=HybridConfig(scales=(1.,))
        current=np.full_like(first,80);current[162:240,270:320]=patch[:78]
        ranked=memory.proposals(current,cfg)
        self.assertTrue(ranked and ranked[0].get('partial_template'))
        np.testing.assert_allclose(ranked[0]['box'],[270,162,320,240],atol=2)
        self.assertEqual(memory.proposals(np.full_like(first,80),cfg),[])

    def test_chromatic_proposals_reject_grayscale_lookalike(self):
        rng=np.random.default_rng(53)
        patch=rng.integers(0,255,(70,50,3),dtype=np.uint8)
        first=np.full((180,400,3),70,np.uint8);first[30:100,30:80]=patch
        mask=np.zeros(first.shape[:2],bool);mask[30:100,30:80]=True
        memory=IdentityMemory(first,mask)
        current=np.full_like(first,70)
        current[40:110,300:350]=patch
        gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
        current[40:110,30:80]=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        cfg=HybridConfig(scales=(1.,))
        ranked=memory.proposals(current,cfg)
        self.assertTrue(ranked)
        np.testing.assert_allclose(ranked[0]['box'],[300,40,350,110],atol=1)
        self.assertTrue(all(c['box'][0]>100 for c in ranked))
        cfg.recovery_color=False
        grayscale=memory.proposals(current,cfg)
        self.assertGreaterEqual(len(grayscale),2)
        self.assertLess(grayscale[0]['score']-grayscale[1]['score'],cfg.proposal_margin)

    def test_return_confirmation_survives_missing_template_peak(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from hybrid_tracker import HybridTracker
        frame=np.random.default_rng(31).integers(0,255,(100,100,3),dtype=np.uint8)
        mask=np.zeros((100,100),bool);mask[20:70,20:70]=True
        predictor=SimpleNamespace(condition_state={},frame_idx=0)
        base=SimpleNamespace(reset=lambda:None,predictor=predictor,active=True,state='TRACKING',shape=frame.shape[:2])
        calls=[]
        def step(*args):
            calls.append(len(args))
            if len(args)==3:
                if calls.count(3)>1:
                    return dict(mask=None,mask_stability=0.)
                predictor.condition_state={'candidate':True};predictor.frame_idx=0
            else:
                self.assertTrue(predictor.condition_state.get('candidate'))
                predictor.frame_idx+=1
            return dict(mask=mask.copy(),mask_stability=1.,valid=True,dx=-5.,dy=-5.,box=[20,20,69,69])
        base.step=step
        tracker=HybridTracker(base)
        tracker.seed_frame=frame.copy();tracker.seed_mask=mask.copy()
        tracker.last_box=np.array([20,20,70,70],float);tracker.last_stamp=0.
        tracker.motion.reset(tracker.last_box,0.)
        tracker.bad=3;tracker.state='LOST'
        proposals=Mock(side_effect=[[dict(box=[20,20,70,70],score=1.,rotation=0)],[],[]])
        tracker.identity=SimpleNamespace(proposals=proposals,competing_score=lambda *a:None,
                                         similarity=lambda f:1.,target_id='original',views=[])
        for stamp in (2.,2.1):
            result=tracker.step(frame,stamp)
            self.assertFalse(result['valid'])
            self.assertEqual(result['transition_reason'],'candidate_probation')
            self.assertEqual(predictor.condition_state,{})
        result=tracker.step(frame,2.2)
        self.assertTrue(result['valid']);self.assertEqual(result['status'],'RECOVERED')
        self.assertEqual(result['target_id'],'original')
        self.assertEqual(proposals.call_count,1)
        self.assertEqual(calls,[3,3,2,3,2])
        self.assertTrue(predictor.condition_state['candidate'])
        self.assertIsNone(tracker.proposal_state)

    def test_handheld_defaults_keep_motion_filter_optional(self):
        import json
        from pathlib import Path
        self.assertFalse(HybridConfig().kalman)
        config=Path(__file__).resolve().parents[1]/'configs/hybrid.json'
        self.assertFalse(HybridConfig(**json.loads(config.read_text())).kalman)

    def test_nested_scales_share_location_but_neighbors_compete(self):
        from recovery.proposals import same_location
        self.assertTrue(same_location([0,0,100,100],[35,35,65,65]))
        self.assertFalse(same_location([0,0,30,30],[40,0,70,30]))

    def test_motion_reset_needs_two_strong_observations(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from hybrid_tracker import HybridTracker
        tracker=HybridTracker(SimpleNamespace(reset=lambda:None,last_mask_stability=1.))
        tracker.hybrid.kalman=True
        frame=np.random.default_rng(23).integers(0,255,(100,100,3),dtype=np.uint8)
        mask=np.zeros((100,100),bool);mask[20:70,20:70]=True
        tracker.seed_frame=frame.copy();tracker.seed_mask=mask.copy()
        tracker.last_box=np.array([20,20,70,70],float)
        tracker.last_stamp=0.;tracker.motion.reset(tracker.last_box,0.)
        with patch('small_target.SmallTargetTracker.assess',return_value=(True,'continuity',tracker.last_box,.95)), patch.object(tracker.motion,'innovation',return_value=100.):
            first=tracker.assess(frame,mask,.1)
            self.assertFalse(first[0]);self.assertEqual(first[1],'kalman_gate')
            tracker.bad=1
            second=tracker.assess(frame,mask,.2)
            self.assertTrue(second[0]);self.assertEqual(second[1],'verified_motion_reset')
        tracker.bad=0
        with patch('small_target.SmallTargetTracker.assess',return_value=(True,'continuity',tracker.last_box,.7)), patch.object(tracker.motion,'innovation',return_value=100.):
            for stamp in (.3,.4,.5):
                self.assertFalse(tracker.assess(frame,mask,stamp)[0])

    def test_search_crop_coordinates_and_rotation(self):
        rng=np.random.default_rng(19)
        patch=rng.integers(20,235,(30,18,3),dtype=np.uint8)
        frame=np.zeros((180,300,3),np.uint8);frame[50:80,180:198]=patch
        mask=np.zeros(frame.shape[:2],bool);mask[50:80,180:198]=True
        memory=IdentityMemory(frame,mask);cfg=HybridConfig(scales=(1.,))
        full=memory.proposals(frame,cfg)
        local=memory.proposals(frame,cfg,[180,50,198,80],local=True,radius_px=35)
        np.testing.assert_allclose(full[0]['box'],local[0]['box'],atol=1)
        rotated=np.zeros_like(frame);rotated[110:128,50:80]=np.rot90(patch)
        found=memory.proposals(rotated,cfg,rotation=1)
        np.testing.assert_allclose(found[0]['box'],[50,110,80,128],atol=1)
        rotated_mask=np.zeros_like(mask);rotated_mask[110:128,50:80]=True
        from small_target import appearance
        self.assertGreater(memory.similarity(appearance(rotated,rotated_mask)[0]),.9)
        rotated[50:80,180:198]=patch
        self.assertGreater(memory.competing_score(rotated,cfg,found[0]),.99)
    def test_partial_texture_and_distractor(self):
        rng=np.random.default_rng(17)
        frame=np.zeros((100,120,3),np.uint8)
        frame[20:80,30:90]=rng.integers(20,230,(60,60,3),dtype=np.uint8)
        mask=np.zeros(frame.shape[:2],bool);mask[20:80,30:90]=True
        partial=mask.copy();partial[:,60:]=False
        current=frame.copy();current[20:80,60:90]=0
        evidence=partial_support(frame,mask,current,partial,[30,20,90,80])
        self.assertGreater(evidence['similarity'],.99)
        self.assertAlmostEqual(evidence['visible_ratio'],.5)
        current[20:80,30:60]=rng.integers(20,230,(60,30,3),dtype=np.uint8)
        self.assertLess(partial_support(frame,mask,current,partial,[30,20,90,80])['similarity'],.9)
    def test_partial_display_does_not_restore_control(self):
        from mask_display import MaskDisplay
        from check_smooth_video import sam
        data=sam(0);data.update(target_mode='SAM_ONLY',status='UNCERTAIN',is_confirmed=False,dx=None,dy=None)
        shown=MaskDisplay().step((0,0.,data['frame']),data)
        self.assertIsNotNone(shown['mask'])
        self.assertIsNone(shown['dx']);self.assertIsNone(shown['dy'])
    def test_uncertain_counts_one_control_lock_episode(self):
        from session_audit import analyze
        rows=[dict(frame=i,time_s=i*.1,status=s,valid=v,is_confirmed=c) for i,(s,v,c) in enumerate([
            ('TRACKING',True,True),('UNCERTAIN',True,False),('REDETECTING',False,False),('RECOVERED',True,True)])]
        result=analyze(rows)
        self.assertEqual((result['losses'],result['resumes']),(1,1))
    def test_variable_dt_and_outlier(self):
        k=BoxKalman();k.reset([0,0,20,20],0)
        for t in (.1,.25,.4,.8):k.predict(t);k.update([t*20,0,t*20+20,20])
        self.assertGreater(k.x[4],5)
        self.assertGreater(k.innovation([900,900,920,920]),25)
        with self.assertRaises(ValueError):k.predict(.2)
    def test_controller_hold_limits_stale(self):
        c=SimulatedGimbal();last=0.
        for i in range(50):
            r=c.step([1,1],i*.02,True);self.assertLessEqual(abs(r['rate_x']),.6)
            self.assertLessEqual(abs(r['rate_x']-last),.041);last=r['rate_x']
        self.assertEqual(c.step(None,2.,False)['rate_x'],0)
        self.assertEqual(safe_output(dict(valid=True,is_confirmed=True,age_ms=500,gimbal_command=r))['gimbal_command']['mode'],'HOLD')
    def test_duplicate_candidates_keep_margin(self):
        rng=np.random.default_rng(4);patch=rng.integers(30,240,(24,24,3),dtype=np.uint8)
        frame=np.zeros((150,240,3),np.uint8);frame[50:74,40:64]=patch;frame[50:74,170:194]=patch
        mask=np.zeros(frame.shape[:2],bool);mask[50:74,40:64]=1
        memory=IdentityMemory(frame,mask);cfg=HybridConfig(scales=(1.,))
        ranked=memory.proposals(frame,cfg)
        self.assertGreaterEqual(len(ranked),2);self.assertLess(ranked[0]['score']-ranked[1]['score'],cfg.proposal_margin)
    def test_lk_translation(self):
        rng=np.random.default_rng(5);f=rng.integers(0,255,(120,160,3),dtype=np.uint8)
        m=np.zeros(f.shape[:2],bool);m[20:100,20:140]=1
        current=cv2.warpAffine(f,np.float32([[1,0,3],[0,1,2]]),(160,120))
        flow=estimate_motion(f,current,m);self.assertIsNotNone(flow)
        np.testing.assert_allclose(flow['matrix'][:,2],[3,2],atol=.7)
    def test_scale_aware_measurement_and_gallery(self):
        k=BoxKalman(relative_measurement=.03);k.reset([0,0,150,470],0)
        self.assertGreater(k.measurement_covariance()[3,3],k.measurement_covariance()[1,1])
        self.assertGreater(k.innovation([900,0,1050,470]),25)
        with self.assertRaises(ValueError):HybridConfig(confirmation_frames=0)
    def test_telemetry_watchdog(self):
        import tempfile,time,json
        from pathlib import Path
        from camera import TelemetryWriter
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'target.json';writer=TelemetryWriter(path)
            try:
                writer.submit(dict(valid=True,is_confirmed=True,age_ms=0,gimbal_command=dict(mode='TRACK',rate_x=.2,rate_y=0)))
                time.sleep(.4)
                self.assertEqual(json.loads(path.read_text())['gimbal_command']['mode'],'HOLD')
            finally:writer.close()
    def test_cuda_unavailable_clear_error(self):
        from unittest.mock import patch
        from source_camera_tracker import SourceCameraTracker
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'CUDA unavailable'):SourceCameraTracker()
    def test_camera_disconnect_releases(self):
        from unittest.mock import patch,MagicMock
        from camera import LatestCapture
        cap=MagicMock();cap.isOpened.return_value=True;cap.get.return_value=30.;cap.read.return_value=(False,None)
        with patch('cv2.VideoCapture',return_value=cap):
            capture=LatestCapture(0,dict(backend='dshow'));capture.start_read.set();capture.thread.join(2)
            self.assertTrue(capture.done);cap.release.assert_called()
    def test_labelled_metrics(self):
        from recovery.metrics import identity_metrics
        rows=[dict(frame=0,time_s=0,status='TRACKING',x1=0,y1=0,x2=10,y2=10,dx=0,dy=0),
              dict(frame=1,time_s=.1,status='RECOVERED',x1=20,y1=0,x2=30,y2=10,dx=0,dy=0)]
        gt=[dict(frame=0,visible=True,box=[0,0,10,10]),dict(frame=1,visible=False)]
        result=identity_metrics(rows,gt)
        self.assertEqual(result['success_iou50_percent'],100)
        self.assertEqual(result['false_locks_when_absent'],1)

if __name__=='__main__':unittest.main()
