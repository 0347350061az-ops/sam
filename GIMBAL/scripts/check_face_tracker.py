"""Headless regression checks for face occlusion and automatic recovery."""
import unittest
import numpy as np
from face_tracker import FaceTracker


class Base:
    def reset(self):
        pass

    def step(self, frame, stamp, box):
        return dict(mask=np.ones(frame.shape[:2], bool))


class Detector:
    faces = None

    def setInputSize(self, size):
        pass

    def detect(self, frame):
        return None, self.faces


class FaceRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tracker = FaceTracker(Base())
        self.detector = Detector()
        self.tracker.detector = self.detector
        self.frame = np.zeros((180, 320, 3), np.uint8)
        rng = np.random.default_rng(12)
        self.face = rng.integers(0, 256, (80, 60, 3), dtype=np.uint8)
        self.other = rng.integers(0, 256, (80, 60, 3), dtype=np.uint8)
        self.frame[30:110, 40:100] = self.face
        self.detector.faces = np.array([[40, 30, 60, 80]])
        result = self.tracker.step(self.frame, 0, [38, 28, 103, 112])
        self.assertTrue(result['valid'])
        self.assertFalse(result['mask'][:, 101:].any())

    def hide(self, count):
        self.detector.faces = None
        for i in range(count):
            result = self.tracker.step(self.frame, i+1)
            self.assertFalse(result['valid'])
            self.assertIsNone(result['dx'])

    def test_long_occlusion_and_moved_return(self):
        self.hide(100)
        self.frame[:] = 0
        self.frame[40:120, 200:260] = self.face
        self.detector.faces = np.array([[200, 40, 60, 80]])
        for i in range(2):
            self.assertFalse(self.tracker.step(self.frame, 101+i)['valid'])
        result = self.tracker.step(self.frame, 103)
        self.assertTrue(result['valid'])
        self.assertEqual(result['box'], [200, 40, 260, 120])
        self.assertEqual(result['reacquired'], 1)
        self.assertFalse(result['mask'][:, :200].any())

    def test_other_face_at_old_position_rejected(self):
        self.hide(20)
        self.frame[30:110, 40:100] = self.other
        self.detector.faces = np.array([[40, 30, 60, 80]])
        for i in range(5):
            self.assertFalse(self.tracker.step(self.frame, 21+i)['valid'])

    def test_ambiguous_faces_and_interrupted_confirmation(self):
        self.hide(2)
        self.frame[30:110, 200:260] = self.face
        self.detector.faces = np.array([[40, 30, 60, 80], [200, 30, 60, 80]])
        self.assertFalse(self.tracker.step(self.frame, 3)['valid'])
        self.detector.faces = np.array([[40, 30, 60, 80]])
        self.assertFalse(self.tracker.step(self.frame, 4)['valid'])
        self.hide(1)
        self.detector.faces = np.array([[40, 30, 60, 80]])
        self.assertFalse(self.tracker.step(self.frame, 5)['valid'])
        self.assertFalse(self.tracker.step(self.frame, 6)['valid'])
        self.assertTrue(self.tracker.step(self.frame, 7)['valid'])

    def test_reselection_discards_previous_identity(self):
        self.hide(20)
        self.frame[30:110, 200:260] = self.other
        self.detector.faces = np.array([[200, 30, 60, 80]])
        self.assertTrue(self.tracker.step(self.frame, 21, [230, 70, 230, 70])['valid'])
        self.assertEqual(self.tracker.reacquired, 0)


if __name__ == '__main__':
    unittest.main()
