"""Detection-only regressions: scale fallback and original-coordinate mapping."""
import numpy as np

from facelab.detection import detect_faces


ROW = np.array([[100, 80, 120, 140, 125, 115, 175, 115,
                 150, 145, 130, 175, 170, 175, .9]], dtype=np.float32)


class Detector:
    def __init__(self, found_at):
        self.found_at = found_at
        self.calls = []

    def setInputSize(self, size):
        self.size = size

    def detect(self, image):
        assert self.size == (image.shape[1], image.shape[0])
        self.calls.append(self.size)
        faces = ROW.copy() if self.size == self.found_at else None
        return None, faces


def test_retries_large_image_when_full_size_finds_no_face():
    detector = Detector((640, 480))
    faces = detect_faces(detector, np.zeros((960, 1280, 3), dtype=np.uint8))
    assert detector.calls == [(1280, 960), (640, 480)]
    np.testing.assert_allclose(faces[:, :14], ROW[:, :14] * 2)
    np.testing.assert_allclose(faces[:, 14], ROW[:, 14])


def test_preserves_original_result_when_faces_are_found():
    detector = Detector((1280, 960))
    faces = detect_faces(detector, np.zeros((960, 1280, 3), dtype=np.uint8))
    assert detector.calls == [(1280, 960)]
    np.testing.assert_array_equal(faces, ROW)


def test_does_not_upscale_small_images_or_invent_faces():
    detector = Detector(None)
    assert detect_faces(detector, np.zeros((240, 320, 3), dtype=np.uint8)) is None
    assert detector.calls == [(320, 240)]


def test_uses_separate_xy_ratios_after_rounding():
    detector = Detector((640, 480))
    faces = detect_faces(detector, np.zeros((961, 1281, 3), dtype=np.uint8))
    np.testing.assert_allclose(faces[:, [0, 2, 4, 6, 8, 10, 12]],
                               ROW[:, [0, 2, 4, 6, 8, 10, 12]] * (1281 / 640))
    np.testing.assert_allclose(faces[:, [1, 3, 5, 7, 9, 11, 13]],
                               ROW[:, [1, 3, 5, 7, 9, 11, 13]] * (961 / 480))
