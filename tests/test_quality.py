from io import BytesIO

import numpy as np
from PIL import Image
import pytest

from facelab.quality import quality_for_face
from facelab.vision import FaceEngine


def codes(result):
    return {issue['code'] for issue in result['issues']}


def test_smooth_face_warns_about_blur():
    result = quality_for_face(Image.new('RGB', (200, 200), (130, 130, 130)), (10, 10, 150, 150))
    assert 'blur' in codes(result)
    assert result['metrics']['blur_score'] == 0


@pytest.mark.parametrize('value,expected', [(15, 'dark'), (245, 'bright')])
def test_exposure_warnings(value, expected):
    result = quality_for_face(Image.new('RGB', (200, 200), (value, value, value)), (10, 10, 150, 150))
    assert expected in codes(result)


def test_small_face_warning_and_sharp_reference():
    data = np.random.default_rng(4).integers(70, 180, (200, 200, 3), dtype=np.uint8)
    image = Image.fromarray(data)
    assert 'small' in codes(quality_for_face(image, (0, 0, 60, 60)))
    assert quality_for_face(image, (0, 0, 180, 180))['issues'] == []


def test_outside_face_box_rejected():
    with pytest.raises(ValueError):
        quality_for_face(Image.new('RGB', (20, 20)), (40, 40, 10, 10))


def test_quality_preview_never_extracts_features():
    engine = FaceEngine()

    class Detector:
        def setInputSize(self, size):
            pass

        def detect(self, image):
            return None, np.array([[10,10,90,90,30,35,70,35,50,55,35,75,65,75,.99]], dtype=np.float32)

    class NoRecognition:
        def __getattr__(self, name):
            raise AssertionError('Quality preview must not call recognition')

    engine._detector = Detector()
    engine._recognizer = NoRecognition()
    buffer = BytesIO()
    Image.new('RGB', (120, 120), (130, 130, 130)).save(buffer, format='PNG')
    analysis = engine.inspect(buffer.getvalue())
    assert len(analysis.faces) == 1
    assert analysis.faces[0].embedding is None
    assert analysis.faces[0].quality['issues']


def test_selected_face_extracts_only_its_own_features():
    engine = FaceEngine()

    class Detector:
        def setInputSize(self, size):
            pass

        def detect(self, image):
            return None, np.array([[10,10,40,40,20,20,40,20,30,30,20,40,40,40,.99],
                                   [65,10,40,40,75,20,95,20,85,30,75,40,95,40,.98]], dtype=np.float32)

    class Recognizer:
        def __init__(self):
            self.selected = []

        def alignCrop(self, image, detection):
            self.selected.append(int(detection[0]))
            return image

        def feature(self, aligned):
            return np.array([1.,0.])

    engine._detector, engine._recognizer = Detector(), Recognizer()
    buffer = BytesIO()
    Image.new('RGB', (120,120), (130,130,130)).save(buffer, format='PNG')
    result = engine.process(buffer.getvalue(), face_index=1)
    assert len(result.faces) == 1 and result.faces[0].box == (65,10,40,40)
    assert engine._recognizer.selected == [65]
    with pytest.raises(ValueError, match='выбор'):
        engine.process(buffer.getvalue(), face_index=2)
    assert engine._recognizer.selected == [65]
