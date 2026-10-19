"""Optional real-model regression: installed models, including Cyrillic paths."""
from io import BytesIO

from PIL import Image
import pytest

from facelab.vision import FaceEngine


def test_real_models_load_from_unicode_project_path():
    engine=FaceEngine()
    if not engine.ready:
        pytest.skip('Download model weights to run the real-model regression.')
    data=BytesIO()
    Image.new('RGB',(320,320),'white').save(data,format='PNG')
    assert engine.process(data.getvalue()).faces == []
