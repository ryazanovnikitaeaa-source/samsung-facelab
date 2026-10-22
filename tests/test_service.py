from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from facelab.service import FaceService
from facelab.storage import Store
from facelab.vision import Analysis, Face, decode_image


def picture(color):
    buf = BytesIO()
    Image.new('RGB', (120, 120), color).save(buf, format='PNG')
    return buf.getvalue()


class FakeEngine:
    def inspect(self, data):
        analysis = self.process(data)
        for face in analysis.faces:
            face.embedding = None
        return analysis

    def process(self, data):
        image, digest = decode_image(data)
        color = image.getpixel((0, 0))
        count = 0 if color[2] > 200 else 2 if color[0] == color[1] else 1
        vector = np.array([1., 0.]) if color[0] > color[1] else np.array([0., 1.])
        return Analysis(image, digest, [Face((10,10,90,90), vector, .99) for _ in range(count)])


@pytest.fixture
def service(tmp_path):
    return FaceService(Store(tmp_path / 'test.sqlite3'), FakeEngine())


def enroll(service, name='Анна'):
    return service.enroll(name, [('ref.png', picture((200, 0, 0)))])


def test_enrollment_recognition_unknown_and_delete(service):
    person = enroll(service)
    assert service.recognize(picture((180, 0, 0)), .5)['faces'][0]['name'] == 'Анна'
    assert service.recognize(picture((0, 180, 0)), .5)['faces'][0]['person_id'] is None
    assert service.store.people()[0]['samples'] == 1
    service.delete(person['id'])
    assert service.store.people() == []


def test_atomic_registration_rejects_group_photo(service):
    with pytest.raises(ValueError, match='ровно одно'):
        service.enroll('А', [('a.png',picture((200,0,0))), ('b.png',picture((50,50,0)))])
    assert service.store.people() == []


def test_duplicate_name_and_duplicate_images(service):
    enroll(service)
    with pytest.raises(ValueError, match='именем'):
        service.enroll(' анна ', [('new.png',picture((190,0,0)))])
    with pytest.raises(ValueError, match='повтор'):
        service.enroll('Б', [('a.png',picture((0,200,0))),('b.png',picture((0,200,0)))])


def test_reject_invalid_image_and_no_face(service):
    with pytest.raises(ValueError):
        service.recognize(b'bad jpeg', .5)
    with pytest.raises(ValueError, match='ровно одно'):
        service.enroll('A', [('blank.png',picture((0,0,255)))])


def test_evaluation_rejects_reference_and_duplicate_test(service):
    p = enroll(service)
    with pytest.raises(ValueError, match='эталон'):
        service.evaluate([('ref.png', picture((200,0,0)))], p['id'], .5)
    result = service.evaluate([('new.png',picture((180,0,0)))], p['id'], .5)
    assert result['metrics']['accuracy'] == 1
    with pytest.raises(ValueError, match='уже'):
        service.evaluate([('new.png',picture((180,0,0)))], p['id'], .5)


def test_evaluation_does_not_mix_thresholds_and_gallery_mutation_clears(service):
    p=enroll(service)
    service.evaluate([('a.png',picture((180,0,0)))],p['id'],.5)
    with pytest.raises(ValueError, match='порог'):
        service.evaluate([('b.png',picture((170,0,0)))],p['id'],.6)
    service.enroll('Б', [('b.png',picture((0,200,0)))])
    assert service.report()['metrics']['total'] == 0


def test_unknown_evaluation_and_unknown_person_id(service):
    enroll(service)
    result=service.evaluate([('unknown.png',picture((0,200,0)))],None,.5)
    assert result['metrics']['correct']==1
    with pytest.raises(ValueError):
        service.evaluate([('x.png',picture((170,0,0)))],'missing',.5)


def test_identical_pixels_different_encoding_have_same_digest():
    data=picture((200,0,0))
    image=Image.open(BytesIO(data))
    buf=BytesIO(); image.save(buf,format='BMP')
    assert decode_image(data)[1] == decode_image(buf.getvalue())[1]


def test_validation_records_candidate_and_quality_without_mixing_purposes(service):
    person = enroll(service)
    report = service.evaluate([('new.png', picture((180,0,0)))], person['id'], .5, purpose='validation')
    assert report['purpose'] == 'validation'
    assert report['cases'][0]['candidate_id'] == person['id']
    assert 'quality' in report['cases'][0]
    assert report['calibration']['available']
    with pytest.raises(ValueError, match='режим'):
        service.evaluate([('other.png',picture((170,0,0)))], person['id'], .5, purpose='test')
    assert service.report()['metrics']['total'] == 1


def test_quality_warnings_do_not_block_valid_enrollment(service):
    person = enroll(service)
    assert person['quality_warnings']
    assert service.store.people()[0]['id'] == person['id']


def test_small_detected_face_warns_but_does_not_block_registration(service):
    class SmallFaceEngine(FakeEngine):
        def process(self,data):
            result=super().process(data)
            result.faces[0].box=(10,10,40,40)
            return result

    service.engine=SmallFaceEngine()
    person=enroll(service)
    assert any(issue['code']=='small' for entry in person['quality_warnings'] for issue in entry['quality']['issues'])


def test_reports_have_ordered_revisions_and_instance_key(service):
    first=service.report()
    second=service.report()
    assert second['report_revision'] > first['report_revision']
    assert second['report_instance'] == first['report_instance']
    assert first['report_instance']
