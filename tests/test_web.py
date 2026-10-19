from fastapi.testclient import TestClient

from facelab.web import create_app
from facelab.service import FaceService
from facelab.storage import Store
from test_service import FakeEngine, picture


def client(tmp_path):
    service=FaceService(Store(tmp_path/'db.sqlite3'),FakeEngine())
    return TestClient(create_app(service=service)), service


def test_full_http_flow(tmp_path):
    c,_=client(tmp_path)
    assert c.get('/').status_code==200
    assert c.get('/api/status').json()['people_count']==0
    r=c.post('/api/people',data={'name':'Анна'},files=[('files',('a.png',picture((200,0,0)),'image/png'))])
    assert r.status_code==200, r.text
    person=r.json()['person']
    r=c.post('/api/recognize',data={'threshold':'.5'},files={'file':('new.png',picture((180,0,0)),'image/png')})
    assert r.status_code==200 and r.json()['faces'][0]['name']=='Анна'
    assert c.get(f"/api/people/{person['id']}/thumbnail").headers['content-type']=='image/jpeg'
    assert c.delete(f"/api/people/{person['id']}").status_code==200


def test_validation_returns_readable_errors(tmp_path):
    c,_=client(tmp_path)
    r=c.post('/api/recognize',files={'file':('bad.png',b'broken','image/png')})
    assert r.status_code==400 and isinstance(r.json()['detail'],str)
    assert c.delete('/api/people/missing').status_code==404


def test_external_host_and_origin_rejected(tmp_path):
    c,_=client(tmp_path)
    assert c.get('/api/people',headers={'host':'evil.example'}).status_code==400
    r=c.post('/api/people',headers={'origin':'https://evil.example'},data={'name':'A'},files={'files':('a.png',picture((200,0,0)),'image/png')})
    assert r.status_code==403


def test_report_exports_and_no_nan(tmp_path):
    c,_=client(tmp_path)
    assert c.get('/api/report').json()['metrics']['accuracy'] is None
    assert 'text/csv' in c.get('/api/report.csv').headers['content-type']


def test_threshold_out_of_range(tmp_path):
    c,_=client(tmp_path)
    r=c.post('/api/recognize',data={'threshold':'2'},files={'file':('a.png',picture((200,0,0)),'image/png')})
    assert r.status_code in [400,422]
