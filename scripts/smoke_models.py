"""Real-model technical smoke test. This is NOT an accuracy benchmark."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PIL import Image, ImageEnhance
import cv2
from facelab.service import FaceService
from facelab.storage import Store
from facelab.vision import FaceEngine

SAMPLES={
    'astronaut.png':'https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/astronaut.png',
    'grace_hopper.jpg':'https://raw.githubusercontent.com/matplotlib/matplotlib/v3.10.0/lib/matplotlib/mpl-data/sample_data/grace_hopper.jpg',
}


def encode(image):
    out=BytesIO(); image.save(out,format='PNG'); return out.getvalue()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args()
    sample_dir=ROOT/'tmp'/'samples'; sample_dir.mkdir(parents=True,exist_ok=True)
    for filename,url in SAMPLES.items():
        target=sample_dir/filename
        if not target.exists():
            if not args.download:
                raise SystemExit('Run once with --download to fetch public technical samples.')
            request=urllib.request.Request(url,headers={'User-Agent':'FaceLab-test/1.0'})
            with urllib.request.urlopen(request,timeout=60) as response:
                target.write_bytes(response.read())
    engine=FaceEngine()
    a=Image.open(sample_dir/'astronaut.png').convert('RGB')
    b=Image.open(sample_dir/'grace_hopper.jpg').convert('RGB')
    altered=ImageEnhance.Brightness(a).enhance(.87)
    altered.save(sample_dir/'astronaut-altered.png')
    group=Image.new('RGB',(1024,600),'#e6e9e0')
    a_copy=a.copy(); b_copy=b.copy(); a_copy.thumbnail((480,560)); b_copy.thumbnail((480,560))
    group.paste(a_copy,(10,20)); group.paste(b_copy,(530,20)); group.save(sample_dir/'group.png')
    blank=Image.new('RGB',(512,512),'#dce5cf'); blank.save(sample_dir/'blank.png')
    with tempfile.TemporaryDirectory() as temporary:
        service=FaceService(Store(Path(temporary)/'db.sqlite3'),engine)
        person=service.enroll('Пример A',[('astronaut.png',encode(a))])
        same=service.recognize(encode(altered))
        other=service.recognize(encode(b))
        multi=service.recognize(encode(group))
        no_faces=service.recognize(encode(blank))
        assert len(same['faces'])==1 and same['faces'][0]['person_id']==person['id'],same
        assert len(other['faces'])==1 and other['faces'][0]['person_id'] is None,other
        assert len(multi['faces'])==2 and sum(f['person_id'] is not None for f in multi['faces'])==1
        assert no_faces['faces']==[]
        report={
            'kind':'technical_smoke_test_not_accuracy_benchmark',
            'opencv':cv2.__version__,
            'checks':{'augmented_reference_recognized':True,'different_portrait_rejected':True,
                      'two_faces_detected':True,'blank_has_no_faces':True},
            'same_similarity':same['faces'][0]['similarity'],
            'other_similarity':other['faces'][0]['similarity'],
            'sources':SAMPLES,
            'limitation':'Only two source portraits and an augmented copy. No independent identity accuracy claim.',
        }
        (ROOT/'tmp'/'model-smoke-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=True,indent=2))


if __name__=='__main__':
    main()
