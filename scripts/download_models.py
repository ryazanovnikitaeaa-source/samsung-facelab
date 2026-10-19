"""Download official weights atomically and verify their LFS SHA-256."""
from hashlib import sha256
from pathlib import Path
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
MODELS={
    'face_detection_yunet_2023mar.onnx':('face_detection_yunet','8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'),
    'face_recognition_sface_2021dec.onnx':('face_recognition_sface','0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79'),
}


def main():
    destination=ROOT/'models'
    destination.mkdir(exist_ok=True)
    for filename,(folder,expected) in MODELS.items():
        target=destination/filename
        if target.exists() and sha256(target.read_bytes()).hexdigest()==expected:
            print(f'OK: {filename}')
            continue
        url=f'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/{folder}/{filename}'
        print(f'Downloading {filename} ...',flush=True)
        request=urllib.request.Request(url,headers={'User-Agent':'FaceLab-educational/1.0'})
        temporary=target.with_suffix('.part')
        digest=sha256()
        try:
            with urllib.request.urlopen(request,timeout=90) as response, temporary.open('wb') as out:
                while chunk:=response.read(1024*1024):
                    out.write(chunk); digest.update(chunk)
            if digest.hexdigest()!=expected:
                raise RuntimeError(f'SHA-256 mismatch for {filename}. Model was not installed.')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f'Verified: {filename}',flush=True)
    print('Models are ready. Recognition runs offline.')


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print(f'Model setup failed: {exc}',file=sys.stderr)
        sys.exit(1)
