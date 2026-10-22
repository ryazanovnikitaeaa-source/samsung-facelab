"""YuNet detection, SFace alignment/features and bounded image decoding."""
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from threading import Lock
import warnings

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .detection import detect_faces
from .matching import normalize
from .quality import quality_for_face

ROOT = Path(__file__).resolve().parents[1]
DETECTOR = 'face_detection_yunet_2023mar.onnx'
RECOGNIZER = 'face_recognition_sface_2021dec.onnx'
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000


@dataclass
class Face:
    box: tuple
    embedding: np.ndarray | None
    detector_score: float
    quality: dict | None = None


@dataclass
class Analysis:
    image: Image.Image
    image_hash: str
    faces: list[Face]


def decode_image(data):
    if not data or len(data) > MAX_FILE_BYTES:
        raise ValueError('Файл пуст или превышает 10 МБ.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as raw:
                if raw.format not in {'JPEG','PNG','WEBP','BMP'}:
                    raise ValueError('Поддерживаются JPG, PNG, WEBP и BMP.')
                if raw.width * raw.height > MAX_PIXELS:
                    raise ValueError('Слишком большое изображение: максимум 20 мегапикселей.')
                image = ImageOps.exif_transpose(raw).convert('RGB')
                image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise ValueError('Не удалось прочитать изображение. Выберите корректный JPG, PNG, WEBP или BMP.') from exc
    digest = sha256(f'{image.width}x{image.height}:'.encode() + image.tobytes()).hexdigest()
    image.thumbnail((1600,1600), Image.Resampling.LANCZOS)
    return image, digest


class FaceEngine:
    def __init__(self, model_dir=None):
        self.model_dir = Path(model_dir or ROOT/'models')
        self._lock = Lock()
        self._detector = None
        self._recognizer = None

    @property
    def ready(self):
        return all((self.model_dir / f).is_file() for f in [DETECTOR,RECOGNIZER])

    def _load(self):
        if self._detector is not None:
            return
        if not self.ready:
            raise RuntimeError('Модели не установлены. Запустите setup.bat для их загрузки.')
        # Python handles Unicode Windows paths; OpenCV's native path reader does not.
        config = np.empty(0,dtype=np.uint8)
        detector_bytes = np.frombuffer((self.model_dir/DETECTOR).read_bytes(),dtype=np.uint8)
        recognizer_bytes = np.frombuffer((self.model_dir/RECOGNIZER).read_bytes(),dtype=np.uint8)
        try:
            detector = cv2.FaceDetectorYN.create('onnx',detector_bytes,config,(320,320),.85,.3,5000)
            recognizer = cv2.FaceRecognizerSF.create('onnx',recognizer_bytes,config)
        except cv2.error as exc:
            raise RuntimeError('Не удалось загрузить модели. Повторно запустите setup.bat и перезапустите приложение.') from exc
        self._detector,self._recognizer = detector,recognizer

    def process(self, data, face_index=None):
        return self._analyze(data, extract_features=True, face_index=face_index)

    def inspect(self, data):
        """Detection and image quality only; no identity features or matching."""
        return self._analyze(data, extract_features=False)

    def _analyze(self, data, extract_features, face_index=None):
        image, digest = decode_image(data)
        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        faces = []
        with self._lock:
            self._load()
            detections = detect_faces(self._detector, bgr)
            candidates = []
            if detections is not None:
                for detection in detections:
                    x,y,w,h = [int(v) for v in detection[:4]]
                    x1,y1 = max(0,x),max(0,y)
                    x2,y2 = min(image.width,x+w),min(image.height,y+h)
                    if x2>x1 and y2>y1:
                        box = (x1,y1,x2-x1,y2-y1)
                        candidates.append((detection,box))
            if face_index is not None:
                if type(face_index) is not int or not 0 <= face_index < len(candidates):
                    raise ValueError('Некорректный выбор лица. Повторите предварительную проверку снимка.')
                candidates = [candidates[face_index]]
            for detection,box in candidates:
                embedding = None
                if extract_features:
                    aligned = self._recognizer.alignCrop(bgr, detection)
                    embedding = normalize(self._recognizer.feature(aligned).copy())
                faces.append(Face(box,embedding,float(detection[-1]),quality_for_face(image,box)))
        return Analysis(image,digest,faces)
