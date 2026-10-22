"""Advisory image-quality heuristics. These do not estimate recognition accuracy."""
import cv2
import numpy as np


def quality_for_face(image, box):
    x, y, width, height = (int(value) for value in box)
    left, top = max(0, x), max(0, y)
    right, bottom = min(image.width, x + width), min(image.height, y + height)
    if right <= left or bottom <= top:
        raise ValueError('Область лица находится вне изображения.')
    rgb = np.asarray(image.crop((left, top, right, bottom)).convert('RGB'))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    normalized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_LINEAR)
    sharpness = float(cv2.Laplacian(normalized, cv2.CV_64F).var())
    brightness = float(gray.mean())
    dark_fraction = float((gray < 30).mean())
    bright_fraction = float((gray > 235).mean())
    issues = []
    if min(right - left, bottom - top) < 96:
        issues.append({'code':'small', 'message':'Лицо маленькое. Попробуйте более крупный портрет.'})
    if sharpness < 50:
        issues.append({'code':'blur', 'message':'Мало мелких деталей. Проверьте резкость и отсутствие размытия.'})
    if brightness < 55 or dark_fraction > .45:
        issues.append({'code':'dark', 'message':'Область лица тёмная. Попробуйте более равномерное освещение.'})
    if brightness > 205 or bright_fraction > .45:
        issues.append({'code':'bright', 'message':'Область лица очень светлая. Проверьте пересвет.'})
    return {
        'level':'warning' if issues else 'ok', 'issues':issues,
        'metrics':{'face_width':right-left, 'face_height':bottom-top,
                   'blur_score':round(sharpness, 2), 'brightness':round(brightness, 1),
                   'dark_fraction':round(dark_fraction, 4), 'bright_fraction':round(bright_fraction, 4)},
        'method':'Резкость: дисперсия Лапласиана на области 128×128; яркость: среднее серого 0–255. '
                 'Эвристики помогают выбрать снимок, но не гарантируют качество узнавания.',
    }
