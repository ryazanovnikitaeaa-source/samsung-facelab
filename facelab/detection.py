"""Find faces, retrying smaller scales only when the original has no result."""
import cv2


def detect_faces(detector, image):
    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)
    if faces is not None and len(faces):
        return faces.copy()

    # YuNet can miss a large close-up at full resolution. Retain its score
    # threshold and retry at lower resolutions instead of accepting weaker hits.
    for longest_side in (640, 320):
        if max(width, height) <= longest_side:
            continue
        ratio = longest_side / max(width, height)
        size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
        resized = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        detector.setInputSize(size)
        _, faces = detector.detect(resized)
        if faces is None or not len(faces):
            continue
        faces = faces.copy()
        # Bounding box and five landmarks must all refer to the original image.
        # Independent ratios account for rounding in rectangular resized images.
        faces[:, 0:14:2] *= width / size[0]
        faces[:, 1:14:2] *= height / size[1]
        return faces
    return None
