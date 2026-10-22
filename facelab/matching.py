"""Model-independent cosine matching and honest open-set metrics."""
import math

import numpy as np

DEFAULT_THRESHOLD = 0.363


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or not math.isfinite(norm) or norm < 1e-8:
        raise ValueError('Некорректные признаки лица.')
    return vector / norm


def identify(vector, gallery, threshold=DEFAULT_THRESHOLD):
    if not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError('Порог должен быть конечным числом от -1 до 1.')
    vector = normalize(vector)
    best_id, best_score = None, -2.
    for person_id, references in gallery.items():
        for reference in references:
            score = float(np.clip(np.dot(vector, normalize(reference)), -1., 1.))
            if score > best_score:
                best_id, best_score = person_id, score
    if best_id is None:
        return {'person_id': None, 'candidate_id': None, 'similarity': None}
    return {'person_id': best_id if best_score >= threshold else None,
            'candidate_id': best_id, 'similarity': best_score}


def summarize(cases):
    total = len(cases)
    known = sum(c['expected_id'] is not None for c in cases)
    unknown = total - known
    correct = sum(c['expected_id'] == c['predicted_id'] for c in cases)
    false_accepts = sum(c['expected_id'] is None and c['predicted_id'] is not None for c in cases)
    false_rejects = sum(c['expected_id'] is not None and c['predicted_id'] is None for c in cases)
    wrong_identity = sum(c['expected_id'] is not None and c['predicted_id'] is not None
                         and c['expected_id'] != c['predicted_id'] for c in cases)
    return {'total': total, 'known': known, 'unknown': unknown, 'correct': correct,
            'accuracy': correct / total if total else None,
            'false_accepts': false_accepts, 'false_accept_rate': false_accepts / unknown if unknown else None,
            'false_rejects': false_rejects, 'false_reject_rate': false_rejects / known if known else None,
            'wrong_identity': wrong_identity,
            'mean_ms': sum(c['elapsed_ms'] for c in cases) / total if total else None}
