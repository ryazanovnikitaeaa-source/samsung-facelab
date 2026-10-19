import numpy as np
import pytest

from facelab.matching import identify, summarize


def test_nearest_person_and_threshold():
    gallery = {'a': [np.array([1., 0.])], 'b': [np.array([0., 1.])]}
    assert identify(np.array([5., 0.]), gallery, .5)['person_id'] == 'a'
    assert identify(np.array([-1., -1.]), gallery, .5)['person_id'] is None


def test_empty_gallery_and_exact_boundary():
    assert identify(np.array([1., 0.]), {}, .5)['similarity'] is None
    assert identify(np.array([1., 0.]), {'a': [np.array([1., 0.])]}, 1.)['person_id'] == 'a'


def test_best_reference_not_average():
    gallery = {'a': [np.array([1., 0.]), np.array([0., 1.])]}
    assert identify(np.array([0., 1.]), gallery, .9)['person_id'] == 'a'


@pytest.mark.parametrize('threshold', [float('nan'), float('inf'), -1.1, 1.1])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        identify(np.array([1., 0.]), {}, threshold)


def test_invalid_embedding():
    with pytest.raises(ValueError):
        identify(np.array([0., 0.]), {}, .5)


def test_metrics_distinguish_false_accept_reject_and_wrong_identity():
    cases = [
        {'expected_id':'a', 'predicted_id':'a', 'elapsed_ms':10},
        {'expected_id':'a', 'predicted_id':None, 'elapsed_ms':20},
        {'expected_id':'a', 'predicted_id':'b', 'elapsed_ms':30},
        {'expected_id':None, 'predicted_id':'a', 'elapsed_ms':40},
        {'expected_id':None, 'predicted_id':None, 'elapsed_ms':50},
    ]
    m = summarize(cases)
    assert m['correct'] == 2 and m['accuracy'] == .4
    assert m['false_accepts'] == 1 and m['false_accept_rate'] == .5
    assert m['false_rejects'] == 1 and m['false_reject_rate'] == pytest.approx(1/3)
    assert m['wrong_identity'] == 1
    assert m['mean_ms'] == 30


def test_missing_denominators_are_null():
    assert summarize([])['accuracy'] is None
    assert summarize([{'expected_id':'a','predicted_id':'a','elapsed_ms':1}])['false_accept_rate'] is None
