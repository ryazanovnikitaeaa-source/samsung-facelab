from copy import deepcopy

import pytest

from facelab.calibration import compare_thresholds


def case(expected, candidate, score):
    return {'expected_id': expected, 'candidate_id': candidate, 'similarity': score,
            'predicted_id': None, 'elapsed_ms': 1}


def test_replays_best_candidate_even_when_original_threshold_rejected_it():
    cases = [case('a','a',.7), case('b','a',.8), case(None,'a',.4)]
    original = deepcopy(cases)
    result = compare_thresholds(cases, 'validation', .75)
    assert result['available']
    assert .4 < result['recommended_threshold'] <= .7
    point = next(p for p in result['points'] if p['threshold'] == .5)
    assert point['metrics']['correct'] == 2
    assert point['metrics']['wrong_identity'] == 1
    assert point['balanced_accuracy'] == .75
    assert cases == original


def test_no_threshold_recommendation_from_final_test():
    result = compare_thresholds([case('a','a',.8),case(None,'a',.2)], 'test', .363)
    assert not result['available']
    assert result['recommended_threshold'] is None
    assert result['points'] == []


def test_known_only_can_show_curve_but_cannot_recommend():
    result = compare_thresholds([case('a','a',.8)], 'validation', .363)
    assert result['available']
    assert result['recommended_threshold'] is None


def test_legacy_rows_are_not_guessed():
    result = compare_thresholds([{'expected_id':'a','predicted_id':None,'similarity':.3,'elapsed_ms':1}], 'validation', .363)
    assert not result['available']
    assert result['excluded'] == 1


def test_invalid_similarity_excluded_and_empty_collection_safe():
    result = compare_thresholds([case('a','a',float('nan'))], 'validation', .363)
    assert not result['available'] and result['excluded'] == 1
    assert compare_thresholds([], 'validation', .363)['points'] == []


def test_default_and_actual_threshold_are_included():
    result = compare_thresholds([case('a','a',.8), case(None,'a',.2)], 'validation', .477)
    thresholds = [p['threshold'] for p in result['points']]
    assert thresholds == sorted(set(thresholds))
    assert .363 in thresholds and .477 in thresholds
