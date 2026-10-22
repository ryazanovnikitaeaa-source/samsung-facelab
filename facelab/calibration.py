"""Replay saved candidates on validation data without rerunning face models."""
import math

from .matching import DEFAULT_THRESHOLD, summarize


def compare_thresholds(cases, purpose, current_threshold=DEFAULT_THRESHOLD):
    eligible = []
    for case in cases:
        if 'candidate_id' not in case:
            continue
        score, candidate = case.get('similarity'), case['candidate_id']
        valid = candidate is None and score is None
        if candidate is not None:
            valid = (isinstance(score, (int, float)) and math.isfinite(score) and -1 <= score <= 1)
        if valid:
            eligible.append(case)
    result = {'available':False, 'eligible':len(eligible), 'excluded':len(cases)-len(eligible),
              'points':[], 'recommended_threshold':None, 'recommendation_metrics':None,
              'criterion':'Среднее между долей верных имён среди известных и долей верных отказов среди неизвестных. '
                          'При равенстве — меньше ложных узнаваний, затем ближе к 0.363.',
              'reason':''}
    if purpose != 'validation':
        result['reason'] = 'Порог выбирается на валидации. Итоговый тест используется только для проверки выбранного порога.'
        return result
    if not eligible:
        result['reason'] = ('В старых результатах нет исходного кандидата. Сохраните отчёт и начните новую валидацию.'
                            if cases else 'Добавьте новые снимки в режиме «Валидация».')
        return result
    thresholds = {round(.2 + index*.01, 3) for index in range(61)} | {DEFAULT_THRESHOLD}
    if current_threshold is not None and math.isfinite(current_threshold) and -1 <= current_threshold <= 1:
        thresholds.add(round(current_threshold, 6))
    for threshold in sorted(thresholds):
        replay = []
        for case in eligible:
            score = case.get('similarity')
            prediction = case['candidate_id'] if score is not None and score >= threshold else None
            replay.append({'expected_id':case['expected_id'], 'predicted_id':prediction,
                           'elapsed_ms':case.get('elapsed_ms', 0)})
        metrics = summarize(replay)
        known_correct = sum(c['expected_id'] is not None and c['expected_id']==c['predicted_id'] for c in replay)
        known_rate = known_correct/metrics['known'] if metrics['known'] else None
        balanced = ((known_rate + 1 - metrics['false_accept_rate'])/2
                    if known_rate is not None and metrics['false_accept_rate'] is not None else None)
        result['points'].append({'threshold':threshold, 'metrics':metrics,
                                 'known_correct_rate':known_rate, 'balanced_accuracy':balanced})
    result['available'] = True
    candidates = [point for point in result['points']
                  if point['balanced_accuracy'] is not None and .2 <= point['threshold'] <= .8]
    if candidates:
        best = min(candidates, key=lambda point:(-point['balanced_accuracy'], point['metrics']['false_accept_rate'],
                                                abs(point['threshold']-DEFAULT_THRESHOLD), -point['threshold']))
        result['recommended_threshold'] = best['threshold']
        result['recommendation_metrics'] = best['metrics']
        result['reason'] = 'Рекомендация рассчитана на валидации в диапазоне 0.2–0.8. Проверьте её на отдельном итоговом наборе.'
    else:
        result['reason'] = 'Для рекомендации нужны и известные, и неизвестные люди. Сейчас показаны только доступные кривые.'
    return result
