"""Printable, anonymized experiment report. No photos, names or raw IDs."""
from datetime import datetime, timedelta, timezone
from html import escape


def percent(value):
    return '—' if value is None else f'{value * 100:.1f}%'.replace('.', ',')


def threshold_chart(calibration):
    points = calibration.get('points', [])
    if not points:
        return ''
    left, top, width, height = 70, 20, 980, 220
    low = min(point['threshold'] for point in points)
    high = max(point['threshold'] for point in points)

    def x(value):
        return left + (value-low)/(high-low or 1)*width

    def y(value):
        return top+(1-value)*height

    elements = []
    for tick in [0, .25, .5, .75, 1]:
        elements.append(f'<path d="M{left} {y(tick):.2f}H{left+width}" stroke="#dce3d5"/>')
        elements.append(f'<text x="58" y="{y(tick)+5:.2f}" text-anchor="end">{int(tick*100)}%</text>')
    for tick in [low, (low+high)/2, high]:
        elements.append(f'<text x="{x(tick):.2f}" y="275" text-anchor="middle">{tick:.3f}</text>')
    for key, color in [('accuracy','#275946'), ('false_accept_rate','#bd7144'), ('false_reject_rate','#657bb4')]:
        coordinates = ' '.join(f'{x(p["threshold"]):.2f},{y(p["metrics"][key]):.2f}'
                               for p in points if p['metrics'][key] is not None)
        if coordinates:
            elements.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="3"/>')
    recommendation = calibration.get('recommended_threshold')
    if recommendation is not None:
        elements.append(f'<path d="M{x(recommendation):.2f} {top}V{top+height}" stroke="#84946c" stroke-dasharray="5 5"/>')
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 300" role="img" '
            'aria-label="Зависимость верных ответов, ложных узнаваний и отказов от порога">'
            + ''.join(elements) + '</svg><p class="legend">Зелёная линия: верные ответы. '
            'Оранжевая: ложные узнавания. Синяя: ложные отказы. По горизонтали — порог.</p>')


def render_report(report, author=''):
    metrics = report['metrics']
    cases = report['cases']
    calibration = report.get('calibration', {})
    purpose = 'Валидация' if report.get('purpose') == 'validation' else 'Итоговый тест'
    author = ''.join(char for char in author if char.isprintable())[:120].strip()
    try:
        moment = datetime.fromisoformat(report['generated_at']).astimezone(timezone(timedelta(hours=3)))
        date = moment.strftime('%d.%m.%Y %H:%M МСК')
    except (ValueError, KeyError, TypeError):
        date = 'Не указана'
    gallery = report.get('gallery', {})
    threshold = report.get('threshold')
    threshold_text = '—' if threshold is None else f'{threshold:.3f}'
    summary = [
        ('Режим', purpose), ('Порог проверки', threshold_text),
        ('Людей в базе', str(gallery.get('people', '—'))), ('Эталонных снимков', str(gallery.get('samples', '—'))),
        ('Всего тестовых снимков', str(metrics['total'])), ('Известные / неизвестные', f'{metrics["known"]} / {metrics["unknown"]}'),
        ('Верные ответы', f'{metrics["correct"]} из {metrics["total"]} ({percent(metrics["accuracy"])})'),
        ('Ложные узнавания', f'{metrics["false_accepts"]} из {metrics["unknown"]} ({percent(metrics["false_accept_rate"])})'),
        ('Ложные отказы', f'{metrics["false_rejects"]} из {metrics["known"]} ({percent(metrics["false_reject_rate"])})'),
        ('Ошибки выбора имени', str(metrics['wrong_identity'])),
        ('Среднее время обработки', '—' if metrics['mean_ms'] is None else f'{metrics["mean_ms"]:.1f} мс'),
    ]
    summary_html = ''.join(f'<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>' for label,value in summary)
    participants = {}

    def label(person_id):
        if person_id is None:
            return 'Неизвестный'
        if person_id not in participants:
            participants[person_id] = f'Участник {len(participants)+1}'
        return participants[person_id]

    rows = []
    for index, case in enumerate(cases, 1):
        expected, predicted = case['expected_id'], case['predicted_id']
        score = case.get('similarity')
        quality = case.get('quality')
        issues = 'Нет данных'
        if quality is not None:
            issues = '; '.join(issue['message'] for issue in quality.get('issues', [])) or 'Замечаний нет'
        values = [f'Снимок {index}', label(expected), label(predicted), '—' if score is None else f'{score:.3f}',
                  'Верно' if expected==predicted else 'Ошибка', issues]
        rows.append('<tr>' + ''.join(f'<td>{escape(value)}</td>' for value in values) + '</tr>')
    details = ('<table class="cases"><thead><tr><th>Случай</th><th>Ожидалось</th><th>Ответ</th>'
               '<th>Сходство</th><th>Результат</th><th>Качество снимка</th></tr></thead><tbody>'
               + ''.join(rows) + '</tbody></table>') if rows else '<p>Нет результатов. Добавьте независимые тестовые снимки.</p>'
    recommendation = calibration.get('recommended_threshold')
    recommendation_text = ('Рекомендованный порог на сетке: '+f'{recommendation:.3f}'
                           if recommendation is not None else 'Рекомендация порога отсутствует')
    quality_cases = [case for case in cases if case.get('quality') is not None]
    warned = sum(bool(case['quality'].get('issues')) for case in quality_cases)
    quality_text = (f'Замечания к изображению: {warned} из {len(quality_cases)} оценённых снимков.'
                    if quality_cases else 'Оценки качества изображения пока нет.')
    models = report.get('models', {})
    return f'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FaceLab — отчёт об эксперименте</title><style>
*{{box-sizing:border-box}}body{{font-family:"Segoe UI",Arial,sans-serif;color:#263f35;background:#eef1e8;margin:0;padding:40px}}
main{{max-width:1060px;margin:auto;background:#fff;padding:48px}}h1{{font-size:34px;line-height:1.2;margin:8px 0 18px}}h2{{font-size:22px;margin:34px 0 16px}}
p{{font-size:14px;line-height:1.8}}.meta,.note,.legend{{color:#667b61}}.meta{{margin:0}}table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{padding:11px 12px;border-bottom:1px solid #dce3d5;text-align:left;vertical-align:top;overflow-wrap:anywhere}}th{{font-weight:600}}
.summary th{{width:52%}}.cases{{font-size:12px}}.cases td:last-child{{max-width:240px}}svg{{display:block;width:100%;height:auto;margin-top:22px}}
svg text{{font-family:"Segoe UI",Arial,sans-serif;font-size:15px;fill:#65775c}}.note{{padding:14px 18px;background:#f2f5ec;font-size:13px}}
@media print{{body{{background:white;padding:0}}main{{padding:0;max-width:none}}h2{{break-after:avoid}}tr{{break-inside:avoid}}.note{{background:white;border:1px solid #dce3d5}}@page{{size:A4;margin:17mm}}}}
</style></head><body><main>
<p class="meta">FaceLab {escape(report.get('version',''))} · {escape(date)}</p>
<h1>Отчёт об эксперименте</h1><p>Автор: {escape(author) if author else 'не указан'}<br>Режим: <strong>{purpose}</strong></p>
<p class="note">Участники и названия снимков обезличены. Изображения и исходные идентификаторы в отчёт не включены.
Для PDF используйте печать браузера и «Сохранить как PDF».</p>
<h2>Состав набора и результаты</h2><table class="summary">{summary_html}</table>
<p>Модели: {escape(models.get('detection','YuNet'))}, {escape(models.get('recognition','SFace'))}.
Использованы предобученные веса. Добавление эталонов не переобучает нейросеть.</p>
<h2>Сравнение порогов</h2><p><strong>{recommendation_text}</strong></p>
<p>{escape(calibration.get('reason','Нет данных для сравнения.'))}</p>
{threshold_chart(calibration)}
<p>{escape(calibration.get('criterion',''))}</p>
<p>Случаев для сравнения: {calibration.get('eligible',0)}. Исключено из сравнения: {calibration.get('excluded',0)}.</p>
<h2>Случаи и ошибки</h2>{details}<p>{escape(quality_text)}</p>
<h2>Методика и ограничения</h2>
<p>{escape(report.get('protocol',''))}</p>
<p>Доля ложных узнаваний считается среди неизвестных, ложных отказов — среди известных. Ошибка выбора другого имени учитывается отдельно.
Если примеров соответствующего типа нет, метрика не определена. Сходство не является вероятностью правильного ответа.</p>
<p>Оценка резкости и освещения эвристическая. Она не гарантирует правильного узнавания. Итоговая точность на новых данных
может отличаться от результата этой выборки. Порог выбирается только на валидации и затем проверяется на отдельном тесте.</p>
<p class="meta">Метод: OpenCV YuNet + SFace. Документация: https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html</p>
</main></body></html>'''
