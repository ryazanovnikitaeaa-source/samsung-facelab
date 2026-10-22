from facelab.calibration import compare_thresholds
from facelab.matching import summarize
from facelab.reporting import render_report


def report(cases=None):
    cases = cases or []
    return {'generated_at':'2026-10-22T12:00:00+00:00','purpose':'validation',
            'metrics':summarize(cases),'threshold':.363 if cases else None,
            'cases':cases,'calibration':compare_thresholds(cases,'validation',.363),
            'models':{'detection':'YuNet','recognition':'SFace'},
            'gallery':{'people':2,'samples':6},'version':'1.1.0','protocol':'Отдельные снимки'}


def test_empty_report_is_printable_and_honest():
    html = render_report(report())
    assert '<!doctype html>' in html.lower()
    assert '@media print' in html
    assert 'Нет результатов' in html
    assert 'nan' not in html.lower()


def test_html_escapes_author_and_does_not_include_personal_names_or_filenames():
    cases = [{'filename':'private-portrait.png','expected_id':'p1','expected_name':'PRIVATE NAME',
              'predicted_id':'p1','predicted_name':'PRIVATE NAME','candidate_id':'p1',
              'similarity':.8,'correct':True,'elapsed_ms':5,'threshold':.363}]
    html = render_report(report(cases), author='<script>alert(1)</script>')
    assert '<script>' not in html
    assert '&lt;script&gt;' in html
    assert 'PRIVATE NAME' not in html and 'private-portrait.png' not in html
    assert 'Участник 1' in html


def test_validation_report_has_actual_curve_and_denominators():
    cases = [{'expected_id':'p','predicted_id':'p','candidate_id':'p','similarity':.8,
              'elapsed_ms':1,'correct':True,'threshold':.363},
             {'expected_id':None,'predicted_id':None,'candidate_id':'p','similarity':.2,
              'elapsed_ms':1,'correct':True,'threshold':.363}]
    html = render_report(report(cases))
    assert '<svg' in html
    assert 'Валидация' in html
    assert '0 из 1' in html
    assert 'Итоговая точность на новых данных' in html
