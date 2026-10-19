"""Enrollment, open-set recognition and independent evaluation."""
import base64
from datetime import datetime, timezone
from io import BytesIO
from threading import RLock
from time import perf_counter
import unicodedata

from .matching import DEFAULT_THRESHOLD, identify, summarize


def image_url(image, size=None):
    image = image.copy()
    if size:
        image.thumbnail(size)
    buffer = BytesIO()
    image.save(buffer,format='JPEG',quality=90)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


class FaceService:
    def __init__(self, store, engine):
        self.store, self.engine = store, engine
        self.lock = RLock()

    def enroll(self, name, files):
        name = unicodedata.normalize('NFC',' '.join(name.split()))
        if not 1 <= len(name) <= 60 or any(unicodedata.category(c).startswith('C') for c in name):
            raise ValueError('Укажите имя длиной от 1 до 60 символов без управляющих знаков.')
        if name.casefold() == 'неизвестный':
            raise ValueError('Выберите другое имя: «Неизвестный» используется для отказа от узнавания.')
        if not 1 <= len(files) <= 20:
            raise ValueError('Выберите от 1 до 20 фотографий. Для качества рекомендуем 3–5 разных снимков.')
        with self.lock:
            if self.store.name_exists(name.casefold()):
                raise ValueError('Человек с таким именем уже существует. Выберите другое имя.')
            seen = self.store.reference_hashes()
            samples, thumbnail = [], None
            for filename,data in files:
                analysis = self.engine.process(data)
                if len(analysis.faces) != 1:
                    raise ValueError(f'«{filename}»: для регистрации нужно ровно одно лицо. Найдено: {len(analysis.faces)}.')
                if analysis.image_hash in seen:
                    raise ValueError(f'«{filename}»: повтор изображения. Нужны разные снимки, отсутствующие в базе.')
                seen.add(analysis.image_hash)
                face=analysis.faces[0]
                x,y,w,h = face.box
                if min(w,h) < 48:
                    raise ValueError(f'«{filename}»: лицо слишком маленькое. Выберите более крупный портрет.')
                samples.append((analysis.image_hash,face.embedding.tolist()))
                if thumbnail is None:
                    crop=analysis.image.crop((x,y,x+w,y+h))
                    crop.thumbnail((160,160))
                    out=BytesIO(); crop.save(out,format='JPEG',quality=85)
                    thumbnail=out.getvalue()
            person=self.store.add(name,name.casefold(),thumbnail,samples)
            return person

    def _recognize(self, analysis, threshold):
        gallery=self.store.gallery()
        names={p['id']:p['name'] for p in self.store.people()}
        # Also validate the threshold when no faces are present.
        identify([1.], {}, threshold)
        results=[]
        for face in analysis.faces:
            result=identify(face.embedding,gallery,threshold)
            result.update({'name':names.get(result['person_id'],'Неизвестный'),
                           'box':list(face.box),'detector_score':face.detector_score})
            results.append(result)
        return results

    def recognize(self, data, threshold=DEFAULT_THRESHOLD):
        with self.lock:
            start=perf_counter()
            analysis=self.engine.process(data)
            faces=self._recognize(analysis,threshold)
            elapsed=(perf_counter()-start)*1000
            return {'image':image_url(analysis.image),'width':analysis.image.width,'height':analysis.image.height,
                    'faces':faces,'threshold':threshold,'elapsed_ms':round(elapsed,1),
                    'gallery_empty':not bool(self.store.people())}

    def delete(self, person_id):
        with self.lock:
            self.store.delete(person_id)

    def evaluate(self, files, expected_id, threshold=DEFAULT_THRESHOLD):
        if not 1<=len(files)<=20:
            raise ValueError('Выберите от 1 до 20 тестовых фотографий.')
        with self.lock:
            people={p['id']:p['name'] for p in self.store.people()}
            if not people:
                raise ValueError('Сначала добавьте хотя бы одного человека в базу.')
            if expected_id is not None and expected_id not in people:
                raise ValueError('Выбранный человек отсутствует в базе.')
            existing=self.store.evaluations()
            if existing and abs(existing[0]['threshold']-threshold)>1e-9:
                raise ValueError('Изменён порог. Сначала начните новую проверку, чтобы не смешивать результаты.')
            seen={c['image_hash'] for c in existing}
            references=self.store.reference_hashes()
            cases=[]
            for filename,data in files:
                start=perf_counter()
                analysis=self.engine.process(data)
                if analysis.image_hash in references:
                    raise ValueError(f'«{filename}»: это эталон из базы. Для проверки нужен другой снимок.')
                if analysis.image_hash in seen:
                    raise ValueError(f'«{filename}»: снимок уже участвовал в этой проверке.')
                if len(analysis.faces)!=1:
                    raise ValueError(f'«{filename}»: для оценки нужно ровно одно лицо. Найдено: {len(analysis.faces)}.')
                prediction=self._recognize(analysis,threshold)[0]
                cases.append({'filename':filename,'image_hash':analysis.image_hash,
                              'expected_id':expected_id,'expected_name':people.get(expected_id,'Неизвестный'),
                              'predicted_id':prediction['person_id'],'predicted_name':prediction['name'],
                              'similarity':prediction['similarity'],'correct':expected_id==prediction['person_id'],
                              'threshold':threshold,'elapsed_ms':round((perf_counter()-start)*1000,1)})
                seen.add(analysis.image_hash)
            self.store.add_evaluations(cases)
            return self.report()

    def report(self):
        with self.lock:
            cases=self.store.evaluations()
            return {'generated_at':datetime.now(timezone.utc).isoformat(),'metrics':summarize(cases),
                    'threshold':cases[0]['threshold'] if cases else None,'cases':cases,
                    'models':{'detection':'YuNet 2023mar','recognition':'SFace 2021dec'},
                    'protocol':'Одно лицо на фото; эталоны и повторные изображения исключены по хешу RGB-пикселей. '
                               'Кадры одной серии и пересохранённые с потерями копии необходимо исключить вручную. '
                               'Порог настраивается на валидации; итоговый тест должен быть отдельным.'}
