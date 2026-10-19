# Источники и сторонние компоненты

## Модели
1. **YuNet**, OpenCV Zoo, обнаружение лиц. Файл `face_detection_yunet_2023mar.onnx`. Лицензия MIT, copyright Shiqi Yu. [Репозиторий](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet). Копия лицензии: `licenses/YuNet-MIT.txt`.
2. **SFace**, OpenCV Zoo, признаки лица. Файл `face_recognition_sface_2021dec.onnx`. Лицензия Apache-2.0. [Репозиторий](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface). Копия лицензии: `licenses/SFace-Apache-2.0.txt`.
3. [Официальное руководство OpenCV](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html): обнаружение, выравнивание, признаки, косинусное сравнение. Значение 0.363 указано для протокола на LFW и служит только начальной настройкой проекта. Чужие опубликованные показатели точности не выдаются за результаты FaceLab.

Контрольные суммы ONNX сверены с Git LFS-указателями официального репозитория:
```text
YuNet 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
SFace 0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79
```

## Технические изображения
- **Astronaut / Eileen Collins**: [описание scikit-image](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut), [оригинал в репозитории](https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/data/astronaut.png). В документации источник NASA и public domain.
- **Grace Hopper**: [пример Matplotlib](https://github.com/matplotlib/matplotlib/blob/v3.10.0/lib/matplotlib/mpl-data/sample_data/grace_hopper.jpg), фотография ВМС США из набора примеров Matplotlib. Используется только как другой портрет технической проверки.
- Групповой кадр в демонстрации составлен из этих двух портретов. Изменённая яркость первого портрета используется только для проверки конвейера и **не является независимым тестовым снимком**.
- Скриншоты в презентации и видео показывают реально работающий FaceLab на технических примерах. Это не измерение точности на пользовательском наборе.

## Программные компоненты
- [OpenCV](https://opencv.org/), выполнение нейросетей на CPU.
- [FastAPI](https://fastapi.tiangolo.com/tutorial/request-files/), локальный HTTP API и загрузка файлов.
- [SQLite](https://www.sqlite.org/), локальная база.
- [Pillow](https://pillow.readthedocs.io/), чтение и нормализация изображений.
- [NumPy](https://numpy.org/), векторы и сравнение признаков.

## Помощь при разработке
При разработке кода, интерфейса, тестов и материалов использовался ИИ-ассистент Codex. Веса моделей разработаны третьими лицами. Подготовка собственного набора, валидация, анализ ошибок и проверка понимания для защиты требуют участия студента. Нельзя утверждать, что модели обучены студентом с нуля.
