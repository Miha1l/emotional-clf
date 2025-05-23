# Распознавание эмоциональной речи

Данное консольное приложения предназначено для обучения и тестирования моделей HuBERT
для задачи классификации эмоций на примере датасета Dusha.
В данный момент приложения поддерживает только бинарную классификацию на эмоции радости и печали.

## Обученные модели

Для обучения использовался набор данных [Dusha(crowd)](https://github.com/salute-developers/golos/tree/master/dusha#dusha-dataset)

| Описание модели                              | Ссылка                                                                                                       |
|----------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| Модель для классификации                     | [hubert-clf](https://drive.google.com/file/d/1xiTUK7hO8Q4JNrZjVdhcRla8mHE4poJ7/view?usp=drive_link)          |
| Модель классификации, обученная на триплетах | [hubert-triplets-clf](https://drive.google.com/file/d/1GhvEO9ipMNiNR4EcWEKjnpJpV82PPEb_/view?usp=drive_link) |

## Использование приложения
### Режим predict

В этом режиме можно получить предсказания для переданного набора аудиофайлов.

Пример запуска:

```commandline
python main.py predict -d ./samples -m ./trained_models/hubert-base-dusha-ft-bin-clf
```

### Режим train

В этом режиме происходит дообучение модели HuBERT Base для бинарной классификации.
Перед запуском необходимо подготовить `.csv` файл в формате, зависящим от выбранной функции потерь:

_CrossEntropy_:

| audio       | label |
|-------------|-------|
| sample1.wav | 0     |
| sample2.wav | 1     |

Где 0 соответствует эмоции радости, 1 - печали.

_Triplet_:

| anchor        | positive      | negative      |
|---------------|---------------|---------------|
| positive1.wav | positive2.wav | sad1.wav      |
| sad1.wav      | sad2.wav      | positive2.wav |

Пример запуска:

```commandline
python main.py train -d ./samples -o train.csv --epochs 8
```

### Режим test

В этом режиме происходит тестирование переданной модели классификации.
Перед запуском необходимо подготовить `.csv` файл в формате, аналогичном формату для обучения.
Расчитываемые метрики: Accuracy, Recall, Precision, F1. Метрики сохраняются по умолчанию в файл `metrics.json`.

Пример запуска:

```commandline
python main.py test -d ./samples -f ./samples/train.csv -m ./trained_models/hubert-base-dusha-ft-bin-clf
```