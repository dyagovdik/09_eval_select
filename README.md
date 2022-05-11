RS School Machine Learning course

09 module Evaluation-selection homework

В этом проекте используется [Forest train dataset](https://www.kaggle.com/competitions/forest-cover-type-prediction) датасет.

## Краткое руководство:
1. Создайте гит-клон репозитория на ваш компьютер.
2. Загрузите [Forest train dataset](https://www.kaggle.com/competitions/forest-cover-type-prediction) датасет, сохраните *.csv локально (по умолчанию "path is *data/.csv"* в корне репозитория).
3. Убедитесь, что у вас установлен Python 3.9 and [Poetry](https://python-poetry.org/docs/) (я использовал 1.1.13).
4. Установите зависимости проекта (*запустите эту и следующие команды в терминале, из корня клонированного репозитория*):
```sh
poetry install --no-dev
```
5. Запустите train с помощью этой команды:
```sh
poetry run train -d <path to csv with data> -s <path to save trained model>
```
Вы можете настроить дополнительные параметры (например, гиперпараметры) в CLI. Чтобы получить их полный список, используйте help:
```sh
poetry run train --help
```
По умолчанию установлены следующие параметры
  * -d, --dataset-path FILE         [default: data/train.csv]
  * -s, --save-model-path FILE      [default: data/model.joblib]
  * --random-state INTEGER          [default: 42]
  * --test-split-ratio FLOAT RANGE  [default: 0.2; 0<x<1]
  * --use-model TEXT                [default: LogReg, можно выбрать RFC]
  * --use-scaler BOOLEAN            [default: True]
  * --use-feature-selection INTEGER [default: 0, можно выбрать 1 - Principal component analysis, 2 - SelectFromModel]
  * --max_iter INTEGER              [default: 1000]
  * --logregc FLOAT                 [default: 1.0]
  * --n_estimators INTEGER          [default: 100]
  * --criterion TEXT                [default: entropy]
  * --max_depth INTEGER             [default: 10]
  * --bootstrap BOOLEAN             [default: True]

6. Запустите пользовательский интерфейс MLflow, чтобы просмотреть информацию о проведенных экспериментах:
```sh
poetry run mlflow ui
```

## Разработка

Код в этом репозитории протестирован, отформатирован с помощью black, flake8 и прошел проверку типов mypy перед комментированием в репозиторий.

Установите все требования (включая требования dev) :
```
poetry install
```
После этого вы можете пользоваться установленными инструментами:
```
poetry run black (or flake8, mypy, pytest) *.py
poetry run black (or flake8, mypy, pytest) srs/name/*.py
```
Для удобства можно запустить все сеансы тестирования одной строкой: 
```
poetry run nox
```

## Для проверяющего
1 (№ задания). В проекте использован [Forest train dataset](https://www.kaggle.com/competitions/forest-cover-type-prediction) датасет.

2. Проект оформлен как пакет Python, в том числе использован макет src.

3. Код собственно опубликован на Github, в том числе с оставлением поэтапных комментариев (более 30).

4. В проекте для управления пакетами и зависимостями используется *Poetry*.

5. Датасет сохранен локально в "path is *data/.csv"* в корне репозитория и сама папка добавлена в .gitignore.

6. Написан скрипт, который обучает модель и сохраняет ее в файл. Скрипт запускается из терминала, получает некоторые аргументы, такие как путь к данным, конфигурации модели и т.д. Для создания CLI использован click. Скрипт также добавлен в pyproject.toml.

7. Выбраны и рассчитаны 3 показателя (accuracy, precision, f1score), в том числе с использование кросс-валидации.

8. Проводены эксперименты, которые можно отследить в MLFlow - three different sets of hyperparameters for each model, two different feature engineering techniques for each model, two different ML models. Скриншоты прилагаются

![MLFlow](https://user-images.githubusercontent.com/99845094/167859394-f8f465d4-3e06-4f74-9f1a-2e8dda5a72ac.PNG)

Чтобы получить такие же результаты нужно запустить следующие команды:
```
poetry run train
poetry run train --max_iter 1200 --logregc 12
poetry run train --max_iter 1500 --logregc 15

poetry run train --use-feature-selection 1
poetry run train --use-feature-selection 1 --max_iter 1200 --logregc 12
poetry run train --use-feature-selection 1 --max_iter 1500 --logregc 15

poetry run train --use-feature-selection 2
poetry run train --use-feature-selection 2 --max_iter 1200 --logregc 12
poetry run train --use-feature-selection 2 --max_iter 1500 --logregc 15

poetry run train --use-model RFC
poetry run train --use-model RFC --max_depth 15 --n_estimators 150
poetry run train --use-model RFC --max_depth 20 --n_estimators 200

poetry run train --use-model RFC --use-feature-selection 1
poetry run train --use-model RFC --use-feature-selection 1 --max_depth 15 --n_estimators 150
poetry run train --use-model RFC --use-feature-selection 1 --max_depth 20 --n_estimators 200

poetry run train --use-model RFC --use-feature-selection 2
poetry run train --use-model RFC --use-feature-selection 2 --max_depth 15 --n_estimators 150
poetry run train --use-model RFC --use-feature-selection 2 --max_depth 20 --n_estimators 200

```
9.-

10. Я постарался написать README удобным для Вас.

11. Для запуска тестов необходимо выполнить следующую команду:
```
poetry run pytest
```
12. Код отфоматирован с помощью black и выровнен с помощью flake8.

![black](https://user-images.githubusercontent.com/99845094/167859091-c06b72c9-a04c-4f14-b328-9be36ee7c444.PNG)

![flake8](https://user-images.githubusercontent.com/99845094/167859121-1988834c-0af7-49ee-9db1-b12ee3e35385.PNG)

13. Код аннториван с помощью mypy.

![mypy](https://user-images.githubusercontent.com/99845094/167859144-8646069e-376d-4b83-9d30-62067da3d7bb.PNG)

14. Все сеансы тестирования и форматированы сведены в одну команду с помощью nox.

![nox](https://user-images.githubusercontent.com/99845094/167859655-03bfc990-f843-4ae3-a3ac-a55791ebb0a2.PNG)

15.-
