# HW7 · Квантизация и дистилляция Qwen3

[Отчёт](hw7_report.md) · [Метрики девяти экспериментов](artifacts_compression_hw/results.csv) · [Чекпоинты на Яндекс Диске](https://disk.360.yandex.ru/d/ylESTDEwpMIgLQ)

| Чекпоинт на Диске | Описание | Quality на 256 примерах dev |
|---|---|---:|
| `awq_massive/` | Qwen3-8B, AWQ W4A16 asymmetric, group size 128, калибровка на 128 примерах MASSIVE train | 0,329541 |
| `stufent_sft/` | Qwen3-8B после дистилляции от Qwen3-32B, 1024 примера, lr 5e-5, 2 эпохи | 0,605893 |

## Запуск

Из корня клонированного репозитория запустите на Linux-девбоксе с Python 3.11 и CUDA GPU (исходный расчёт использовал A100 80 ГБ):

```bash
cd hw7
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_hw7.py --plan
python run_hw7.py --output ./runs/experiment-1
```

Скрипт загружает MASSIVE ru-RU, WikiText и публичные Qwen3-8B/32B, формирует train/dev, получает ответы учителя, обучает учеников и сохраняет результаты. Для кэша загрузок можно задать HF_HOME. Для каждого полного запуска укажите новую папку.

Предсказания, метрики и модели появятся в runs/experiment-1/artifacts_compression_hw. 


