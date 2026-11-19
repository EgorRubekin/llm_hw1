# ДЗ-6. Ускорение инференса Qwen3-8B

[Отчёт о результатах](report/report.md) и [все завершённые замеры в CSV](report/measurements.csv).

Исходные 640 запросов из `data/requests.jsonl` использованы без изменений. SHA-256 файла `67c48379e5ae2608f7ca08650fd346facdc01f766a3e8eecea6459d7cec00ebc`.

`serve.sh` задаёт пять вариантов сервера. Baseline использует настройки vLLM по умолчанию. Два варианта меняют планирование prefill или режим производительности. N-gram запускается на Model Runner V1, рядом проверяется V1 без спекдекодирования.

## Запуск с Docker

Нужен исходный образ vLLM из Dockerfile и отдельная папка для окружения AIPerf, ревизии Qwen3-8B и кешей компиляции. На devbox11 использовался уже локально доступный образ с тем же digest и временное хранилище в `/dev/shm`.

```bash
bash prepare_devbox.sh "$SCRATCH"
bash run_tuning.sh "$GPU_UUID" "$SCRATCH"
bash run_campaign.sh "$GPU_UUID" "$SCRATCH" spec
bash run_campaign.sh "$GPU_UUID" "$SCRATCH" load
python3 summarize.py
```

Все этапы идут последовательно. Для одного замера есть `run_series.sh GPU_UUID SCRATCH CONFIG ЭТАП RPM`. Этап `tuning` прогоняет 640 запросов, `spec` — 160, `load` вызывает `load_test.sh` на 300 секунд. Успешные замеры записываются под `results/<этап>/<конфигурация>/<RPM>/profile/`, там лежат агрегаты и записи запросов. Связанные журналы, команды запуска vLLM и метрики сервера лежат рядом с папками RPM. Прерванные попытки сохранены в `failed/`.

Для сборки графиков нужен Matplotlib. Все графики в отчёте построены из экспортов AIPerf и серверных счётчиков, сохранённых рядом с запусками.
