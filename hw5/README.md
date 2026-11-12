# RL для вызова инструментов

Qwen3.5-2B, ToolACE, TRL. [Исходный проект](https://github.com/DenisMRH/llm-hse-course/tree/main/hw5_rl).

## Запуск

Нужны Docker, NVIDIA GPU и доступ к Hugging Face. Все библиотеки есть в Dockerfile.

```bash
docker build -t toolcall-rl .
docker run --rm -it --gpus device=0 --ipc=host \
  -e RL_EVAL_BATCH_SIZE=32 -e RL_USE_VLLM=1 \
  -v "$PWD":/workspace -v rlvr-hf-cache:/cache toolcall-rl
./run.sh
```

Скрипт готовит данные, оценивает базовую модель, обучает три конфигурации и собирает результаты.

Отдельный эксперимент:

```bash
python3 data.py
python3 train.py configs/exact.json
python3 evaluate.py runs/exact/merged --output results/exact
python3 summarize.py
```

Для других запусков замените `exact` на `partial` или `dr_grpo`. Возобновление: `python3 train.py configs/exact.json --resume runs/exact/checkpoints/checkpoint-25` — укажите существующий checkpoint.

## Данные и настройки

ToolACE revision: `6bda777c88d21e5a204703c1ee45597a8fa4f734`. Модель и revision совпадают с evaluate.py. Seed: 73. Версии библиотек заданы в Dockerfile и сохраняются в manifest.json.

Используется первая пара user/assistant. Имена инструментов заменяются на tool_0, tool_1 и т. д. Дубликаты удаляются; около 10% групп по хешу запроса и инструментов попадает в validation. Лимит промпта — 2048 токенов, ответа — 512. BFCL в обучении не используется.

| Конфигурация | Loss | Нормировка награды | Награда |
|---|---|---|---|
| exact | grpo | group | Полное совпадение вызовов |
| partial | grpo | group | 0.7 × F1 вызовов + 0.3 × F1 имён |
| dr_grpo | dr_grpo | none | Как partial |

Порядок вызовов не важен, повторы учитываются. Если вызов не нужен, награда — 1 за непустой ответ без вызовов. Некорректные и обрезанные ответы получают 0.

100 шагов, lr 5e-6, 4 генерации на запрос, generation batch 8, LoRA r=16, beta=0. Каждые 25 шагов проводится validation. Лучший по validation reward адаптер объединяется с базовыми весами.

## Файлы результатов

- `data/eda.json` — статистика данных.
- `runs/<name>/` — манифест, история обучения и сведения о выбранном checkpoint.
- `results/<name>/` — metrics.json и predictions.jsonl.
- `report/` — таблица, график наград, анализ примеров BFCL и PDF.

Прогоны выполнены на A100 80GB PCIe на devbox11. Использован образ с версиями из Dockerfile и causal-conv1d==1.7.0, собранным для A100. Батч валидации — 32, rollout — vLLM colocate. Валидация выполнялась каждые 25 шагов, сохранение для восстановления — каждые 5 шагов (RL_RECOVERY_STEPS=5).

В архиве оставлены код, конфигурации, EDA, PDF-отчёт, метрики и ответы BFCL, манифесты, истории обучения и сведения о выборе чекпойнтов. Обучающие выборки создаёт data.py; адаптеры и полные веса создаёт train.py. Они не включены в архив.

Перед повторным обучением перенесите существующие runs/ и results/ в отдельную папку: train.py не перезаписывает завершённые запуски. Указанные пути checkpoint в selection.json относятся к проведённым запускам; сами checkpoint не приложены.

Готовый отчёт находится в report/report.pdf. Для пересборки графика через summarize.py нужен Matplotlib, для пересборки PDF через render_report.sh — Pandoc и XeLaTeX с Arial. Для просмотра готового PDF они не нужны.
