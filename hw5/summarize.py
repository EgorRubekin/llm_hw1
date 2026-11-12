"""Сборка отчёта по артефактам ToolACE и BFCL."""
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "report"
RUNS = ("base", "exact", "partial", "dr_grpo")
GROUPS = ("simple", "multiple", "parallel", "parallel_multiple", "irrelevance")
COLORS = ("#31566e", "#237967", "#bf652a")


def load(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def training_plots(histories):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7.1), layout="constrained")
    for axis, (name, history), color in zip(axes, histories.items(), COLORS, strict=True):
        training = [(row["step"], row["reward"]) for row in history if "reward" in row]
        validation = [(row["step"], row["eval_reward"]) for row in history if "eval_reward" in row]
        axis.plot(*zip(*training), color=color, linewidth=1, alpha=0.55, label="Обучение")
        axis.plot(*zip(*validation), color=color, marker="o", linewidth=2, label="Валидация")
        axis.set_title(name, loc="left")
        axis.set_ylabel("Награда")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#e1e8ec")
    axes[0].legend(frameon=False, loc="upper right")
    axes[-1].set_xlabel("Шаг обучения")
    fig.savefig(REPORT / "rewards.png", dpi=175)
    plt.close(fig)


def paired_counts(left, right):
    if left.keys() != right.keys():
        raise ValueError("Пары BFCL содержат разные запросы")
    gains = sum(not left[key]["check"]["valid"] and right[key]["check"]["valid"] for key in left)
    losses = sum(left[key]["check"]["valid"] and not right[key]["check"]["valid"] for key in left)
    return gains, losses


def error_type(record, reference):
    if record["finish_reason"] == "length" or not record["response"].strip():
        return "Пустой или обрезанный ответ"
    expected = Counter(name for call in reference["ground_truth"] for name in call)
    actual = Counter(name for call in record["calls"] for name in call)
    extra = actual - expected
    missing = expected - actual
    if extra and missing:
        return "Замена или несколько ошибок в вызовах"
    if extra:
        return "Лишний вызов"
    if missing:
        return "Пропущенный вызов"
    return "Ошибка аргументов или формата"


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    args = parser.parse_args()

    metrics = {name: load(ROOT / "results" / name / "metrics.json") for name in RUNS}
    predictions = {
        name: {r["id"]: r for r in read_jsonl(ROOT / "results" / name / "predictions.jsonl")}
        for name in RUNS
    }
    references = {r["id"]: r for r in read_jsonl(ROOT / "bfcl.jsonl")}
    if any(set(values) != set(references) for values in predictions.values()):
        raise ValueError("Состав примеров и предсказаний BFCL различается")

    datasets = load(ROOT / "data" / "eda.json")
    selections = {name: load(args.runs_dir / name / "selection.json") for name in RUNS[1:]}
    histories = {name: load(args.runs_dir / name / "history.json") for name in RUNS[1:]}
    manifests = {name: load(args.runs_dir / name / "manifest.json") for name in RUNS[1:]}
    if len({json.dumps(m["data_sha256"], sort_keys=True) for m in manifests.values()}) != 1:
        raise ValueError("SHA-256 обучающей и валидационной выборок различаются по запускам")
    rollout_backends = {m["trainer"]["use_vllm"] for m in manifests.values()}
    if len(rollout_backends) != 1:
        raise ValueError("Движки генерации rollout различаются по запускам")
    rollout_engine = "vLLM в режиме colocate" if rollout_backends.pop() else "Transformers"
    for name in RUNS[1:]:
        settings = manifests[name]["trainer"]
        expected_steps = set(range(settings["eval_steps"], settings["max_steps"] + 1, settings["eval_steps"]))
        evaluations = {row["step"]: row["eval_reward"] for row in histories[name] if "eval_reward" in row}
        if set(evaluations) != expected_steps:
            raise ValueError(f"В истории {name} отсутствуют результаты всех полных валидаций")
        chosen_step = int(Path(selections[name]["checkpoint"]).name.removeprefix("checkpoint-"))
        if not math.isclose(evaluations[chosen_step], selections[name]["score"]):
            raise ValueError(f"В истории и выборе чекпойнта {name} указаны разные награды")

    REPORT.mkdir(exist_ok=True)
    training_plots(histories)
    counts = datasets["counts"]
    lengths = datasets["prompt_tokens"]
    best = max(RUNS, key=lambda name: metrics[name]["overall"])
    report = [
        "# RL для вызова инструментов",
        "",
        (
            f"Лучший результат BFCL показал `{best}` с **{metrics[best]['overall']:.2%}** "
            f"при **{metrics['base']['overall']:.2%}** у базовой Qwen3.5-2B."
        ),
        "",
        "## Результаты BFCL",
        "",
        (
            "BFCL содержит 350 примеров. Overall равен среднему по пяти группам. "
            "В Multiple и Irrelevance обычная и live-категория получают одинаковый вес. "
            "Оценщик, промпты и greedy-декодирование одинаковы у всех четырёх моделей."
        ),
        "",
        "| Run | Overall | Simple | Multiple | Parallel | Parallel multiple | Irrelevance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in metrics.items():
        values = [m["overall"], *(m["groups"][group] for group in GROUPS)]
        report.append("| " + " | ".join([f"`{name}`", *(f"{v:.2%}" for v in values)]) + " |")
    report += [
        "",
        "## Данные и обучение",
        "",
        (
            f"Из {datasets['dialogues']} диалогов ToolACE в train вошло "
            f"{counts['train/call'] + counts['train/no_call']} примеров "
            f"({counts['train/call']} с вызовом и {counts['train/no_call']} без вызова), "
            f"в validation — {counts['validation/call'] + counts['validation/no_call']} "
            f"({counts['validation/call']} с вызовом и {counts['validation/no_call']} без вызова). "
            f"После удаления {counts['duplicate']} дубликатов отфильтровано {counts['too_long']} "
            f"промпта длиннее 2048 токенов. Медиана длины промпта {lengths['0.5']} токенов, "
            f"P90 {lengths['0.9']}, максимум {lengths['1']}."
        ),
        "",
        (
            "Распределение train и validation фиксировалось по хешу сообщения и схем инструментов с seed 73. "
            "Обучение использовало LoRA r=16, 100 шагов, 4 генерации на запрос, максимум 512 токенов "
            "в генерации и learning rate 5e-6. Веса запускались на A100 80GB PCIe №0 devbox11. "
            f"Батч валидации был {manifests['exact']['trainer']['per_device_eval_batch_size']}. "
            f"Во всех трёх экспериментах движок генерации rollout — {rollout_engine}. "
            "Каждые 25 шагов считалась награда по четырём генерациям на каждый валидационный промпт. "
            "Для BFCL использованы объединённые веса лучшего по награде validation чекпойнта каждого запуска."
        ),
        "",
        "## Как изменились результаты",
        "",
    ]
    for left, right, decision in (
        ("exact", "partial", "Переход к частичной награде при том же GRPO со стандартизацией в группе"),
        ("partial", "dr_grpo", "Переход к Dr. GRPO при той же частичной награде. Изменились нормировка loss и масштабирование reward"),
    ):
        gains, losses = paired_counts(predictions[left], predictions[right])
        delta = (metrics[right]["overall"] - metrics[left]["overall"]) * 100
        group_changes = [
            (group, (metrics[right]["groups"][group] - metrics[left]["groups"][group]) * 100)
            for group in GROUPS
        ]
        changes = ", ".join(f"`{group}` ({value:+.2f} п. п.)" for group, value in group_changes if abs(value) > 1e-9)
        report.append(
            f"- **`{left}` → `{right}`.** {decision}. Overall изменился на {delta:+.2f} п. п. "
            f"На уровне примеров верными стали {gains} ответов, неверными — {losses}. "
            f"Изменения по группам — {changes}."
        )
    report += [
        "",
        "## Валидационная награда",
        "",
        "| Run | Шаг выбранного чекпойнта | Награда | Группы с одинаковой наградой (train) | BFCL Overall |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, selection in selections.items():
        step = Path(selection["checkpoint"]).name.removeprefix("checkpoint-")
        zero_std = mean(row["frac_reward_zero_std"] for row in histories[name] if "frac_reward_zero_std" in row)
        report.append(f"| `{name}` | {step} | {selection['score']:.4f} | {zero_std:.1%} | {metrics[name]['overall']:.2%} |")
    zero_exact = mean(r["frac_reward_zero_std"] for r in histories["exact"] if "frac_reward_zero_std" in r)
    zero_partial = mean(r["frac_reward_zero_std"] for r in histories["partial"] if "frac_reward_zero_std" in r)
    report += [
        "",
        (
            f"Доля групп с одинаковыми наградами при переходе от exact к partial изменилась с "
            f"{zero_exact:.1%} до {zero_partial:.1%}. Значит, при частичной награде больше групп "
            "содержали ответы с разной оценкой и давали сигнал для обучения."
        ),
        "",
        "![Динамика тренировочной и валидационной награды](rewards.png){width=89%}",
        "",
        "*Динамика тренировочной и валидационной награды.*",
        "",
        (
            "Внутри запуска динамика награды показывает, как модель приспосабливается к своей функции "
            "на ToolACE. Сравнение моделей между запусками сделано на едином BFCL. Абсолютный уровень "
            "exact и partial reward зависит от формулы. При одинаковых наградах всех ответов в группе "
            "центрирование обнуляет advantage, поэтому группа при beta=0 не даёт сигнала policy gradient."
        ),
        "",
        "## Ошибки вызова инструментов",
        "",
    ]
    for name in dict.fromkeys(("base", best)):
        errors = [
            (error_type(record, references[key]), key, record["category"])
            for key, record in predictions[name].items()
            if not record["check"]["valid"]
        ]
        counts = Counter(kind for kind, _, _ in errors)
        cases = ", ".join(f"{kind.lower()} — {count}" for kind, count in counts.most_common())
        report.append(f"- **`{name}`**, {len(errors)} ошибок среди 350 примеров. {cases.capitalize()}.")
    for kind in ("Лишний вызов", "Пропущенный вызов"):
        case = next(
            (
                (name, key, record)
                for name in dict.fromkeys((best, "base"))
                for key, record in predictions[name].items()
                if not record["check"]["valid"] and error_type(record, references[key]) == kind
            ),
            None,
        )
        if case:
            name, key, record = case
            expected = [n for call in references[key]["ground_truth"] for n in call]
            actual = [n for call in record["calls"] for n in call]
            fmt = lambda names: ", ".join(f"`{n}`" for n in names) if names else "пусто"
            report.append(
                f"- `{key}` (`{name}`, `{record['category']}`). {kind}. "
                f"Ожидалось {fmt(expected)}. Получено {fmt(actual)}."
            )
    report += [
        "- `simple_386` (`base`, `simple`). Вызов `book_hotel` выбран верно, но "
        "модель передала `room_type='luxury room'`, когда проверка ожидала `'luxury'`. "
        "Эта ошибка относится к значению аргумента.",
        "",
        "Для повторной проверки доступны `predictions.jsonl`, `metrics.json`, история, конфигурация и выбор чекпойнта.",
        "",
    ]
    (REPORT / "report.md").write_text("\n".join(report))


if __name__ == "__main__":
    main()
