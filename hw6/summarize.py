"""Таблицы, графики и отчёт по завершённым замерам AIPerf."""
import csv
import re
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from metrics import RESULTS, SLA_SECONDS, best_config, runs

OUT = Path(__file__).resolve().parent / "report"
CONFIGURATIONS = {
    "baseline": "Параметры сервера по умолчанию",
    "prefill1024": "`--max-num-batched-tokens 1024`",
    "interactive": "`--performance-mode interactivity`",
    "plain_v1": "Model Runner V1, без async scheduling и спекдекодирования",
    "ngram": "Model Runner V1, N-gram с окном 2-5 и до 4 токенов за шаг",
}
COLORS = {"baseline": "#31566e", "ngram": "#bf652a"}


def write_results(rows):
    records = [asdict(row) for row in rows]
    with (OUT / "measurements.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def style_axis(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#e1e8ec", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.set_xlabel("Нагрузка, RPM")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))


def spec_plots(pairs):
    figure, axes = plt.subplots(2, 1, figsize=(8.8, 7), sharex=True, layout="constrained")
    rates = [plain.rpm for plain, _ in pairs]
    for config, index in (("baseline", 0), ("ngram", 1)):
        axes[0].plot(
            rates,
            [pair[index].p90 for pair in pairs],
            marker="o",
            linewidth=2,
            color=COLORS[config],
            label={"baseline": "Без спекдекодирования", "ngram": "N-gram"}[config],
        )
    axes[0].set_ylabel("P90 E2E, с")
    axes[0].legend(frameon=False)
    axes[0].set_title("Задержка со спекулятивным декодированием и без него", loc="left")
    axes[1].axhline(1, color="#7c8b90", linewidth=1, linestyle="--")
    axes[1].plot(rates, [plain.p90 / spec.p90 for plain, spec in pairs], marker="o", color="#237967")
    axes[1].set_ylabel("P90 без спекдека / P90 N-gram")
    for axis in axes:
        style_axis(axis)
        axis.set_xticks(rates)
    axes[0].set_xlabel("")
    figure.savefig(OUT / "speculative_decoding.png", dpi=180)
    plt.close(figure)


def sla_plot(load, best, fail):
    figure, axis = plt.subplots(figsize=(8.8, 4.8), layout="constrained")
    axis.axhline(SLA_SECONDS, linestyle="--", color="#687980", label="SLA, 15 с")
    axis.plot([r.rpm for r in load], [r.p90 for r in load], color="#9aadb5", linewidth=1.5)
    axis.scatter(
        [r.rpm for r in load], [r.p90 for r in load],
        c=["#237967" if r.conforms else "#b84938" for r in load], zorder=3,
    )
    for row, color, offset, align in (
        (best, "#237967", (0, -22), "center"),
        (fail, "#b84938", (-10, 13), "right"),
    ):
        axis.annotate(
            f"{row.rpm} RPM · {row.p90:.2f} с",
            (row.rpm, row.p90), xytext=offset, textcoords="offset points",
            ha=align, color=color, weight="bold",
        )
    axis.margins(x=0.18, y=0.2)
    axis.set_title(f"Нагрузочный тест · {load[0].config}", loc="left")
    axis.set_ylabel("P90 E2E, с")
    axis.legend(frameon=False, loc="upper left")
    style_axis(axis)
    axis.set_xticks([row.rpm for row in load])
    axis.tick_params(axis="x", labelrotation=35)
    figure.savefig(OUT / "load_test.png", dpi=180)
    plt.close(figure)


def spec_tokens(rpm):
    root = RESULTS / "spec" / "ngram"

    def counters(path):
        values = {}
        for line in path.read_text().splitlines():
            match = re.match(
                r"vllm:spec_decode_num_(draft_tokens|accepted_tokens)_total(?:\{[^}]*\})? ([^ ]+)",
                line,
            )
            if match:
                name, value = match.groups()
                values[name] = values.get(name, 0) + float(value)
        return values

    before = counters(root / f"metrics-{rpm}-before.txt")
    after = counters(root / f"metrics-{rpm}-after.txt")
    draft = after.get("draft_tokens", 0) - before.get("draft_tokens", 0)
    accepted = after.get("accepted_tokens", 0) - before.get("accepted_tokens", 0)
    return f"{100 * accepted / draft:.1f}%" if draft > 0 else "—"


def main():
    tuning = runs("tuning")
    spec = runs("spec")
    load = sorted(runs("load"), key=lambda row: row.rpm)
    plain_by_rpm = {r.rpm: r for r in spec if r.config == "plain_v1" and r.requests == 160 and not r.errors}
    spec_by_rpm = {r.rpm: r for r in spec if r.config == "ngram" and r.requests == 160 and not r.errors}
    pairs = [(plain_by_rpm[rpm], spec_by_rpm[rpm]) for rpm in sorted(plain_by_rpm.keys() & spec_by_rpm.keys())]
    winner = best_config()
    baseline = next(r for r in tuning if r.config == "baseline")
    fastest = next(r for r in tuning if r.config == winner)
    best = max((r for r in load if r.conforms), key=lambda r: r.rpm)
    fail = min((r for r in load if r.rpm > best.rpm and not r.conforms), key=lambda r: r.rpm)
    if len(pairs) < 3 or any(r.config != winner for r in load):
        raise ValueError("Не хватает сопоставимых точек или конфигурация SLA отличается от лучшей")

    OUT.mkdir(exist_ok=True)
    write_results(tuning + spec + load)
    spec_plots(pairs)
    sla_plot(load, best, fail)

    lines = [
        "# Ускорение инференса Qwen3-8B",
        "",
        (
            f"При 120 RPM минимальная средняя E2E-задержка составила **{fastest.mean:.3f} с** "
            f"для `{winner}` при **{baseline.mean:.3f} с** у baseline. "
            f"Ускорение составило **{baseline.mean / fastest.mean:.3f}×**. "
            f"На нагрузочном тесте SLA выполнен до **{best.rpm} RPM** из проверенных точек."
        ),
        "",
        "## Методика",
        "",
        (
            "Все результаты получены на A100 80GB PCIe №0 на devbox11. "
            "Один экземпляр vLLM 0.22.0, AIPerf 0.13.0, Qwen/Qwen3-8B ревизии "
            "`b968826d9c46dd6066d109eabc6255188de91218`. "
            "Во всех запросах отключён thinking, температура 0, seed 42, максимум 512 токенов. "
            "AIPerf подавал неизменённые тела запросов из `data/requests.jsonl` последовательно "
            "с постоянным интервалом и потоковой выдачей ответа."
        ),
        "",
        (
            "Префиксный кеш сбрасывался перед каждым запуском AIPerf и после него. "
            "Каждый запуск делал 40 запросов прогрева, а затем считал метрики по основной фазе. "
            "E2E измеряется от начала запроса до последнего содержательного фрагмента ответа. "
            "Выход модели мог завершиться по EOS раньше лимита 512."
        ),
        "",
        "## 1. Замеры при 120 RPM",
        "",
        "Во всех строках этого этапа один и тот же набор из 640 запросов. Ускорение S равно средней E2E baseline, делённой на среднюю E2E конфигурации.",
        "",
        "| Конфигурация | Параметры | Запросы | Ошибки | Mean E2E, с | P90 E2E, с | S |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(tuning, key=lambda r: list(CONFIGURATIONS).index(r.config)):
        speed = baseline.mean / row.mean if not row.errors and row.requests == 640 else None
        rendered_speed = f"{speed:.3f}×" if speed is not None else "—"
        lines.append(
            f"| `{row.config}` | {CONFIGURATIONS[row.config]} | {row.requests} | {row.errors} | "
            f"{row.mean:.3f} | {row.p90:.3f} | {rendered_speed} |"
        )
    prefill = next(r for r in tuning if r.config == "prefill1024")
    interactivity = next(r for r in tuning if r.config == "interactive")
    lines += [
        "",
        (
            f"Переход к бюджету 1024 дал {prefill.mean:.3f} с в среднем. При этом средний TTFT "
            f"вырос с {baseline.mean_ttft_ms:.1f} до {prefill.mean_ttft_ms:.1f} мс. "
            "При меньшем бюджете длинные промпты чаще приходится разбивать на несколько шагов prefill."
        ),
        "",
        (
            f"В режиме interactivity TTFT составил {interactivity.mean_ttft_ms:.1f} мс, близко к baseline. "
            f"Средний интервал между токенами снизился с {baseline.mean_itl_ms:.2f} до "
            f"{interactivity.mean_itl_ms:.2f} мс. Длина ответов была примерно одинаковой, "
            "поэтому это согласуется с тем, что выигрыш в E2E пришёлся главным образом на декодирование."
        ),
        "",
        "## 2. Спекулятивное декодирование",
        "",
        (
            "Сопоставлены `plain_v1` и `ngram`. N-gram в vLLM 0.22.0 требует Model Runner V1, "
            "поэтому обе конфигурации этой серии используют V1. N-gram автоматически отключает "
            "async scheduling, поэтому в контрольной конфигурации он тоже отключён. "
            "Подбор кандидатов использует совпадения "
            "n-грамм в контексте запроса. Qwen3-8B проверяет предложенные токены. "
            "Серия использует по 160 запросов из одного и того же последовательного набора на каждом RPM. "
            "Столбец принятия токенов взят из счётчиков vLLM и включает прогрев."
        ),
        "",
        "| RPM | P90 без спекдека, с | P90 N-gram, с | Отношение P90 | Принятые токены N-gram |",
        "|---:|---:|---:|---:|---:|",
    ]
    for plain, draft in pairs:
        lines.append(f"| {plain.rpm} | {plain.p90:.3f} | {draft.p90:.3f} | {plain.p90 / draft.p90:.3f}× | {spec_tokens(plain.rpm)} |")
    lines += ["", "![P90 и относительное ускорение при разной нагрузке](speculative_decoding.png)", ""]
    margins = [(plain.rpm, draft.p90 - plain.p90) for plain, draft in pairs]
    crossings = [(r1, r2) for (r1, d1), (r2, d2) in zip(margins, margins[1:]) if d1 <= 0 < d2]
    if crossings:
        low, high = crossings[0]
        lines.append(f"Первый переход от выигрыша к проигрышу наблюдался между **{low} и {high} RPM**. При {high} RPM P90 у N-gram уже выше.")
    elif all(delta > 0 for _, delta in margins):
        lines.append(f"N-gram уступил `plain_v1` по P90 на всех проверенных нагрузках, начиная с **{margins[0][0]} RPM**.")
    elif all(delta <= 0 for _, delta in margins):
        lines.append(f"N-gram оставался быстрее или на уровне `plain_v1` по P90 вплоть до **{margins[-1][0]} RPM**. Пересечения в этом диапазоне не наблюдалось.")
    else:
        lines.append("На самых низких проверенных нагрузках P90 с N-gram был выше. На части следующих точек знак разницы изменился.")
    first_plain, first_draft = pairs[0]
    last_plain, last_draft = pairs[-1]
    lines += [
        "",
        (
            f"При {first_plain.rpm} RPM разница по P90 составила {first_draft.p90 - first_plain.p90:.2f} с, "
            f"при {last_plain.rpm} RPM она выросла до {last_draft.p90 - last_plain.p90:.2f} с. "
            f"В последней точке средний ITL увеличился с {last_plain.mean_itl_ms:.2f} до "
            f"{last_draft.mean_itl_ms:.2f} мс при близких средних длинах ответа "
            f"({last_plain.output_tokens:.0f} и {last_draft.output_tokens:.0f} токенов). "
            f"Из предложенных токенов принято {spec_tokens(last_plain.rpm)}. "
            "Стоимость построения и проверки кандидатов, по-видимому, оказалась выше экономии "
            "от принятых токенов. При высокой нагрузке разница усилилась. "
            "[Параметры спекдекодирования vLLM 0.22.0](https://docs.vllm.ai/en/v0.22.0/features/speculative_decoding/)"
        ),
        "",
        "## 3. Нагрузочный тест",
        "",
        (
            f"Лучшая по средней задержке конфигурация `{winner}` проверена скриптом `load_test.sh`. "
            "Нагрузку подавали 300 секунд, затем клиент дождался ответов. Для зачёта точки "
            "требовались P90 < 15 с, отсутствие ошибок и число завершённых запросов в пределах двух "
            "от 5 × RPM. Между проходящей и непроходящей точками RPM уточнялся шагом 5."
        ),
        "",
        "| Задано, RPM | Ответы | Ошибки | Mean E2E, с | P90 E2E, с | SLA |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in load:
        lines.append(
            f"| {row.rpm} | {row.requests} | {row.errors} | {row.mean:.3f} | "
            f"{row.p90:.3f} | {'выполнен' if row.conforms else 'нарушен'} |"
        )
    lines += [
        "",
        "![Задержка при 300-секундной нагрузке и граница SLA](load_test.png)",
        "",
        f"Подтверждённая точка — **{best.rpm} RPM**, P90 **{best.p90:.3f} с**. "
        f"При **{fail.rpm} RPM** P90 составил **{fail.p90:.3f} с**. "
        "Граница относится к этому серверу, данному набору запросов и проверенной сетке RPM.",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
