"""Разбор агрегатов AIPerf и проверка полноты замеров."""
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
TIME_SCALE = {"s": 1, "sec": 1, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "ns": 1e-9}
SLA_SECONDS = 15
LOAD_SECONDS = 300


@dataclass(frozen=True)
class Run:
    phase: str
    config: str
    rpm: int
    requests: int
    errors: int
    mean: float
    p90: float
    duration: float
    throughput_rps: float
    output_tokens: float | None
    mean_ttft_ms: float | None
    mean_itl_ms: float | None

    @property
    def conforms(self) -> bool:
        expected = self.rpm * LOAD_SECONDS / 60
        return (
            self.phase == "load"
            and self.errors == 0
            and self.duration >= LOAD_SECONDS - 1
            and abs(self.requests - expected) <= 2
            and self.p90 < SLA_SECONDS
        )


def metric(data, key, statistic="avg", default=None):
    value = data.get(key, default)
    return value.get(statistic, default) if isinstance(value, dict) else value


def seconds(data, key, statistic):
    value = data[key]
    return float(value[statistic]) * TIME_SCALE[value["unit"]]


def read_run(file: Path) -> Run:
    phase, config, rpm = file.relative_to(RESULTS).parts[:3]
    source = json.loads(file.read_text())
    data = source.get("metrics", source)
    return Run(
        phase=phase,
        config=config,
        rpm=int(rpm),
        requests=int(metric(data, "request_count")),
        errors=int(metric(data, "error_request_count", default=0)),
        mean=seconds(data, "request_latency", "avg"),
        p90=seconds(data, "request_latency", "p90"),
        duration=seconds(data, "benchmark_duration", "avg"),
        throughput_rps=float(metric(data, "request_throughput")),
        output_tokens=metric(data, "output_sequence_length"),
        mean_ttft_ms=seconds(data, "time_to_first_token", "avg") * 1000 if "time_to_first_token" in data else None,
        mean_itl_ms=seconds(data, "inter_token_latency", "avg") * 1000 if "inter_token_latency" in data else None,
    )


def runs(phase):
    pattern = f"{phase}/*/*/profile/profile_export_aiperf.json"
    return [read_run(file) for file in sorted(RESULTS.glob(pattern))]


def best_config():
    tested = [r for r in runs("tuning") if r.rpm == 120 and r.requests == 640 and not r.errors]
    if not any(r.config == "baseline" for r in tested):
        raise SystemExit("Нет полного замера baseline при 120 RPM")
    return min(tested, key=lambda r: (r.mean, r.p90)).config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("best", "load", "dump"))
    parser.add_argument("config", nargs="?")
    parser.add_argument("rpm", nargs="?", type=int)
    args = parser.parse_args()
    if args.command == "best":
        print(best_config())
    elif args.command == "load":
        file = RESULTS / "load" / args.config / str(args.rpm) / "profile/profile_export_aiperf.json"
        result = read_run(file)
        print(f"{'pass' if result.conforms else 'fail'} p90={result.p90:.3f} requests={result.requests} errors={result.errors}")
    else:
        print(json.dumps({phase: [asdict(r) for r in runs(phase)] for phase in ("tuning", "spec", "load")}, indent=2))


if __name__ == "__main__":
    main()
