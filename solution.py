import os
import sys
import gc
import json
import math
import time
import statistics

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    default_data_collator,
)
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Don't change this parameter
MAX_TRAINING_TIME_SECONDS = 60 * 15
MAX_LENGTH = 512
INPUT_IDS = 'input_ids'
ATTENTION_MASK = 'attention_mask'
LABELS = 'labels'

# Don't change these parameters
TOKENIZER_NAME = "ai-forever/rugpt3small_based_on_gpt2"
OUTPUT_DIR = "./output_dir"
NUM_SHARDS = 32
VALIDATION_SIZE = 5000

RUNS_DIR = f"{OUTPUT_DIR}/gpt2-1b-russian"

TRAINING_CONFIG = {
    'max_steps': 1200,
    'learning_rate': 6e-4,
    'lr_scheduler_type': 'cosine',
    'warmup_ratio': 0.05,

    'per_device_train_batch_size': 32,
    'gradient_accumulation_steps': 1,
    'per_device_eval_batch_size': 64,

    'optim': 'adamw_torch_fused',
    'weight_decay': 0.1,
    'adam_beta1': 0.9,
    'adam_beta2': 0.95,
    'max_grad_norm': 1.0,

    'bf16': True,
    'tf32': True,
    'torch_compile': True,
    'gradient_checkpointing': False,
    'dataloader_num_workers': 4,
    'dataloader_pin_memory': True,

    'logging_steps': 10,
    'include_num_input_tokens_seen': True,
    'eval_strategy': 'steps',
    'eval_steps': 300,
    'save_strategy': 'steps',
    'save_steps': 300,
    'save_total_limit': 1,
    'save_only_model': True,
    'load_best_model_at_end': True,
    'metric_for_best_model': 'eval_loss',
    'greater_is_better': False,
    'report_to': 'none',
}


class TimeoutCallback(TrainerCallback):
    """Callback to stop training after a specified timeout."""
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout_seconds:
                control.should_training_stop = True
                control.should_evaluate = True
                control.should_save = True
                print(f"Training stopped after {elapsed:.2f} seconds")
        return control


class StepTimeCallback(TrainerCallback):
    """Замеряет время шага (без первых шагов с компиляцией) и пиковую память."""
    def __init__(self, skip_first=5):
        self.skip_first = skip_first
        self.times = []
        self.first_step_time = None
        self._t = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._t0 = time.time()
        torch.cuda.reset_peak_memory_stats()

    def on_step_begin(self, args, state, control, **kwargs):
        self._t = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        dt = time.time() - self._t
        if state.global_step == 1:
            self.first_step_time = time.time() - self._t0  # включает компиляцию
        if state.global_step > self.skip_first:
            self.times.append(dt)

    def summary(self, args):
        if not self.times:
            return {}
        med = statistics.median(self.times)
        tokens_per_step = (args.per_device_train_batch_size
                           * args.gradient_accumulation_steps * MAX_LENGTH)
        return {
            "first_step_s(incl. compile)": round(self.first_step_time or 0, 1),
            "median_step_s": round(med, 4),
            "tokens_per_s(incl. pad)": int(tokens_per_step / med),
            "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        }


def prepare_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_function(examples, tokenizer):
    tok = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    labels = [
        [t if m == 1 else -100 for t, m in zip(ids, mask)]
        for ids, mask in zip(tok[INPUT_IDS], tok[ATTENTION_MASK])
    ]
    return {INPUT_IDS: tok[INPUT_IDS], ATTENTION_MASK: tok[ATTENTION_MASK], LABELS: labels}


def save_as_parquets(ds, output_dir=OUTPUT_DIR, num_shards=NUM_SHARDS):
    os.makedirs(output_dir, exist_ok=True)
    for index in range(num_shards):
        shard = ds.shard(num_shards=num_shards, index=index, contiguous=True)
        shard.to_parquet(f"{output_dir}/{index:05d}.parquet")
        print(f"Saved shard {index + 1}/{num_shards}")


def prepare_dataset(num_proc=None):
    tokenizer = prepare_tokenizer()
    dataset = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train")
    tokenized = dataset.map(
        tokenize_function,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        batch_size=1000,
        num_proc=num_proc or max(1, (os.cpu_count() or 2) - 1),
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    save_as_parquets(tokenized)


def load_tokenized_dataset(data_dir=OUTPUT_DIR):
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".parquet"))
    data_files = [os.path.join(data_dir, f) for f in files]
    dataset = load_dataset('parquet', data_files=data_files)
    return dataset['train']


def split_dataset(dataset, validation_size=VALIDATION_SIZE):
    dataset_size = len(dataset)
    train_dataset = dataset.select(range(validation_size, dataset_size))
    eval_dataset = dataset.select(range(validation_size))
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    return train_dataset, eval_dataset


def create_model(tokenizer):
    # Don't change this parameter
    MODEL_CONFIG = {
        'hidden_size': 2048,
        'num_hidden_layers': 12,
        'num_attention_heads': 16,
        'num_key_value_heads': 8,
        'intermediate_size': 8192,
        'head_dim': 128,
        'hidden_act': 'silu',
        'initializer_range': 0.02,
        'scale_attn_weights': True,
        'use_cache': True,
    }
    config = Qwen3Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **MODEL_CONFIG
    )
    model = Qwen3ForCausalLM._from_config(
        config,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16
    )
    print(f"Model pad token id: {model.config.pad_token_id}")
    with torch.no_grad():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total params: {total_params:,}")
    return model


PROMPTS = [
    "Москва — столица",
    "В 1812 году",
    "Александр Сергеевич Пушкин родился",
    "Искусственный интеллект — это",
]


@torch.no_grad()
def generate_samples(model, tokenizer, prompts=PROMPTS, max_new_tokens=80):
    model.eval()
    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        for mode, kwargs in {
            "greedy": dict(do_sample=False),
            "sampling": dict(do_sample=True, temperature=0.8, top_p=0.95),
        }.items():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                **kwargs,
            )
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            print(f"\n[{mode}] {prompt!r}\n→ {text}")
            results.append({"prompt": prompt, "mode": mode, "text": text})
    return results


def cleanup():
    """Освободить GPU между экспериментами в одном ядре Jupyter."""
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch._dynamo.reset()  # сбросить кэш torch.compile
    except Exception:
        pass


_DATA_CACHE = {}


def get_datasets():
    """Датасет грузится один раз за сессию ядра."""
    if "data" not in _DATA_CACHE:
        _DATA_CACHE["data"] = split_dataset(load_tokenized_dataset())
    return _DATA_CACHE["data"]


def train_model(exp_name="baseline", overrides=None, report_to="none",
                probe=False, probe_steps=40, do_generate=True):
    """
    probe=True: короткий прогон без eval/save, только замер скорости.
    Возвращает dict с метриками.
    """
    tokenizer = prepare_tokenizer()
    train_dataset, eval_dataset = get_datasets()
    model = create_model(tokenizer)

    cfg = {
        **TRAINING_CONFIG,
        'output_dir': f'{RUNS_DIR}/{exp_name}',
        'run_name': exp_name,
        'report_to': report_to,
        **(overrides or {}),
    }
    if probe:
        cfg.update({
            'max_steps': probe_steps, 'eval_strategy': 'no', 'save_strategy': 'no',
            'load_best_model_at_end': False, 'report_to': 'none', 'logging_steps': 5,
        })
    print("Training config:", json.dumps(cfg, indent=2, ensure_ascii=False))
    training_args = TrainingArguments(**cfg)

    step_timer = StepTimeCallback()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        callbacks=[TimeoutCallback(timeout_seconds=MAX_TRAINING_TIME_SECONDS),  # dont change
                   step_timer],
    )

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    speed = step_timer.summary(training_args)
    print("Speed:", speed)

    results = {"exp_name": exp_name, "speed": speed, "train_wall_time_s": round(train_time, 1),
               "global_step": trainer.state.global_step}

    if not probe:
        print("Running final evaluation...")
        eval_results = trainer.evaluate()
        eval_results["perplexity"] = math.exp(eval_results["eval_loss"])
        print(f"Final evaluation results: {eval_results}")
        results["eval"] = eval_results
        trainer.save_state()  # -> output_dir/trainer_state.json

        out = training_args.output_dir
        trainer.save_model(f"{out}/final")
        if do_generate:
            results["samples"] = generate_samples(trainer.model, tokenizer)
        with open(f"{out}/final_results.json", "w") as f:
            json.dump({"config": cfg, **results}, f, ensure_ascii=False, indent=2)

    del trainer, model
    cleanup()
    return results


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "prepare":
        prepare_dataset()
    elif mode == "train":
        name = sys.argv[2] if len(sys.argv) > 2 else "baseline"
        ov = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
        train_model(name, ov)