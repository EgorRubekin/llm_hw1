import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path
from data import MODEL, REVISION, SEED


def main():
    p = argparse.ArgumentParser()
    p.add_argument('config', type=Path)
    p.add_argument('--data', type=Path, default=Path('data'))
    p.add_argument('--resume', help='Explicit trainer checkpoint path')
    a = p.parse_args()
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    from rewards import make_reward
    if not torch.cuda.is_available(): raise RuntimeError('Training requires a CUDA GPU in the provided Docker image')
    config = json.loads(a.config.read_text())
    output = Path('runs') / a.config.stem
    if output.exists() and any(output.iterdir()) and not a.resume:
        raise ValueError(f'{output} already contains a run; use --resume or move it')
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side='left')
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    datasets = {s: Dataset.from_json(str(a.data / (s + '.jsonl'))) for s in ('train', 'validation')}
    if set(datasets['train']['group']) & set(datasets['validation']['group']): raise ValueError('Split leakage')
    rollout_settings = {}
    model_source = MODEL
    if os.environ.get('RL_USE_VLLM') == '1':
        from huggingface_hub import snapshot_download
        model_source = snapshot_download(MODEL, revision=REVISION)
        rollout_settings = {
            'vllm_mode': 'colocate',
            'vllm_gpu_memory_utilization': 0.38,
            'vllm_max_model_length': 2560,
            'vllm_importance_sampling_correction': False,
        }
    args = GRPOConfig(
        output_dir=str(output / 'checkpoints'), seed=SEED, data_seed=SEED,
        model_init_kwargs={'revision': REVISION, 'dtype': 'bfloat16'}, bf16=True,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        per_device_eval_batch_size=int(os.environ.get('RL_EVAL_BATCH_SIZE', '4')),
        num_generations=4, generation_batch_size=8,
        max_completion_length=512, temperature=.8, beta=0.0,
        learning_rate=5e-6, max_steps=100, warmup_ratio=.05,
        loss_type=config['loss_type'], scale_rewards=config['scale_rewards'],
        use_vllm=bool(rollout_settings), gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy='steps', eval_steps=25, save_strategy='steps', save_steps=25,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model='reward',
        greater_is_better=True, logging_steps=1, report_to='none', remove_unused_columns=False,
        **rollout_settings,
    )
    manifest = {'experiment': config, 'trainer': args.to_dict(), 'base_model': MODEL,
                'base_revision': REVISION, 'data_sha256': {s: hashlib.sha256((a.data / (s + '.jsonl')).read_bytes()).hexdigest() for s in datasets},
                'versions': {n: version(n) for n in ('torch', 'transformers', 'trl', 'peft', 'vllm', 'datasets')}}
    manifest_path = output / 'manifest.json'
    if a.resume and manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Resume configuration or dataset differs from original run')
    manifest_path.write_text(json.dumps(manifest, indent=2))
    class RewardValidationTrainer(GRPOTrainer):
        """Валидация по награде на всех генерациях TRL."""

        @torch.no_grad()
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            self._prepare_inputs(inputs)
            return None, None, None

    trainer_class = RewardValidationTrainer if rollout_settings else GRPOTrainer
    trainer = trainer_class(model=model_source, args=args, processing_class=tokenizer,
        train_dataset=datasets['train'], eval_dataset=datasets['validation'],
        reward_funcs=make_reward(tokenizer, config['reward'], args.max_completion_length),
        peft_config=LoraConfig(r=16, lora_alpha=32, target_modules=['q_proj','k_proj','v_proj','o_proj'], task_type='CAUSAL_LM'))
    recovery_steps = int(os.environ.get('RL_RECOVERY_STEPS', '0'))
    if recovery_steps > 0:
        class RecoveryCheckpoint(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step % recovery_steps == 0:
                    control.should_save = True
                return control

        trainer.add_callback(RecoveryCheckpoint())
        print(f'Сохраняем чекпойнт для восстановления каждые {recovery_steps} шагов.')
    trainer.train(resume_from_checkpoint=a.resume)
    trainer.save_state()
    (output / 'history.json').write_text(json.dumps(trainer.state.log_history, indent=2))
    (output / 'selection.json').write_text(json.dumps({'criterion': 'validation reward', 'checkpoint': trainer.state.best_model_checkpoint, 'score': trainer.state.best_metric}, indent=2))
    model = trainer.accelerator.unwrap_model(trainer.model)
    model.save_pretrained(output / 'adapter')
    merged = model.merge_and_unload()
    merged.save_pretrained(output / 'merged', safe_serialization=True, max_shard_size='2GB')
    tokenizer.save_pretrained(output / 'merged')


if __name__ == '__main__': main()
