"""Run the MASSIVE experiments on a CUDA devbox with public Hugging Face data."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/hw7'))
    parser.add_argument('--plan', action='store_true', help='Print commands without allocating a GPU')
    args = parser.parse_args()
    output = args.output.resolve()
    commands = [('hw7_core.py', stage) for stage in ('data', 'baseline', 'teacher')]
    commands.append(('hw7_core.py', 'teacher-answers', '--samples', '2048'))
    for name in ('awq_massive', 'awq_wikitext', 'gptq_massive', 'smoothquant_massive'):
        commands.append(('hw7_continue.py', 'quant', '--name', name))
    for name in ('sft_n512_lr0.0001_ep1', 'sft_n512_lr5e-05_ep1', 'sft_n1024_lr5e-05_ep2'):
        commands.append(('hw7_continue.py', 'sft', '--name', name))
    commands.append(('hw7_continue.py', 'artifacts'))
    if not args.plan:
        import torch
        if not torch.cuda.is_available():
            parser.error('A CUDA GPU is required')
        if output.exists() and any(output.iterdir()):
            parser.error('Choose an empty --output directory for a fresh experiment')
        output.mkdir(parents=True, exist_ok=True)
    for module, *argv in commands:
        print(module, *argv, flush=True)
        if not args.plan:
            subprocess.run([sys.executable, str(CODE/module), *argv], cwd=output,
                           env=dict(os.environ, HW7_ROOT=str(output)), check=True)

if __name__ == '__main__':
    main()
