"""Resume the user's HW7 experiments with validated inputs and atomic completion.
Existing hw7_full.py is preserved; hw7_core.py is its local derivative.
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path
import hw7_core as c

METRICS = ('quality','intent_accuracy','slot_micro_f1','frame_exact_match','valid_json_rate')
QUANT = {
 'awq_massive': dict(method='awq', calibration='massive', samples=128, bits=4, group_size=128, symmetric=False),
 'awq_wikitext': dict(method='awq', calibration='wikitext', samples=128, bits=4, group_size=128, symmetric=False),
 'gptq_massive': dict(method='gptq', calibration='massive', samples=128, bits=4, group_size=128, symmetric=True),
 'smoothquant_massive': dict(method='smoothquant', calibration='massive', samples=128, bits=8, group_size=None, symmetric=True, smoothing_strength=.5),
}
SFT = {
 'sft_n512_lr0.0001_ep1': (512, 1e-4, 1.),
 'sft_n512_lr5e-05_ep1': (512, 5e-5, 1.),
 'sft_n1024_lr5e-05_ep2': (1024, 5e-5, 2.),
}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def validate_eval(name):
    dev=c.rows('dev')
    p=c.OUT/'eval'/f'{name}.predictions.jsonl'
    predictions=[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    if [r['id'] for r in predictions] != [r['id'] for r in dev]:
        raise ValueError(f'{name}: incomplete or mismatched dev IDs')
    if any(r['gold']!=d['gold'] for r,d in zip(predictions,dev)):
        raise ValueError(f'{name}: mismatched gold')
    metrics=c.metric([r['prediction'] for r in predictions],[r['gold'] for r in dev])
    mp=c.OUT/'eval'/f'{name}.metrics.json'
    if mp.exists():
        saved=json.loads(mp.read_text())
        if any(abs(saved[k]-metrics[k])>1e-10 for k in METRICS) or saved['n']!=metrics['n']:
            raise ValueError(f'{name}: saved metrics do not reproduce')
    return metrics


def complete(name):
    p=c.OUT/'eval'/f'{name}.metrics.json'
    if not p.exists():return False
    validate_eval(name)
    return True


def checkpoint(path):
    path=Path(path)
    config=json.loads((path/'config.json').read_text())
    index=path/'model.safetensors.index.json'
    files=set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
    for f in files:
        if not (path/f).is_file() or not (path/f).stat().st_size:
            raise ValueError(f'Incomplete checkpoint: {path/f}')
    if not (path/'tokenizer_config.json').exists():raise ValueError('Missing tokenizer config')
    return config


def preflight():
    for name in ('baseline','teacher'):print(name,validate_eval(name),flush=True)
    train=c.rows('train');dev=c.rows('dev')
    if {r['id'] for r in train}&{r['id'] for r in dev}:raise ValueError('Train/dev overlap')
    answers=[json.loads(s) for s in (c.OUT/'teacher_answers'/'answers.jsonl').read_text().splitlines() if s.strip()]
    lookup={r['id']:r for r in train}; seen=set(); valid=0
    for r in answers:
        if r['id'] in seen or r['id'] not in lookup:raise ValueError('Duplicate or non-train teacher answer')
        seen.add(r['id'])
        if any(r[k]!=lookup[r['id']][k] for k in ('text','gold')):raise ValueError('Teacher row mismatch')
        if not r['prompt_ids'] or not r['answer_ids']:raise ValueError('Empty teacher tokens')
        valid+=c.parse(r['teacher_answer'],c.schema()) is not None
    if valid<1024:raise ValueError(f'Need 1024 valid teacher answers, have {valid}; generate more first')
    cfg=checkpoint(c.OUT/'quantized'/'awq_massive')
    expected=QUANT['awq_massive']
    weights=next(iter(cfg['quantization_config']['config_groups'].values()))['weights']
    if any(weights[k]!=expected[k] for k in ('group_size','symmetric')) or weights['num_bits']!=4:
        raise ValueError('Existing AWQ config differs from experiment plan')
    print('preflight OK',len(train),len(dev),len(answers),valid,flush=True)
    c.save(c.OUT/'resume_audit.json',dict(train_n=len(train),dev_n=len(dev),teacher_n=len(answers),valid_teacher_n=valid,
        dev_sha256=digest(c.OUT/'prepared'/'dev.jsonl'),answers_sha256=digest(c.OUT/'teacher_answers'/'answers.jsonl')))


def recipe(cfg):
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
    ignore=['lm_head']
    if cfg['method']=='awq':
        from llmcompressor.modifiers.awq import AWQModifier
        return [AWQModifier(targets='Linear',scheme='W4A16_ASYM',ignore=ignore)]
    if cfg['method']=='gptq':return [GPTQModifier(targets='Linear',scheme='W4A16',ignore=ignore)]
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
    return [SmoothQuantModifier(smoothing_strength=cfg['smoothing_strength']),
            QuantizationModifier(targets='Linear',scheme='W8A8',ignore=ignore)]


def quant(name):
    from transformers import set_seed
    set_seed(c.SEED)
    from awq_compat import install
    install()
    if complete(name):print('Verified complete:',name);return
    cfg=QUANT[name]; dest=c.OUT/'quantized'/name
    if dest.exists():checkpoint(dest)
    else:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from llmcompressor import oneshot
        tok=AutoTokenizer.from_pretrained(c.BASE)
        data=c.calibration(cfg['calibration'],tok,cfg['samples'])
        input_ids = list(data['input_ids'])
        c.save(c.OUT/'calibration'/f'{name}.json', {
            'config': cfg,
            'input_ids': input_ids,
            'token_lengths': [len(ids) for ids in input_ids],
        })
        model=AutoModelForCausalLM.from_pretrained(c.BASE,torch_dtype=torch.bfloat16,device_map='cuda:0')
        c.save(c.OUT/'calibration'/f'{name}.model.json',{'model_id':c.BASE,'resolved_revision':getattr(model.config,'_commit_hash',None)})
        oneshot(model=model,processor=tok,dataset=data,recipe=recipe(cfg),max_seq_length=2048,num_calibration_samples=cfg['samples'])
        temp=dest.with_name(dest.name+'.inprogress')
        if temp.exists():raise FileExistsError(f'Inspect incomplete previous output before retry: {temp}')
        model.save_pretrained(temp,save_compressed=cfg['method']!='smoothquant');tok.save_pretrained(temp)
        checkpoint(temp);temp.rename(dest)
        del model;c.gc.collect();torch.cuda.empty_cache()
    c.save(c.OUT/'quantized'/f'{name}.config.json',{**cfg,'base_model':c.BASE,'ignore':['lm_head'],'seed':c.SEED,
       'evaluation_backend':'transformers-bf16-decompressed','storage_format':('dense-fake-quant' if cfg['method']=='smoothquant' else 'pack-quantized'),'compatibility_patch':'awq_compat.py',
       'checkpoint_quantization_config':checkpoint(dest)['quantization_config']})
    c.evaluate(name,dest,quantized=True)
    validate_eval(name)


def train(name):
    if complete(name):print('Verified complete:',name);return
    from transformers import set_seed
    set_seed(c.SEED)
    n,lr,epochs=SFT[name]
    path=c.OUT/'training'/name/'model'
    if path.exists():
        checkpoint(path);c.evaluate(name,path)
    else:c.train_student(n,lr,epochs)
    validate_eval(name)


def save_artifacts():
    names = ['baseline', 'teacher', *QUANT, *SFT]
    results = [{'name': name, **validate_eval(name)} for name in names]
    c.save(c.OUT/'results.json', results)
    import csv
    with (c.OUT/'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['name', *METRICS, 'n'])
        writer.writeheader()
        writer.writerows(results)
    best = max((row for row in results if row['name'] in SFT), key=lambda row: row['quality'])
    src=c.OUT/'training'/best['name']/'model';checkpoint(src)
    target=c.OUT/'student_sft'
    if target.is_symlink():target.unlink()
    elif target.exists():raise FileExistsError('Existing student_sft preserved; choose destination explicitly')
    target.symlink_to(src.relative_to(c.OUT),target_is_directory=True)
    c.save(c.OUT/'student_dev.json',best)
    manifest=[]
    for directory in (c.OUT/'quantized'/'awq_massive',target):
        for file in sorted(directory.rglob('*')):
            if file.is_file():manifest.append({'path':str(file.relative_to(c.OUT)),'bytes':file.stat().st_size,'sha256':digest(file)})
    c.save(c.OUT/'submission_manifest.json',{'files':manifest,'retained_int4':'quantized/awq_massive','student':best['name']})


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','quant','sft','artifacts']);p.add_argument('--name');a=p.parse_args()
    from importlib.metadata import version,PackageNotFoundError
    environment={}
    for package in ['torch','transformers','compressed-tensors','llmcompressor','peft','datasets','accelerate']:
        try:environment[package]=version(package)
        except PackageNotFoundError:pass
    c.save(c.OUT/'environments'/f'{a.stage}_{a.name or "default"}.json',environment)
    if a.stage=='preflight':preflight()
    elif a.stage=='quant':quant(a.name)
    elif a.stage=='sft':train(a.name)
    else:save_artifacts()

if __name__=='__main__':main()
