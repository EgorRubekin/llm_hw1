"""Reproducible staged HW7 experiments. Run only inside Aviflow GPU components.
Each stage writes a separate checkpoint / metrics; failures remain visible in task logs.
No external homework solution is used.
"""
import argparse, gc, hashlib, json, os, random, re, shutil, time
from collections import Counter
from pathlib import Path

ROOT = Path(os.environ.get('HW7_ROOT', Path(os.path.abspath(__file__)).parent)); OUT = ROOT / 'artifacts_compression_hw'; OUT.mkdir(parents=True,exist_ok=True)
SEED=42; BASE='Qwen/Qwen3-8B'; TEACHER='Qwen/Qwen3-32B'; MAX_NEW=128
DATASET_REV='ed58ac423a2f4121720918bf5301577edce4ffd3'
SLOT_RE=re.compile(r'\[([^:\[\]]+?)\s*:\s*([^\[\]]*?)\]')

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)

def data():
    from datasets import load_dataset
    base=f'hf://datasets/AmazonScience/massive@{DATASET_REV}/ru-RU'
    raw=load_dataset('parquet',data_files={s:f'{base}/{s}/0000.parquet' for s in ('train','validation')})
    intents=list(raw['train'].features['intent'].names)
    slots=sorted({m.group(1).strip() for row in raw['train'] for m in SLOT_RE.finditer(row['annot_utt'])})
    def convert(row):
        gold={'intent':intents[row['intent']],'slots':[[m.group(1).strip(),m.group(2).strip()] for m in SLOT_RE.finditer(row['annot_utt'])]}
        return {'id':str(row['id']),'text':row['utt'],'gold':json.dumps(gold,ensure_ascii=False,separators=(',',':'))}
    train=[convert(x) for x in raw['train']]
    dev=[convert(x) for x in raw['validation'].shuffle(seed=SEED).select(range(min(256,len(raw['validation']))))]
    target=OUT/'prepared';target.mkdir(exist_ok=True)
    for name,rows in (('train',train),('dev',dev)):
        (target/f'{name}.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
    save(target/'schema.json',{'intents':intents,'slot_types':slots,'seed':SEED,'dataset_revision':DATASET_REV,'dev_ids':[x['id'] for x in dev]})
    print('DATA',len(train),len(dev),len(intents),len(slots),flush=True)

def rows(name):
    p=OUT/'prepared'/f'{name}.jsonl'
    if not p.is_file():raise FileNotFoundError(f'prepare-data must run first: {p}')
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]

def schema():return json.loads((OUT/'prepared'/'schema.json').read_text(encoding='utf-8'))

def system():
    x=schema()
    return ('Определи intent и извлеки slots. Верни только JSON вида {"intent":"...","slots":[["тип","значение"],...]}. '
            'Значения slots копируй из запроса. Если slots нет, верни пустой список. Не добавляй объяснения.\n'
            'Допустимые intents: '+', '.join(x['intents'])+'\nДопустимые типы slots: '+', '.join(x['slot_types']))

def prompt(tok,text):
    ids=tok.apply_chat_template([{'role':'system','content':system()},{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    if len(ids)>2048:raise ValueError(f'prompt too long {len(ids)}')
    return ids

def generate(model,tok,batch):
    import torch
    ps=[prompt(tok,x['text']) for x in batch]
    width=max(map(len,ps)); pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    inp=torch.tensor([[pad]*(width-len(p))+p for p in ps],device=model.device,dtype=torch.long)
    mask=torch.tensor([[0]*(width-len(p))+[1]*len(p) for p in ps],device=model.device,dtype=torch.long)
    with torch.inference_mode():
        output=model.generate(input_ids=inp,attention_mask=mask,do_sample=False,max_new_tokens=MAX_NEW,pad_token_id=pad,eos_token_id=tok.eos_token_id)
    answers=[]
    for seq in output[:,width:].tolist():
        if tok.eos_token_id in seq:seq=seq[:seq.index(tok.eos_token_id)+1]
        answers.append((tok.decode(seq,skip_special_tokens=True).strip(),seq))
    return ps,answers

def parse(text,x):
    try:
        v=json.loads(text.strip())
        if not isinstance(v,dict) or set(v)!={'intent','slots'} or v['intent'] not in x['intents'] or not isinstance(v['slots'],list):return None
        if any(not isinstance(s,list) or len(s)!=2 or s[0] not in x['slot_types'] or not isinstance(s[1],str) or not s[1].strip() for s in v['slots']):return None
        return v
    except (ValueError,TypeError):return None

def metric(pred,gold):
    if not gold or len(pred)!=len(gold):raise ValueError('Need nonempty equal length predictions and gold')
    x=schema(); cor=exact=valid=tp=fp=fn=0
    for p,g in zip(pred,gold):
        truth=parse(g,x); found=parse(p,x)
        if truth is None:raise ValueError('invalid gold')
        a=Counter((t,' '.join(v.casefold().split())) for t,v in truth['slots'])
        b=Counter((t,' '.join(v.casefold().split())) for t,v in found['slots']) if found else Counter()
        tp+=sum((a&b).values());fp+=sum((b-a).values());fn+=sum((a-b).values())
        if found:
            valid+=1;ok=found['intent']==truth['intent'];cor+=ok;exact+=int(ok and a==b)
    n=len(gold);f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else float(valid==n)
    acc=cor/n
    return dict(quality=(acc+f1)/2,intent_accuracy=acc,slot_micro_f1=f1,frame_exact_match=exact/n,valid_json_rate=valid/n,n=n)

def load_model(name,quantized=False):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,AutoConfig,CompressedTensorsConfig
    kw={'torch_dtype':torch.bfloat16,'device_map':'cuda:0'}
    if quantized:
        from awq_compat import install
        install()
    if quantized and getattr(AutoConfig.from_pretrained(str(name)),'quantization_config',None):
        kw['quantization_config']=CompressedTensorsConfig(run_compressed=False)
    return AutoModelForCausalLM.from_pretrained(str(name),**kw).eval(),AutoTokenizer.from_pretrained(str(name))

def eval_model(label,model,tok,items,batch_size=1):
    pred=[]
    path=OUT/'eval'/f'{label}.predictions.jsonl';path.parent.mkdir(exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for i in range(0,len(items),batch_size):
            _,answers=generate(model,tok,items[i:i+batch_size])
            for row,(ans,_) in zip(items[i:i+batch_size],answers):
                pred.append(ans);f.write(json.dumps({'id':row['id'],'gold':row['gold'],'prediction':ans},ensure_ascii=False)+'\n')
            print(label,i+len(answers),'/',len(items),flush=True)
    scores=metric(pred,[r['gold'] for r in items]);save(OUT/'eval'/f'{label}.metrics.json',scores)
    print(label,scores,flush=True)
    return scores

def evaluate(label,model_name,quantized=False):
    model,tok=load_model(model_name,quantized)
    try:return eval_model(label,model,tok,rows('dev'),batch_size=1 if label=='teacher' else 4)
    finally:
        del model;gc.collect()

def teacher_answers(n=2048):
    train=rows('train');random.Random(SEED).shuffle(train);train=train[:n]
    model,tok=load_model(TEACHER)
    path=OUT/'teacher_answers'/'answers.jsonl';path.parent.mkdir(exist_ok=True)

    done=set();mode='a' if path.exists() else 'w'
    if path.exists():
        for s in path.read_text(encoding='utf-8').splitlines():
            if s.strip():done.add(str(json.loads(s)['id']))
    try:
        with path.open(mode,encoding='utf-8') as f:
            for idx,r in enumerate(train):
                if r['id'] in done:continue
                ps,ans=generate(model,tok,[r]); text,ids=ans[0]
                f.write(json.dumps({**r,'teacher_answer':text,'prompt_ids':ps[0],'answer_ids':ids},ensure_ascii=False)+'\n');f.flush()
                if idx%32==0:print('teacher answers',idx+1,'/',len(train),flush=True)
    finally:del model;gc.collect()
    actual=[json.loads(s)['id'] for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
    if len(set(actual)&{r['id'] for r in train})!=len(train):raise RuntimeError('incomplete teacher answers')
    print('teacher answers ready',len(train),flush=True)

def calibration(source,tok,n=128):
    from datasets import Dataset,load_dataset
    train=rows('train');random.Random(SEED).shuffle(train)
    if source=='massive': texts=[x['text'] for x in train[:n]]
    elif source=='wikitext':
        wiki=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='train')
        texts=[x['text'].strip() for x in wiki if x['text'].strip() and not x['text'].lstrip().startswith('=')]
        random.Random(SEED).shuffle(texts)
        if len(texts)<n:raise ValueError('not enough WikiText examples')
    else:raise ValueError(source)
    tokenized=[]
    for text in texts:
        try:ids=prompt(tok,text)
        except ValueError:continue
        tokenized.append(ids)
        if len(tokenized)==n:break
    if len(tokenized)!=n:raise ValueError('Not enough calibration examples under prompt limit')
    return Dataset.from_dict({'input_ids':tokenized,'attention_mask':[[1]*len(p) for p in tokenized]})

def quantize(method,source,n=128):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier,QuantizationModifier
    tok=AutoTokenizer.from_pretrained(BASE)
    ds=calibration(source,tok,n)
    if method=='awq':
        from llmcompressor.modifiers.awq import AWQModifier
        recipe=[AWQModifier(targets='Linear',scheme='W4A16_ASYM',ignore=['lm_head'])]
    elif method=='gptq':recipe=[GPTQModifier(targets='Linear',scheme='W4A16',ignore=['lm_head'])]
    elif method=='smoothquant':
        from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
        recipe=[SmoothQuantModifier(smoothing_strength=0.5),QuantizationModifier(targets='Linear',scheme='W8A8',ignore=['lm_head'])]
    else:raise ValueError(method)
    label=f'{method}_{source}';dest=OUT/'quantized'/label
    if (dest/'config.json').exists():print('reuse quantized',dest,flush=True)
    else:
        dest.mkdir(parents=True,exist_ok=True)
        model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.bfloat16,device_map='cuda:0')
        try:
            oneshot(model=model,processor=tok,dataset=ds,recipe=recipe,max_seq_length=2048,num_calibration_samples=n)
            model.save_pretrained(dest,save_compressed=method!='smoothquant')
            tok.save_pretrained(dest)
        finally:del model;gc.collect()
    save(OUT/'quantized'/f'{label}.config.json',{'method':method,'calibration':source,'samples':n,'base_model':BASE})
    evaluate(label,dest,quantized=True)

def train_student(n,lr,epochs):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,Trainer,TrainingArguments
    from peft import LoraConfig,get_peft_model,TaskType
    src=OUT/'teacher_answers'/'answers.jsonl'
    if not src.exists():raise FileNotFoundError(src)
    x=schema();all_rows=[json.loads(s) for s in src.read_text(encoding='utf-8').splitlines() if s.strip()]
    valid=[r for r in all_rows if parse(r['teacher_answer'],x) is not None]
    selected=valid[:n]
    if len(selected)<n:raise ValueError(f'only {len(selected)} valid teacher labels, requested {n}')
    tok=AutoTokenizer.from_pretrained(BASE)
    class Samples(torch.utils.data.Dataset):
        def __len__(self):return len(selected)
        def __getitem__(self,i):
            row=selected[i];p=list(row['prompt_ids']);a=list(row['answer_ids'])
            if not a or a[-1]!=tok.eos_token_id:a.append(tok.eos_token_id)
            return {'input_ids':p+a,'attention_mask':[1]*(len(p)+len(a)),'labels':[-100]*len(p)+a}
    for row in selected:
        if row['prompt_ids'] != prompt(tok,row['text']):raise ValueError('Teacher/student prompt tokenization differs')
        if tok.decode(row['answer_ids'],skip_special_tokens=True).strip()!=row['teacher_answer'].strip():raise ValueError('Teacher/student answer tokenization differs')
    def collate(batch):
        pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        width=max(len(r['input_ids']) for r in batch)
        return {k:torch.tensor([r[k]+[(-100 if k=='labels' else 0 if k=='attention_mask' else pad)]*(width-len(r[k])) for r in batch],dtype=torch.long) for k in ('input_ids','attention_mask','labels')}
    tag=f'sft_n{n}_lr{lr:g}_ep{epochs:g}';path=OUT/'training'/tag;path.mkdir(parents=True,exist_ok=True)
    model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.bfloat16)
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type=TaskType.CAUSAL_LM))
    args=TrainingArguments(output_dir=str(path/'checkpoints'),per_device_train_batch_size=1,gradient_accumulation_steps=8,learning_rate=lr,num_train_epochs=epochs,bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},seed=SEED,data_seed=SEED,save_strategy='no',logging_steps=10,report_to='none',remove_unused_columns=False)
    trainer=Trainer(model=model,args=args,train_dataset=Samples(),data_collator=collate)
    trainer.train();save(path/'training_log.json',trainer.state.log_history)
    model=model.merge_and_unload();model.config.use_cache=True
    ckpt=path/'model';ckpt.mkdir(exist_ok=True);model.save_pretrained(ckpt,safe_serialization=True);tok.save_pretrained(ckpt)
    del trainer,model;gc.collect();torch.cuda.empty_cache()
    scores=evaluate(tag,ckpt)
    save(path/'config.json',{'n':n,'lr':lr,'epochs':epochs,'base_model':BASE,'scores':scores})

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['data','baseline','teacher','teacher-answers','quant','sft']);p.add_argument('--method',choices=['awq','gptq','smoothquant']);p.add_argument('--source',choices=['massive','wikitext'],default='massive');p.add_argument('--samples',type=int,default=128);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--epochs',type=float,default=1.0)
    a=p.parse_args();random.seed(SEED)
    if a.stage=='data':data()
    elif a.stage in ('baseline','teacher'):evaluate(a.stage,BASE if a.stage=='baseline' else TEACHER)
    elif a.stage=='teacher-answers':teacher_answers(a.samples)
    elif a.stage=='quant':quantize(a.method,a.source,a.samples)
    elif a.stage=='sft':train_student(a.samples,a.lr,a.epochs)

if __name__=='__main__':main()

