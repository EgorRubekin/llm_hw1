import argparse
import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

MODEL = 'Qwen/Qwen3.5-2B'
REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'
DATA_REVISION = '6bda777c88d21e5a204703c1ee45597a8fa4f734'
SEED = 73


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def normalize(value):
    if isinstance(value, list):
        return [normalize(v) for v in value]
    if not isinstance(value, dict):
        return value
    out = {k: normalize(v) for k, v in value.items() if v is not None}
    if isinstance(out.get('type'), str):
        out['type'] = {'dict': 'object', 'float': 'number', 'tuple': 'array', 'any': 'string'}.get(out['type'], out['type'])
    return out


def references(answer, mapping):
    answer = answer.strip()
    if not answer:
        raise ValueError('empty answer')
    if not answer.startswith('['):
        if '<tool' in answer or re.match(r'^[\w.]+\(', answer):
            raise ValueError('ambiguous call')
        return []
    tokens = []
    i = 0
    quote = None
    escaped = False
    depth = 0
    while i < len(answer):
        c = answer[i]
        if quote:
            tokens.append(c)
            if escaped:
                escaped = False
            elif c == '\\':
                escaped = True
            elif c == quote:
                quote = None
            i += 1
            continue
        if c in '\"\'':
            quote = c
        if depth == 1 and (i == 1 or answer[i-1] in ', '):
            found = next((n for n in sorted(mapping, key=len, reverse=True)
                          if answer.startswith(n + '(', i)), None)
            if found:
                tokens.append(mapping[found]); i += len(found); continue
        if c in '[({': depth += 1
        if c in '])}': depth -= 1
        tokens.append(c); i += 1
    tree = ast.parse(''.join(tokens), mode='eval').body
    if not isinstance(tree, ast.List) or not tree.elts:
        raise ValueError('not a nonempty call list')
    def literal(node):
        class JsonNames(ast.NodeTransformer):
            def visit_Name(self, item):
                if item.id not in ('true', 'false', 'null'):
                    raise ValueError('nonliteral argument')
                return ast.copy_location(ast.Constant({'true': True, 'false': False, 'null': None}[item.id]), item)
        return ast.literal_eval(JsonNames().visit(node))
    result = []
    for call in tree.elts:
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id not in mapping.values() or call.args:
            raise ValueError('unsupported call')
        args = {}
        for kw in call.keywords:
            if kw.arg is None or kw.arg in args:
                raise ValueError('duplicate or expanded arguments')
            args[kw.arg] = literal(kw.value)
        result.append({'name': call.func.id, 'arguments': args})
    return result


def convert(dialogue):
    system = dialogue['system']
    marker = system.index('Here is a list of functions')
    start = system.index('[', marker)
    schemas, _ = json.JSONDecoder().raw_decode(system[start:])
    if not schemas or len({s['name'] for s in schemas}) != len(schemas):
        raise ValueError('empty or duplicate tools')
    mapping = {s['name']: f'tool_{i}' for i, s in enumerate(schemas)}
    tools = [{'type': 'function', 'function': {
        'name': mapping[s['name']], 'description': s.get('description') or '',
        'parameters': normalize(s.get('parameters') or {'type': 'object', 'properties': {}})
    }} for s in schemas]
    turns = dialogue['conversations']
    if len(turns) < 2 or turns[0]['from'] != 'user' or turns[1]['from'] != 'assistant':
        raise ValueError('unsupported opening roles')
    calls = references(turns[1]['value'], mapping)
    messages = [{'role': 'user', 'content': turns[0]['value']}]
    group = digest({'messages': messages, 'tools': tools})
    return {'group': group, 'messages': messages, 'tool_spec': json.dumps(tools),
            'reference': json.dumps(calls), 'kind': 'call' if calls else 'no_call'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path)
    p.add_argument('--output', type=Path, default=Path('data'))
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--max-prompt-tokens', type=int, default=2048)
    a = p.parse_args()
    if a.source is None:
        from huggingface_hub import hf_hub_download
        a.source = Path(hf_hub_download('Team-ACE/ToolACE', 'data.json', repo_type='dataset', revision=DATA_REVISION))
    source = json.loads(a.source.read_text())
    tokenizer = None
    if not a.audit_only:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    buckets = {'train': [], 'validation': []}
    counts = Counter(); seen = set(); lengths = []
    for dialogue in source:
        try:
            row = convert(dialogue)
            if row['group'] in seen:
                counts['duplicate'] += 1; continue
            seen.add(row['group'])
            if tokenizer is not None:
                row['prompt'] = tokenizer.apply_chat_template(row['messages'], tools=json.loads(row['tool_spec']), tokenize=False, add_generation_prompt=True, enable_thinking=False)
                size = len(tokenizer.encode(row['prompt'], add_special_tokens=False))
                if size > a.max_prompt_tokens:
                    counts['too_long'] += 1; continue
                lengths.append(size)
            split = 'validation' if int(digest([SEED, row['group']])[:8], 16) % 10 == 0 else 'train'
            buckets[split].append(row); counts[split + '/' + row['kind']] += 1
        except (ValueError, TypeError, KeyError, SyntaxError) as exc:
            counts['rejected/' + type(exc).__name__] += 1
    a.output.mkdir(parents=True, exist_ok=True)
    stats = {'source': 'Team-ACE/ToolACE', 'source_revision': DATA_REVISION if a.source.name == 'data.json' else 'local file; see SHA256',
             'source_sha256': hashlib.sha256(a.source.read_bytes()).hexdigest(), 'seed': SEED,
             'dialogues': len(source), 'counts': dict(counts), 'audit_only': a.audit_only,
             'prompt_tokens': {str(q): sorted(lengths)[int(q * (len(lengths)-1))] for q in (0, .5, .9, 1)} if lengths else {}}
    (a.output / 'eda.json').write_text(json.dumps(stats, indent=2))
    if not a.audit_only:
        if not all(buckets.values()): raise ValueError('empty split')
        for split, rows in buckets.items():
            (a.output / (split + '.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    print(json.dumps(stats, indent=2))


if __name__ == '__main__': main()
