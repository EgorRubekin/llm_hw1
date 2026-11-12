import json
from collections import Counter


def fingerprint(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def score(actual, expected, mode):
    left, right = Counter(map(fingerprint, actual)), Counter(map(fingerprint, expected))
    exact = float(left == right)
    if mode == 'exact' or not expected:
        return exact
    if mode != 'partial': raise ValueError(mode)
    names_a = Counter(x['name'] for x in actual)
    names_b = Counter(x['name'] for x in expected)
    denom = len(actual) + len(expected)
    call_f1 = 2 * sum((left & right).values()) / denom
    name_f1 = 2 * sum((names_a & names_b).values()) / denom
    return .7 * call_f1 + .3 * name_f1


def make_reward(tokenizer, mode, limit):
    from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    parser = Qwen3CoderToolParser(tokenizer)
    def reward(completions, reference, tool_spec, completion_ids=None, **kwargs):
        results = []
        ids = completion_ids if completion_ids is not None else [None] * len(completions)
        for text, target, spec, tokens in zip(completions, reference, tool_spec, ids, strict=True):
            if not isinstance(text, str):
                text = ''.join(m.get('content') or '' for m in text)
            if not text.strip() or (tokens is not None and len(tokens) >= limit and tokens[-1] != tokenizer.eos_token_id):
                results.append(0.0); continue
            request = ChatCompletionRequest.model_validate({'model': 'policy', 'messages': [{'role': 'user', 'content': 'score'}], 'tools': json.loads(spec)})
            parser.tools = request.tools
            try:
                parsed = parser.extract_tool_calls(text, request)
                calls = [{'name': c.function.name, 'arguments': json.loads(c.function.arguments)} for c in parsed.tool_calls]
                valid = all(text.count(a) == text.count(b) for a, b in [('<tool_call>', '</tool_call>'), ('<function=', '</function>'), ('<parameter=', '</parameter>')])
                valid &= text.count('<tool_call>') == len(calls)
                if not calls and any(tag in text for tag in ('<tool', '<function', '<parameter')): valid = False
                results.append(score(calls, json.loads(target), mode) if valid else 0.0)
            except (ValueError, TypeError, KeyError):
                results.append(0.0)
        return results
    reward.__name__ = 'tool_' + mode
    return reward
