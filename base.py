import re
import functools
from itertools import permutations, product, groupby
from collections import defaultdict
from dataclasses import dataclass
from openai import AsyncOpenAI
from together import AsyncTogether
import anyio
from anthropic import AsyncAnthropic
from tenacity import AsyncRetrying, wait_fixed, before_sleep_log
import logging

logger = logging.getLogger(__name__)


def seval(l, r, op):
    if op == '+':
        return l + r, True
    elif op == '-':
        return l - r, l >= r
    elif op == '*':
        return l * r, True
    elif op == '/':
        return int(l / r) if r != 0 else None, r != 0 and l % r == 0
    else:
        raise ValueError(f'Invalid operator {op}')

### Generator ###

@dataclass
class Expression:
    expression: str
    result: int
    score: int
    steps: str = None

@functools.cache
def mix(numbers, operations):
    if len(numbers) == 1:
        assert len(operations) == 0
        return [(f'{numbers[0]}',[], numbers[0], True)]
    else:
        mixes = []
        for i in range(1, len(numbers)):
            for left_template, left_steps, left_val, left_valid in mix(numbers[:i], operations[:i-1]):
                for right_template, right_steps, right_val, right_valid in mix(numbers[i:], operations[i:]):
                    op = operations[i-1]
                    if left_valid and right_valid:
                        val, valid = seval(left_val, right_val, op)
                        if valid:
                            steps = left_steps + right_steps + [f"{left_val} {op} {right_val} = {val}"]
                            template = f"({left_template} {op} {right_template})"
                            mixes.append((template, steps, val, True))
        return mixes

def get_all_expressions_new(base_numbers):
    n = len(base_numbers)
    expressions = []
    for num_op in range(1, n):
        for operations in product(['+', '-', '*', '/'], repeat=num_op):
            score = sum([1 if op in ['+', '*'] else 2 if op == '-' else 3 for op in operations]) + 5
            score += 6 if len(set(operations)) == 4 else 0
            for numbers in permutations(base_numbers, num_op+1):
                for expression, steps, val, _ in mix(numbers, operations):
                    expressions.append(Expression(expression, val, score, '\n'.join(steps)))
    return expressions

def get_sorted_expressions(base_numbers):
    results = defaultdict(lambda: [[], {}])
    sorted_expressions = sorted(get_all_expressions_new(base_numbers), key=lambda x: x.result)
    grouped = groupby(sorted_expressions, lambda x: x.result)
    for key, group in grouped:
        assert key not in results
        group = list(group)
        max_score = max(x.score for x in group)
        if len(base_numbers) == 5 and max_score != 18:
            continue
        results[key][0] = sorted(group, key=lambda x: x.score)
        results[key][1]['diff'] = sum(x.score for x in group)/len(group)**2
        results[key][1]['max'] = max_score

    return sorted(list(results.items()), key=lambda x: x[1][1]['diff'])

### Evaluator ###

def expr_to_shot(base_numbers, target, simple, simple_score, best, best_score):
    simple_str = f"""Simple solution ({simple_score} points):
{simple}
"""
    best_str = f"""
Best solution ({best_score} points):
{best}
"""
    header = f"""Example:
Target number: {target}
Base numbers: {', '.join(map(str, base_numbers))}

"""
    if (not simple) or simple_score == best_score:
        return header + best_str
    else:
        return header + simple_str + best_str

def check_answer(message, target, base_numbers):
    try:
        last_block = re.findall(r'((?:\s*(?:\n|^)\s*\d+\s*[+\-*\/]\s*\d+\s*=\s*\d+\s*)+)(?:\n|$)', message.strip())[-1]
    except:
        print('No answer block found')
        return 0, 'wrong_format'

    avilable_numbers = base_numbers.copy()
    score = 0
    used_operations = set()
    for line in last_block.strip().split('\n'):
        if line.isspace() or not line:
            continue
        try:
            oper1, operator, oper2, result = re.fullmatch(r'(\d+)\s*([+\-*\/])\s*(\d+)\s*=\s*(\d+)', line.strip()).groups()
        except:
            raise ValueError('This should not happen', line)
        try:
            if float(oper1) != int(float(oper1)) or float(oper2) != int(float(oper2)) or float(result) != int(float(result)):
                print('The numbers should be integers', line)
                return 0, 'illegal_intermediate_number'
        except OverflowError:
            print('The numbers are too big', line)
            return 0, 'illegal_intermediate_number'
        oper1, oper2, result = int(oper1), int(oper2), int(result)
        if oper1 < 0 or oper2 < 0 or result < 0:
            print('The numbers should be positive', line)
            return 0, 'illegal_intermediate_number'
        if seval(oper1, oper2, operator)[0] != result:
            print('The calculation is not correct', line)
            return 0, 'wrong_calculation'
        try:
            avilable_numbers.remove(int(oper1))
            avilable_numbers.remove(int(oper2))
        except:
            print('You are using a number you should not', line)
            return 0, 'illegal_number_usage'
        avilable_numbers.append(int(result))

        if operator == '+':
            score += 1
        elif operator == '*':
            score += 1
        elif operator == '-':
            score += 2
        elif operator == '/':
            score += 3
        else:
            print('The operator is not valid', line)
            return 0, 'illegal_operator'

        used_operations.add(operator)

    if len(used_operations) == 4:
        score += 6

    assert score <= 13 or len(base_numbers) > 5

    if result != target:
        print('The result is not the target number')
        return 0, 'wrong_result'
    score += 5

    return score, 'correct'

async def openai_call_wrapper(temp, max_tokens, top_p, client, message, model):
    res = await client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'user', 'content': message}
        ],
        temperature=temp,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    return res.choices[0].message.content

async def claude_call_wrapper(temp, max_tokens, top_p, client, message, model):
    res = await client.messages.create(
        model=model,
        messages=[
            {'role': 'user', 'content': message}
        ],
        temperature=temp,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    return res.content[0].text

def get_client_models(cfg):
    if cfg.api == 'openai':
        if cfg.get('base_url', None):
            client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        else:
            client = AsyncOpenAI(api_key=cfg.api_key)
        client.call_wrapper = functools.partial(openai_call_wrapper, cfg.temperature, cfg.max_tokens, cfg.top_p)
    elif cfg.api == 'together':
        client = AsyncTogether(api_key=cfg.api_key)
        client.call_wrapper = functools.partial(openai_call_wrapper, cfg.temperature, cfg.max_tokens, cfg.top_p)
    elif cfg.api == 'claude':
        client = AsyncAnthropic(api_key=cfg.api_key)
        client.call_wrapper = functools.partial(claude_call_wrapper, cfg.temperature, cfg.max_tokens, cfg.top_p)
    else:
        raise ValueError('Invalid API')
    models = cfg.models
    return client, models

async def call_api_single(client, message, model, results, idx):
        async for attempt in AsyncRetrying(wait=wait_fixed(3),
                                           before_sleep=before_sleep_log(logger, logging.WARNING),
                                           reraise=True):
            with attempt:
                ans = await client.call_wrapper(client, message, model)
                results[idx]= {'answer': ans, 'model': model}


async def call_api(client, models, messages):
    results = {}
    async with anyio.create_task_group() as tg:
        for i in range(len(messages)):
            for model in models:
                tg.start_soon(call_api_single, client, messages[i], model, results, i)
    return results