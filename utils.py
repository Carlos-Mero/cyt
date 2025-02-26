from math_verify import parse, verify
from typing import Iterable, Any
import json
import logging
import re
from accelerate import Accelerator

from datasets import Dataset, concatenate_datasets, load_dataset

def find_boxed(pred_str: str):
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a

def load_jsonl(file: str) -> Iterable[Any]:
    with open (file, "r", encoding='utf-8') as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print(f"failed to load json: {line}")
                exit()

def extract_ans_gsm8k(example):
    """This function extracts the last number from a ground truth solution as the golden answer in gsm8k"""
    numbers = re.findall(r'\d+', example['answer'])
    example['answer'] = numbers[-1] if numbers else None
    return example

def load_datasets(dspaths):
    dss = []
    for dspath in dspaths:
        if dspath == "datasets/math500.jsonl":
            ds = Dataset.from_list([e for e in load_jsonl(dspath)])
            dss.append(ds)
        elif dspath == "openai/gsm8k":
            ds = (load_dataset(dspath, 'main'))['train']
            ds = ds.map(extract_ans_gsm8k)
            ds = ds.rename_column('question', 'problem')
            dss.append(ds)
        else:
            raise NotImplementedError("Unknown dataset!")
    return concatenate_datasets(dss)

class Evaluator():
    def __init__(self, max_clength: int = 1024, stepsize: int = 2, min_clength: int = 512, ema: float = 0.9, tokenizer = None):
        self.logger = logging.getLogger("evaluator")
        self.max_clength = max_clength
        self.ema = ema
        self.stepsize = stepsize
        self.min_clength = min_clength
        self.tokenizer = tokenizer
        self.accelerator = Accelerator()

    def __call__(self, prompts, completions, answer, **kwargs):
        completions = [completion if isinstance(completion, str) else completion[0]['content'] for completion in completions]
        avg_len = (sum(len(self.tokenizer.tokenize(s)) for s in completions) / len(completions))
        self.max_clength = int(self.max_clength * self.ema + avg_len * (1.0 - self.ema))
        len_modifier = [2.0 if len(c) <= self.max_clength else 1.0 for c in completions]
        golds = [parse(ans) for ans in answer]
        answers = [parse(find_boxed(completion)) for completion in completions]
        rewards = [1.0 if verify(g, a) else -1.0 for (g, a) in zip(golds, answers)]
        rewards = [r * lg for (r, lg) in zip(rewards, len_modifier)]
        self.logger.info(f"rewards: {rewards}")
        if self.accelerator.is_main_process:
            logidx = rewards.index(max(rewards))
            self.logger.info(f"sample output cot with the best reward:\n{prompts[logidx]}\n{completions[logidx]}")
            self.logger.info(f"current self max length: {self.max_clength}")
        return rewards

    @property
    def __name__(self):
        return "Evaluator"
