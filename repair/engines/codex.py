from typing import Dict, Any, List, Tuple, Callable
import os

import openai
import numpy as np
import tiktoken
import tqdm

from repair.utils import (
    gcc_compile,
    CompileResult,
    RepairTaskRecord,
    RepairEngine,
    BenchmarkRunner,
)


class CodexEngine(object):

    def __init__(
        self,
        openai_key: str,
        temperature: float = 0,
        n: int = 1,
        maxtokens: int = 200,
        stop: str = "///",
        engine: str = "code-davinci-002",
    ):
        openai.api_key = openai_key
        self.temperature = temperature
        self.maxtokens = maxtokens
        self.n = n
        self.top_p = 1.0
        self.presence_penalty = 0.0
        self.frequency_penalty = 0.0
        self.stop = stop
        self.engine = engine

    def complete(self, prompt: str, **kwargs):
        try:
            response = openai.Completion.create(
                engine=self.engine,
                prompt=prompt,
                max_tokens=kwargs.get("maxtokens", self.maxtokens),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                presence_penalty=kwargs.get("presence_penalty",
                                            self.presence_penalty),
                frequency_penalty=kwargs.get("frequency_penalty",
                                             self.frequency_penalty),
                stop=kwargs.get("stop", self.stop),
                n=kwargs.get("n", self.n),
                logprobs=1,
            )
            unsorted_completions = [
                (completion["text"],
                 np.mean(completion["logprobs"]["token_logprobs"]))
                for completion in response["choices"]
            ]

            # higher average prob should come first
            sorted_completions = sorted(unsorted_completions,
                                        key=lambda x: x[1],
                                        reverse=True)
            result_dicts = []
            for completion_txt, avg_logprob in sorted_completions:
                result_dicts.append({
                    "completion": completion_txt,
                    "avg_logprob": avg_logprob,
                })
            return result_dicts
        except Exception as e:
            print("Skipping exception")
            print(e)
            return None


class CodexRepair(RepairEngine, BenchmarkRunner):

    def run_benchmark(
        self,
        cases: List[RepairTaskRecord],
        **kwargs,
    ) -> List[List[str]]:
        predictions = [
            self.repair(t.source, **kwargs) for t in tqdm.tqdm(cases)
        ]
        return [[p["repair"] for p in group] for group in predictions]


class CodexBaseRepair(CodexRepair):
    """
    Based on https://beta.openai.com/examples/default-fix-python-bugs
    """

    def __init__(self, *args, **kwargs):
        self.codex = CodexEngine(*args, **kwargs)

    def get_prompt(self, code: str, **kwargs) -> str:
        # Based on https://beta.openai.com/examples/default-fix-python-bugs
        prompt = """//// Fix bugs in the below code\n"""
        prompt += f"/// Buggy C\n{code}\n\n"
        prompt += "/// Fixed C"
        if 'fixed' in kwargs and kwargs['fixed'] is not None:
            prompt += f"\n{kwargs['fixed']}\n\n"

        return prompt

    def get_repair_from_completion_(
            self, completion_dict: Dict[str, Any]) -> Dict[str, Any]:
        comp = completion_dict["completion"]
        # remove first \n and last \n, result from our prompt style
        if comp[0] == "\n":
            comp = comp[1:]
        if comp[-1] == "\n":
            comp = comp[:-1]
        return {"repair": comp, "score": completion_dict["avg_logprob"]}

    def deduplicate_repairs_(
            self, repair_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        unique_repair_dicts = []
        already_included = set()
        for rd in repair_dicts:
            repair_str = rd["repair"]
            if repair_str not in already_included:
                unique_repair_dicts.append(rd)
                already_included.add(repair_str)
        return unique_repair_dicts

    def repair(self, code: str, **kwargs):
        # FIXME: warn if number of tokens too few for length of code
        # or set to num tokens + K
        # Use: https://github.com/openai/tiktoken , which is fast openai tokenizer
        # to estimate tokens for code and pass in as maxtokens=<val>
        prompt = self.get_prompt(code, **kwargs)
        completion_dicts = self.codex.complete(prompt, **kwargs)
        if completion_dicts is None:
            return []

        repair_dicts = [
            self.get_repair_from_completion_(c) for c in completion_dicts
        ]
        repair_dicts = self.deduplicate_repairs_(repair_dicts)
        return repair_dicts


class CodexWithErrorInfo(CodexBaseRepair):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_cache = {}

    def _get_error_info(self, code: str) -> str:
        if code not in self.error_cache:
            self.error_cache[code] = gcc_compile(code).error
        return self.error_cache[code]

    def get_prompt(self, code: str, **kwargs):
        prompt = """//// Fix bugs in the below code\n"""
        prompt += f"/// Buggy C\n{code}\n\n"
        error_msg = self._get_error_info(code)
        prompt += f"/// Error Message\n{error_msg}\n\n"
        prompt += "/// Fixed C"
        if 'fixed' in kwargs and kwargs['fixed'] is not None:
            prompt += f"\n{kwargs['fixed']}\n\n"

        return prompt


class FewShotSelector(object):

    def __init__(self, example_bank: List[RepairTaskRecord]):
        self.example_bank = example_bank

    def select_shots(self, code: str, k: int) -> List[RepairTaskRecord]:
        raise NotImplementedError()


class FixedFewShots(FewShotSelector):

    def select_shots(self, code: str, k: int) -> List[RepairTaskRecord]:
        return self.example_bank[:k]


class RandomFewShots(FewShotSelector):

    def select_shots(self, code: str, k: int) -> List[RepairTaskRecord]:
        return np.random.choice(self.example_bank,
                                size=min(len(self.example_bank), k),
                                replace=False)


# FIXME: implement a similarity-based shot selector using
# some of the utlities in codesimilarity.py
class SimilarityFewShots(FewShotSelector):

    def __init__(self, example_bank: List[RepairTaskRecord]):
        raise NotImplementedError()

    def select_shots(self, code: str, k: int) -> List[RepairTaskRecord]:
        raise NotImplementedError()


class CodexWithFewShots(CodexBaseRepair):

    def __init__(self, shot_selector: FewShotSelector, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shot_selector = shot_selector
        self.prompt_helper = CodexWithErrorInfo(*args, **kwargs)

    def get_prompt(self, code: str, **kwargs):
        few_shots = self.shot_selector.select_shots(code, kwargs.get("k", 3))
        prompt = ""
        for shot in few_shots:
            prompt += self.prompt_helper.get_prompt(shot.source,
                                                    fixed=shot.target)

        prompt += self.prompt_helper.get_prompt(code)
        return prompt


GPT_TOKENIZER = None

def gpt_tokenize(code: str) -> List[str]:
    global GPT_TOKENIZER
    if GPT_TOKENIZER is None:
        GPT_TOKENIZER = tiktoken.get_encoding("gpt2")
    return GPT_TOKENIZER.encode(code)