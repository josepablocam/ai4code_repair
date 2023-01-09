from typing import Dict, Any, List, Tuple

import openai
import numpy as np

from repair.utils import gcc_compile, CompileResult, DeepFixRecord


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


class CodexBaseRepair(object):
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
            self.error_cache = gcc_compile(code).error
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


class CodexWithFewShots(CodexBaseRepair):
    """
    Based on https://beta.openai.com/examples/default-fix-python-bugs
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_helper = CodexWithErrorInfo(*args, **kwargs)

    def get_prompt(self, code: str, **kwargs):
        few_shots = kwargs.get('fewshots', None) or []
        prompt = ""
        for (buggy_shot, fixed_shot) in few_shots:
            prompt += self.prompt_helper.get_prompt(buggy_shot,
                                                    fixed=fixed_shot)

        prompt += self.prompt_helper.get_prompt(code)
        return prompt


def generate_basic_example_bank(
        entries: List[DeepFixRecord],
        size=100,
        engine=None) -> List[Tuple[DeepFixRecord, str]]:
    if engine is None:
        engine = CodexWithErrorInfo()
    example_bank = []
    compile_cache = {}

    for entry in entries:
        if len(example_bank) >= size:
            return example_bank
        results = engine.repair(entry.code, n=5)
        for r in results:
            candidate_code = r["repair"]
            if candidate_code not in compile_cache:
                compile_cache[candidate_code] = gcc_compile(candidate_code)
            if compile_cache[candidate_code]:
                # satisfied the compiler
                example = (entry, candidate_code)
                example_bank.append(example)
                # add single example per case
                break
    return example_bank


def create_random_fewshots(target: List[DeepFixRecord],
                           all_programs: List[DeepFixRecord]):
    pass


def create_embedding_fewshots(target: List[DeepFixRecord],
                              all_programs: List[DeepFixRecord]):
    pass
