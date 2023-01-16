from typing import List
import os

from transformers import AutoTokenizer, AutoModel
import torch

from repair.utils import CompileResult, gcc_compile, get_torch_device

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class CodeBertEmbedder(object):

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base")
        self.model = self.model.to(get_torch_device())
        self.model.eval()
        self.cache_compile_results = {}

    def prepare_msg_input(self, msg: str):
        # FIXME: better preparation (e.g. distinguish between NL/code)
        return msg

    def embed_compile_results(self, compile_results: List[CompileResult]):
        """
        following https://github.com/microsoft/CodeBERT
        """
        if isinstance(compile_results, CompileResult):
            compile_results = [compile_results]

        msgs = [r.error for r in compile_results]
        if any(m is None for m in msgs):
            raise Exception("Embedding compiler error message requires error")

        txt_inputs = [self.prepare_msg_input(m) for m in msgs]
        encoded_inputs = self.tokenizer(
            txt_inputs,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
        ).to(get_torch_device())
        with torch.no_grad():
            result = self.model(**encoded_inputs)
        # aggregate with average
        # FIXME: consider other approaches to aggregating or using pooled output
        embeddings = result[0].mean(dim=1)
        return embeddings

    def embed(self, programs: List[str]):
        msgs = []
        for prog in programs:
            if prog not in self.cache_compile_results:
                self.cache_compile_results[prog] = gcc_compile(prog)
            msgs.append(self.cache_compile_results[prog])
        return self.embed_compile_results(msgs)
