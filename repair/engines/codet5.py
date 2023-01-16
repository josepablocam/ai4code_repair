import os
import re
from typing import List, Dict

from transformers import AutoTokenizer, T5ForConditionalGeneration


from repair.utils import BenchmarkRunner, RepairEngine, gcc_compile, get_torch_device

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class CodeT5ClozeRepair(RepairEngine, BenchmarkRunner):
    """
    https://arxiv.org/pdf/2207.08281.pdf
    """
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-base')
        self.model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')

    def _localize_line(self, code):
        result = gcc_compile(code)
        if result.ok:
            raise Exception("To localize must have compile error")

        line_regex = re.compile(".c:([0-9]+):[0-9]+")
        matches = line_regex.findall(result.error)
        # turn to 0-indexed
        line_nums = set(int(m) - 1 for m in matches)
        if len(line_nums) == 0:
            # default to first line
            # FIXME: better fall back
            return [0]
        return line_nums
        
    
    def _add_code_masks(self, code):
        # FIXME: extend to not mask whole line
        # but just range indicated by error message
        line_nums = self._localize_line(code)
        lines = code.split("\n")
        mask_tokens = []
        for ix, l_ix in enumerate(line_nums):
            mask_token = f"<extra_id_{ix}>"
            lines[l_ix] = mask_token #+ ";"
            mask_tokens.append(mask_token)
        return ["\n".join(lines), mask_tokens]

    def _decoded_to_mask_tokens_map(self, decoded: str, mask_tokens: List[str]):
        mapping = {}
        for mask_token in mask_tokens:
            start_ix = decoded.find(mask_token)
            # offset by length of mask itself
            start_ix += len(mask_token)
            end_ix = decoded[start_ix:].find("<extra_id_")
            if end_ix >= 0:
                mask_value = decoded[start_ix:(start_ix + end_ix)]
            else:
                # rest of string
                mask_value = decoded[start_ix:]
            # remove any special tokens
            special_tokens = [self.tokenizer.eos_token, self.tokenizer.pad_token]
            for special_tok in special_tokens:
                mask_value = mask_value.replace(special_tok, "")
            mapping[mask_token] = mask_value
        return mapping

    def _decode_code_with_cloze(self, code: str, mask_token_map: Dict[str, str]):
        for token_name, token_value in mask_token_map.items():
            code = code.replace(token_name, token_value)
        return code

    def repair(self, code: str, **kwargs):
        # Replace entire line that has error
        # extract line from errormessage
        new_code, mask_tokens = self._add_code_masks(code)
        encoded_inputs = self.tokenizer(
            new_code,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
        ).to(get_torch_device())
        # FIXME: we can set the average length of mask_token to be average length
        # of each line tokenized
        generated = self.model.generate(
            **encoded_inputs, 
            max_length=20 * len(mask_tokens),
            num_beams=kwargs.get("num_beams", 3),
            # default to as many as beam size
            num_return_sequences=kwargs.get("num_return_sequences", kwargs.get("num_beams", 3)),
            early_stopping=kwargs.get("early_stopping", True),
        )
        results = []
        for seq in generated:
            decoded = self.tokenizer.decode(seq, skip_special_tokens=False)
            mask_token_map = self._decoded_to_mask_tokens_map(decoded, mask_tokens)
            filled_code = self._decode_code_with_cloze(new_code, mask_token_map)
            results.append({"repair": filled_code})
        return results


class CodeT5FineTunedRepair(RepairEngine, BenchmarkRunner):
    pass

