import os
import logging
import re
import time
from typing import List, Dict, Union, Any, Optional

import numpy as np
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import tqdm
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    pass
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup

from repair.utils import (BenchmarkRunner, RepairEngine, gcc_compile,
                          get_torch_device, RepairTaskRecord, get_train_data)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

DEFAULT_CODET5_VERSION = "Salesforce/codet5-small"


class BaseCodeT5Repair(RepairEngine, BenchmarkRunner):

    def run_benchmark(
        self,
        cases: List[RepairTaskRecord],
        **kwargs,
    ) -> List[List[str]]:
        all_sources = [t.source for t in cases]
        if kwargs.get("verbose", False):
            # show progress bar, may help if slow (i.e. no gpu)
            predictions = [
                self.repair(s, **kwargs) for s in tqdm.tqdm(all_sources)
            ]
        else:
            predictions = self.repair(all_sources, **kwargs)
        return [[p["repair"] for p in group] for group in predictions]


class CodeT5ClozeRepair(BaseCodeT5Repair):
    """
    https://arxiv.org/pdf/2207.08281.pdf
    """

    def __init__(self, model_version=None):
        self.model_version = DEFAULT_CODET5_VERSION if model_version is None else model_version
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_version)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_version)
        self.model = self.model.to(get_torch_device())
        self.model.eval()

    def _localize_line(self, code):
        result = gcc_compile(code)
        if result.ok:
            raise Exception("To localize must have compile error")

        line_regex = re.compile(".c:([0-9]+):[0-9]+: error:")
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
            lines[l_ix] = mask_token  #+ ";"
            mask_tokens.append(mask_token)
        return ["\n".join(lines), mask_tokens]

    def _decoded_to_mask_tokens_map(self, decoded: str,
                                    mask_tokens: List[str]):
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
            special_tokens = [
                self.tokenizer.eos_token, self.tokenizer.pad_token
            ]
            for special_tok in special_tokens:
                mask_value = mask_value.replace(special_tok, "")
            mapping[mask_token] = mask_value
        return mapping

    def _fill_code_cloze(self, code: str, mask_token_map: Dict[str, str]):
        for token_name, token_value in mask_token_map.items():
            code = code.replace(token_name, token_value)
        return code

    def repair(self, code: Union[str, List[str]], **kwargs):
        single_program = False
        if isinstance(code, str):
            str_inputs = [code]
            single_program = True
        else:
            assert isinstance(code, list) and isinstance(code[0], str)
            str_inputs = code

        # Replace entire line that has error
        # extract line from errormessage
        batched_mask_tokens = []
        batched_masked_code = []

        for code in str_inputs:
            new_code, mask_tokens = self._add_code_masks(code)
            batched_masked_code.append(new_code)
            batched_mask_tokens.append(mask_tokens)

        num_inputs = len(str_inputs)

        # run as batch with codet5
        encoded_inputs = self.tokenizer(
            batched_masked_code,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
        ).to(get_torch_device())

        # FIXME: set max length to be some reasonable value based on average length of masked line
        max_length = int(
            20 * np.ceil(np.mean([len(ts) for ts in batched_mask_tokens])))
        with torch.no_grad():
            generated = self.model.generate(
                **encoded_inputs,
                max_length=kwargs.get("max_length", max_length),
                num_beams=kwargs.get("num_beams", 3),
                # default to as many as beam size
                num_return_sequences=kwargs.get("num_return_sequences",
                                                kwargs.get("num_beams", 3)),
                early_stopping=kwargs.get("early_stopping", True),
            )
            # TODO: check better reshaping
            batched_generated = generated.reshape(num_inputs, -1,
                                                  generated.shape[-1])

        results = []
        # create repairs by replacing masks in masked code with the mask values generated by codet5
        for seqs, masked_code, mask_tokens in zip(batched_generated,
                                                  batched_masked_code,
                                                  batched_mask_tokens):
            acc = []
            for seq in seqs:
                decoded = self.tokenizer.decode(seq, skip_special_tokens=False)
                mask_token_map = self._decoded_to_mask_tokens_map(
                    decoded, mask_tokens)
                filled_code = self._fill_code_cloze(masked_code,
                                                    mask_token_map)
                acc.append({"repair": filled_code})
            results.append(acc)
        if single_program:
            results = results[0]
        return results


def _load_labeled_dataset_tokenized(
    cases: List[RepairTaskRecord],
    tokenizer: Any,
    max_length: Optional[int] = None,
    task_prefix: Optional[str] = None,
):
    source_data, target_data = zip(*[[case.source, case.target]
                                     for case in cases])
    assert not any(e is None for e in target_data), "All data must be labeled"
    # add a "Fix prefix" so that fine-tuned model knows to "switch" tasks
    if task_prefix is not None:
        source_data = [task_prefix + source for source in source_data]
    encoded_source = tokenizer(
        source_data,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=max_length,
    )['input_ids'].to(get_torch_device())

    encoded_target = tokenizer(
        target_data,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=max_length,
    )['input_ids'].to(get_torch_device())

    return TensorDataset(encoded_source, encoded_target)


class CodeT5FineTunedRepair(BaseCodeT5Repair):

    def __init__(self, model_version=None, fine_tuned_path=None, task_prefix=None):
        self.model_version = DEFAULT_CODET5_VERSION if model_version is None else model_version
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_version)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_version)
        self.task_prefix = "Fix C: " if task_prefix is None else task_prefix

        if fine_tuned_path is not None:
            print(f"Loading fine tuned model from {fine_tuned_path}")
            self.model.load_state_dict(torch.load(fine_tuned_path))

        self.model = self.model.to(get_torch_device())

    def _save_checkpoint(self, checkpoint_path):
        print(f"Saving checkpoint model to {checkpoint_path}")
        torch.save(self.model.state_dict(), checkpoint_path)

    def repair(self, code: Union[str, List[str]], **kwargs):
        single_program = False
        if isinstance(code, str):
            str_inputs = [code]
            single_program = True
        else:
            assert isinstance(code, list) and isinstance(code[0], str)
            str_inputs = code

        str_inputs = [self.task_prefix + s for s in str_inputs]

        num_inputs = len(str_inputs)
        # run as batch with codet5
        encoded_inputs = self.tokenizer(
            str_inputs,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
        ).to(get_torch_device())

        # FIXME: set max length to be some reasonable value
        mean_length = encoded_inputs["attention_mask"].sum(
            dim=1).to(float).mean().item()
        max_length = int(20 * mean_length)
        with torch.no_grad():
            generated = self.model.generate(
                **encoded_inputs,
                max_length=kwargs.get("max_length", max_length),
                num_beams=kwargs.get("num_beams", 3),
                # default to as many as beam size
                num_return_sequences=kwargs.get("num_return_sequences",
                                                kwargs.get("num_beams", 3)),
                early_stopping=kwargs.get("early_stopping", True),
            )
            batched_generated = generated.reshape(num_inputs, -1,
                                                  generated.shape[-1])

        results = []
        # create repairs by replacing masks in masked code with the mask values generated by codet5
        for seqs in batched_generated:
            acc = []
            for seq in seqs:
                decoded = self.tokenizer.decode(seq, skip_special_tokens=True)
                acc.append({"repair": decoded})
            results.append(acc)
        if single_program:
            results = results[0]
        return results

    def finetune(
        self,
        train_data=None,
        num_epochs=4,
        batch_size=8,
        n_gpu=1,
        weight_decay=0.0,
        learning_rate=5e-5,
        adam_epsilon=1e-8,
        warmup_steps=5,
        gradient_accumulation_steps=1,
        log_dir=None,
        checkpoint_path=None,
    ):
        # this is a simplified version OF https://github.com/salesforce/CodeT5/blob/main/run_gen.py
        # FIXME: add in validation and save best checkpoint rather than take last
        t0 = time.time()

        self.model.train()

        if n_gpu > 1:
            # for DataParallel
            model = torch.nn.DataParallel(model)

        tb_writer = None
        if log_dir is not None:
            tb_writer = SummaryWriter(log_dir)

        # Prepare training data loader
        if train_data is None:
            train_data = get_train_data()
        train_dataset = _load_labeled_dataset_tokenized(
            train_data,
            self.tokenizer,
            task_prefix=self.task_prefix,
        )
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset,
                                      sampler=train_sampler,
                                      batch_size=4,
                                      pin_memory=False)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in self.model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            weight_decay
        }, {
            'params': [
                p for n, p in self.model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            0.0
        }]

        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=learning_rate,
                          eps=adam_epsilon)
        num_train_optimization_steps = num_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        print("***** Running training *****")
        print(f"  Num examples = {train_example_num}")
        print(f"  Batch size = {batch_size}")
        print(f"  Batch num = {np.ceil(train_example_num / batch_size)}")
        print(f"  Num epoch = {num_epochs}")

        global_step = 0
        for cur_epoch in range(0, num_epochs):
            bar = tqdm.tqdm(train_dataloader,
                            total=len(train_dataloader),
                            desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            self.model.train()

            for step, batch in enumerate(bar):
                batch = tuple(t.to(get_torch_device()) for t in batch)
                source_ids, target_ids = batch
                source_mask = source_ids.ne(self.tokenizer.pad_token_id)
                target_mask = target_ids.ne(self.tokenizer.pad_token_id)

                outputs = self.model(input_ids=source_ids,
                                     attention_mask=source_mask,
                                     labels=target_ids,
                                     decoder_attention_mask=target_mask)
                loss = outputs.loss

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.

                global_step += 1
                if tb_writer is not None:
                    tb_writer.add_scalar('batch_loss', loss.item(), global_step)

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                tr_loss += loss.item()

                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if nb_tr_steps % gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    train_loss = round(
                        tr_loss * gradient_accumulation_steps /
                        (nb_tr_steps + 1), 4)
                    bar.set_description("[{}] Train loss {}".format(
                        cur_epoch, round(train_loss, 3)))
            if checkpoint_path is not None:
                self._save_checkpoint(checkpoint_path)
        t1 = time.time()
        print("Total training time:", t1 - t0)
