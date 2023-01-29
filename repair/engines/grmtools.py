from dataclasses import dataclass
from enum import Enum
import re
import subprocess
from typing import Any, Dict, List

import tqdm

from repair.utils import (
    to_tmp_cfile,
    RepairEngine,
    BenchmarkRunner,
    RepairTaskRecord,
    wrt_root
)

LEX_FILE = wrt_root("resources/mini-c99.l")
YACC_FILE = wrt_root("resources/mini-c99.y")


class EditOp(Enum):
    INSERT = 0
    DELETE = 1
    SHIFT = 2


@dataclass
class GRMTEdit:
    op: EditOp
    token: str


@dataclass
class GRMTEditSeq:
    line: int  # zero-indexed
    col: int  # zero-indexed
    steps: List[GRMTEdit]


def _parse_nimbleparse_logs(log) -> List[GRMTEditSeq]:
    log_lines = log.strip().split("\n")
    acc = []
    in_fix_seq = False
    line, col = None, None

    START_OF_REPAIR = "Parsing error at"

    while len(log_lines) > 0:
        log_line = log_lines.pop(0).strip()

        if log_line.startswith(START_OF_REPAIR):
            in_fix_seq = True
            line, col = re.findall("[0-9]+", log_line)
            # nimblparse reports line/col 1 indexed, change to zero
            line = int(line) - 1
            col = int(col) - 1
        elif in_fix_seq:
            # FIXME: consider other options not just first edit sequence
            # note that there are dependencies across locations, and
            # edit sequences at position i + 1, assume you have applied
            # the first edit sequence at position i
            assert log_line[0] == "1"
            seq = _parse_nimbleparse_edit_line(log_line)
            edit_seq = GRMTEditSeq(line, col, seq)
            acc.append(edit_seq)

            # consume reset of edit sequences (ignoring them)
            while (len(log_lines) > 0
                   and not log_lines[0].strip().startswith(START_OF_REPAIR)):
                log_lines.pop(0)
            in_fix_seq = False
        else:
            continue

    return acc


def _parse_nimbleparse_edit_line(seq_str: str) -> List[GRMTEdit]:
    seq = []
    # remove leading number
    offset = seq_str.find(":") + 1
    seq_str = seq_str[offset:].strip()
    op_map = {
        "Insert": EditOp.INSERT,
        "Delete": EditOp.DELETE,
        "Shift": EditOp.SHIFT
    }

    n = len(seq_str)
    i = 0
    while i < n:
        op = None
        tok = None
        for k, v in op_map.items():
            if seq_str[i:].startswith(k):
                op = v
                # point after the op (i.e. to a space)
                i += len(k)
                break
        # point to char after the space (possible start of tok)
        i += 1
        # consume until next Insert/Delete/Shift
        j = i
        while j < n and not seq_str.startswith(tuple(op_map.keys()), j):
            j += 1
        # j now points to start of next op or end
        if j == n:
            # end of string, so i onwards is the token
            tok = seq_str[i:]
        else:
            # remove 2 to account for space and comma separator preceding op
            tok = seq_str[i:(j - 2)]
        i = j
        assert op is not None
        assert tok is not None
        seq.append(GRMTEdit(op, tok))
    return seq


def _apply_nimbleparse_edit_seq(code: str, edit_seq: GRMTEditSeq) -> str:
    lines = code.split("\n")
    prefix_lines = lines[:edit_seq.line]
    # suffix_lines = lines[edit_seq.line + 1:]
    fix_lines = lines[edit_seq.line:]
    fix_line = "\n".join(fix_lines)
    i = edit_seq.col
    for step in edit_seq.steps:
        i, fix_line = _apply_edit(fix_line, i, step)

    new_lines = list(prefix_lines)
    new_lines.append(fix_line)
    # new_lines.extend(suffix_lines)

    return "\n".join(new_lines)


def _get_token_str(step: GRMTEdit, context: Any) -> str:
    tok = step.token
    # FIXME: use context to suggest a token instead of defaults for lexer tokens
    # this will be useful for INSERT operations where grmtools can't suggest
    # FIXME: one idea is to leave create "mask" values (that will lex as right)
    # token type but that you can then mask out with something like CodeT5
    defaults = {"ID": "_id", "STRING_LITERAL": '"s"', "NUM": "0"}
    return defaults.get(tok, tok.lower())


def _apply_edit(code: str, i: int, step: GRMTEdit, context=None) -> str:
    tok = _get_token_str(step, context=context)
    if step.op == EditOp.SHIFT:
        j = code.find(tok, i)
        assert j >= 0, "Shifting token must exist"
        j += len(tok)
        # no need to change string
        return j, code

    prefix = code[:i]
    rest = code[i:]
    if step.op == EditOp.INSERT:
        # length of token and before/after space
        i = i + len(tok) + 2
        new_code = prefix + " " + tok + " " + rest
        return i, new_code
    elif step.op == EditOp.DELETE:
        # just remove token
        j = rest.find(tok)
        assert j >= 0, "Deletion token must exist"
        rest = rest[:j] + rest[(j + len(tok)):]
        return i, (prefix + rest)
    else:
        raise Exception("Unknown edit type", str(step.op))


def _run_nimbleparse(code) -> str:
    """
    Return repair sequence logs if any (empty if nimbleparse succeeds)
    """
    with to_tmp_cfile(code) as path:
        log_output = ""
        try:
            subprocess.check_output(
                ["nimbleparse", "-q", LEX_FILE, YACC_FILE, path],
                stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as output:
            log_output = output.stdout.decode()
        return log_output


DEBUG = True


class GRMTRepair(RepairEngine, BenchmarkRunner):

    def repair(self, code: str, **kwargs) -> List[Dict[str, Any]]:
        curr_depth = kwargs.get("_curr_depth", 0)
        max_depth = kwargs.get("max_depth", 3)

        try:
            logs = _run_nimbleparse(code)
            edit_seqs = _parse_nimbleparse_logs(logs)
            if len(edit_seqs) > 0:
                # note that because locations can change based on
                # repair, it is easier to run nimbleparse multiple
                # times after making the first seq of edits, rather`
                # than try to keep locations consistent throughout
                code = _apply_nimbleparse_edit_seq(code, edit_seqs[0])

            if len(edit_seqs) > 1:
                kwargs = dict(kwargs)
                kwargs["_curr_depth"] = curr_depth + 1
                kwargs["max_depth"] = max_depth
                # call with newly changed code
                return self.repair(code, **kwargs)

            # FIXME: consider applying a C formatter to make code nicer
            return [{"repair": code}]
        except Exception as err:
            if DEBUG:
                import pdb
                pdb.post_mortem()
            return [{"repair": code}]

    def run_benchmark(self, cases: List[RepairTaskRecord],
                      **kwargs) -> List[List[str]]:
        if kwargs.get("verbose", False):
            cases = tqdm.tqdm(cases)
        return [[r["repair"] for r in self.repair(case.source)]
                for case in cases]
