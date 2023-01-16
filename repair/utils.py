from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager
import os
import subprocess
import pickle
from typing import Optional, List, Tuple, Any, Union, Callable, Optional, Dict
from tempfile import NamedTemporaryFile

import editdistance
from pygments.lexers import CLexer
import tree_sitter
import numpy as np
import torch
from tree_sitter import Language, Parser
import pandas as pd
import zss

# Some defaults
# limit on token edit distance between code pairs
MAX_TOKEN_EDIT_DISTANCE = 5
# to avoid super slow queries
MAX_NUM_GPT_TOKENS = 200
# compiled C parser from tree sitter
TREE_SITTER_LIB = "resources/cparser.so"
# tree sitter parser instance
TREE_SITTER_PARSER = None
# folder with all .pkl data files
DATA_FOLDER = "data/"
# some functionality is limited when debugging (just for author dev)
DEBUG = True


@dataclass
class RepairTaskRecord(object):
    code_id: str
    user_id: str
    problem_id: str
    source: str
    error: str
    errorcount: int
    target: Optional[str] = None


@dataclass
class CompileResult(object):
    ok: bool
    error: Optional[str]
    errorcount: int


@contextmanager
def to_tmp_cfile(code):
    file_obj = NamedTemporaryFile(mode="w",
                                  encoding="utf-8",
                                  delete=False,
                                  suffix=".c")
    file_obj.write(code)
    file_obj.close()
    try:
        yield file_obj.name
    finally:
        os.remove(file_obj.name)


# based on https://bitbucket.org/iiscseal/deepfix/src/master/util/helpers.py
def gcc_compile(code: str, timeout_seconds: int = 30) -> CompileResult:
    with to_tmp_cfile(code) as f_name:
        out_file_name = f_name.replace(".c", ".out")
        shell_string = f"gcc -w -std=c99 -pedantic {f_name} -lm -o {out_file_name}"

        error_msg = ""
        try:
            subprocess.check_output(shell_string,
                                    timeout=timeout_seconds,
                                    shell=True,
                                    stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            error_msg = e.output.decode()

        if os.path.exists(out_file_name):
            os.remove(out_file_name)

        if len(error_msg.strip()) == 0:
            return CompileResult(True, None, 0)
        else:
            errorcount = error_msg.count("error:")
            return CompileResult(False, error_msg, errorcount)


# output of pygments tokenizers
PygToken = Tuple[Any, str]


def tokenize(code: str, remove_whitespace=True) -> List[PygToken]:
    tokens = list(CLexer().get_tokens(code))
    return [t for t in tokens if not str(t[0]).endswith(".Whitespace")]


def token_edit_distance(inp1: Union[str, List[PygToken]],
                        inp2: Union[str, List[PygToken]]) -> float:
    if isinstance(inp1, str):
        inp1 = tokenize(inp1)

    if isinstance(inp2, str):
        inp2 = tokenize(inp2)

    # only use token content, not type, since type can
    # be impacted by syntax error itself
    inp1 = [txt for _, txt in inp1]
    inp2 = [txt for _, txt in inp2]
    return float(editdistance.eval(inp1, inp2))


def diff(from_code: str, to_code: str) -> str:
    with to_tmp_cfile(from_code) as from_path, to_tmp_cfile(
            to_code) as to_path:
        diff_str = ""
        try:
            subprocess.check_output(["diff", "-u", from_path, to_path],
                                    stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as output:
            diff_raw_output = output.stdout.decode()
            # remove first two lines (just have file names)
            diff_str = "\n".join(diff_raw_output.split("\n")[2:])
        return diff_str


def _setup_treesitter_language() -> Language:
    if not os.path.exists(TREE_SITTER_LIB):
        Language.build_library(TREE_SITTER_LIB, ["resources/tree-sitter-c"])
    c_language = Language(TREE_SITTER_LIB, "c")
    return c_language


def _set_treesitter_parser() -> None:
    lang = _setup_treesitter_language()
    parser = Parser()
    parser.set_language(lang)
    global TREE_SITTER_PARSER
    TREE_SITTER_PARSER = parser


def _get_treesitter_parser() -> Parser:
    if TREE_SITTER_PARSER is None:
        _set_treesitter_parser()
    return TREE_SITTER_PARSER


def _convert_treesitter_to_zss(t: tree_sitter.Node) -> zss.Node:
    children = t.children
    if len(children) == 0:
        return zss.Node(t.text.decode("utf-8"))
    new_node = zss.Node(t.type)
    for child in children:
        new_child = _convert_treesitter_to_zss(child)
        new_node.addkid(new_child)
    return new_node


def tree_to_list(t: zss.Node):
    return (t.label, [tree_to_list(c) for c in t.children])


def get_parse_tree(code: str) -> zss.Node:
    parser = _get_treesitter_parser()
    tree = parser.parse(bytes(code, "utf-8"))
    return _convert_treesitter_to_zss(tree.root_node)


def tree_edit_distance(inp1: Union[str, zss.Node],
                       inp2: Union[str, zss.Node]) -> float:
    if isinstance(inp1, str):
        inp1 = get_parse_tree(inp1)

    if isinstance(inp2, str):
        inp2 = get_parse_tree(inp2)

    return zss.simple_distance(inp1, inp2)


def basic_check_prediction(predicted: str,
                           buggy: Optional[str] = None,
                           distance_fn: Optional[Callable[[str, str],
                                                          float]] = None,
                           max_distance: Optional[float] = None) -> bool:
    compile_result = gcc_compile(predicted)
    if not compile_result.ok:
        return False
    # check distance if any
    if distance_fn is not None:
        return distance_fn(predicted, buggy) <= max_distance
    else:
        return True


@dataclass
class PredictionAnnotation(object):
    prediction: str
    buggy: str
    compile_result: CompileResult
    distance: float


def run_basic_annotation(predicted: List[List[str]], buggy: List[str]):
    annotations = []
    for preds, buggy in zip(predicted, buggy):
        annot = []
        for p in preds:
            compile_result = gcc_compile(p)
            # FIXME: consider how this might change if we use a different
            # distance function (e.g. `tree_edit_distance`) and/or threshold
            distance = token_edit_distance(p, buggy)
            entry = PredictionAnnotation(p, buggy, compile_result, distance)
            annot.append(entry)
        annotations.append(annot)
    return annotations


def basic_results_table(
        annotated: List[List[PredictionAnnotation]]) -> pd.DataFrame:
    #                          top-1 | top-3 | top-5
    # Compile
    # Compile + Distance
    max_distance = MAX_TOKEN_EDIT_DISTANCE

    def get_first_true_ix(seq):
        try:
            return seq.index(True)
        except ValueError:
            return np.inf

    stats = {}
    stats['compile'] = np.array([
        get_first_true_ix([p.compile_result.ok for p in group])
        for group in annotated
    ])
    stats['compile+distance'] = np.array([
        get_first_true_ix([
            p.compile_result.ok and p.distance <= max_distance for p in group
        ]) for group in annotated
    ])
    recs = []
    for cutoff in [1, 3, 5]:
        for stat, vals in stats.items():
            pct = np.mean(vals < cutoff)
            recs.append((stat, f"top-{cutoff}", pct))
    df = pd.DataFrame(recs, columns=["stat", "cutoff", "pct"])
    pvt = df.pivot_table(index="stat", columns="cutoff",
                         values="pct").reset_index()
    pvt.columns.name = ""
    return pvt


def get_train_data():
    with open(os.path.join(DATA_FOLDER, "train.pkl"), "rb") as fin:
        return pickle.load(fin)


def get_test_data():
    with open(os.path.join(DATA_FOLDER, "test.pkl"), "rb") as fin:
        cases = pickle.load(fin)
        if DEBUG:
            cases = cases[:5]
        return cases


class RepairEngine(object):

    def repair(self, code: str, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError()


class BenchmarkRunner(object):

    def run_benchmark(self, cases: List[RepairTaskRecord],
                      **kwargs) -> List[List[str]]:
        raise NotImplementedError()


def run_benchmark(system: BenchmarkRunner, **kwargs):
    test_data = get_test_data()
    predicted = system.run_benchmark(test_data, **kwargs)
    buggy = [t.source for t in test_data]
    annot = run_basic_annotation(predicted, buggy)
    summary = basic_results_table(annot)
    return summary, annot


def get_torch_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"
