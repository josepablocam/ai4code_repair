from dataclasses import dataclass
import os
import subprocess
from typing import Optional, List, Tuple, Any, Union, Callable, Optional
from tempfile import NamedTemporaryFile

import editdistance
from pygments.lexers import CLexer
import tree_sitter
import numpy as np
from tree_sitter import Language, Parser
import pandas as pd
import zss


@dataclass
class DeepFixRecord(object):
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


# based on https://bitbucket.org/iiscseal/deepfix/src/master/util/helpers.py
def gcc_compile(code: str, timeout_seconds: int = 30) -> CompileResult:
    f = NamedTemporaryFile(mode="w",
                           encoding="utf-8",
                           suffix=".c",
                           delete=False)
    f.write(code)
    f.flush()

    out_file_name = f.name.replace(".c", ".out")

    shell_string = f"gcc -w -std=c99 -pedantic {f.name} -lm -o {out_file_name}"

    error_msg = ""
    try:
        subprocess.check_output(shell_string,
                                timeout=timeout_seconds,
                                shell=True,
                                stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        error_msg = e.output.decode()

    if len(error_msg.strip()) == 0:
        return CompileResult(True, None, 0)
    else:
        errorcount = error_msg.count("error:")
        return CompileResult(False, error_msg, errorcount)


# output of pygments tokenizers
PygToken = Tuple[Any, str]


def tokenize(code: str) -> List[PygToken]:
    return list(CLexer().get_tokens(code))


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


TREE_SITTER_LIB = "resources/cparser.so"
TREE_SITTER_PARSER = None


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



def tree_edit_distance(inp1: Union[str, zss.Node], inp2: Union[str, zss.Node]) -> float:
    if isinstance(inp1, str):
        inp1 = get_parse_tree(inp1)

    if isinstance(inp2, str):
        inp2 = get_parse_tree(inp2)

    return zss.simple_distance(inp1, inp2)



def basic_check_prediction(predicted: str, buggy: Optional[str]=None, distance_fn: Optional[Callable[[str, str], float]]=None, max_distance: Optional[float]=None) -> bool:
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
    compile_result: CompileResult
    distance: float


def run_basic_annotation(predicted: List[List[str]], buggy: List[str]):
    annotations = []
    for preds, buggy in zip(predicted, buggy):
        annot = []
        for p in preds:
            compile_result = gcc_compile(p)
            distance = token_edit_distance(p, buggy)
            entry = PredictionAnnotation(p, compile_result, distance)
            annot.append(entry)
        annotations.append(annot)
    return annotations


def basic_results_table(annotated: List[List[PredictionAnnotation]]) -> pd.DataFrame:
    #                          top-1 | top-3 | top-5
    # Compile
    # Compile + Distance
    max_distance = 5
    def get_first_true_ix(seq):
        try:
            return seq.index(True)
        except ValueError:
            return np.inf
    stats = {}
    stats['compile'] = np.array([get_first_true_ix([p.compile_result.ok for p in group]) for group in annotated])
    stats['compile+distance'] = np.array([get_first_true_ix([p.compile_result.ok  and p.distance <= max_distance for p in group]) for group in annotated])
    recs = []
    for cutoff in [1, 3, 5]:
        for stat, vals in stats.items():
            pct = np.mean(vals < cutoff)
            recs.append((stat, f"top-{cutoff}", pct))
    df = pd.DataFrame(recs, columns=["stat", "cutoff", "pct"])
    return df.pivot_table(keys="stat", columns="cutoff", values="pct").reset_index()