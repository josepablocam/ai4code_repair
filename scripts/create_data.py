from argparse import ArgumentParser
import os
import random
import pickle
import sqlite3
from multiprocessing import Pool
from typing import Optional, Tuple, List

import tqdm

from repair.utils import (
    RepairTaskRecord,
    gcc_compile,
    token_edit_distance,
    MAX_TOKEN_EDIT_DISTANCE,
    MAX_NUM_GPT_TOKENS,
)
from repair.engines.codex import gpt_tokenize


def filter_to_source_within_gpt_token_max(progs: List[RepairTaskRecord]):
    # also avoid keeping only super small programs
     return [p for p in progs if 20 < len(gpt_tokenize(p.source)) <= MAX_NUM_GPT_TOKENS]


def get_programs_with_errors(conn, max_num=1000):
    cursor = conn.cursor()
    query = f"""
SELECT *
FROM Code 
WHERE errorcount > 0
ORDER BY RANDOM()
LIMIT {max_num}
"""
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    progs = [RepairTaskRecord(*row) for row in results]
    return progs


def create_test_data(conn, max_num=1000, filter_max_gpt_tokens=False):
    progs = get_programs_with_errors(conn, max_num)
    if filter_max_gpt_tokens:
        progs = filter_to_source_within_gpt_token_max(progs)
    return progs

def get_programs_with_no_errors(conn, max_num=10000):
    cursor = conn.cursor()
    query = f"""
SELECT *
FROM Code 
WHERE errorcount = 0
ORDER BY RANDOM()
LIMIT {max_num}
"""
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    progs = [RepairTaskRecord(*row) for row in results]
    return progs


def add_noise(code: str) -> Tuple[str, str]:
    ops = [
        "remove-line",
        "replace-line",
        "remove-char",
        "replace-char",
        "insert-char",
    ]
    chosen_op = random.choice(ops)
    if chosen_op in {"remove-line", "replace-line"}:
        lines = code.split("\n")
        n = len(lines)
        ix1 = random.randint(0, n - 1)
        if chosen_op == "remove-line":
            new_lines = lines[:ix1] + lines[(ix1 + 1):]
        else:
            assert chosen_op == "replace-line"
            ix2 = random.randint(0, n - 1)
            new_lines = list(lines)
            new_lines[ix1] = new_lines[ix2]
        return chosen_op, "\n".join(new_lines)
    else:
        n = len(code)
        ix1 = random.randint(0, n - 1)
        new_code = ""
        if chosen_op == "remove-char":
            new_code = code[:ix1] + code[(ix1 + 1):]
        elif chosen_op == "replace-char":
            ix2 = random.randint(0, n - 1)
            new_chars = list(code)
            new_chars[ix1] = new_chars[ix2]
            new_code = "".join(new_chars)
        else:
            assert chosen_op == "insert-char"
            # just delimiters
            options = ["(", ")", "{", "}", ";", "."]
            new_char = random.choice(options)
            new_code = code[:ix1] + new_char + code[(ix1 + 1):]

        return chosen_op, new_code


def generate_paired_obs(case: RepairTaskRecord) -> Optional[RepairTaskRecord]:
    # introduce some random noise
    _, broken_code = add_noise(case.source)
    comp_result = gcc_compile(broken_code)
    if comp_result.ok:
        # skip if didn't actually break compilation
        return None
    # check if satisfies max edit distance
    dist = token_edit_distance(broken_code, case.source)
    if dist > MAX_TOKEN_EDIT_DISTANCE:
        return None
    return RepairTaskRecord(
        case.code_id,
        case.user_id,
        case.problem_id,
        broken_code,
        comp_result.error,
        comp_result.errorcount,
        case.source,
    )


# See https://bitbucket.org/iiscseal/deepfix/src/master/data_processing/training_data_generator.py
# for a better training generator
def create_training_data(
    conn,
    max_num=1000,
    ncpus=2,
    exclude: List[RepairTaskRecord] = None,
    filter_max_gpt_tokens=False,
):
    no_errors = get_programs_with_no_errors(conn, max_num=max_num)

    # remove any programs that werein our exclude data
    if exclude is not None:
        exclude_set = {(e.problem_id, e.user_id) for e in exclude}
        no_errors = [
            p for p in no_errors
            if (p.problem_id, p.user_id) not in exclude_set
        ]

    if filter_max_gpt_tokens:
        no_errors = filter_to_source_within_gpt_token_max(no_errors)

    if ncpus == 1:
        train_data = [generate_paired_obs(e) for e in tqdm.tqdm(no_errors)]
    else:
        train_data = Pool(ncpus).map(generate_paired_obs, tqdm.tqdm(no_errors))
    train_data = [entry for entry in train_data if entry is not None]
    return train_data


def get_args():
    parser = ArgumentParser(
        description="Create DeepFix-based benchmarks for syntax repair")
    parser.add_argument("--input",
                        type=str,
                        help="DeepFix database",
                        default="data/prutor-deepfix-09-12-2017.db")
    parser.add_argument("--max_test",
                        type=int,
                        help="Max number of test cases",
                        default=100)
    parser.add_argument("--max_train",
                        type=int,
                        help="Max number of train cases",
                        default=10000)
    parser.add_argument("--seed",
                        type=int,
                        help="RNG seed for shuffling",
                        default=42)
    parser.add_argument("--output_folder",
                        type=str,
                        help="Folder for pickled output",
                        default="data/")
    parser.add_argument(
        "--filter_max_gpt_tokens",
        action="store_true",
        help=f"Filter to maximum number of GPT-tokenizer tokens for any observation({MAX_NUM_GPT_TOKENS})",
    )

    parser.add_argument("--ncpus",
                        type=int,
                        help="Number of CPUs for parallel",
                        default=2)
    return parser.parse_args()


def main():
    args = get_args()
    conn = sqlite3.connect(args.input)
    random.seed(args.seed)
    test_data = create_test_data(
        conn,
        max_num=args.max_test,
        filter_max_gpt_tokens=args.filter_max_gpt_tokens,
    )
    train_data = create_training_data(
        conn,
        max_num=args.max_train,
        ncpus=args.ncpus,
        exclude=test_data,
        filter_max_gpt_tokens=args.filter_max_gpt_tokens,
    )

    for (subset_progs, name) in zip([test_data, train_data],
                                    ["test", "train"]):
        file_path = os.path.join(args.output_folder, name + ".pkl")
        print(
            f"Writing {len(subset_progs)} compilation error tasks as pickled file to {file_path}"
        )
        with open(file_path, "wb") as fout:
            pickle.dump(subset_progs, fout)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
