from argparse import ArgumentParser
import pickle
import sqlite3

from repair.utils import DeepFixRecord

def get_programs_with_errors(conn):
    cursor = conn.cursor()
    query = """
SELECT *
FROM Code 
WHERE errorcount > 0
"""
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    benchmark = [DeepFixRecord(*row) for row in results]
    return benchmark


def get_args():
    parser = ArgumentParser(description="Create DeepFix-based benchmarks for syntax repair")
    parser.add_argument("--input", type=str, help="DeepFix database", default="data/prutor-deepfix-09-12-2017.db")
    parser.add_argument("--output", type=str, help="Pickle output", default="data/syntax-benchmark.pkl")
    return parser.parse_args()


def main():
    args = get_args()
    conn = sqlite3.connect(args.input)
    benchmark = get_programs_with_errors(conn)
    print(f"Writing {len(benchmark)} compilation error tasks as pickled file to {args.output}")
    with open(args.output, "wb") as fout:
        pickle.dump(benchmark, fout)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()



