"""
run_pipeline.py — Pipeline runner for the applied project.

Executes each notebook in order using nbconvert, saving the executed
version with outputs back to the same file. Run from the project root:

    python run_pipeline.py

To start from a specific notebook (0-based index):

    python run_pipeline.py --from 3

To run a single notebook only:

    python run_pipeline.py --only 5
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

NOTEBOOKS = [
    "notebooks/1_data_cleaning.ipynb",
    "notebooks/2_cointegration_test.ipynb",
    "notebooks/3_pairs_strategy.ipynb",
    "notebooks/4_portfolio_construction.ipynb",
    "notebooks/5_trading_signal.ipynb",
    "notebooks/6_backtests.ipynb",        
    "notebooks/7_combined_results.ipynb",
]

def run_notebook(path: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  Running: {path}")
    print(f"{'='*60}")
    start = time.time()

    result = subprocess.run(
        [
            sys.executable, "-m", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=3600",
            "--ExecutePreprocessor.kernel_name=python3",
            path,
        ],
        capture_output=True,
        text=True,
    )

    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"  Completed in {elapsed:.1f}s")
        return True
    else:
        print(f"  Failed after {elapsed:.1f}s")
        print(result.stderr[-2000:] if result.stderr else "(no stderr)")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run the full notebook pipeline.")
    parser.add_argument(
        "--only", type=int, default=None,
        help="Run only the notebook at this 0-based index.",
    )
    parser.add_argument(
        "--from", dest="from_idx", type=int, default=0,
        help="Start from this 0-based notebook index.",
    )
    args = parser.parse_args()

    if args.only is not None:
        notebooks = [NOTEBOOKS[args.only]]
    else:
        notebooks = NOTEBOOKS[args.from_idx:]

    root = Path(__file__).parent

    for nb in notebooks:
        success = run_notebook(str(root / nb))
        if not success:
            print(f"\nPipeline stopped at: {nb}")
            idx = NOTEBOOKS.index(nb)
            print(f"Re-run from here with:  python run_pipeline.py --from {idx}")
            sys.exit(1)

    print(f"\nAll {len(notebooks)} notebook(s) completed successfully.")


if __name__ == "__main__":
    main()
