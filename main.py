import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run_steps(step):
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    for step in range(1, 6) if step == 'all' else [int(step)]:
        script = next(ROOT.glob(f'{step:02d}_*/run.py'))
        subprocess.run([sys.executable, str(script)], env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description='Run one step, or the training pipeline (1–5).')
    parser.add_argument('step', choices=['all', '1', '2', '3', '4', '5', '6'])
    args = parser.parse_args()
    run_steps(args.step)


if __name__ == '__main__':
    main()
