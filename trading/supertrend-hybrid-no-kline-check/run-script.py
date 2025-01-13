import os, platform
import subprocess
import time
from datetime import datetime, timedelta

def run_task():
    run_task_start = datetime.now()

    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    system = platform.system()

    # Use os.path.join to construct the virtual environment python interpreter path and script path
    if system == 'Windows':
        venv_python = os.path.join(root_path, '.venv', 'Scripts', 'python')
    else:
        venv_python = os.path.join(root_path, '.venv', 'bin', 'python')

    script_path = os.path.join(root_path, 'trading', 'supertrend-hybrid-no-kline-check', 'interval-predict.py')

    subprocess.run([venv_python, script_path])

    run_task_end = datetime.now()
    print(f'Running task took {run_task_end - run_task_start}')

while True:
    now = datetime.now()

    minutes_to_add = 15 - (now.minute % 15)
    next_run = now + timedelta(minutes=minutes_to_add)
    next_run = next_run.replace(second=0, microsecond=0)

    # If time has just passed the next run mark (e.g. 15:00:04)
    if next_run <= now:
        next_run += timedelta(minutes=15)

    wait_seconds = (next_run - now).total_seconds()

    print(f'Current time: {now}. Next run at: {next_run}. Waiting {wait_seconds} seconds.')

    time.sleep(wait_seconds)

    print(f'Running task at {next_run}...')
    run_task()
