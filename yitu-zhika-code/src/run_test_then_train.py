"""
Phase2 Combined Runner: Test then Train
- Step 1: Run test_phase2_pipeline.py (validates full pipeline)
- Step 2: If test passes, start Phase2 multitask training (100 epochs)
Output goes to stdout (redirected to phase2_run.log by launcher).
"""
import os
import sys
import subprocess
from datetime import datetime

# === Environment ===
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# === Paths ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # src/
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                   # project root
os.chdir(PROJECT_ROOT)

PYTHON = sys.executable

def ts():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

print(f"[{ts()}] === Phase2 Combined Runner ===", flush=True)
print(f"[{ts()}] Project root: {PROJECT_ROOT}", flush=True)
print(f"[{ts()}] Python: {PYTHON}", flush=True)

# === Step 1: Pipeline Test ===
print(f"\n[{ts()}] --- Step 1: Pipeline Test ---", flush=True)
test_script = os.path.join('src', 'test_phase2_pipeline.py')

try:
    result = subprocess.run(
        [PYTHON, test_script],
        capture_output=True, text=True, encoding='utf-8', timeout=300
    )
    if result.stdout:
        for line in result.stdout.split('\n'):
            if line.strip():
                print(f"  {line}", flush=True)
    if result.stderr:
        print("  [stderr]:", flush=True)
        for line in result.stderr.split('\n'):
            if line.strip():
                print(f"  {line}", flush=True)

    if result.returncode != 0:
        print(f"\n[{ts()}] Pipeline test FAILED (exit code {result.returncode}). Training aborted.", flush=True)
        sys.exit(1)
    else:
        print(f"\n[{ts()}] Pipeline test PASSED!", flush=True)

except subprocess.TimeoutExpired:
    print(f"\n[{ts()}] Pipeline test timed out (5 min). Training aborted.", flush=True)
    sys.exit(1)
except Exception as e:
    print(f"\n[{ts()}] Error running test: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# === Step 2: Training ===
print(f"\n[{ts()}] --- Step 2: Phase2 Multitask Training ---", flush=True)
print(f"[{ts()}] Config: configs/default.yaml", flush=True)
print(f"[{ts()}] Generator: checkpoints/phase1/final_model.pth", flush=True)
print(f"[{ts()}] Training started.\n", flush=True)

train_cmd = [
    PYTHON,
    os.path.join('src', 'training', 'train_multitask.py'),
    '--config', 'configs/default.yaml',
    '--generator_ckpt', 'checkpoints/phase1/final_model.pth'
]

process = subprocess.Popen(
    train_cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding='utf-8'
)

try:
    for line in process.stdout:
        print(line, end='', flush=True)
    process.wait()
    print(f"\n[{ts()}] Training finished. Exit code: {process.returncode}", flush=True)
except KeyboardInterrupt:
    print(f"\n[{ts()}] Interrupted. Terminating...", flush=True)
    process.terminate()
    process.wait()
    print(f"[{ts()}] Terminated. Exit code: {process.returncode}", flush=True)
