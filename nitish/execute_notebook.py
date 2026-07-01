"""Execute the notebook programmatically using nbclient"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import nbformat
import nbclient
from pathlib import Path

NB_PATH = Path(r"d:\College\3rd_Year\Hackathons\RED ROB\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\dataset_analysis.ipynb")

print(f"Reading notebook: {NB_PATH}")
nb = nbformat.read(NB_PATH, as_version=4)
print(f"Cells to execute: {len(nb.cells)}")

client = nbclient.NotebookClient(
    nb,
    timeout=900,
    kernel_name="python3",
    resources={"metadata": {"path": str(NB_PATH.parent)}}
)

print("Executing notebook (this may take 5-10 minutes)...")
client.execute()

print("Writing executed notebook...")
with open(NB_PATH, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Done! Notebook saved: {NB_PATH}")
print(f"Notebook size: {NB_PATH.stat().st_size / 1e6:.1f} MB")
