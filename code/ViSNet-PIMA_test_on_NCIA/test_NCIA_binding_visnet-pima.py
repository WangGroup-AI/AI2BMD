import os
import re
import sys
import csv
import importlib.metadata
from pathlib import Path

import numpy as np
from ase.io import read as ase_read
from ase.units import kcal, mol


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "2.0"

import jax
jax.config.update("jax_default_matmul_precision", "float32")

# Use source mlip
_orig = importlib.metadata.version


def _patch(name):
    if name == "mlip":
        return "dev"
    return _orig(name)


importlib.metadata.version = _patch
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR / "mlip"))

from mlip.inference.batched_inference import run_batched_inference
from mlip.models import Visnet
from mlip.models.model_io import load_model_from_zip


EXTXYZ_DIR = REPO_ROOT / "data" / "NCI-Atlas"
MODEL_PATH = REPO_ROOT / "data" / "ViSNet-PIMA-jax-checkpoint" / "4+1_c128.zip"
OUTPUT_CSV = REPO_ROOT / "results" / "log_NCIA" / "ncia_binding_visnet-pima.csv"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
BATCH_SIZE = 32
CUTOFF = 10.0
EV_TO_KCAL = mol / kcal
ALLOWED_SYMBOLS = {
    "H", "Li", "B", "C", "N", "O", "F", "Na", "Mg",
    "Si", "P", "S", "Cl", "K", "Ca", "Br", "I",
}

PAIR_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)=("[^"]*"|\'[^\']*\'|\S+)')


def parse_comment(comment):
    data = {}
    for key, value in PAIR_RE.findall(comment):
        data[key] = value.strip("\"'")
    return data


def read_comments(path):
    comments = []
    with open(path, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            n_atoms = int(line)
            comments.append(f.readline().strip())
            for _ in range(n_atoms):
                f.readline()
    return comments


def parse_indices(value):
    indices = []
    for part in re.split(r"[ ,;]+", str(value).strip()):
        if "-" in part:
            start, end = part.split("-", 1)
            indices.extend(range(int(start) - 1, int(end)))
        elif part:
            indices.append(int(part) - 1)
    return indices


def predict_energies(frames, force_field):
    predictions = run_batched_inference(
        frames,
        force_field,
        batch_size=BATCH_SIZE,
        cutoff_distance=CUTOFF,
    )
    return np.array([float(pred.energy) for pred in predictions])


def set_nonperiodic_box(frames):
    for frame in frames:
        frame.set_cell([100.0, 100.0, 100.0])
        frame.set_pbc(False)


def allowed_selection(frame, indices):
    return all(frame[i].symbol in ALLOWED_SYMBOLS for i in indices)


def evaluate_file(path, force_field):
    frames = ase_read(str(path), index=":", format="extxyz")
    comments = read_comments(path)

    metadata = [{**frame.info, **parse_comment(comment)} for frame, comment in zip(frames, comments)]
    selections_a = [parse_indices(data["selection_a"]) for data in metadata]
    selections_b = [parse_indices(data["selection_b"]) for data in metadata]
    keep = [
        allowed_selection(frame, sel_a) and allowed_selection(frame, sel_b)
        for frame, sel_a, sel_b in zip(frames, selections_a, selections_b)
    ]
    frames = [frame for frame, ok in zip(frames, keep) if ok]
    metadata = [data for data, ok in zip(metadata, keep) if ok]
    selections_a = [sel for sel, ok in zip(selections_a, keep) if ok]
    selections_b = [sel for sel, ok in zip(selections_b, keep) if ok]
    refs = np.array([float(data["benchmark_Eint"]) for data in metadata])

    if not frames:
        return np.array([])

    frames_a = [frame[sel] for frame, sel in zip(frames, selections_a)]
    frames_b = [frame[sel] for frame, sel in zip(frames, selections_b)]

    set_nonperiodic_box(frames)
    set_nonperiodic_box(frames_a)
    set_nonperiodic_box(frames_b)

    e_total = predict_energies(frames, force_field)
    e_a = predict_energies(frames_a, force_field)
    e_b = predict_energies(frames_b, force_field)
    pred_binding = (e_total - e_a - e_b) * EV_TO_KCAL
    errors = pred_binding - refs

    return errors


def subset_name(path):
    name = path.stem
    return name[5:] if name.startswith("NCIA_") else name


def metric_row(name, errors):
    if len(errors) == 0:
        return {
            "subset": name,
            "mae": "",
            "rmse": "",
        }

    return {
        "subset": name,
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
    }


def format_metric(value):
    return "" if value == "" else f"{value:.6f}"


def main():
    print("Loading model and parameters")
    force_field = load_model_from_zip(Visnet, MODEL_PATH)

    all_errors = []
    rows = []
    for path in sorted(EXTXYZ_DIR.glob("*.xyz")):
        print(f"Processing {path.name}")
        errors = evaluate_file(path, force_field)
        all_errors.extend(errors)
        row = metric_row(subset_name(path), errors)
        rows.append(row)

        if len(errors):
            print(
                f"MAE={row['mae']:.6f}, RMSE={row['rmse']:.6f} kcal/mol"
            )
        else:
            print("  No frames to evaluate.")

    all_errors = np.array(all_errors)
    if len(all_errors):
        total_row = metric_row("weighted_total", all_errors)
        rows.append(total_row)
        print(
            f"Overall: MAE={total_row['mae']:.6f}, RMSE={total_row['rmse']:.6f} kcal/mol"
        )
    else:
        print("No frames to evaluate.")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["subset", "mae", "rmse"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                **row,
                "mae": format_metric(row["mae"]),
                "rmse": format_metric(row["rmse"]),
            }
            for row in rows
        )
    print(f"Saved metrics to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
