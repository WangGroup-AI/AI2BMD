#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ase.units import kcal, mol


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
EV_TO_KCAL_PER_MOL = mol / kcal

MD22_RUNS = (
    ("Ac-Ala3-NHMe", "log_md22_Ac_Ala3_NHMe"),
    ("DHA", "log_md22_DHA"),
    ("Stachyose", "log_md22_Stachyose"),
    ("AT-AT", "log_md22_AT_AT"),
    ("AT-AT-CG-CG", "log_md22_AT_AT_CG_CG"),
    ("Buckyball catcher", "log_md22_Buckyball_Catcher"),
    ("Double-walled nanotube", "log_md22_Double_Walled_Nanotube"),
)

TRP_CAGE_RUNS = (
    ("20% finetuned with pretraining", "pima_ft_20pct"),
    ("40% finetuned with pretraining", "pima_ft_40pct"),
    ("60% finetuned with pretraining", "pima_ft_60pct"),
    ("80% finetuned with pretraining", "pima_ft_80pct"),
    ("100% finetuned with pretraining", "pima_ft_100pct"),
    ("100% finetuned without pretraining", "pima_ft_ablation"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Display ViSNet-PIMA reproduction results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Results directory (default: repository results directory).",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help=(
            "Directory for the rendered result tables as CSV files "
            "(default: <results-dir>/show_results_tables)."
        ),
    )
    return parser.parse_args()


def load_predictions(path, unit_scale=1.0):
    if not path.is_file():
        raise FileNotFoundError(f"Missing prediction file: {path}")

    predictions = torch.load(path, map_location="cpu", weights_only=False)
    required = ("y_pred", "y_true", "dy_pred", "dy_true")
    missing = [key for key in required if key not in predictions]
    if missing:
        raise KeyError(f"{path} is missing keys: {', '.join(missing)}")

    if predictions["y_pred"].shape != predictions["y_true"].shape:
        raise ValueError(f"Energy tensor shape mismatch in {path}")
    if predictions["dy_pred"].shape != predictions["dy_true"].shape:
        raise ValueError(f"Force tensor shape mismatch in {path}")
    if predictions["dy_true"].ndim != 2 or predictions["dy_true"].shape[1] != 3:
        raise ValueError(f"Expected force tensors with shape [N, 3] in {path}")

    energy_error = np.abs(
        predictions["y_true"].numpy() - predictions["y_pred"].numpy()
    )
    force_error = np.abs(
        predictions["dy_true"].numpy() - predictions["dy_pred"].numpy()
    )

    return {
        "energy_mae": energy_error.mean() * unit_scale,
        "force_mae": force_error.mean() * unit_scale,
    }


def load_ncia(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing NCIA metrics file: {path}")

    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "Subset": row["subset"],
                    "Evaluated": int(row["evaluated"]),
                    "Skipped": int(row["skipped"]),
                    "MAE (kcal/mol)": float(row["mae"]),
                    "RMSE (kcal/mol)": float(row["rmse"]),
                }
            )
    if not rows:
        raise ValueError(f"No NCIA metrics found in {path}")
    return rows


def collect_results(results_dir):
    md22_rows = []
    for label, directory in MD22_RUNS:
        metrics = load_predictions(results_dir / directory / "inference_results.pt")
        md22_rows.append(
            {
                "Dataset": label,
                "Energy MAE (kcal/mol)": metrics["energy_mae"],
                "Force MAE (kcal/mol/Å)": metrics["force_mae"],
            }
        )

    chig = load_predictions(
        results_dir / "log_Chig" / "inference_results.pt",
        unit_scale=EV_TO_KCAL_PER_MOL,
    )
    chig_rows = [
        {
            "Dataset": "Chignolin",
            "Energy MAE (kcal/mol)": chig["energy_mae"],
            "Force MAE (kcal/mol/Å)": chig["force_mae"],
        }
    ]

    trp_rows = []
    for label, directory in TRP_CAGE_RUNS:
        metrics = load_predictions(
            results_dir
            / "log_learning_curve_and_ablation_studys"
            / directory
            / "inference_results.pt",
            unit_scale=1000.0,
        )
        trp_rows.append(
            {
                "Model": label,
                "Scalar MAE (meV)": metrics["energy_mae"],
                "Force MAE (meV/Å)": metrics["force_mae"],
            }
        )

    ncia_rows = load_ncia(results_dir / "log_NCIA" / "ncia_binding_visnet-pima.csv")
    return md22_rows, chig_rows, trp_rows, ncia_rows


def get_result_tables(results_dir):
    """Return the result tables as numeric pandas DataFrames.

    This function is the importable interface for Jupyter and JupyterBook.
    The calculation and unit conversion are still performed by collect_results.
    """
    results_dir = Path(results_dir).expanduser().resolve()
    md22_rows, chig_rows, trp_rows, ncia_rows = collect_results(results_dir)
    return {
        "md22": pd.DataFrame(md22_rows),
        "chignolin": pd.DataFrame(chig_rows),
        "trp_cage": pd.DataFrame(trp_rows),
        "ncia": pd.DataFrame(ncia_rows),
    }


def save_result_tables(tables, csv_dir):
    """Save each displayed table as a separate CSV file."""
    csv_dir.mkdir(parents=True, exist_ok=True)
    filenames = {
        "md22": "md22_results.csv",
        "chignolin": "chignolin_results.csv",
        "trp_cage": "trp_cage_results.csv",
        "ncia": "ncia_results.csv",
    }
    float_formats = {
        "md22": "%.4f",
        "chignolin": "%.4f",
        "trp_cage": "%.2f",
        "ncia": "%.2f",
    }
    paths = {}
    for name, filename in filenames.items():
        path = csv_dir / filename
        tables[name].to_csv(
            path,
            index=False,
            float_format=float_formats[name],
        )
        paths[name] = path
    return paths


def render_results(results_dir, tables):
    """Render all result DataFrames with header and bottom borders.

    The same function is used by the command-line entry point and notebooks,
    so both environments receive the same bordered representation.
    """
    def render(frame, decimals=4):
        lines = frame.to_markdown(
            index=False,
            tablefmt="grid",
            floatfmt=f".{decimals}f",
            colalign=("left", *("center",) * (len(frame.columns) - 1)),
        ).splitlines()
        if len(lines) <= 3:
            return "\n".join(lines)

        # Keep the top border, header separator, data rows, and bottom border;
        # remove only the separators between individual data rows.
        bordered_lines = lines[:3]
        bordered_lines.extend(line for line in lines[3:-1] if line.startswith("|"))
        bordered_lines.append(lines[-1])
        return "\n".join(bordered_lines)

    sections = [
        "ViSNet-PIMA reproduction results",
        f"Results directory: {results_dir}",
        "",
        "MD22 in Table 1",
        render(tables["md22"]),
        "",
        "AIMD-Chig in Fig. 2a and Fig. 2b",
        render(tables["chignolin"]),
        "",
        "Trp-cage learning curve and ablation study in Table S6 and Fig. S8",
        render(tables["trp_cage"], decimals=2),
        "",
        "NCI Atlas interaction energies in Extended Data Table 4",
        render(tables["ncia"], decimals=2),
    ]
    return "\n".join(sections)


def main():
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    csv_dir = (
        args.csv_dir.expanduser().resolve()
        if args.csv_dir is not None
        else results_dir / "show_results_tables"
    )

    tables = get_result_tables(results_dir)
    csv_paths = save_result_tables(tables, csv_dir)
    print(render_results(results_dir, tables))

if __name__ == "__main__":
    main()
