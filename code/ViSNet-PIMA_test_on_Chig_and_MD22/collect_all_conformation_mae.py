#!/usr/bin/env python3
"""Collect per-conformation energy and Cartesian force MAEs for all splits.

The inference and unit-conversion conventions follow ``train.py``:

* ``train.py`` defaults are applied first, followed by the YAML file and then
  checkpoint hyperparameters;
* the same ``DataModule`` prepares the train/validation/test split;
* forces are obtained from the model's energy derivative;
* MD22 values retain their native units, while non-MD22 values are converted
  with the same ``kcal/mol`` factor used by ``train.py``.

For conformation i with N_i atoms, the four reported errors are

    delta_E_i  = |E_pred_i - E_true_i|
    delta_Fi,a = (1 / N_i) sum_j |F_pred_i,j,a - F_true_i,j,a|

for Cartesian directions a in {x, y, z}. Dataset indices are zero-based.
Each dataset is saved both as a human-readable CSV file and as a structured
NumPy ``.npy`` file that supports memory-mapped loading.
"""

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from ase.units import kcal, mol
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from visnet.data import DataModule
from visnet.models.model import create_model
from visnet.utils import save_argparse


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "data" / "checkpoint"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "all_conformation_mae_old"
EV_TO_KCAL_PER_MOL = mol / kcal
SPLIT_NAMES = ("train", "val", "test")
DETAIL_HEADER = (
    "split",
    "conformation_index_within_split",
    "dataset_index",
    "num_atoms",
    "delta_E_kcal_per_mol",
    "delta_Fx_kcal_per_mol_per_angstrom",
    "delta_Fy_kcal_per_mol_per_angstrom",
    "delta_Fz_kcal_per_mol_per_angstrom",
)
DETAIL_DTYPE = np.dtype(
    [
        ("split", "U5"),
        ("conformation_index_within_split", np.int64),
        ("dataset_index", np.int64),
        ("num_atoms", np.int32),
        ("delta_E_kcal_per_mol", np.float64),
        ("delta_Fx_kcal_per_mol_per_angstrom", np.float64),
        ("delta_Fy_kcal_per_mol_per_angstrom", np.float64),
        ("delta_Fz_kcal_per_mol_per_angstrom", np.float64),
    ]
)
# These two options are absent from the released checkpoints/configurations,
# so train.py supplies its argparse defaults when reconstructing the model.
TRAIN_MODEL_DEFAULTS = {
    "PIMA_Block": True,
    "PIMA_charge_charge": False,
}


@dataclass(frozen=True)
class RunSpec:
    label: str
    dataset: str
    dataset_arg: str | None
    config_name: str
    checkpoint_name: str
    dataset_root: str
    result_log_dir: str

    @property
    def output_name(self):
        if self.dataset == "MD22":
            return f"MD22_{self.label}"
        return self.label


RUN_SPECS = (
    RunSpec(
        "Ac-Ala3-NHMe",
        "MD22",
        "Ac_Ala3_NHMe",
        "ViSNet-MD22-Ac_Ala3_NHMe.yml",
        "Ac-Ala3-NHMe.ckpt",
        "md22-dataset",
        "log_md22_ViSNet-MD22-Ac_Ala3_NHMe",
    ),
    RunSpec(
        "DHA",
        "MD22",
        "DHA",
        "ViSNet-MD22-DHA.yml",
        "DHA.ckpt",
        "md22-dataset",
        "log_md22_DHA",
    ),
    RunSpec(
        "Stachyose",
        "MD22",
        "stachyose",
        "ViSNet-MD22-stachyose.yml",
        "stachyose.ckpt",
        "md22-dataset",
        "log_md22_Stachyose",
    ),
    RunSpec(
        "AT-AT",
        "MD22",
        "AT_AT",
        "ViSNet-MD22-AT_AT.yml",
        "AT-AT.ckpt",
        "md22-dataset",
        "log_md22_AT_AT",
    ),
    RunSpec(
        "AT-AT-CG-CG",
        "MD22",
        "AT_AT_CG_CG",
        "ViSNet-MD22-AT_AT_CG_CG.yml",
        "AT-AT-CG-CG.ckpt",
        "md22-dataset",
        "log_md22_AT_AT_CG_CG",
    ),
    RunSpec(
        "Buckyball-catcher",
        "MD22",
        "buckyball_catcher",
        "ViSNet-MD22-buckyball_catcher.yml",
        "buckyball-catcher.ckpt",
        "md22-dataset",
        "log_md22_Buckyball_Catcher",
    ),
    RunSpec(
        "Double-walled-nanotube",
        "MD22",
        "double_walled_nanotube",
        "ViSNet-MD22-double_walled_nanotube.yml",
        "double-walled-nanotube_2.ckpt",
        "md22-dataset",
        "log_md22_Double_Walled_Nanotube",
    ),
    RunSpec(
        "Chignolin",
        "Chignolin",
        None,
        "ViSNet-Chignolin.yml",
        "Chignolin.ckpt",
        "AIMD-Chig",
        "log_Chig",
    ),
)


def parse_args():
    labels = [spec.label for spec in RUN_SPECS]
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every train/validation/test conformation and save "
            "per-conformation delta_E, delta_Fx, delta_Fy, and delta_Fz."
        )
    )
    parser.add_argument(
        "--datasets",
        choices=("all", "md22", "chignolin", *labels),
        default="all",
        help="Datasets to evaluate (default: all MD22 subsets and Chignolin).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing the original inference log directories.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing the checkpoint files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: results/all_conformation_mae).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device (default: CUDA when available).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the checkpoint seed when regenerating splits.",
    )
    parser.add_argument(
        "--regenerate-splits",
        action="store_true",
        help=(
            "Regenerate splits through DataModule instead of loading each "
            "original results/<log_dir>/splits.npz file."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the checkpoint inference batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override the checkpoint DataLoader worker count.",
    )
    return parser.parse_args()


def select_specs(selection):
    if selection == "all":
        return RUN_SPECS
    if selection == "md22":
        return tuple(spec for spec in RUN_SPECS if spec.dataset == "MD22")
    if selection == "chignolin":
        return tuple(spec for spec in RUN_SPECS if spec.dataset == "Chignolin")
    return tuple(spec for spec in RUN_SPECS if spec.label == selection)


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(requested)


def load_run(spec, cli_args, run_dir):
    config_path = SCRIPT_DIR / "examples_MD22_AIMD-Chig" / spec.config_name
    checkpoint_path = cli_args.checkpoint_dir / spec.checkpoint_name
    dataset_root = REPO_ROOT / "data" / spec.dataset_root
    original_splits = cli_args.results_dir / spec.result_log_dir / "splits.npz"

    for path in (config_path, checkpoint_path, dataset_root):
        if not path.exists():
            raise FileNotFoundError(f"Required input does not exist: {path}")

    with config_path.open(encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_hparams = checkpoint.get("hyper_parameters")
    if not isinstance(checkpoint_hparams, dict):
        raise ValueError(f"Checkpoint has no hyper_parameters dictionary: {checkpoint_path}")

    # Match train.py command precedence: --conf, then --load-model, then
    # explicit command-line overrides.
    effective = dict(TRAIN_MODEL_DEFAULTS)
    effective.update(config)
    effective.update(checkpoint_hparams)
    effective.update(
        load_model=str(checkpoint_path),
        dataset_root=str(dataset_root),
        log_dir=str(run_dir),
        task="inference",
    )

    if effective.get("dataset") != spec.dataset:
        raise ValueError(
            f"Dataset mismatch for {spec.label}: expected {spec.dataset}, "
            f"checkpoint specifies {effective.get('dataset')}."
        )
    if effective.get("dataset_arg") != spec.dataset_arg:
        raise ValueError(
            f"Dataset argument mismatch for {spec.label}: expected "
            f"{spec.dataset_arg!r}, checkpoint specifies "
            f"{effective.get('dataset_arg')!r}."
        )
    if not effective.get("derivative"):
        raise ValueError(f"Force derivatives are disabled for {spec.label}.")

    if cli_args.seed is not None:
        effective["seed"] = cli_args.seed
    if cli_args.batch_size is not None:
        if cli_args.batch_size <= 0:
            raise ValueError("--batch-size must be positive.")
        effective["inference_batch_size"] = cli_args.batch_size
    if cli_args.num_workers is not None:
        if cli_args.num_workers < 0:
            raise ValueError("--num-workers cannot be negative.")
        effective["num_workers"] = cli_args.num_workers

    if cli_args.regenerate_splits:
        effective["splits"] = None
    else:
        if not original_splits.is_file():
            raise FileNotFoundError(
                f"Original split file is missing: {original_splits}. "
                "Use --regenerate-splits to generate a new split."
            )
        effective["splits"] = str(original_splits)

    effective["conf"] = str(config_path)
    namespace = argparse.Namespace(**effective)
    save_argparse(namespace, str(run_dir / "input.yaml"), exclude=["conf"])
    return namespace, checkpoint


def validate_complete_partition(data_module):
    split_indices = {
        "train": np.asarray(data_module.idx_train, dtype=np.int64),
        "val": np.asarray(data_module.idx_val, dtype=np.int64),
        "test": np.asarray(data_module.idx_test, dtype=np.int64),
    }
    combined = np.concatenate(tuple(split_indices.values()))
    unique = np.unique(combined)
    expected = np.arange(len(data_module.dataset), dtype=np.int64)

    if combined.size != unique.size:
        raise ValueError("Train, validation, and test splits contain overlapping indices.")
    if not np.array_equal(np.sort(unique), expected):
        missing = np.setdiff1d(expected, unique)
        extra = np.setdiff1d(unique, expected)
        raise ValueError(
            "Splits do not cover the complete dataset: "
            f"missing={missing.size}, extra={extra.size}."
        )
    return split_indices


def load_model_from_checkpoint(args, checkpoint, device):
    # This is the same construction/state-loading procedure as
    # visnet.models.model.load_model, while reusing the already loaded
    # checkpoint to avoid reading every large checkpoint twice.
    model = create_model(vars(args))
    state_dict = {
        re.sub(r"^model\.", "", key): value
        for key, value in checkpoint["state_dict"].items()
    }
    incompatible = model.load_state_dict(state_dict, strict=False)
    # The released checkpoints predate the currently declared ``charge`` MLP.
    # It is not executed when PIMA_charge_charge=False, which is the train.py
    # default used by these runs.  Permit only those four inactive parameters;
    # any other mismatch would make the reproduced inference unreliable.
    allowed_missing = set()
    if args.PIMA_Block and not args.PIMA_charge_charge:
        allowed_missing = {
            "representation_model.charge.0.weight",
            "representation_model.charge.0.bias",
            "representation_model.charge.2.weight",
            "representation_model.charge.2.bias",
        }
    unapproved_missing = set(incompatible.missing_keys) - allowed_missing
    if unapproved_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model state mismatch: "
            f"missing={sorted(unapproved_missing)}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return model.to(device).eval()


def format_number(value):
    return format(float(value), ".17g")


def round_delta(value):
    return float(format(float(value), ".12f"))


def format_delta(value):
    return format(float(value), ".12f")


def evaluate_split(
    model,
    data_module,
    split_name,
    dataset_indices,
    args,
    device,
    writer,
    records,
    record_offset,
):
    subset = getattr(data_module, f"{split_name}_dataset")
    loader = DataLoader(
        dataset=subset,
        batch_size=args.inference_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    unit_scale = 1.0 if args.dataset == "MD22" else EV_TO_KCAL_PER_MOL
    cursor = 0
    energy_error_sum = 0.0
    force_error_sum = np.zeros(3, dtype=np.float64)
    atom_count = 0

    progress = tqdm(loader, desc=f"{args.dataset}:{args.dataset_arg or ''} {split_name}")
    for batch in progress:
        batch = batch.to(device)
        with torch.set_grad_enabled(args.derivative):
            pred, deriv = model(batch)

        if deriv is None:
            raise RuntimeError("The model did not return force derivatives.")

        energy_abs = (pred.reshape(-1) - batch.y.reshape(-1)).abs()
        force_abs = (deriv - batch.dy).abs()
        ptr = batch.ptr.detach().cpu().numpy()
        num_graphs = int(batch.num_graphs)
        if energy_abs.numel() != num_graphs:
            raise ValueError(
                f"Expected one energy per conformation, received "
                f"{energy_abs.numel()} energies for {num_graphs} conformations."
            )

        energy_cpu = energy_abs.detach().cpu().numpy() * unit_scale
        force_cpu = force_abs.detach().cpu().numpy() * unit_scale
        batch_indices = dataset_indices[cursor : cursor + num_graphs]

        for local_index, dataset_index in enumerate(batch_indices):
            atom_start = int(ptr[local_index])
            atom_end = int(ptr[local_index + 1])
            num_atoms = atom_end - atom_start
            if num_atoms <= 0:
                raise ValueError(f"Conformation {dataset_index} has no atoms.")
            directional_mae = force_cpu[atom_start:atom_end].mean(axis=0)
            delta_values = (
                round_delta(energy_cpu[local_index]),
                round_delta(directional_mae[0]),
                round_delta(directional_mae[1]),
                round_delta(directional_mae[2]),
            )
            record = (
                split_name,
                cursor + local_index,
                int(dataset_index),
                num_atoms,
                *delta_values,
            )
            records[record_offset + cursor + local_index] = record
            formatted_errors = tuple(format_delta(value) for value in delta_values)
            writer.writerow(record[:4] + formatted_errors)

        energy_error_sum += float(energy_cpu.sum(dtype=np.float64))
        force_error_sum += force_cpu.sum(axis=0, dtype=np.float64)
        atom_count += force_cpu.shape[0]
        cursor += num_graphs

        del pred, deriv, energy_abs, force_abs, batch

    if cursor != len(dataset_indices):
        raise ValueError(
            f"Processed {cursor} {split_name} conformations, expected "
            f"{len(dataset_indices)}."
        )

    return {
        "split": split_name,
        "conformations": cursor,
        "atoms": atom_count,
        "energy_mae": energy_error_sum / cursor,
        "force_x_mae": force_error_sum[0] / atom_count,
        "force_y_mae": force_error_sum[1] / atom_count,
        "force_z_mae": force_error_sum[2] / atom_count,
        "force_mae": force_error_sum.sum() / (atom_count * 3),
    }


def evaluate_run(spec, cli_args, device):
    run_dir = cli_args.output_dir / spec.output_name
    run_dir.mkdir(parents=True, exist_ok=True)
    args, checkpoint = load_run(spec, cli_args, run_dir)

    pl.seed_everything(args.seed, workers=True)
    data_module = DataModule(args)
    data_module.prepare_dataset()
    split_indices = validate_complete_partition(data_module)
    model = load_model_from_checkpoint(args, checkpoint, device)

    csv_path = run_dir / f"{spec.output_name}_all_splits_mae.csv"
    npy_path = run_dir / f"{spec.output_name}_all_splits_mae.npy"
    temporary_csv_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    temporary_npy_path = npy_path.with_suffix(npy_path.suffix + ".tmp")
    records = np.empty(len(data_module.dataset), dtype=DETAIL_DTYPE)
    summary_rows = []
    record_offset = 0
    with temporary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(DETAIL_HEADER)
        for split_name in SPLIT_NAMES:
            summary = evaluate_split(
                model,
                data_module,
                split_name,
                split_indices[split_name],
                args,
                device,
                writer,
                records,
                record_offset,
            )
            summary_rows.append(summary)
            record_offset += summary["conformations"]

    if record_offset != len(data_module.dataset):
        raise RuntimeError(f"Incomplete output for {spec.label}.")

    with temporary_npy_path.open("wb") as handle:
        np.save(handle, records, allow_pickle=False)
    temporary_csv_path.replace(csv_path)
    temporary_npy_path.replace(npy_path)

    print(f"Saved {len(data_module.dataset):,} conformations to:")
    print(f"  CSV: {csv_path}")
    print(f"  NPY: {npy_path}")
    del model, checkpoint, data_module
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary_rows


def write_summary(output_dir, rows):
    path = output_dir / "summary.csv"
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset",
                "split",
                "conformations",
                "atoms",
                "energy_mae_kcal_per_mol",
                "force_x_mae_kcal_per_mol_per_angstrom",
                "force_y_mae_kcal_per_mol_per_angstrom",
                "force_z_mae_kcal_per_mol_per_angstrom",
                "force_mae_kcal_per_mol_per_angstrom",
            ]
        )
        for dataset, row in rows:
            writer.writerow(
                [
                    dataset,
                    row["split"],
                    row["conformations"],
                    row["atoms"],
                    format_number(row["energy_mae"]),
                    format_number(row["force_x_mae"]),
                    format_number(row["force_y_mae"]),
                    format_number(row["force_z_mae"]),
                    format_number(row["force_mae"]),
                ]
            )
    temporary_path.replace(path)
    print(f"Saved split summaries to {path}")


def main():
    args = parse_args()
    args.results_dir = args.results_dir.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    all_summary_rows = []
    for spec in select_specs(args.datasets):
        print(f"\nEvaluating all splits for {spec.label}")
        for row in evaluate_run(spec, args, device):
            all_summary_rows.append((spec.output_name, row))
    write_summary(args.output_dir, all_summary_rows)


if __name__ == "__main__":
    main()
