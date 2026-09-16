#!/usr/bin/env python3
"""Compare AI2BMD-PIMA predictions with multi-frame M062X XYZ labels."""

from __future__ import annotations

import argparse
import io
import gzip
import os
import pickle
import sys
import tempfile
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore", message=r"Unknown hyperparameter: ckpt_type=.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*\.jittable.*deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r"Converting a tensor with requires_grad=True to a scalar.*", category=UserWarning)

SRC_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SRC_DIR.parents[2] / "data" / "M062X_protein"
DEFAULT_CKPT_PATH = SRC_DIR / "ViSNet" / "checkpoints"
FIXED_CKPT_TYPE = "new"
FIXED_FRAGMENT_CALC = "fragment"
INPUT_SUBDIRECTORY_ORDER = ("chignolin", "trp-cage", "ww", "abd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--input-pdb", type=Path)
    parser.add_argument("--input-xyz", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device-strategy", choices=["excess-compute", "small-molecule", "large-molecule"], default="large-molecule")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args()


def pair_files(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    if args.input_pdb:
        pdb_path = args.input_pdb.expanduser().resolve()
        xyz_path = (args.input_xyz or pdb_path.with_suffix(".xyz")).expanduser().resolve()
        return [(pdb_path, xyz_path)]
    input_dir = args.input_dir.expanduser().resolve()
    pairs = [(pdb, pdb.with_suffix(".xyz"))
             for pdb in input_dir.rglob("*_align.pdb")
             if pdb.with_suffix(".xyz").is_file()]

    def sort_key(pair: tuple[Path, Path]) -> tuple[int, str, str]:
        pdb_path = pair[0]
        directory = pdb_path.parent.name.casefold()
        try:
            order = INPUT_SUBDIRECTORY_ORDER.index(directory)
        except ValueError:
            order = len(INPUT_SUBDIRECTORY_ORDER)
        return order, directory, pdb_path.name.casefold()

    return sorted(pairs, key=sort_key)


def validate_args(args: argparse.Namespace, pairs: list[tuple[Path, Path]]) -> None:
    if not pairs:
        raise FileNotFoundError("No matching *_align.pdb/*.xyz pairs were found")
    for pdb_path, xyz_path in pairs:
        if not pdb_path.is_file() or not xyz_path.is_file():
            raise FileNotFoundError(f"Missing pair: {pdb_path}, {xyz_path}")
    if args.start < 0 or args.stride <= 0:
        raise ValueError("--start must be non-negative and --stride must be positive")
    if args.stop is not None and args.stop < args.start:
        raise ValueError("--stop must be greater than or equal to --start")
    args.ckpt_path = DEFAULT_CKPT_PATH.resolve()
    if not args.ckpt_path.is_dir():
        raise FileNotFoundError(args.ckpt_path)
    args.output_dir = (args.output_dir or SRC_DIR.parent / "ai2bmd-comparison").expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)


def load_api(pdb_path: Path, runtime_dir: Path, args: argparse.Namespace, long_range_calc: str) -> dict[str, Any]:
    from AIMD import arguments

    cli = ["--prot-file", str(pdb_path), "--base-dir", str(runtime_dir), "--log-dir", str(runtime_dir),
           "--ckpt-path", str(args.ckpt_path), "--ckpt-type", FIXED_CKPT_TYPE,
           "--solvent-method", "FF19SB", "--no-solvent", "--fragcalc", "fragment",
           "--frag-nonbonded-calc", long_range_calc, "--device-strategy", args.device_strategy]
    original_argv = sys.argv
    sys.argv = [original_argv[0], *cli]
    try:
        initialization_output = io.StringIO()
        with redirect_stdout(initialization_output):
            configured = arguments.init()
        for line in initialization_output.getvalue().splitlines():
            if not line.startswith("DeviceStrategy:"):
                print(line)
        from AIMD.protein import Protein
        from Calculators.calculator import FragmentCalculator, patch_check_state
        from Calculators.device_strategy import DeviceStrategy
        from utils.utils import read_protein
    finally:
        sys.argv = original_argv
    return {"args": configured, "Protein": Protein, "FragmentCalculator": FragmentCalculator,
            "patch_check_state": patch_check_state, "DeviceStrategy": DeviceStrategy,
            "read_protein": read_protein}


def read_frames(pdb_path: Path, xyz_path: Path, args: argparse.Namespace) -> tuple[list[Any], list[Any]]:
    from ase.io import read

    selection = slice(args.start, args.stop, args.stride)
    pdb_frames = list(read(str(pdb_path), index=selection))
    xyz_frames = list(read(str(xyz_path), index=selection, format="extxyz"))
    if len(pdb_frames) != len(xyz_frames):
        raise ValueError(f"{pdb_path.name}: selected PDB/XYZ frame counts differ")
    if not pdb_frames:
        raise ValueError(f"{pdb_path.name}: selected frame range is empty")
    for index, (pdb, xyz) in enumerate(zip(pdb_frames, xyz_frames)):
        if len(pdb) != len(xyz) or not np.array_equal(pdb.numbers, xyz.numbers):
            raise ValueError(f"{pdb_path.name}: atom order/count differs at frame {index}")
        try:
            xyz.get_forces()
            xyz.get_potential_energy()
        except Exception as error:
            raise ValueError(f"{xyz_path.name}: frames need energy and forces in extxyz") from error
    return pdb_frames, xyz_frames


def atom_identity(atoms: Any, index: int) -> tuple[int, str, str]:
    return (int(atoms.arrays["residuenumbers"][index]),
            str(atoms.arrays["residuenames"][index]).strip(),
            str(atoms.arrays["atomtypes"][index]).strip())


def ai2bmd_order_permutation(input_atoms: Any, ordered_atoms: Any) -> np.ndarray:
    input_indices = {}
    for index in range(len(input_atoms)):
        identity = atom_identity(input_atoms, index)
        if identity in input_indices:
            raise ValueError(f"Duplicate atom identity in PDB: {identity}")
        input_indices[identity] = index
    permutation = np.asarray(
        [input_indices[atom_identity(ordered_atoms, index)] for index in range(len(ordered_atoms))],
        dtype=np.int64,
    )
    if len(permutation) != len(input_atoms):
        raise ValueError("Atom count changed during AI2BMD atom reordering")
    return permutation


def is_ai2bmd_order(atoms: Any) -> bool:
    first_residue = np.min(atoms.arrays["residuenumbers"])
    atom_names = [str(name).strip() for name, number in
                  zip(atoms.arrays["atomtypes"], atoms.arrays["residuenumbers"])
                  if number == first_residue]
    return atom_names == ["CH3", "C", "O", "H1", "H2", "H3"]


def infer_pair(pdb_path: Path, xyz_path: Path, args: argparse.Namespace, long_range_calc: str) -> dict[str, Any]:
    from ase.io import read, write
    from utils.utils import reorder_coord_amber2tinker

    pdb_frames, xyz_frames = read_frames(pdb_path, xyz_path, args)
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    with tempfile.TemporaryDirectory(prefix="ai2bmd-infer-") as temp:
        runtime = Path(temp)
        import shutil
        shutil.copyfile(SRC_DIR / "utils" / "seq_dict_FF19SB.pkl", runtime / "seq_dict.pkl")
        first_path = runtime / "reference.pdb"
        write(str(first_path), pdb_frames[0], format="proteindatabank")
        if not is_ai2bmd_order(pdb_frames[0]):
            reorder_coord_amber2tinker(str(first_path))
        ordered_input = read(str(first_path), index=0)
        ordered_to_input = ai2bmd_order_permutation(pdb_frames[0], ordered_input)
        api = load_api(first_path, runtime, args, long_range_calc)
        api["patch_check_state"]()
        reference = api["Protein"](api["read_protein"](str(first_path)), pdb4params=str(first_path))
        calculator = api["FragmentCalculator"](
            is_root_calc=True, nbcalc_type=long_range_calc,
            ckpt_path=api["args"].ckpt_path, ckpt_type=FIXED_CKPT_TYPE,
            checks=api["args"].checks)
        reference.calc = calculator
        calculator.bonded_calculator.fragment_method.fragment(reference)
        calculator.nonbonded_calculator.set_parameters(reference)
        api["DeviceStrategy"].set_work_partitions(reference.fragments_start, reference.fragments_end)
        energies, forces, positions = [], [], []
        frame_iterator = pdb_frames
        if tqdm is not None:
            frame_iterator = tqdm(
                pdb_frames,
                desc=f"{'AI2BMD-PIMA' if long_range_calc == 'pima' else 'AI2BMD'} inference",
                unit="frame",
                dynamic_ncols=True,
            )
        for frame in frame_iterator:
            reference.set_positions(frame.get_positions()[ordered_to_input])
            energies.append(float(np.asarray(reference.get_potential_energy()).reshape(-1)[0]))
            ordered_forces = np.asarray(reference.get_forces(), dtype=np.float64)
            frame_forces = np.empty_like(ordered_forces)
            frame_forces[ordered_to_input] = ordered_forces
            forces.append(frame_forces)
            positions.append(frame.get_positions().copy())

    ref_energy = np.asarray([float(frame.get_potential_energy()) for frame in xyz_frames])
    ref_forces = np.stack([np.asarray(frame.get_forces(), dtype=np.float64) for frame in xyz_frames])
    return {"input_pdb": str(pdb_path), "input_xyz": str(xyz_path),
            "long_range_calculator": long_range_calc,
            "checkpoint_path": str(args.ckpt_path), "checkpoint_type": FIXED_CKPT_TYPE,
            "units": {"energy": "eV", "forces": "eV/angstrom", "positions": "angstrom"},
            "atomic_numbers": pdb_frames[0].numbers.astype(np.int64),
            "positions": np.stack(positions), "pred_energies": np.asarray(energies),
            "pred_forces": np.stack(forces), "ref_energies": ref_energy, "ref_forces": ref_forces}


def metrics(result: dict[str, Any]) -> dict[str, float]:
    de = result["pred_energies"] - result["ref_energies"]
    df = result["pred_forces"] - result["ref_forces"]
    return {"energy_mae_eV": float(np.mean(np.abs(de))), "energy_rmse_eV": float(np.sqrt(np.mean(de ** 2))),
            "force_mae_eV_A": float(np.mean(np.abs(df))), "force_rmse_eV_A": float(np.sqrt(np.mean(df ** 2))),
            "force_max_abs_eV_A": float(np.max(np.abs(df)))}


def release_work_queue() -> None:
    """Release the process-wide queue before initializing the next protein."""
    from utils import utils as utils_module

    utils_module.WorkQueue.finalise()
    utils_module._workqueue_instance = None


def save_pickle(path: Path, result: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def main() -> None:
    from ase.units import kcal, mol

    args = parse_args(); pairs = pair_files(args); validate_args(args, pairs)
    ev_to_kcal_per_mol = mol / kcal
    for pdb_path, xyz_path in pairs:
        name = pdb_path.parent.name
        print(f"[{name}] inferring {pdb_path.name}")
        try:
            pima_result = infer_pair(pdb_path, xyz_path, args, "pima")
            release_work_queue()
            mm_result = infer_pair(pdb_path, xyz_path, args, "mm")
        finally:
            release_work_queue()
        save_pickle(args.output_dir / f"{name}.ai2bmd-pima.pkl.gz", pima_result)
        save_pickle(args.output_dir / f"{name}.ai2bmd.pkl.gz", mm_result)
        pima_metrics = metrics(pima_result)
        mm_metrics = metrics(mm_result)
        atom_count = len(pima_result["atomic_numbers"])
        pima_energy_mae = (pima_metrics["energy_mae_eV"] / atom_count *
                           ev_to_kcal_per_mol)
        mm_energy_mae = (mm_metrics["energy_mae_eV"] / atom_count *
                         ev_to_kcal_per_mol)
        pima_force_mae = pima_metrics["force_mae_eV_A"] * ev_to_kcal_per_mol
        mm_force_mae = mm_metrics["force_mae_eV_A"] * ev_to_kcal_per_mol
        print(
            f"[{name}] frames={len(pima_result['pred_energies'])}, "
            f"AI2BMD-PIMA energy MAE={pima_energy_mae:.6g} kcal/mol/atom, "
            f"AI2BMD energy MAE={mm_energy_mae:.6g} kcal/mol/atom, "
            f"AI2BMD-PIMA force MAE={pima_force_mae:.6g} kcal/mol/Å, "
            f"AI2BMD force MAE={mm_force_mae:.6g} kcal/mol/Å"
        )
        print()
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
