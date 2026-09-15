import argparse
import os
from os import path as osp

import torch
from Calculators.device_strategy import DeviceStrategy
import utils
from Calculators import error_injection


_args = None


def get():
    if not _args:
        raise Exception("Arguments are not initialized. Call initialize() first.")
    return _args


def init(argv=None):
    """Initializes the argument registry. If no argv is supplied (default), 
    parses arguments from process command line.
    The initialization result will be kept in the module-level member `_args`,
    so that the settings can be retrieved from other modules with get().
    """
    global _args

    _src_dir = utils.src_dir()
    parser = argparse.ArgumentParser(description="DL Molecular Simulation.")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=os.getcwd(),
        help="A directory for running simulation",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="A directory for saving results",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=osp.join(_src_dir, "ViSNet/checkpoints"),
        help="A directory including well-trained pytorch models",
    )
    parser.add_argument(
        "--ckpt-type",
        type=str,
        default="new",
        choices=["old", "new", "nll"],
        help="Checkpoint type, old/new/nll",
    )
    parser.add_argument(
        "--prot-file",
        type=str,
        default=osp.abspath(f"{_src_dir}/../testcases/1_rep_trp.c0.pdb"),
        help="Protein file for simulation",
    )
    parser.add_argument(
        "--temp-k",
        type=int,
        default=300,
        help="Simulation temperature in Kelvin",
    )
    parser.add_argument(
        "--sim-steps",
        type=int,
        default=1000000,
        help="Simulation steps for simulation",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=1,
        help="TimeStep (fs) for simulation",
    )
    parser.add_argument(
        "--preeq-steps",
        type=int,
        default=1000,
        help="Pre-equilibration simulation steps for each constraint",
    )
    parser.add_argument(
        "--constraints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Constrain hydrogen bonds",
    )
    parser.add_argument(
        "--monitor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Monitor protein properties during simulation",
    )
    parser.add_argument(
        "--checks",
        type=str,
        default=["DLNaN", "PairwiseDist", "BondLen"],
        nargs="+",
        choices=["DLNaN", "PairwiseDist", "BondLen"],
        help="Validity checks to perform",
    )
    parser.add_argument(
        "--mm-method",
        type=str,
        default="tinker-GPU",
        choices=["amber", "tinker", "tinker-GPU"],
        help="MM calculator for the nonbonded energy",
    )
    parser.add_argument(
        "--solvent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use solvent or not",
    )
    parser.add_argument(
        "--solvent-method",
        type=str,
        default="AMOEBA",
        choices=["AMOEBA", "FF19SB"],
        help="Method to use for preprocessing the protein",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for simulation",
    )
    parser.add_argument(
        "--start-from-sander",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use sander (Amber) to conduct pre-equilibration computation after pre-processing, and disables internal pre-equilibration.",
    )
    parser.add_argument(
        "--restart",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Restart the simulation",
    )
    parser.add_argument(
        "--build-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build PDB frames from traj after simulation",
    )
    parser.add_argument(
        "--record-per-steps",
        type=int,
        default=100,
        help="Interval for writing out frame data",
    )
    parser.add_argument(
        "--write-solvent",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write solvent into traj file",
    )
    parser.add_argument(
        "--device-strategy",
        type=str,
        default="small-molecule",
        choices=["excess-compute", "small-molecule", "large-molecule"],
        help="""The compute device allocation strategy.
        excess-compute=Assume compute resources are more than sufficient for
                ViSNet inference. Reserves last GPU for solvent/non-bonded
                computation.
        small-molecule=Maximise resources for ViSNet.
        large-molecule=Maximise resources for ViSNet, while also maximising
                concurrency and usage of GPUs for computation.
        """,
    )
    parser.add_argument(
        "--work-strategy",
        type=str,
        default="combined",
        choices=["combined"],
        help="""The work allocation strategy.
        combined=Distribute work evenly amongst both types of fragments.
        """,
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=9999,
        help="""Define the maximum chunk size (in units of atoms) for
        ACE-NME/dipeptide fragments.  The data will be split and processed
        according to these sizes.
        """,
    )
    parser.add_argument(
        "--inject-errors",
        type=str,
        default="",
        help="""A tuple of integers <grace,off,on,periodic> that specifies force error injection behavior,
        for test purposes. See src/Calculators/error_injection.py for details."""
    )
    parser.add_argument(
        "--rollback-steps",
        type=int,
        default=10,
        help="In case of MLFF error, how many steps to go backward in the trajectory (including the current step)",
    )
    parser.add_argument(
        "--rollback-mm-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In case of MLFF error, the fallback method will be MM only, instead of QM and MM)",
    )
    parser.add_argument(
        "--rollback-save",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="In case of MLFF error, the affected frames will be first saved as active learning unlabelled data, before being discarded",
    )
    parser.add_argument(
        "--restart-rollback-buffer",
        type=int,
        default=10,
        help="On restart, copy the last N frames from previous trajectory into the current. Alleviates the 'cannot rollback at the beginning of a restart run' problem.",
    )
    parser.add_argument(
        "--fragcalc",
        type=str,
        default="fragment",
        choices=["fragment", "visnet", "orca"],
        help="""The ASE calculator to use in the QMMM simulation, for the fragments part.
        fragment: use AI2BMD FragmentCalculator.
        visnet: disables fragmentation, and feed all QM atoms through a single ViSNet model.
        orca: disables fragmentation, and feed all QM atoms through a ASE driver for ORCA (DFT).
        """,
    )
    parser.add_argument(
        "--frag-nonbonded-calc",
        type=str,
        default="pima",
        choices=["mm", "pme", 'pima'],
        help="Nonbonded calculator for fragments; required when fragmentation is enabled.",
    )
    parser.add_argument(
        "--charge",
        type=int,
        default=None,
        help="""The charge of the system. Only used if --fragcalc orca"""
    )
    parser.add_argument(
        "--multiplicity",
        type=int,
        default=None,
        help="""The multiplicity of the system. Only used if --fragcalc orca"""
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action='count',
        default=0,
        help="""Verbosity level"""
    )
    parser.add_argument(
        "--restart-steps",
        type=int,
        default=1,
        help="""Steps to restart from. default is 1."""
    )
    parser.add_argument(
        "--minimization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="""Run AI2BMD with energy minimization"""
    )
    parser.add_argument(
        "--save-pima",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="""Save pima dataset"""
    )

    _args = parser.parse_args(argv)
    _args.prot_name = osp.basename(_args.prot_file)[:-4]
    if _args.log_dir is None:
        _args.log_dir = osp.join(_args.base_dir, f"Logs-{_args.prot_name}")
    os.makedirs(_args.log_dir, exist_ok=True)
    _args.base_dir = osp.abspath(_args.base_dir)
    _args.log_dir = osp.abspath(_args.log_dir)
    _args.ckpt_path = osp.abspath(_args.ckpt_path)
    _args.prot_file = osp.abspath(_args.prot_file)
    _args.utils_dir = osp.join(_src_dir, "utils")

    strategy_feedback = DeviceStrategy.initialize(
        _args.device_strategy,
        _args.work_strategy,
        _args.mm_method,
        torch.cuda.device_count(),
        _args.chunk_size,
    )
    _args.mm_method = strategy_feedback['solvent-method']

    error_injection.initialize(_args.inject_errors)
    assert _args.rollback_steps > 1, "Do not rollback just the current step."

    # start_from_sander mandates restart, and therefore disables internal preeq
    if _args.start_from_sander:
        _args.restart = True

    # clear checks if monitoring is disabled
    if _args.monitor is False:
        _args.checks = []

    return _args

