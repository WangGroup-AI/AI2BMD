import os
import json
import io
import shutil
from abc import ABC, abstractmethod

import numpy as np
from ase import units
from ase.calculators.calculator import Calculator
from ase.calculators.amber import Amber
from ase.constraints import Hookean
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.geometry import find_mic
from ase.md.langevin import Langevin
from ase.md.md import MolecularDynamics
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from AIMD import arguments, envflags
from AIMD.protein import Protein
from AIMD.qmmm import AsyncQMMM
from Calculators.calculator import FragmentCalculator
from Calculators.device_strategy import DeviceStrategy
from Calculators.monitor import FragmentMonitorError
from Calculators.tinker_async import TinkerAsyncCalculator
from Calculators.visnet_calculator import ViSNetCalculator
from utils.system import get_physical_core_count
from utils.utils import (
    MDObserver,
    PDBAnalyzer,
    RNGPool,
    WorkQueue,
    get_default_orca_calculator,
    kcalmol2ev,
    read_protein,
)


class BaseSimulator(ABC):
    def __init__(
        self, prot: Protein, log_path: str, preeq_steps: int = 200, temp_k: int = 300
    ) -> None:
        self.prot = prot
        self.log_path = log_path
        self.simulation_save_path = os.path.join(log_path, "SimulationTmp")
        os.makedirs(self.simulation_save_path, exist_ok=True)
        self.simulation_results_path = os.path.join(log_path, "SimulationResults")
        os.makedirs(self.simulation_results_path, exist_ok=True)
        self.qm_save_path = os.path.join(log_path, "ActiveLearningData")
        os.makedirs(self.qm_save_path, exist_ok=True)
        self.nowat_pdb = self.prot.nowater_PDB_path
        # This restart is deliberately kept next to the preprocessing files,
        # and contains the complete system (including solvent) independently
        # of the production trajectory's write-solvent setting.
        mm_dir = os.path.dirname(self.nowat_pdb)
        prot_stem = os.path.basename(self.nowat_pdb).removesuffix("-preeq-nowat.pdb")
        self.quick_restart_path = os.path.join(mm_dir, f"{prot_stem}-quick-restart.traj")
        self.preeq_steps = preeq_steps
        self.prot.set_pbc(True)
        self.temp_k = temp_k
        self.rollback_steps = arguments.get().rollback_steps
        self.rollback_factor = 1
        self.fail_step_stack = []
        self.rollback_buffer: int = arguments.get().restart_rollback_buffer
        assert self.rollback_buffer >= 0, "Invalid restart rollback buffer setting"

    def get_qm_idx(self):
        return list(range(len(read_protein(self.nowat_pdb))))

    def need_fragmentation(self):
        return isinstance(self.qmcalc, FragmentCalculator)

    def initialize_fragcalc(self):
        if self.need_fragmentation():
            self.qmcalc.bonded_calculator.fragment_method.fragment(self.qmatoms)
            self.qmcalc.nonbonded_calculator.set_parameters(self.qmatoms)

            start, end = self.qmatoms.fragments_start, self.qmatoms.fragments_end
        else:
            start, end = [0], [len(self.qmatoms)]

        # set work partitions based on dipeptides/ACE-NMEs
        DeviceStrategy.set_work_partitions(start, end)

        # Keep a human-readable record of the fragmentation used for this run.
        fragment_dir = self.prot.save_path
        os.makedirs(fragment_dir, exist_ok=True)
        fragment_info = {
            "qm_atom_count": len(self.qmatoms),
            "fragment_count": len(start),
            "fragments": [
                {
                    "start": int(s),
                    "end": int(e),
                    "atom_indices": list(range(int(s), int(e))),
                }
                for s, e in zip(start, end)
            ],
        }
        with open(os.path.join(fragment_dir, "fragments.json"), "w") as fp:
            json.dump(fragment_info, fp, indent=2)

    def set_calculator(self, **kwargs) -> None:
        os.chdir(self.simulation_save_path)
        self.make_calculator(**kwargs)
        self.initialize_fragcalc()

    @abstractmethod
    def make_calculator(self, **kwargs) -> Calculator:
        pass

    def make_fragment_calculator(self, is_root_calc: bool, **kwargs) -> FragmentCalculator:
        fragcalc_type = arguments.get().fragcalc
        if fragcalc_type == "fragment":
            return FragmentCalculator(is_root_calc=is_root_calc, **kwargs)
        elif fragcalc_type == "visnet":
            return ViSNetCalculator(is_root_calc=is_root_calc, **kwargs)
        elif fragcalc_type == "orca":
            charge = arguments.get().charge
            mult = arguments.get().multiplicity

            return get_default_orca_calculator("full", charge, mult)
        else:
            raise ValueError(f"Unrecognized fragment calculator type {fragcalc_type}")

    def simulate(
        self,
        prot_name: str,
        simulation_steps: int,
        time_step: float,
        record_per_steps: int,
        hydrogen_constraints: bool,
        checks: list[str],
        seed: int,
        restart: bool,
        build_frames: bool,
        restart_steps: int,
        save_pima: bool,
        write_solvent: bool = False,
    ):
        restart_traj_path = os.path.join(self.simulation_results_path, f"{prot_name}-traj.traj")

        if restart:
            with Trajectory(restart_traj_path) as ori_traj:
                restart_frame_count = len(ori_traj)
                restart_last_frame = ori_traj[-restart_steps]
                self.prot.set_positions(restart_last_frame.get_positions())
                self.prot.set_velocities(restart_last_frame.get_velocities())
        else:
            restart_frame_count = 0
            MaxwellBoltzmannDistribution(self.prot, temperature_K=self.temp_k, rng=np.random.RandomState(seed))

        # initialize rng pool
        rng_pool = RNGPool(seed=seed, shape=(len(self.prot), 3), count=2)

        MolDyn = Langevin(
            self.prot,
            timestep=time_step * units.fs,
            temperature_K=self.temp_k,
            friction=0.001 / units.fs,
            rng=rng_pool,
        )

        if restart:
            moldyn_traj_filename = os.path.join(self.simulation_results_path, f"{prot_name}-traj-restart.traj")
        else:
            moldyn_traj_filename = os.path.join(self.simulation_results_path, f"{prot_name}-traj.traj")

        moldyn_traj = Trajectory(moldyn_traj_filename, "w", self.prot)

        observer = MDObserver(
            a=self.prot,
            q=self.qmatoms,
            md=MolDyn,
            traj=moldyn_traj,
            rng=rng_pool,
            step_offset=restart_frame_count,
            write_solvent=write_solvent,
        )
        MolDyn.attach(observer.save_traj_copy, interval=record_per_steps)
        MolDyn.attach(observer.write_traj, interval=record_per_steps)
        MolDyn.attach(observer.printenergy, interval=record_per_steps)
        if save_pima:
            MolDyn.attach(observer.get_hidden_embeddings, interval=record_per_steps)
        MolDyn.attach(observer.renameALdata, interval=1)
        MolDyn.attach(observer.fill_rng_pool, interval=1)

        quick_restart = False
        if not restart and self.preeq_steps != 0 and os.path.isfile(self.quick_restart_path):
            try:
                with Trajectory(self.quick_restart_path, "r") as quick_traj:
                    if len(quick_traj) != 1 or len(quick_traj[0]) != len(self.prot):
                        raise ValueError("invalid atom count")
                    quick_frame = quick_traj[0]
                    self.prot.set_positions(quick_frame.get_positions())
                    velocities = quick_frame.get_velocities()
                    if velocities is None:
                        raise ValueError("missing velocities")
                    self.prot.set_velocities(velocities)
                quick_restart = True
                print(f"Pre-equilibration step already done, skip...")
            except (OSError, ValueError, IndexError) as exc:
                print(f"Invalid quick pre-equilibration restart, rerun pre-equilibration: {exc}")

        if (not restart) and (not quick_restart) and (self.preeq_steps != 0):
            init_constraint = self.prot.constraints.copy()
            indices_to_constrain = self.get_qm_idx()
            restraints = [10, 5, 1, 0.5, 0.1]
            solute_atoms = len(indices_to_constrain)
            solvent_atoms = len(self.prot) - solute_atoms
            print(
                f"Total atoms in system: {len(self.prot)} "
                f"(solute: {solute_atoms}, solvent: {solvent_atoms})"
            )
            print("Start pre-equilibration")
            for restraint in restraints:
                print(
                    f"Pre-equilibration with {restraint} eV/A² for {self.preeq_steps} steps"
                )
                constraints = []
                ref_positions = self.prot.positions
                for idx in indices_to_constrain:
                    pos = ref_positions[idx]
                    constraint = Hookean(a1=idx, a2=pos, k=restraint * kcalmol2ev, rt=0)
                    constraints.append(constraint)
                self.prot.constraints.extend(constraints)
                MolDyn.run(self.preeq_steps)
                self.prot.constraints = init_constraint.copy()
            print("Pre-equilibration finished!")
            # Write the complete state before any solvent filtering used by
            # MDObserver.  ASE Trajectory stores positions, velocities, cell,
            # and all atom arrays in this single-frame restart.
            os.makedirs(os.path.dirname(self.quick_restart_path), exist_ok=True)
            with Trajectory(self.quick_restart_path, "w", self.prot) as quick_traj:
                quick_traj.write(self.prot)
            print(f"Saved quick pre-equilibration restart: {self.quick_restart_path}")

        if hydrogen_constraints is True:
            pdb_analyzer = PDBAnalyzer(self.nowat_pdb)
            hydrogen_bonds = pdb_analyzer.find_bonded_atoms("H")
            hydrogen_constraints = []

            for pair in hydrogen_bonds:
                # * Hookean constraints
                hydrogen_constraint = Hookean(
                    a1=pair[0], a2=pair[1], k=pair[3], rt=pair[2]
                )
                hydrogen_constraints.append(hydrogen_constraint)

            self.prot.constraints.extend(hydrogen_constraints)

        if restart:
            print(f"Re-start simulation for {simulation_steps} steps")
        else:
            print(f"Start simulation for {simulation_steps} steps")

        rollback_buffer_len = min(self.rollback_buffer, restart_frame_count)
        if rollback_buffer_len:
            # Double caution: when MolDyn.nsteps != 0, it does not call "write" on step 0.
            # This is actually a better behavior than our current practice, where step 0 replicates the last frame
            # of the previous trajectory, and this will make rollback behave consistently because there's no
            # repeated frames.
            print(f"Adding {rollback_buffer_len} frames from previous trajectory to current...")
            # write orig_traj[-rollback_buffer_len .. -1] to moldyn_traj
            with Trajectory(restart_traj_path) as ori_traj:
                for i in range(rollback_buffer_len):
                    moldyn_traj.write(ori_traj[i - rollback_buffer_len])
            # adjust MolDyn.nsteps and simulation_steps
            simulation_steps += rollback_buffer_len - 1
            MolDyn.nsteps = rollback_buffer_len - 1
            # nsteps != 0, ASE will not write step 0.
            if rollback_buffer_len == 1:
                # very unlikely, but we check this and warn anyway
                print("[simulator] Warning: rollback_buffer length is 1, step 0 will be duplicated!")
            # adjust observer offset to exclude the rollback buffer
            observer.step_offset -= rollback_buffer_len
        elif self.preeq_steps != 0:
            # adjust simulation_steps to exclude preeq steps
            simulation_steps += MolDyn.nsteps

        while True:
            try:
                MolDyn.run(simulation_steps - MolDyn.nsteps)
                break
            except FragmentMonitorError as ex:
                self.rollback(MolDyn, moldyn_traj_filename, moldyn_traj, ex)
                print(f"Rollback complete. {simulation_steps - MolDyn.nsteps} to go.")
                # additional fill required for rollback
                rng_pool.fill()

        print("Simulation finished!")
        WorkQueue.finalise()
        moldyn_traj.close()

        # note: if we add rollback buffer to the beginning, we should not remove them here,
        # because these frames are more updated than the previous trajectory now.
        # when we're done with all the segments of the whole trajectory, we can identify
        # these "buffered" segments by checking len(moldyn_traj) and see if there're additional (n-1) frames,
        # and overwrite those from the previous segment.

        if build_frames and not restart:
            self.build_frames_from_traj(prot_name)

        if not envflags.DEBUG_RC:
            shutil.rmtree(
                os.path.join(self.log_path, "SimulationTmp"), ignore_errors=True
            )

    def rollback(
        self,
        MolDyn: MolecularDynamics,
        moldyn_traj_filename: str,
        moldyn_traj: Trajectory,
        ex: FragmentMonitorError,
    ):
        # drop the failing frame, plus ex_nsteps (computed in get_rollback_steps)
        # for example, if ex_nsteps==2:
        #
        # current timeline:  ...  32   33   34  35 <-- monitor fails here, MolDyn.nsteps == 34
        #
        #                         .------ new MolDyn.nsteps
        #                         |
        #                         v   [_______] use fallback method
        # rollback timeline: ...  32  [33] [34] 35 <-- fallback released, back to normal computation
        #                    `-----'
        #                    Frame 0 ... Frame 32 are kept, 33 frames in total
        ex_nsteps = self.get_rollback_steps(MolDyn.nsteps)
        print(f"Simulation will now rollback {ex_nsteps} steps...")
        # stepback_fn drives the monitor step back with computed ex_nsteps
        ex.stepback_fn(ex_nsteps, MolDyn.nsteps + 1)
        MolDyn.nsteps -= ex_nsteps
        keep_frames_count = MolDyn.nsteps + 1
        assert MolDyn.nsteps >= 0, "cannot rollback further, already at the beginning"
        moldyn_traj.close()
        # self.dump_traj_python(moldyn_traj_filename, range(keep_frames_count))
        self.truncate_traj_native(moldyn_traj_filename, keep_frames_count)
        with Trajectory(moldyn_traj_filename, "r") as reader:
            rollback_last_frame = reader[keep_frames_count - 1]
            self.prot.set_positions(rollback_last_frame.get_positions())
            self.prot.set_velocities(rollback_last_frame.get_velocities())
        moldyn_traj._open(moldyn_traj_filename, "a")

    def dump_traj_python(self, moldyn_traj_filename, frames_range):
        tmp_save_filename = f"{moldyn_traj_filename}.tmp-save"
        frames_kept = 0
        with Trajectory(moldyn_traj_filename, "r") as reader:
            with Trajectory(tmp_save_filename, "w") as writer:
                for i in frames_range:
                    writer.write(reader[i])
                    frames_kept += 1
                print(f"{frames_kept} / {len(reader)} frames kept.")
        shutil.move(tmp_save_filename, moldyn_traj_filename)

    def truncate_traj_native(self, moldyn_traj_filename, keep_frames_count):
        ret = os.system(f"truncate_traj {moldyn_traj_filename} {keep_frames_count}")
        if 0 != ret:
            raise RuntimeError(f"truncate_traj_native failed with code {ret}")

    # The failure steps are kept in a stack, with nstep decreasing monotonously towards the stack top.
    # The stack keeps track of situations where another failure happens when processing the previous ones.
    #
    # Algorithm:
    #   1) pop stack top while the failure on stack is older than current (made progress)
    #      - if we empty the stack, it means we have cleared all the older errors and made actual progress.
    #        we then reset rollback factor to 1.
    #      - otherwise, it means we are still on the stack of an older root cause. keep the rollback factor as-is.
    #   2) if stack top is equal to the current, it means that we have hit a deadloop, and have to increase the rollback factor.
    #      - otherwise, push current to stack.
    #
    # Example, assume rollback_steps == 10:
    #
    #   - [nnn] in the timeline is the value of moldyn_nsteps at the failure
    #
    # timeline 0:                                       ... 391 [392] x       stack: [392]              (push 392)
    # timeline 1:           ... 383 ... [387]  x                              stack: [392, 387]         (push 387)
    # timeline 2:   ... 378 ...                   [389] x                     stack: [392, 389]         (pop 387)
    # timeline 3:       ... 380 ...                             [392] x       stack: [392]              (pop 389, increase rollback factor)
    def get_rollback_steps(self, moldyn_nsteps: int) -> int:

        # algorithm step 1
        while len(self.fail_step_stack):
            if self.fail_step_stack[-1] >= moldyn_nsteps:
                break
            self.fail_step_stack = self.fail_step_stack[:-1]
        if not len(self.fail_step_stack):
            # made some progress, reset rollback factor to 1
            self.rollback_factor = 1

        # algorithm step 2
        if len(self.fail_step_stack) and self.fail_step_stack[-1] == moldyn_nsteps:
            self.rollback_factor += 1
            print(f"Rollback can't make progress, increasing factor to {self.rollback_factor}")
        else:
            self.fail_step_stack.append(moldyn_nsteps)

        return self.rollback_factor * self.rollback_steps

    def build_frames_from_traj(self, prot_name):
        """Create one periodic, protein-centered multi-model PDB trajectory."""
        print("Building PDB trajectory from trajectory...")
        traj_path = os.path.join(self.simulation_results_path, f"{prot_name}-traj.traj")
        pdb_path = os.path.join(self.simulation_results_path, f"{prot_name}-traj.pdb")
        if os.path.exists(pdb_path):
            os.remove(pdb_path)

        protein_indices = self.get_qm_idx()
        with Trajectory(traj_path) as simutraj:
            for atoms in simutraj:
                atoms = atoms.copy()
                # Trajectory frames may lose PDB topology arrays; restore them
                # from the simulation protein so ASE writes real residue data.
                source_indices = (np.arange(len(self.prot)) if len(atoms) == len(self.prot)
                                  else np.asarray(self.get_qm_idx()))
                if len(atoms) == len(source_indices):
                    for name in ("atomtypes", "residuenames", "residuenumbers", "chainids"):
                        if name in self.prot.arrays:
                            atoms.set_array(name, self.prot.arrays[name][source_indices].copy())
                atoms.set_pbc(True)
                residue_numbers = atoms.arrays.get("residuenumbers")

                def make_whole(indices):
                    """Put a molecule's atoms on the same periodic image."""
                    if len(indices) < 2:
                        return
                    origin = atoms.positions[indices[0]].copy()
                    for index in indices[1:]:
                        delta = atoms.positions[index] - origin
                        delta, _ = find_mic(delta, atoms.cell, pbc=True)
                        atoms.positions[index] = origin + delta

                # AMOEBA coordinates are centered around the origin.  Rebuild
                # the protein as one object, rather than wrapping each residue
                # independently (which would scatter it at box boundaries).
                make_whole(np.asarray(protein_indices, dtype=int))
                protein_com = atoms[protein_indices].get_center_of_mass()
                box_center = np.sum(atoms.cell.array, axis=0) / 2.0
                atoms.translate(box_center - protein_com)

                # Rebuild and wrap solvent molecules as units, preserving O-H
                # geometry while placing each complete molecule in the cell.
                if residue_numbers is not None:
                    protein_set = set(protein_indices)
                    for residue in np.unique(residue_numbers):
                        indices = np.flatnonzero(residue_numbers == residue)
                        if all(int(index) in protein_set for index in indices):
                            continue
                        make_whole(indices)
                        frac = atoms.cell.scaled_positions(atoms.positions[indices[0]])
                        shift = np.floor(frac)
                        atoms.positions[indices] -= shift @ atoms.cell.array
                frame_buffer = io.StringIO()
                write(frame_buffer, atoms, format="proteindatabank")
                frame_text = frame_buffer.getvalue()
                frame_text = "".join(
                    line[:76] + "  " + line[78:]
                    if line.startswith(("ATOM", "HETATM")) and len(line) >= 78
                    else line
                    for line in frame_text.splitlines(keepends=True)
                )
                with open(pdb_path, "a") as pdb_file:
                    pdb_file.write(frame_text)

        print("Done building PDB trajectory.")


class SolventSimulator(BaseSimulator):
    def __init__(
        self,
        prot: Protein,
        log_path: str,
        preeq_steps: int,
        temp_k: int,
        utils_dir: str,
        mm_file: str,
        pdb_file: str,
        top_file: str,
        nowat_pdb_file: str,
        nowat_top_file: str,
        mmcalc_type: str,
        solvent_method: str,
        dev_strategy: str,
    ) -> None:
        super().__init__(prot, log_path, preeq_steps, temp_k)

        self.utils_dir = utils_dir
        self.mm_file = mm_file
        self.pdb_file = pdb_file
        self.top_file = top_file
        self.nowat_pdb_file = nowat_pdb_file
        self.nowat_top_file = nowat_top_file
        self.mmcalc_type = mmcalc_type
        self.solvent_method = solvent_method
        self.dev_strategy = dev_strategy

    def make_mm_calculator(self):
        devices = DeviceStrategy.get_solvent_devices()

        if self.mmcalc_type == "amber":
            # prepare the backup
            shutil.copy(self.mm_file, os.path.basename(self.mm_file))
            shutil.copy(self.top_file, os.path.basename(self.top_file))

            cpu_cores = get_physical_core_count()
            # Note: for small molecules (e.g., protein with more than 1000 atoms), using sander.MPI with too many cores will lead to error, more tests are needed
            # Amoeba in amber does not support multi cpu
            if cpu_cores > 1 and self.dev_strategy == "large-molecule" and self.solvent_method == "FF19SB":
                amber_exe = f"mpirun -np {cpu_cores} sander.MPI -O"
            elif cpu_cores > 1 and self.solvent_method == "FF19SB":
                # for small molecules anf ff19sb force field, let's use 8 cores
                cpu_core_small_mol = min(8, cpu_cores)
                amber_exe = f"mpirun -np {cpu_core_small_mol} sander.MPI -O"
            else:
                amber_exe = "sander -O"

            mm_calc = Amber(
                amber_exe=amber_exe,
                infile=os.path.basename(self.mm_file),
                outfile="mm.out",
                topologyfile=os.path.basename(self.top_file),
                incoordfile="mm.crd",
                outcoordfile="mm_dummy.crd",
            )

            mm_calc.add_e_f_args("mden_mm", "mdfrc_mm")
        elif self.mmcalc_type in ["tinker", "tinker-GPU"]:
            mm_calc = TinkerAsyncCalculator(
                pdb_file=self.pdb_file,
                utils_dir=self.utils_dir,
                devices=devices,
            )
        else:
            raise ValueError(f"Unknown mm calculator: {self.mmcalc_type}")
        return mm_calc

    def make_mm_qmregion_calculator(self):
        devices = DeviceStrategy.get_solvent_devices()
        if self.mmcalc_type == "amber":
            # prepare the backup
            shutil.copy(self.mm_file, "mm_qmregion.in")
            shutil.copy(self.nowat_top_file, os.path.basename(self.nowat_top_file))

            cpu_cores = get_physical_core_count()
            # Note: for small proteins, using sander.MPI will lead to error, more tests are needed
            # Amoeba in amber does not support multi cpu
            if cpu_cores > 1 and self.dev_strategy == "large-molecule" and self.solvent_method == "FF19SB":
                amber_exe = f"mpirun -np {cpu_cores} sander.MPI -O"
            else:
                amber_exe = "sander -O"

            mm_qmregion_calc = Amber(
                amber_exe=amber_exe,
                infile="mm_qmregion.in",
                outfile="mm_qmregion.out",
                topologyfile=os.path.basename(self.nowat_top_file),
                incoordfile="mm_qmregion.crd",
                outcoordfile="mm_qmregion_dummy.crd",
            )

            mm_qmregion_calc.add_e_f_args("mden_mm_qmregion", "mdfrc_mm_qmregion")
        elif self.mmcalc_type in ["tinker", "tinker-GPU"]:
            mm_qmregion_calc = TinkerAsyncCalculator(
                pdb_file=self.nowat_pdb_file,
                utils_dir=self.utils_dir,
                devices=devices,
            )
        else:
            raise ValueError(f"Unknown mm calculator: {self.mmcalc_type}")
        return mm_qmregion_calc

    def make_calculator(self, **kwargs):
        self.prot.calc = AsyncQMMM(
            selection=self.get_qm_idx(),
            qmcalc=self.make_fragment_calculator(is_root_calc=False, **kwargs),
            mmcalc1=self.make_mm_qmregion_calculator(),
            mmcalc2=self.make_mm_calculator(),
        )

        self.prot.calc.initialize_qm(self.prot)

        self.qmcalc = self.prot.calc.qmcalc
        self.qmatoms = self.prot.calc.qmatoms

        if isinstance(self.prot.calc.mmcalc1, TinkerAsyncCalculator):
            self.prot.calc.mmcalc1.atoms = self.qmatoms
            self.prot.calc.mmcalc1._start_tinker()
        if isinstance(self.prot.calc.mmcalc2, TinkerAsyncCalculator):
            self.prot.calc.mmcalc2.atoms = self.prot
            self.prot.calc.mmcalc2._start_tinker()


class NoSolventSimulator(BaseSimulator):
    def __init__(
        self,
        prot: Protein,
        log_path: str,
        preeq_steps: int,
        temp_k: int,
        mm_file: str,
        top_file: str,
        nowat_top_file: str,
        **kwargs
    ) -> None:
        super().__init__(prot, log_path, preeq_steps, temp_k)

        self.prot = self.prot[self.get_qm_idx()]

        self.mm_file = mm_file
        self.top_file = top_file
        self.nowat_top_file = nowat_top_file

    def make_calculator(self, **kwargs):
        self.prot.calc = self.make_fragment_calculator(is_root_calc=True, **kwargs)

        self.qmcalc = self.prot.calc
        self.qmatoms = self.prot
