import math
import os
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial, wraps
from os import path as osp
from typing import Any, Callable, List

import numpy as np
import torch
from ase import Atoms, units
from ase.calculators.orca import ORCA
from ase.io import read
from ase.io.trajectory import TrajectoryWriter
from ase.md.md import MolecularDynamics

from AIMD import arguments
from utils.system import get_physical_core_count


def read_protein(fpath: str) -> Atoms:
    r"""
    Convert .pdb file to ase Atoms object.
    """
    assert fpath.endswith(".pdb"), "Error: The file format is not PDB!"
    atoms = read(fpath)
    args = arguments.get()
    if args.solvent_method == 'FF19SB' and args.solvent and args.mm_method == "amber":
        cell = atoms.get_cell()
        x, y, z = cell[0, 0], cell[1, 1], cell[2, 2]
        pme = math.ceil(max(x, y, z)) * 1.5
        for i in range(999):
            if (
                    (pme + i) % 2 == 0
                    and (pme + i) % 3 == 0
                    and (pme + i) % 4 == 0
                    and (pme + i) % 5 == 0
                ):
                    pme = pme + i
                    break
        atoms.set_cell([pme, pme, pme])
    if len(atoms) == 0:
        raise ValueError("Error: The PDB file is empty!")

    return atoms


def fill_atom_symbol(atom_symbol: str) -> str:
    if len(atom_symbol) > 4:
        raise ValueError("atom symbol is too long")
    elif len(atom_symbol) == 4:
        return atom_symbol
    elif 0 < len(atom_symbol) < 4:
        return " " + atom_symbol + " " * (3 - len(atom_symbol))
    elif len(atom_symbol) <= 0:
        raise ValueError("atom symbol is too short")
    return ""


def reorder_atoms(fpath: str):
    r"""
    Reorder atoms in .pdb output from tinker.
    """
    assert fpath.endswith(".pdb"), "Error: The file format is not PDB!"
    with open(fpath, 'r') as f:
        lines = f.readlines()

    output = []
    sidechain = []

    res_id = None
    res_count = 0
    h_found = False

    for l in lines:
        cols = l.split()

        if len(cols) < 8 or cols[0] != 'ATOM':
            output.append(l)
            continue

        if cols[4] != res_id:
            res_count = 0

        # check if sidechain atoms should be written
        if cols[2] == 'H' or cols[2] == 'HA':
            res_count = 0
            h_found = True
        elif h_found is True:
            # write sidechain atoms right after H/HA
            output.extend(sidechain)
            sidechain = []
            h_found = False

        # enumerate N/CA/C/O atoms
        if res_count == 0 and cols[2] == 'N':
            res_count += 1
        elif res_count == 1 and cols[2] == 'CA':
            res_count += 1
        elif res_count == 2 and cols[2] == 'C':
            res_count += 1
        elif res_count == 3 and cols[2] == 'O':
            res_count += 1
        # save rows between N/CA/C/O and H/HA
        elif res_count == 4:
            sidechain.append(l)
            res_id = cols[4]
            continue

        # save line to output
        output.append(l)

        # update current residue id
        res_id = cols[4]

    with open(fpath, 'w') as f:
        for l in output:
            f.write(l)


def standardise_pdb(fpath: str):
    r"""
    Check and rewrite residue numbers in .pdb output from tinker. Wraps residue
    numbers > 9999 to 0 so that ase can process the .pdb file correctly.
    """
    assert fpath.endswith(".pdb"), "Error: The file format is not PDB!"
    with open(fpath, 'r') as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue

            try:
                # ase.io.proteindatabank: extract the residue number
                res_idx = int(line[22:26].split()[0])
            except IndexError:
                break
            else:
                return

    output = []
    with open(fpath, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                # wrap the residue number on 10000
                res_idx = int(line[6:].split()[3])
                end = re.search(r'\b({})\b'.format(res_idx), line[22:]).end() + 22
                output.append(line[:22] + f"{res_idx % 10000: >4}" + line[end:])
            else:
                output.append(line)

    with open(fpath, 'w') as f:
        for l in output:
            f.write(l)


def get_residue_name(prot: Atoms) -> List[str]:
    atom_resname: List[str] = prot.arrays["residuenames"].tolist()
    atom_resid: List[np.int64] = prot.arrays["residuenumbers"].tolist()
    mol_resid = list(set(atom_resid))
    mol_resid.sort(key=atom_resid.index)
    mol_index = [atom_resid.index(id) for id in mol_resid]
    mol_resname = [
        atom_resname[index].strip()
        for index in mol_index
        if atom_resname[index].strip() != "WAT"
    ]
    return mol_resname


# Move pdb coordinates according to the mass center
def translate_coord_pdb(inpfile: str, outfile: str):
    atomic_masses = {
    'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'P': 30.974, 'S': 32.06, 'F': 18.998, 'CL': 35.453, 'NA': 22.990
}
    atoms = []
    masses = []
    with open(inpfile, 'r') as file:
        for line in file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                atom_type = line[76:78].strip()
                mass = atomic_masses.get(atom_type, 0)
                atoms.append([x, y, z])
                masses.append(mass)
    atoms = np.array(atoms)
    masses = np.array(masses)

    # get mass center
    total_mass = np.sum(masses)
    mass_center = np.sum(atoms * masses[:, np.newaxis], axis=0) / total_mass

    # translate atoms
    atoms -= mass_center

    # write pdb
    with open(inpfile, 'r') as file:
        original_lines = file.readlines()

    # get pdc
    pbc_x = math.ceil(np.max(np.abs(atoms[:, 0])) + 50)
    pbc_y = math.ceil(np.max(np.abs(atoms[:, 1])) + 50)
    pbc_z = math.ceil(np.max(np.abs(atoms[:, 2])) + 50)

    with open(outfile, 'w') as file:
        atom_index = 0
        # file.write(f"HEADER    {pbc_x:.3f} {pbc_y:.3f} {pbc_z:.3f}\n")
        for line in original_lines[1:]:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                x, y, z = atoms[atom_index]
                file.write(f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}")
                atom_index += 1
            else:
                file.write(line)

    return pbc_x, pbc_y, pbc_z


def reorder_coord_amber2tinker(fpath: str):
    r"""
    Reorder atoms in .pdb output from tinker.
    """
    assert fpath.endswith(".pdb"), "Error: The file format is not PDB!"
    
    reorder_dict = {
        'ACE': [1, 4, 5, 0, 2, 3],
        'ALA': [0, 2, 8, 9, 1, 3, 4, 5, 6, 7],
        'ARG': [0, 2, 22, 23, 1, 3, 4, 7, 10, 13, 15, 16, 19, 5, 6, 8, 9, 11, 12, 14, 17, 18, 20, 21],
        'ASN': [0, 2, 12, 13, 1, 3, 4, 7, 8, 9, 5, 6, 10, 11],
        'ASP': [0, 2, 10, 11, 1, 3, 4, 7, 8, 9, 5, 6],
        'CYS': [0, 2, 9, 10, 1, 3, 4, 7, 5, 6, 8],
        'GLN': [0, 2, 15, 16, 1, 3, 4, 7, 10, 11, 12, 5, 6, 8, 9, 13, 14],
        'GLU': [0, 2, 13, 14, 1, 3, 4, 7, 10, 11, 12, 5, 6, 8, 9],
        'GLY': [0, 2, 5, 6, 1, 3, 4],
        'HIE': [0, 2, 15, 16, 1, 3, 4, 7, 8, 13, 9, 11, 5, 6, 14, 10, 12],
        'ILE': [0, 2, 17, 18, 1, 3, 4, 10, 6, 13, 5, 11, 12, 7, 8, 9, 14, 15, 16],
        'LEU': [0, 2, 17, 18, 1, 3, 4, 7, 9, 13, 5, 6, 8, 10, 11, 12, 14, 15, 16],
        'LYS': [0, 2, 20, 21, 1, 3, 4, 7, 10, 13, 16, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 19],
        'MET': [0, 2, 15, 16, 1, 3, 4, 7, 10, 11, 5, 6, 8, 9, 12, 13, 14],
        'PHE': [0, 2, 18, 19, 1, 3, 4, 7, 8, 16, 10, 14, 12, 5, 6, 9, 17, 11, 15, 13],
        'PRO': [0, 10, 12, 13, 11, 7, 4, 1, 8, 9, 5, 6, 2, 3],
        'SER': [0, 2, 9, 10, 1, 3, 4, 7, 5, 6, 8],
        'THR': [0, 2, 12, 13, 1, 3, 4, 10, 6, 5, 11, 7, 8, 9],
        'TRP': [0, 2, 22, 23, 1, 3, 4, 7, 8, 21, 10, 12, 19, 13, 17, 15, 5, 6, 9, 11, 20, 14, 18, 16],
        'TYR': [0, 2, 19, 20, 1, 3, 4, 7, 8, 17, 10, 15, 12, 13, 5, 6, 9, 18, 11, 16, 14],
        'VAL': [0, 2, 14, 15, 1, 3, 4, 6, 10, 5, 7, 8, 9, 11, 12, 13],
        'NME': [0, 2, 1, 3, 4, 5],
    } 
            
    output = []
    
    amino_acids = []
    residue_names = ['']
    atom_start = False
    
    with open(fpath, 'r') as f:
        lines = f.readlines()
        
        res_idx = None
        atoms = []
        
        for l in lines:
            cols = l.split()

            if not atom_start and not l.startswith('ATOM') and not l.startswith('HETATM'):
                output.append(l)
                continue
            
            atom_start = True
            
            if atom_start and not l.startswith('ATOM') and not l.startswith('HETATM'):
                continue
            
            if cols[4] != res_idx:
                res_idx = cols[4]
                
                residue_names.append(cols[3])
                amino_acids.append(atoms)
                atoms = []
            
            atoms.append(l)
            
        amino_acids.append(atoms)
        
    for residue, atoms in zip(residue_names[1:], amino_acids[1:]):
        if residue not in reorder_dict:
            output.extend(atoms)
        else:
            reordered_atoms = [atoms[i] for i in reorder_dict[residue]]
            output.extend(reordered_atoms)
        
    with open(fpath, 'w') as f:
        for l in output:
            f.write(l)


def record_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} takes {end - start} seconds")
        return result

    return wrapper


class RNGPool:
    def __init__(self, seed, shape, count):
        self.rng = np.random.default_rng(seed)

        self.pool = deque()
        self.shape = shape
        self.count = count

        self.fill()

    def fill(self):
        while len(self.pool) < self.count:
            self.pool.append(self.rng.standard_normal(self.shape))

    def drain(self):
        return self.pool.popleft()

    def standard_normal(self, size):
        if size == self.shape and len(self.pool):
            return self.drain()
        else:
            return self.rng.standard_normal(size)


class SkipCheckState:
    """Temporarily disables atoms.check_state so that it does not compare"""
    def __init__(self, atoms):
        self.skip_check_state = getattr(atoms, 'skip_check_state', False)
        self.atoms = atoms
        atoms.skip_check_state = True

    def __enter__(self, *_):
        pass

    def __exit__(self, *_):
        self.atoms.skip_check_state = self.skip_check_state

_fragment_step_time = None

_workqueue_instance = None

class WorkQueue():
    def __init__(self):
        global _workqueue_instance
        self.work: deque[Callable] = deque()
        if _workqueue_instance is not None:
            raise RuntimeError("There should be only one WorkQueue instance.")
        _workqueue_instance = self

    def __bool__(self):
        return True

    def __len__(self):
        return len(self.work)

    def submit(self, action):
        return self.work.append(action)

    def drain(self):
        while len(self.work):
            self.work.popleft()()

    @classmethod
    def finalise(cls):
        if _workqueue_instance:
            _workqueue_instance.drain()


def delay_work(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if _workqueue_instance:
            _workqueue_instance.submit(partial(f, *args, **kwargs))
        else:
            # root calc isn't workqueue-aware
            # do not delay the work because nobody will drain the works
            f(*args, **kwargs)

    return wrapper


class MDObserver:
    """An observer class, offering functions that can be attached to ASE MolecularDynamics object, notified at every step."""

    def __init__(self, a: Atoms, q: Atoms, md: MolecularDynamics, traj: TrajectoryWriter, rng: RNGPool, step_offset: int, write_solvent: bool = False):
        self.a = a
        self.q = q
        self.q_indices = np.arange(len(q), dtype=int)
        self.md = md
        self.traj = traj
        self.rng = rng
        self.step_offset = step_offset
        self.write_solvent = write_solvent
        self.prev_step_time = None
        self.copy = None
        self.x_rep_list = []
        self.v_rep_list = []
        self.frag_energy_list = []
        self.frag_force_list = []
        self.prot_energy_list = []
        self.prot_force_list = []
        self.pos_list = []


    def get_md_step(self):
        return self.step_offset + self.md.nsteps

    def save_traj_copy(self):
        self.copy = self.a.copy()

    @delay_work
    def write_traj(self):
        frame = self.copy if self.write_solvent else self.copy[self.q_indices]
        with SkipCheckState(frame):
            self.traj.write(frame)

    def printenergy(self):
        """
        Function to print the potential, kinetic and total energy
        """
        # per atom need / len(a)

        with SkipCheckState(self.a):
            cur_time = time.perf_counter()
            epot = self.a.get_potential_energy().item()
            ekin = self.a.get_kinetic_energy().item()
            temperature = self.a.get_temperature()
            if self.prev_step_time is None:
                steptime = ""
            else:
                steptime = f"time = {(cur_time - self.prev_step_time) * 1000:.1f}ms"
            if _fragment_step_time is None:
                frag_steptime = ""
            else:
                frag_steptime = f"fragtime = {_fragment_step_time * 1000:.1f}ms"
            self.prev_step_time = cur_time
            print(f"Step {self.get_md_step():d}:"
                  f"  Epot = {epot:.3f}eV"
                  f"  Ekin = {ekin:.3f}eV"
                  f" (T = {temperature:3.0f}K)"
                  f"  Etot = {epot+ekin:.3f}eV")
                #   f"  {steptime}"
                #   f"  {frag_steptime}")
    def get_hidden_embeddings(self):
        self.x_rep_list.append(self.q.x_rep)
        self.v_rep_list.append(self.q.v_rep)
        self.frag_energy_list.append(self.q.frag_energy)
        self.frag_force_list.append(self.q.frag_force)
        self.prot_energy_list.append(self.q.prot_energy)
        self.prot_force_list.append(self.q.prot_force)
        self.pos_list.append(self.q.get_positions())
        # np.save('x_rep.npy', np.stack(self.x_rep_list))
        # np.save('v_rep.npy', np.stack(self.v_rep_list))
        # np.save('frag_energy.npy', np.array(self.frag_energy_list))
        # np.save('frag_force.npy', np.stack(self.frag_force_list))
        # np.save('prot_energy.npy', np.array(self.prot_energy_list))
        # np.save('prot_force.npy', np.stack(self.prot_force_list))
        # np.save('prot_pos.npy', np.stack(self.pos_list))

        # np.save('exclude_pair.npy', self.q.exclude_index.numpy())
        # np.save('z.npy', self.q.get_atomic_numbers())


        np.savez('../pima_dataset.npz', z = self.q.get_atomic_numbers(), pos = np.stack(self.pos_list), prot_energy = np.array(self.prot_energy_list), prot_force = np.stack(self.prot_force_list),
                 frag_energy = np.array(self.frag_energy_list), frag_force = np.stack(self.frag_force_list), x_rep = np.stack(self.x_rep_list), v_rep = np.stack(self.v_rep_list),
                 exclude_pair = self.q.exclude_index.numpy(), adjmatrix = self.q.initial_mm_adjmatrix().numpy()
                 )
        
    def renameALdata(self):
        """
        Function to rename the active learning data
        """
        for file in os.listdir("../ActiveLearningData/"):
            if file.startswith("new_"):
                ori_name = file.split("new_")[1]
                os.rename(
                    "../ActiveLearningData/" + file,
                    "../ActiveLearningData/" + "Step" + str(self.get_md_step()) + "_" + ori_name,
                )

    @delay_work
    def fill_rng_pool(self):
        """
        Function to fill the RNG pool
        """
        self.rng.fill()


class PDBAnalyzer:
    def __init__(self, filename):
        self.filename = filename
        self.covalent_radii = {
            "H": 0.31,
            "C": 0.76,
            "N": 0.71,
            "O": 0.66,
            "P": 1.07,
            "S": 1.05,
        }
        self.atoms = self.parse_pdb()

    def parse_pdb(self):
        """Parse a PDB file and return a list of atoms and their coordinates."""
        atoms = []
        with open(self.filename, "r") as f:
            for line in f:
                if line.startswith("ATOM"):
                    atom_name = line[12:16].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    atoms.append((atom_name, np.array([x, y, z])))
        return atoms

    def compute_distance(self, atom1, atom2):
        """Compute the Euclidean distance between two atoms."""
        _, (_, pos1) = atom1
        _, (_, pos2) = atom2
        return np.linalg.norm(pos1 - pos2)

    def find_bonded_atoms(self, target_atom_name):
        """Find atoms that are bonded to a target atom type based on distance."""
        covalent_radius = self.covalent_radii.get(target_atom_name, 0)
        indexed_atoms = [
            (i, atom)
            for i, atom in enumerate(self.atoms)
            if atom[0].startswith(target_atom_name)
        ]
        bonded_atoms = []
        for i1, atom1 in indexed_atoms:
            for i2, atom2 in enumerate(self.atoms):
                if i1 != i2:
                    distance = self.compute_distance((i1, atom1), (i2, atom2))
                    atom2_radius = self.covalent_radii.get(atom2[0][0], 0)
                    idea_length = covalent_radius + atom2_radius
                    if distance <= idea_length + 0.2:
                        bonded_atoms.append((i1, i2, idea_length + 0.2, 15))
        assert len(indexed_atoms) == len(
            bonded_atoms
        ), "Hydrogen constraint: hydrogen covalent bonds != hydrogen num"
        return bonded_atoms


kcalmol2ev = 1 * (units.kcal / units.mol) / units.eV


def src_dir():
    return osp.abspath(osp.join(osp.dirname(__file__), ".."))


# helpers for common numpy -> torch operations
def numpy_to_torch(x: np.array, device: str):
    return torch.from_numpy(x).to(device)

def numpy_list_to_torch(x: list[np.array], device: str):
    return torch.from_numpy(np.concatenate(x)).to(device)


# wrapper for serial/parallel execution
def execution_wrapper(f_args: list[Any], concurrent: bool):
    if concurrent is True:
        with ThreadPoolExecutor(len(f_args)) as executor:
            futures = [executor.submit(f, *args) for f, *args in f_args]

        return [f.result() for f in futures]
    else:
        return [f(*args) for f, *args in f_args]


# ORCA calulator with default settings
ORCABLOCK = \
f"""
%maxcore 1000
%pal nprocs {get_physical_core_count()} end
%scf
Convergence Tight
ConvForced true
MaxIter 300
end
"""

def get_default_orca_calculator(name: str, charge: int, mult: int):
    assert charge is not None
    assert mult is not None

    return ORCA(
        label=f"orcacalc_{name}",
        orcasimpleinput="M062X 6-31G* defgrid3",
        charge=charge,
        mult=mult,
        orcablocks=ORCABLOCK,
    )
