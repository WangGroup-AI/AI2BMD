import functools
import operator
import shutil
from abc import ABC
from os import path as osp

import numpy as np
import torch
import torch_scatter
from ase import Atoms
from ase.calculators.amber import Amber
from ase.units import eV, kcal, mol

import utils
from AIMD import arguments
from AIMD.fragment import FragmentData, FragmentInfo
from utils.reference import fragment_covalent_bonds, fragment_info
from utils.reference import fragment_covalent_bonds, fragment_info, fragment_atoms_str
from utils.utils import get_default_orca_calculator, numpy_list_to_torch, numpy_to_torch


class BaseAggregateSentry(ABC):
    def __init__(self, device: str):
        self.device = device

class PairwiseDistAggregateSentry(BaseAggregateSentry):
    r"""
    Assert the validity of each residue in the system by the pairwise distance
    between every pair of non-covalent-bonded atoms.
    """

    def __init__(self, fragments: FragmentData, device: str):
        super().__init__(device)

        def _calculate_covalent_bond_mask(size, left_idx, right_idx, block):
            """
            calculates the covalent bond masks for a dipeptide.
            """
            mask = np.zeros((block, block), dtype=bool)
            mask[:size, :size] = 1
            mask[np.tril_indices(block)] = 0
            mask[left_idx, right_idx] = 0

            return mask

        sizes = fragments.end - fragments.start
        block = np.max(sizes)
        param = [fragment_covalent_bonds[name] for name in fragments.sym]

        # calculate scatter indices for padding
        index = np.arange(fragments.end[-1])
        delta = np.zeros_like(index)
        delta[fragments.start[1:]] = (block - sizes)[:-1]
        index = index + np.cumsum(delta)

        index = numpy_to_torch(index, self.device)

        # shape of batched atom pairs in fragment
        shape = (len(fragments), block, 3)

        # calculate padded masks to exclude covalent bond distances
        masks = numpy_to_torch(
            np.stack([
                _calculate_covalent_bond_mask(s, p[0], p[1], block)
                for s, p in zip(sizes, param)
            ]),
            self.device,
        )

        # calculate batch indices for segment_coo reduction
        start = [(s * (s - 1) // 2) - len(p[0]) for s, p in zip(sizes, param)]
        start = numpy_to_torch(np.cumsum(start, dtype=int), device=self.device)
        batch = torch.zeros((start[-1],), dtype=torch.int, device=self.device)
        batch = torch.cumsum(batch.scatter(0, start[:-1], 1), dim=0)

        self.index = index
        self.shape = shape
        self.masks = masks
        self.batch = batch

    def __call__(self, positions: torch.tensor, energy: torch.tensor, forces: torch.tensor):
        return self._batch_pdist2(positions, self.index, self.shape, self.masks, self.batch)

    def _batch_pdist2(
        self,
        p: torch.tensor,
        index: torch.tensor,
        shape: tuple[int, int, int],
        mask: torch.tensor,
        batch: torch.tensor,
    ):
        x = torch.zeros(shape, dtype=torch.float, device=self.device).view(-1, 3)
        x = torch_scatter.scatter(p, index, dim=0, out=x).reshape(shape)
        x2 = torch.einsum('ijk,ijk->ij', x, x).unsqueeze(-1)
        d2 = torch.baddbmm(x2.transpose(-2, -1), x, x.transpose(-2, -1), alpha=-2).add_(x2)
        min_d2 = torch_scatter.segment_coo(d2[mask], batch, reduce='min')

        return (min_d2 <= (1.35 * 1.35)).to(dtype=torch.int)

class DLNaNAggregateSentry(BaseAggregateSentry):
    r"""
    Assert the validity of all residues in the system by checking for NaNs in
    atom properties.
    """

    def __init__(self, fragments: FragmentData, device: str):
        super().__init__(device)

        self.index = numpy_to_torch(fragments.batch, device)

    def __call__(self, positions: torch.tensor, energy: torch.tensor, forces: torch.tensor):
        index = self.index

        pos_nan = torch.isnan(positions).to(dtype=torch.int)
        energy_nan = torch.isnan(energy).to(dtype=torch.int)
        forces_nan = torch.isnan(forces).to(dtype=torch.int)

        scalar_nan = energy_nan.squeeze()
        vector_nan = torch.sum(pos_nan, axis=1) + torch.sum(forces_nan, axis=1)
        vector_nan = torch_scatter.segment_coo(vector_nan, index, reduce="sum")

        return scalar_nan + vector_nan

class BondLenAggregateSentry(BaseAggregateSentry):
    r"""
    Assert the validity of all residues in the system by checking the bond lengths.
    """

    def __init__(self, fragments: FragmentData, device: str):
        super().__init__(device)

        covalent_l = []
        covalent_r = []
        amber_covalent_bond_len_lower = []
        amber_covalent_bond_len_upper = []

        for name in fragments.sym:
            params = fragment_covalent_bonds[name]

            covalent_l.append(params[0])
            covalent_r.append(params[1])
            amber_covalent_bond_len_lower.append((params[2] - 0.3) ** 2)
            amber_covalent_bond_len_upper.append((params[2] + 0.3) ** 2)

        length = numpy_to_torch(fragments.start, device)
        length = torch.diff(length.to(dtype=torch.int))

        sizes = numpy_to_torch(np.cumsum([len(v) for v in covalent_l]), device)
        batch = torch.zeros((sizes[-1],), dtype=torch.int, device=device)
        batch = torch.cumsum(batch.scatter(0, sizes[:-1], 1), dim=0)

        self.index = batch

        offset = torch.zeros((sizes[-1],), dtype=torch.int, device=device)
        offset = torch.cumsum(offset.put_(sizes[:-1], length, accumulate=True), dim=0)

        self.covalent_l = numpy_list_to_torch(covalent_l, self.device) + offset
        self.covalent_r = numpy_list_to_torch(covalent_r, self.device) + offset
        self.amber_covalent_bond_len_lower = numpy_list_to_torch(amber_covalent_bond_len_lower, self.device)
        self.amber_covalent_bond_len_upper = numpy_list_to_torch(amber_covalent_bond_len_upper, self.device)

    def __call__(self, positions: torch.tensor, energy: torch.tensor, forces: torch.tensor):
        relative_pos = positions[self.covalent_l] - positions[self.covalent_r]
        d2_covalent_bond = torch.einsum('ij,ij->i', relative_pos, relative_pos)

        lower_bound = d2_covalent_bond < self.amber_covalent_bond_len_lower
        upper_bound = d2_covalent_bond > self.amber_covalent_bond_len_upper

        bounds = lower_bound.to(dtype=torch.int) + upper_bound.to(dtype=torch.int)

        return torch_scatter.segment_coo(bounds, self.index, reduce="sum")


aggregate_sentry_map = {
    "PairwiseDist": PairwiseDistAggregateSentry,
    "DLNaN": DLNaNAggregateSentry,
    "BondLen": BondLenAggregateSentry,
}


class FragmentMonitorError(Exception):
    """Raised when there are errors in the MLFF computation for fragments.

    stepback_fn: (int, int) -> None, is a callback function from exception handler to the thrower to step back the monitors.
        The first integer argument 'back_steps' is the number of steps that we want to rollback, _in addition to_ the
        current step (which will be discarded).
        The second integer argument 'cur_step' is the step where the error happened in the MD simulation.
    
    """

    def __init__(self, stepback_fn):
        super().__init__()
        self.stepback_fn = stepback_fn


class FragmentMonitor:
    """Fragment monitor for individual fragments."""

    def __init__(self, type: str, name: str, idx: int, start: int, end: int):
        self.type = type
        self.name = name
        self.idx = idx
        self.start = start
        self.end = end

        self.fallback_counter = 0
        self.uncertainty_counter = 0
        self.uncertainty_current_step = False
        self.bondlen_count = 0

    def uncertainty_active(self, uncert):
        if uncert > 0.0001:  #decrease the threshold to 0.0001 in order to meet the demand of sensitivity
            # TODO log this event
            print(f'{self.idx} {self.name} unseen: {uncert}')
            self.uncertainty_current_step = True
            return True
        else:
            return False
        
    def bondlen_info(self):
        self.bondlen_count += 1 #count how often this event occurs
        print(f'{self.idx} {self.name} bond length warning: {self.bondlen_count}')
        return np.random.rand() #return a random number to decide whether to check the error

    def fallback_active(self):
        return self.fallback_counter != 0

    def need_check(self):
        return self.fallback_counter == 0 and self.uncertainty_counter == 0

    def step_forward(self):
        if self.fallback_counter:
            self.fallback_counter -= 1
        if self.uncertainty_counter:
            self.uncertainty_counter -= 1
        if self.uncertainty_current_step:
            self.uncertainty_current_step = False
            self.uncertainty_counter += 1

    # assume n_steps = 3
    # counters in these example are logical values for illustration purpose.
    # actual counter needs compensation to adapt to ASE behavior. see "NOTE" below.
    #
    # situation 1: current step is mlff, but error occurs
    # current timline:    ......x     [x=error happened]
    # rollback timeline:  ...fff.     [f=fallback mode, new counter=3]
    # simulator restart:  ---^  ^
    # fallback deactivated: ----'
    #
    # situation 2: current step is already fallback, error occured in another fragment
    # previous timline:   ......x     [x=error happened]
    # current timline:    ...ffx      [remaining counter = 1]
    # rollback timeline:  ..ffff.     [f=fallback mode, new counter=4]
    def step_backward(self, n_steps: int):
        # NOTE: ASE MolDyn objects will call atoms.get_forces() before commiting the first step to
        # compute some initial properties. 
        # In our situation, the first call to get_forces() will drain 1 from monitor counters, and
        # therefore, we need to push back n_steps + 1 in the monitors to compensate that.
        self.fallback_counter += n_steps + 1

    def fallback_calculate(self, atoms: Atoms, energy: float, forces: np.ndarray):
        self.write_xyz(atoms)
        if arguments.get().rollback_mm_only:
            energy_fallback, forces_fallback = self.fallback_mm_calculate(atoms)
        else:
            try:
                energy_fallback, forces_fallback = self.fallback_qm_calculate(atoms)
                energy_mae = np.abs(energy.cpu() - energy_fallback).item() * eV / (kcal / mol)
                forces_mae = np.abs(forces.cpu() - forces_fallback).mean() * eV / (kcal / mol)
                print(forces_mae)
                try:
                    shutil.copy(f"orcacalc_{self.name}.inp", f"/mnt/aimd4b/acl_data_1115/new_{self.name}{self.idx}_E{energy_mae:.3f}_F{forces_mae:.3f}.inp") #..
                    shutil.copy(f"orcacalc_{self.name}.engrad", f"/mnt/aimd4b/acl_data_1115/new_{self.name}{self.idx}_E{energy_mae:.3f}_F{forces_mae:.3f}.engrad")
                except:
                    pass
                shutil.copy(f"orcacalc_{self.name}.inp", f"../ActiveLearningData/new_{self.name}{self.idx}_E{energy_mae:.3f}_F{forces_mae:.3f}.inp")
                shutil.copy(f"orcacalc_{self.name}.engrad", f"../ActiveLearningData/new_{self.name}{self.idx}_E{energy_mae:.3f}_F{forces_mae:.3f}.engrad")
            except RuntimeError:
                print(f"[QMFail]: {self.name}{self.idx}")
                energy_fallback, forces_fallback = self.fallback_mm_calculate(atoms)

        return energy_fallback, forces_fallback

    def fallback_qm_calculate(self, atoms: Atoms):
        charge = fragment_info[self.name][1]
        mult = fragment_info[self.name][2]
        atoms.calc = get_default_orca_calculator(self.name, charge, mult)
        forces_qm = atoms.get_forces()
        energy_qm = atoms.get_potential_energy()
        return energy_qm, forces_qm

    def fallback_mm_calculate(self, atoms: Atoms):
        calc = Amber(
            amber_exe='sander -O ',
            infile=osp.join(utils.src_dir(), 'Fragmentation', 'mm.in'),
            outfile=osp.join('mm.out'),
            topologyfile=osp.join(utils.src_dir(), 'Fragmentation', 'prmtop', f'{fragment_info[self.name][0]}.prmtop'),
        )
        calc.add_e_f_args('mden', 'mdfrc')

        atoms.calc = calc
        forces_mm = atoms.get_forces()
        energy_mm = atoms.get_potential_energy()
        return energy_mm, forces_mm
    def write_xyz(self,atoms):
        # visualize the dipeptide that triggers active learning
        positions = atoms.get_positions()
        with open(f"../ActiveLearningData/new_{self.name}{self.idx}_calc.xyz", "w") as f:
            f.write(f"{atoms.get_positions().shape[0]}\n")
            f.write(f"0 0\n")
            for z_, p_ in zip(fragment_atoms_str[self.name], positions):
                f.write(z_ + "   " + str(p_[0]) + " " + str(p_[1]) + " " + str(p_[2]) + "\n")


monitor_types = ["dipeptides", "ACE_NMEs"]

class MonitorCollection:
    def __init__(self, fragments: FragmentData, checks: list[str], device: str):
        self.device = device

        # dipeptide and ACE-NME fragments are interleaved, dipeptides first
        self.monitors = [
            FragmentMonitor(
                # even indices are dipeptides, odd indices are ACE-NMEs
                type=monitor_types[idx % 2],
                name=fragments.sym[idx],
                # internal index increments separately for dipeptides/ACE-NMEs
                idx=idx // 2,
                start=fragments.start[idx],
                end=fragments.end[idx],
            )
            for idx in range(len(fragments))
        ]

        self.sentries = [
            aggregate_sentry_map[check](fragments, device) for check in checks
        ]

        self.index = numpy_to_torch(fragments.batch, device=device)

    def fallback(self, fragments: FragmentData, energy: torch.tensor, forces: torch.tensor, uncert: torch.tensor = None):
        if uncert is not None:
            uncert = torch_scatter.scatter(uncert, self.index, dim=0, reduce='max')
        else:
            uncert = torch.zeros_like(energy)

        for idx, monitor in enumerate(self.monitors):
            uncert_val = uncert[idx]
            if monitor.fallback_active() or monitor.uncertainty_active(uncert_val):
                atoms = fragments.get_atoms(idx)
                energy_dl = energy[idx]
                forces_dl = forces[monitor.start:monitor.end]

                energy_fallback, forces_fallback = monitor.fallback_calculate(atoms, energy_dl, forces_dl)
                #energy_fallback = numpy_to_torch(energy_fallback, device=self.device)
                forces_fallback = numpy_to_torch(forces_fallback, device=self.device)   #convert to torch.tensor, or will raise error
                energy[idx] = energy_fallback
                forces[monitor.start:monitor.end] = forces_fallback

    def check(self, fragments: FragmentData, energy: torch.tensor, forces: torch.tensor):
        monitor_errors = []

        if len(self.sentries):
            positions = numpy_to_torch(fragments.pos, self.device)

            result = functools.reduce(operator.add, (f(positions, energy, forces) for f in self.sentries))
            result = torch.nonzero(result).squeeze(-1)

            for idx in result.detach().cpu().numpy():
                if self.monitors[idx].need_check():
                    monitor_errors.append(self.monitors[idx])

        return monitor_errors
    def check_idx(self,idx):
        # Just want to check the relationships between monitor's index and their order, no practical use
        print(idx)
        print(self.monitors[idx].idx)
        print(self.monitors[idx].name)

    def step_forward(self):
        for monitor in self.monitors:
            monitor.step_forward()

    def step_backward(self, n_steps: int):
        for monitor in self.monitors:
            if monitor.fallback_active():
                monitor.step_backward(n_steps)
