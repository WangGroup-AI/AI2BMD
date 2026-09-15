import time
from typing import Union

from ase import Atoms
from ase.calculators.calculator import Calculator, compare_atoms
from ase.calculators.amber import Amber

import utils.utils as utils
from AIMD.protein import Protein
from Calculators.device_strategy import DeviceStrategy
from Calculators.bonded import DLBondedCalculator
from Calculators.combiner import DipeptideCombiner
from Calculators.nonbonded import MMNonBondedCalculator
from Calculators.pima import PimaNonBondedCalculator
from Calculators.pme import PMENonBondedCalculator
from utils.utils import WorkQueue, execution_wrapper


def check_state(self: Calculator, atoms: Union[Atoms, Protein], tol=1e-15):
    r"""
    Check for any system changes since last calculation.
    - Skips the check and assume no changes (and therefore no calculations), if 'atoms.check_state' is false.
    """
    
    # don't remove these for now -- prints the stack at check_state
    # print("check_state")
    # id2name = dict([(th.ident, th.name) for th in threading.enumerate()])
    # code = []
    # for threadId, stack in sys._current_frames().items():
    #     code.append("\n# Thread: %s(%d)" % (id2name.get(threadId,""), threadId))
    #     for filename, lineno, name, line in traceback.extract_stack(stack):
    #         code.append('File: "%s", line %d, in %s' % (filename, lineno, name))
    #         if line:
    #             code.append("  %s" % (line.strip()))
    # print("\n".join(code))
    if getattr(atoms, "skip_check_state", False):
        # force turn off check_state
        return []
    else:
        return compare_atoms(self.atoms, atoms, tol=tol, excluded_properties=set(self.ignored_changes))


def patch_check_state():
    Calculator.check_state = check_state


def read_results(self: Amber):
    """read energy and forces"""
    self.read_energy(filename=self.eoutfile)
    self.read_forces(filename=self.foutfile)


def add_e_f_args(self: Amber, eoutfile: str, foutfile: str):
    """add missing arguments in ase interface"""
    self.eoutfile = eoutfile
    self.foutfile = foutfile

    self.command += f" -e {eoutfile} -frc {foutfile}"

def patch_amber_calc():
    Amber.read_results = read_results
    Amber.add_e_f_args = add_e_f_args


nonbonded_calcs = {
    'mm': MMNonBondedCalculator,
    'pme': PMENonBondedCalculator,
    'pima': PimaNonBondedCalculator,
}

class FragmentCalculator(Calculator):
    r"""
    FragmentCalculator is a universal calculator for dipeptide fragments,
    including Bonded Calculator, QM/DL Calculator, and MM (Non-Bonded) Calculator.

    Parameters:
    -----------
        properties: list[str]
            Targets of the calculation.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, is_root_calc, nbcalc_type, **kwargs):
        super().__init__(**kwargs)

        self.is_root_calc = is_root_calc

        if self.is_root_calc is True:
            self.work_queue = WorkQueue()

        # * set bonded calculator
        self.bonded_calculator = DLBondedCalculator(**kwargs)

        # * set non-bonded calculator
        device = DeviceStrategy.get_non_bonded_device()
        if nbcalc_type != 'pima':
            self.nonbonded_calculator = nonbonded_calcs[nbcalc_type](device=device)
        else:
            self.nonbonded_calculator = PimaNonBondedCalculator(kwargs['ckpt_path'], device)
        self.nbcalc_type = nbcalc_type

        # * set fragment strategy
        self.concurrent = DeviceStrategy.fragment_strategy()

        # * set combiner
        self.combiner = DipeptideCombiner()

    def calculate(self, atoms, properties, system_changes):
        time_begin = time.perf_counter()

        if self.is_root_calc:
            Calculator.calculate(self, atoms, properties, system_changes)
            self.work_queue.drain()

        f_args = [
            (self.bonded_calculator,    atoms),
            (self.nonbonded_calculator, atoms),
        ]
        if self.nbcalc_type != 'pima':
            energy, forces = zip(*execution_wrapper(f_args, self.concurrent))
            energy = self.combiner.energy_combine(*energy)
            forces = self.combiner.forces_combine(*forces)
        else:
            #energy_frag, forces_frag = zip(*execution_wrapper(f_args, self.concurrent))
            energy_frag, forces_frag = self.bonded_calculator(atoms)
            energy_nb, forces_nb = self.nonbonded_calculator(atoms)

            energy = self.combiner.energy_combine(energy_frag, energy_nb)
            forces = self.combiner.forces_combine(forces_frag, forces_nb)
        atoms.prot_energy = energy
        atoms.prot_force = forces

        self.results = {
            "energy": energy,
            "forces": forces,
        }

        time_end = time.perf_counter()
        utils._fragment_step_time = time_end - time_begin
