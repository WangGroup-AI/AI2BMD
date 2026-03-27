import os
import shutil
import numpy as np
import time
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import LBFGS, BFGS, BFGSLineSearch
from ase.optimize.minimahopping import MinimaHopping
from ase.constraints import Hookean
from numpy.linalg import eigh

from AIMD import arguments 
from AIMD.protein import Protein
from Calculators.device_strategy import DeviceStrategy
from Calculators.fragment import FragmentCalculator
from Calculators.visnet_calculator import ViSNetCalculator
from utils.pdb import read_protein

args = arguments.init()

class AI2BMDOptimizer:
    def __init__(self, atoms, fmax=0.015, max_cycles=100, max_step_size=0.1):
        self.atoms = atoms
        self.fmax = fmax
        self.max_cycles = max_cycles
        self.max_step_size = max_step_size        
        self.traj_file = "optimization_trace.xyz"
        self.best_file = "best_structure.xyz"

    def run(self):
        all_best_f = 999.0
        for cycle in range(self.max_cycles):
            print(f"\n🚀 [Cycle {cycle+1}] BFGS Optimization")
            
            curr_max_step_size = self.max_step_size
            if cycle == 0:
                opt = BFGS(self.atoms, alpha=40.0, maxstep=self.max_step_size)
            else:
                opt = BFGSLineSearch(self.atoms, alpha=40.0, maxstep=curr_max_step_size)

            stagnant_steps = 0
            best_f = 999.0
            
            for step in opt.irun(fmax=self.fmax):
                write(self.traj_file, self.atoms, append=True)
                curr_f = self.cal_fmax()
                
                if curr_f < best_f:
                    best_f = curr_f
                    print(f"✨ Update local minimum (fmax={best_f:.4f})")
                    if best_f < all_best_f:
                        all_best_f = best_f
                        # Reduce step size for finer optimization when global best is found
                        curr_max_step_size = curr_max_step_size / 10
                        write(self.best_file, self.atoms)
                        print(f"✨ Update global minimum (fmax={all_best_f:.4f})")

                # Check for plateau/stagnation
                if (curr_f - best_f) > 1e-6:
                    stagnant_steps += 1
                else:
                    stagnant_steps = 0
                
                # If stagnant for 100 steps, force a break to trigger breakout logic
                if stagnant_steps >= 100:
                    print(f"⚠️  BFGS trapped in local minimum (fmax={best_f:.4f}). Triggering SGD breakout...")
                    break
            else:
                # If irun finishes normally (converged to fmax)
                print("🎉 Structure successfully converged!")
                return True
            
            print(f"✨ Current global minimum (fmax={all_best_f:.4f})")
            # --- SGD Breakout Phase ---
            print(f"🛠️  [Breakout] Performing forced coordinate restructuring...")
            self.shake_and_push(steps=20, lr=0.01)

        print("❌ Maximum cycles reached without convergence. Please check the stability of the ML potential for this configuration.")
        return False
    
    def cal_fmax(self):
        forces = self.atoms.get_forces()
        fmax = np.sqrt((forces**2).sum(axis=1).max())
        return fmax
    
    def shake_and_push(self, steps=20, lr=0.1): 
        """Perturbs the structure using SGD to escape local minima"""
        for i in range(steps):
            forces = self.atoms.get_forces()
            fmax = self.cal_fmax()
            actual_lr = lr
            
            # Adaptive learning rate based on force magnitude
            if fmax > 100.0:
                actual_lr = lr / 100.0
            elif fmax > 10.0:
                actual_lr = lr / 10.0
                
            pos = self.atoms.get_positions()
            move = actual_lr * forces
            
            # Constraint: Limit the maximum displacement per atom
            max_allowed = 0.05 
            lengths = np.linalg.norm(move, axis=1)
            mask = lengths > max_allowed
            if mask.any():
                move[mask] *= (max_allowed / lengths[mask])[:, np.newaxis]
            
            # Add small random noise to break symmetry
            noise = np.random.normal(0, 0.001, pos.shape)
            
            self.atoms.set_positions(pos + move + noise)
            
            if i % 10 == 0:
                print(f"   - SGD Pushing... Step {i}, Current fmax: {fmax:.4f}")


class ProteinOptimizer:
    def __init__(
        self, prot: Protein, log_path: str, fmax: float = 0.015
    ) -> None:
        self.prot = prot
        self.log_path = log_path
        self.optimization_save_path = os.path.join(log_path, "OptimizationResults")
        os.makedirs(self.optimization_save_path, exist_ok=True)
        
        self.nowat_pdb = self.prot.nowater_PDB_path
        self.fmax = fmax
        self.prot.set_pbc(True)
        self.qmcalc = None

    def get_qm_idx(self):
        """Get indices of all atoms for constraints or analysis"""
        return list(range(len(read_protein(self.nowat_pdb))))

    def need_fragmentation(self):
        """Check if the current calculator requires fragmentation"""
        return isinstance(self.qmcalc, FragmentCalculator)

    def initialize_fragcalc(self):
        """Initialize FragmentCalculator and assign device resources"""
        if self.need_fragmentation():
            # Execute fragmentation logic
            self.qmcalc.bonded_calculator.fragment_method.fragment(self.prot)
            self.qmcalc.nonbonded_calculator.set_parameters(self.prot)

            # Get start and end indices of fragments
            start, end = self.prot.fragments_start, self.prot.fragments_end
        else:
            # Non-fragmented mode (e.g., Global ViSNet or monomer calculation)
            start, end = [0], [len(self.prot)]

        # Set GPU/CPU workload partitions based on fragments
        DeviceStrategy.set_work_partitions(start, end)

    def make_calculator(self, is_root_calc: bool, **kwargs):
        """
        Factory method: Determines which AI force field or Fragment Calculator to use
        """
        mode = arguments.get().mode
        if mode == "fragment":
            return FragmentCalculator(is_root_calc=is_root_calc, **kwargs)
        elif mode == "visnet":
            # Assuming ViSNetCalculator is compatible with FragmentCalculator logic
            return ViSNetCalculator(is_root_calc=is_root_calc, **kwargs)
        else:
            raise ValueError(f"Unsupported mode: {mode}. Please check your arguments.")

    def set_calculator(self, **kwargs) -> None:
        """Set up the calculator and perform initialization"""
        # Change to working directory to prevent temporary file clutter
        os.chdir(self.optimization_save_path)
        
        self.qmcalc = self.make_calculator(is_root_calc=True, **kwargs)
        self.prot.set_calculator(self.qmcalc)
        
        # Initialize fragments and DeviceStrategy allocation
        self.initialize_fragcalc()

    def optimize(self, prot_name: str, max_cycles: int = 1000, max_step_size: float = 0.1):
        print(f"--- Starting Structure Optimization for {prot_name} ---")
        print(f"Mode: {arguments.get().mode} | Target fmax: {self.fmax} eV/A")
        
        opt = AI2BMDOptimizer(atoms=self.prot, fmax=self.fmax, max_cycles=max_cycles, max_step_size=max_step_size)

        try:
            opt.run()
            print("Optimization successfully converged!")

            if args.vib:
                from ase.vibrations import Vibrations
                vib_name = 'vib_best_structure'
                vib = Vibrations(self.prot, name=vib_name)
                
                print("\n--- Starting Hessian Matrix Calculation (Finite Displacement Method) ---")
                vib.run()

                # Output vibration summary
                print("\n--- Vibrational Frequency Results ---")
                vib.summary()

        except Exception as e:
            print(f"Optimization encountered an error: {e}")
