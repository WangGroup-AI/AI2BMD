import atexit
import os
import subprocess
from logging import getLogger

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.units import kcal, mol

from AIMD import envflags
from AIMD.preprocess import run_command
from Calculators.async_utils import AsyncServer


_tinker_instance_id = 0

class TinkerAsyncCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, pdb_file, utils_dir, devices: list[str], **kwargs):
        super().__init__(**kwargs)
        self.pdb_file = os.path.abspath(pdb_file)
        self.utils_dir = utils_dir
        self.prot_name = os.path.basename(pdb_file)[:-4]
        self.devices = devices
        self._tinker_proc = None
        self.atoms: Atoms
        self.server = AsyncServer("tinker")
        self.logger = getLogger("Tinker-Proxy")

        global _tinker_instance_id
        self.instance_id = _tinker_instance_id
        _tinker_instance_id += 1

        if any(map(lambda x: x.startswith('cuda'), devices)):
            self.command_dir = '/usr/local/gpu-m'
        else:
            self.command_dir = '/usr/local/cpu-m'

    def _start_tinker(self):
        print("Initializing tinker...")
        self._write_key(self.atoms)
        self.logger.debug('Key file written!')
        if not os.path.exists(f'{self.prot_name}.xyz'):
            self._generate_xyz_template()
            self.logger.debug('XYZ template generated!')

        # bind devices
        envs = os.environ.copy()
        gpus = []
        for device in self.devices:
            if device.startswith("cuda"):
                _, nr = device.split(':')
                gpus.append(nr)
        if len(gpus) > 1:
            assert self.instance_id < len(gpus)
            envs["CUDA_VISIBLE_DEVICES"] = gpus[self.instance_id]
        elif len(gpus) == 1:
            envs["CUDA_VISIBLE_DEVICES"] = gpus[0]

        outfd = None
        log_args = "<< _EOF\n" if envflags.DEBUG_RC else f" > dynamic{self.instance_id}.log << _EOF\n"
        self._tinker_proc = subprocess.Popen(
            f"{self.command_dir}/tinker9 ai2bmd {self.prot_name}.xyz -k {self.prot_name}.key"
            f"{log_args}"
            f"{self.server.socket_path}\n"  # unix socket for IPC
            f"_EOF",
            shell=True,
            env=envs,
            stdout=outfd,
            stderr=outfd,
        )
        self.logger.debug('Waiting for Tinker to start...')
        self.server.accept()
        sz = np.empty(shape=[2], dtype='int32')
        self.server.recv(sz)

        self.sz_real = sz[0]
        self.sz_energy = sz[1]
        self.sz_double = 8 # hard-coded

        self.n_atoms = len(self.atoms)
        self._tinker_istep = 1
        self._tinker_sbuf = self.server.makebuf([1], 'int32')
        self._tinker_ebuf = self.server.makebuf([1], self.sz_energy)
        self._tinker_xbuf = self.server.makebuf([self.n_atoms], self.sz_real)
        self._tinker_ybuf = self.server.makebuf([self.n_atoms], self.sz_real)
        self._tinker_zbuf = self.server.makebuf([self.n_atoms], self.sz_real)
        self._tinker_gxbuf = self.server.makebuf([self.n_atoms], self.sz_double)
        self._tinker_gybuf = self.server.makebuf([self.n_atoms], self.sz_double)
        self._tinker_gzbuf = self.server.makebuf([self.n_atoms], self.sz_double)

        self.logger.debug(f'Tinker connected! tinker::real size={self.sz_real} tinker::energy_prec size={self.sz_energy}')
        atexit.register(self._shutdown)

    def _shutdown(self):
        if self.server is None or self._tinker_proc is None:
            return
        self.logger.debug("Shutting down tinker...")
        # send the shutdown command, wait for 500ms, and if the program still does not exit, kill it
        self._tinker_sbuf[0] = -1
        self.server.send(self._tinker_sbuf)
        try:
            self._tinker_proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired as err:
            self._tinker_proc.kill()
        self.server.close()
        self.logger.debug("Tinker shutdown complete.")

    def _write_key(self, atoms):
        with open(f'{self.prot_name}.key', 'w') as f:
            f.write(f"""
                parameters {self.utils_dir}/amoebabio18.prm
                neighbor-list
                a-axis {atoms.cell[0, 0]}
                b-axis {atoms.cell[1, 1]}
                c-axis {atoms.cell[2, 2]}
                cutoff 12
                vdw-cutoff 12
                integrator stochastic
                friction 1.0
                save-force
                ewald
                ewald-cutoff 7.0
                fft-package FFTW
                polarization mutual
                polar-eps 0.01
            """)

    def _generate_xyz_template(self):
        run_command(
            f"cp {self.pdb_file} . && "
            f"sed -i '/TER/d' {self.prot_name}.pdb && "
            f"{self.command_dir}/pdbxyz8 {self.prot_name}.pdb",
            self.directory
        )

    def _sync_tinker(self):
        self._tinker_sbuf[0] = self._tinker_istep
        self.server.send(self._tinker_sbuf)
        self.server.recv(self._tinker_sbuf)
        if self._tinker_sbuf[0] != self._tinker_istep:
            raise Exception("tinker_async: status decynchronized")
        self._tinker_istep += 1

    def _write_xyz(self):
        pos = self.atoms.get_positions()
        
        # translate coordinates according to mass center
        atom_types = self.atoms.get_chemical_symbols()
        atomic_masses = {
    'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'P': 30.974, 'S': 32.06, 'F': 18.998, 'CL': 35.453, 'NA': 22.990
}
        atoms = []
        masses = []
        
        for atom_idx,atom_type in enumerate(atom_types):
            mass = atomic_masses.get(atom_type, 0)
            masses.append(mass)
            atoms.append(pos[atom_idx])
        atoms = np.array(atoms)
        masses = np.array(masses)
        
        # get mass center
        total_mass = np.sum(masses)
        mass_center = np.sum(atoms * masses[:, np.newaxis], axis=0) / total_mass

        # translate atoms
        pos -= mass_center
        
        # convert to tinker::real
        self._tinker_xbuf[:] = pos[:,0]
        self._tinker_ybuf[:] = pos[:,1]
        self._tinker_zbuf[:] = pos[:,2]
        self.server.send(self._tinker_xbuf)
        self.server.send(self._tinker_ybuf)
        self.server.send(self._tinker_zbuf)

    def _read_result(self):
        self.server.recv(self._tinker_ebuf)
        self.server.recv(self._tinker_gxbuf)
        self.server.recv(self._tinker_gybuf)
        self.server.recv(self._tinker_gzbuf)
        energy = self._tinker_ebuf[0] * (kcal / mol)
        grad = np.stack([self._tinker_gxbuf, self._tinker_gybuf, self._tinker_gzbuf], axis=-1)
        return energy, -grad * (kcal / mol)

    def calculate(self, atoms, properties, system_changes):
        # Calculator.calculate(self, atoms, properties, system_changes)

        self._sync_tinker()
        self._write_xyz()
        energy, forces = self._read_result()
        self.results["energy"] = energy
        self.results["forces"] = forces
