import itertools
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from os import path as osp

import numpy as np

from AIMD import arguments
from AIMD.fragment import FragmentData
from AIMD.protein import Protein
from Calculators import error_injection
from Calculators.active_learning import ActiveLearningData
from Calculators.combiner import DipeptideBondedCombiner
from Calculators.device_strategy import DeviceStrategy
from Calculators.error_injection import inject_force_errors
from Calculators.monitor import FragmentMonitorError, MonitorCollection
from Calculators.visnet_calculator import ViSNetModelLike, get_visnet_model
from Fragmentation import DistanceFragment
from utils.utils import numpy_to_torch


class DLBondedCalculator:
    r"""
    DLBondedCalculator is a dipeptide bonded calculator based on
     DL calculations supported by ViSNet.
    """

    def __init__(
        self,
        ckpt_path: str,
        ckpt_type: str,
        checks: list[str],
        **kwargs,
    ) -> None:
        self.models: list[ViSNetModelLike] = []
        self.ckpt_path = ckpt_path
        self.ckpt_type = ckpt_type
        assert self.ckpt_type in ["nll", "new", "old"], "invalid ckpt_type"
        self.checks = checks

        # * set fragment method and combiner
        self.fragment_method = DistanceFragment()
        self.combiner = DipeptideBondedCombiner()

        self.monitors = None
        self.history_ringbuf = deque()
        self.history_size = arguments.get().rollback_steps
        self.history_enable = arguments.get().rollback_save

        print("Loading models...")
        model_path = osp.join(self.ckpt_path, f"visnet-uni-{self.ckpt_type}.ckpt")
        self.models = [
            get_visnet_model(model_path, ckpt_type, device)
            for device in DeviceStrategy.get_bonded_devices()
        ]

    def _inference_impl(
        self, data: list[FragmentData], model: ViSNetModelLike
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        return zip(*[model.dl_potential_loader(unit) for unit in data])

    def calculate(
        self, fragments: FragmentData
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Calculate the energy and forces of the dipeptide.
         The basic function of DLBondedCalculator.

        Parameters:
        -----------
            fragments: FragmentData
                combined dipeptide and ACE-NME fragments.
        """
        
        self.ringbuf_put(fragments)

        # monitor & fallback initialization
        if not self.monitors:
            self.initialize_monitors(fragments, DeviceStrategy.get_default_device())

        # retrieve devices and work assignment
        devices = DeviceStrategy.get_bonded_devices()
        work = DeviceStrategy.get_work_partitions()

        # work execution
        n_devices = len(devices)
        partitions = [[] for _ in range(n_devices)]
        for idx, start, end in work:
            partitions[idx].append(fragments[start:end])

        futures: list[Future] = []
        with ThreadPoolExecutor(n_devices) as executor:
            for data, model in zip(partitions, self.models):
                futures.append(executor.submit(self._inference_impl, data, model))

        # collect results
        output = [
            np.concatenate(list(itertools.chain(*item)))
            for item in zip(*[f.result() for f in futures])
        ]

        if error_injection.should_inject():
            print("### INJECTING ERRORS NOW ###")
            inject_force_errors(output[1])

        # convert numpy arrays to torch tensors
        device = DeviceStrategy.get_default_device()

        output = [numpy_to_torch(x, device=device) for x in output]
        energy, forces, uncert, x_rep, v_rep = output

        # monitor step 1: override mlff output with active fallbacks
        self.monitors.fallback(fragments, energy, forces, uncert)

        # monitor step 2: check mlff results
        monitor_errors = self.monitors.check(fragments, energy, forces)
        # instead of stepping back, just print out the dipeptide which bond length monitor is triggered, and move on.
        for monitor in monitor_errors:
            random_num = monitor.bondlen_info()
            if monitor.name != 'AN':
                idx = monitor.idx * 2
            else:
                idx = monitor.idx *2 + 1  #need to check this
            if random_num < 0: #randomly choose the dipeptide to calculate errors
                atoms = fragments.get_atoms(idx)
                energy_dl = energy[idx]
                forces_dl = forces[monitor.start:monitor.end]

                energy_fallback, forces_fallback = monitor.fallback_calculate(atoms, energy_dl, forces_dl)
                forces_fallback = numpy_to_torch(forces_fallback, device=device)   #convert to torch.tensor, or will raise error
                energy[idx] = energy_fallback
                forces[monitor.start:monitor.end] = forces_fallback


        # monitor step 3a: if there are errors, request simulators to rollback, and activate our fallback.
        '''if len(monitor_errors):
            raise FragmentMonitorError(
                lambda back_steps, cur_step: self.step_backward(back_steps, cur_step, monitor_errors))'''

        # monitor step 3b: this step is completed safely. de-activate the expired fallbacks
        self.monitors.step_forward()

        # split results between dipeptides/ACE-NMEs
        dipeptides_energy, ACE_NMEs_energy = (energy[s] for s in fragments.scalar_split())
        dipeptides_forces, ACE_NMEs_forces = (forces[s] for s in fragments.vector_split())
        dipeptides_x, ACE_NMEs_x = (x_rep[s] for s in fragments.vector_split())
        dipeptides_v, ACE_NMEs_v = (v_rep[s] for s in fragments.vector_split())

        return (
            dipeptides_energy,
            dipeptides_forces,
            ACE_NMEs_energy,
            ACE_NMEs_forces,
            dipeptides_x, ACE_NMEs_x,
            dipeptides_v, ACE_NMEs_v
        )

    def __call__(self, prot: Protein) -> tuple[np.ndarray, np.ndarray]:
        fragments = self.fragment_method.get_fragments(prot)
        (
            dipeptides_energies,
            dipeptides_forces,
            ACE_NMEs_energies,
            ACE_NMEs_forces,
            dipeptides_x, ACE_NMEs_x,
            dipeptides_v, ACE_NMEs_v,
        ) = self.calculate(fragments)

        energy = self.combiner.energy_combine(
            dipeptides_energies,
            ACE_NMEs_energies,
        )
        forces = self.combiner.forces_combine(
            len(prot),
            dipeptides_forces,
            ACE_NMEs_forces,
            prot.select_index,
            prot.origin_index,
        )
        prot_x = self.combiner.forces_combine(
            len(prot),
            dipeptides_x, ACE_NMEs_x,
            prot.select_index,
            prot.origin_index,
        )
        prot_v = self.combiner.forces_combine(
            len(prot),
            dipeptides_v, ACE_NMEs_v,
            prot.select_index,
            prot.origin_index,
        )
        prot.x_rep = prot_x
        prot.v_rep = prot_v

        prot.frag_energy = energy
        prot.frag_force = forces

        return energy, forces

    def ringbuf_tighten(self):
        if self.history_enable:
            while len(self.history_ringbuf) > self.history_size:
                self.history_ringbuf.popleft()

    def ringbuf_put(self, fragments: FragmentData):
        if self.history_enable:
            self.history_ringbuf.append(fragments)
            self.ringbuf_tighten()

    def ringbuf_dump(self, new_size, cur_step, monitor_errors):
        if self.history_enable:
            # resize ringbuf first in case simulator doesn't want that many steps
            self.history_size = new_size
            self.ringbuf_tighten()
            ActiveLearningData(self.history_ringbuf, monitor_errors).dump(cur_step)

    def step_backward(self, back_steps, cur_step, monitor_errors):
        r"""
        Continuation function handed over to the outer simulator, when a monitor error occurs.
        When this function is called, it does the following things:
            - multiplexes the step back request to the monitors.
            - dumps the history ringbuffer into ActiveLearningData, and persist that on disk.

        Parameters:
        -----------

            back_steps: int
                Total steps to go backward. Computed by the simulator.
            cur_step: int
                The step where the error happens. Provided by the simulator.
            monitor_errors: list[FragmentMonitor]
                Piggy-backed from self to simulator and then back here. Represents the new errors.

        """
        # 1) currently active monitors need to push back
        self.monitors.step_backward(back_steps)

        # 2) for new errors, push back
        for monitor in monitor_errors:
            monitor.step_backward(back_steps)

        # 3) dump the ring buffer
        self.ringbuf_dump(back_steps, cur_step, monitor_errors)

    def initialize_monitors(self, fragments, device):
        self.monitors = MonitorCollection(fragments, self.checks, device)
