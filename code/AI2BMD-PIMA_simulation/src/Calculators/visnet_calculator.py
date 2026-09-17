#!/usr/bin/env python
import argparse
import atexit
import os
import subprocess
from logging import getLogger
from os import path as osp
from typing import Union

import numpy as np
import torch
import torch.multiprocessing as mp
from ase.calculators.calculator import Calculator

import utils
from AIMD import envflags
from AIMD.fragment import FragmentData
from Calculators.device_strategy import DeviceStrategy
from Calculators.async_utils import AsyncServer, AsyncClient
from ViSNet.model.visnet import load_model
from utils.utils import numpy_to_torch

class ViSNetModel:
    r"""
    Calculate the energy and forces of the system
    using deep learning model, i.e., ViSNet.

    Parameters:
    -----------
        model:
            Deep learning model.
        device: cpu | cuda
            Device to use for calculation.
    """
    implemented_properties = ["energy", "forces"]

    def __init__(self, model, device="cpu"):
        self.model = model
        self.model.eval()
        self.device = device
        self.stream = (
            torch.cuda.Stream(device=device)
            if device.startswith('cuda')
            else None
        )
        self.model.to(self.device)

    def collate(self, frag: FragmentData):
        z = numpy_to_torch(frag.z, self.device)
        pos = numpy_to_torch(frag.pos, self.device)
        batch = numpy_to_torch(frag.batch, self.device)

        return dict(z=z, pos=pos, batch=batch)

    def dl_potential_loader(self, frag_data: FragmentData):
        shapes = [1, 3, 1]

        # note: this context is a no-op if self.stream is None
        with torch.cuda.stream(self.stream):
            with torch.set_grad_enabled(True):
                output = self.model(self.collate(frag_data))
            output_list = []
            for v, d in zip(output[:3], shapes):
                if v is not None:
                    output_list.append(v.detach().cpu().reshape(-1, d).numpy())
                else:
                    output_list.append(np.zeros((output_list[1].shape[0],1)))
            for v in output[3:]:
                output_list.append(v.detach().cpu().numpy())
            output = tuple(output_list)

            # output = tuple(
            #     v.detach().cpu().reshape(-1, d).numpy()
            #     for v, d in zip(output, shapes)
            #     if v is not None
            # )

        return output

    @classmethod
    def from_file(cls, **kwargs):
        if "model_path" not in kwargs:
            raise ValueError("model_path must be provided")

        model_path = kwargs["model_path"]
        ckpt_type = kwargs["ckpt_type"]

        device = kwargs.get("device", "cpu")

        model = load_model(model_path, ckpt_type=ckpt_type)
        out = cls(model, device=device)
        return out


class ViSNetAsyncModel:
    """A proxy object that spawns a subprocess, loads a model, and serve inference requests."""

    def __init__(self, model_path: str, ckpt_type: str, device: str):
        self.model_path = model_path
        self.ckpt_type = ckpt_type
        self.device = device
        self.server = AsyncServer("ViSNet")
        self.logger = getLogger("ViSNet-Proxy")
        envs = os.environ.copy()
        envs["PYTHONPATH"] = f"{osp.abspath(osp.join(osp.dirname(__file__), '..'))}"
        outfd = subprocess.PIPE if not envflags.DEBUG_RC else None
        # use __file__ as process so that viztracer-patched subprocess doesn't track us
        # this file should have chmod +x
        self.proc = subprocess.Popen(
            [
                __file__,
                "--model-path", model_path,
                "--ckpt-type", ckpt_type,
                "--device", device,
                "--socket-path", self.server.socket_path,
            ],
            shell=False,
            env=envs,
            stdout=outfd,
            stderr=outfd,
        )
        self.logger.debug(f'Waiting for worker ({device}) to start...')
        self.server.accept()
        self.logger.debug(f'Worker ({device}) started.')
        atexit.register(self._shutdown)

    def dl_potential_loader(self, data: FragmentData):
        self.server.send_object(data)
        return self.server.recv_object()

    def _shutdown(self):
        self.logger.debug(f"Shutting down worker ({self.device})...")
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
        if self.server:
            self.server.close()
        self.logger.debug(f"Worker ({self.device}) shutdown complete.")


class ViSNetCalculator(Calculator):
    r"""
    Feed the input through a ViSNet model, without fragmentation
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, ckpt_path: str, ckpt_type: str,
                 is_root_calc=True, **kwargs):
        super().__init__(**kwargs)
        self.ckpt_path = ckpt_path
        self.ckpt_type = ckpt_type
        self.is_root_calc = is_root_calc
        assert self.ckpt_type in ["nll", "new", "old"], "invalid ckpt_type"
        self.use_kernel = use_kernel
        self.device = DeviceStrategy.get_bonded_devices()[0]
        model_path = osp.join(self.ckpt_path, f"visnet-uni-{self.ckpt_type}.ckpt")
        self.model = get_visnet_model(model_path, ckpt_type, self.device)

    def calculate(self, atoms, properties, system_changes):
        if self.is_root_calc:
            Calculator.calculate(self, atoms, properties, system_changes)

        data = FragmentData(
            atoms.numbers,
            atoms.positions.astype(np.float32),
            atoms.arrays["residuenames"],
            np.array([0], dtype=int),
            np.array([len(atoms)], dtype=int),
            np.zeros((len(atoms),), dtype=int),
        )

        output = self.model.dl_potential_loader(data)

        self.results = {
            "energy": output[0],
            "forces": output[1],
        }


if __name__ == "__main__":
    mp.set_sharing_strategy("file_system")
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser("ViSNet proxy")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--ckpt-type", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--socket-path", type=str, required=True)
    args = parser.parse_args()

    logger = getLogger("AI2BMD-ViSNet-Worker")
    logger.info(f'model-path: {args.model_path}')
    logger.info(f'ckpt-type: {args.ckpt_type}')
    logger.info(f'device: {args.device}')

    utils.register_print_stack_on_sigusr2(pass_through = False)

    kwargs = {
        'model_path': args.model_path,
        'ckpt_type': args.ckpt_type,
        'device': args.device,
    }
    calculator = ViSNetModel.from_file(**kwargs)
    client = AsyncClient(args.socket_path)
    # start serving
    logger.info('Start serving.')
    try:
        while True:
            data: FragmentData = client.recv_object()
            output = calculator.dl_potential_loader(data)
            # XXX CPU model in a worker is problematic, wrong results, including nan values
            client.send_object(output)
    except Exception:
        exit(0)

ViSNetModelLike = Union[ViSNetModel, ViSNetAsyncModel]
_local_calc: dict[str, ViSNetModel] = {}
_async_calc: dict[str, ViSNetModel] = {}

def get_visnet_model(model_path: str, ckpt_type: str, device: str):
    # return ViSNetAsyncCalculator(model_path, ckpt_type, device)
    # allow up to 1 copy of GPU model to run in the master process
    device_sig = device
    if device_sig.startswith('cuda'):
        device_sig = 'cuda'
    signature = f"{device_sig}-{model_path}"
    if signature in _local_calc: # exists in master
        if device == 'cpu':
            # work around CPU model on worker proxy problem: always reuse local
            return _local_calc[signature]
        else:
            # do not reuse local, but create a proxy
            # return ViSNetAsyncModel(model_path, ckpt_type, device)
            async_key = f"{device}-{model_path}"
            if async_key not in _async_calc:
                _async_calc[async_key] = ViSNetAsyncModel(model_path, ckpt_type, device)
            return _async_calc[async_key]
    else: # doesn't exist in master, create one
        kwargs = {
            'model_path': model_path,
            'ckpt_type': ckpt_type,
            'device': device,
        }
        calc = ViSNetModel.from_file(**kwargs)
        _local_calc[signature] = calc
        return calc
