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
from ase.calculators.calculator import Calculator, all_changes

from AIMD import arguments
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
        # note: this context is a no-op if self.stream is None
        with torch.cuda.stream(self.stream):
            with torch.set_grad_enabled(True):
                e, f = self.model(self.collate(frag_data))

                e = e.detach().cpu().reshape(-1, 1).numpy()
                f = f.detach().cpu().reshape(-1, 3).numpy()

        return e, f

    @classmethod
    def from_file(cls, **kwargs):
        if "model_path" not in kwargs:
            raise ValueError("model_path must be provided")

        model_path = kwargs["model_path"]
        device = kwargs.get("device", "cpu")

        model = load_model(model_path)
        out = cls(model, device=device)
        return out


class ViSNetAsyncModel:
    """A proxy object that spawns a subprocess, loads a model, and serve inference requests."""

    def __init__(self, model_path: str, device: str):
        self.model_path = model_path
        self.device = device
        self.server = AsyncServer("ViSNet")
        self.logger = getLogger("ViSNet-Proxy")
        envs = os.environ.copy()
        
        try:
            _ = envs["PYTHONPATH"]
        except:
            envs["PYTHONPATH"] = ""
        
        envs["PYTHONPATH"] = f"{osp.abspath(osp.join(osp.dirname(__file__), '..'))}:{envs['PYTHONPATH']}"
        outfd = None if arguments.get().verbose >= 3 else subprocess.DEVNULL
        # use __file__ as process so that viztracer-patched subprocess doesn't track us
        # this file should have chmod +x
        self.proc = subprocess.Popen(
            [
                __file__,
                "--model-path", model_path,
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
        model_path = osp.join(self.ckpt_path, f"visnet-uni-{self.ckpt_type}.ckpt")
        self.device = DeviceStrategy.get_bonded_devices()[0]
        self.model = get_visnet_model(model_path, self.device)

    def calculate(self, atoms, properties, system_changes):
        if self.is_root_calc:
            Calculator.calculate(self, atoms, properties, system_changes)

        data = FragmentData(
            atoms.numbers,
            atoms.positions.astype(np.float32),
            np.array([0], dtype=int),
            np.array([len(atoms)], dtype=int),
            np.zeros((len(atoms),), dtype=int),
        )

        e, f = self.model.dl_potential_loader(data)

        self.results = {
            "energy": e,
            "forces": f,
        }

class ViSNetPIMACalculator(Calculator):
    """
    专门为 JAX/PIMA 模型设计的 ASE 计算器子类。
    它伪装成 ViSNet 的接口，但内部完全运行 JAX 逻辑。
    """
    implemented_properties = ["energy", "forces"]

    def __init__(self, ckpt_path: str, device="cpu", is_root_calc=True, **kwargs):
        super().__init__(**kwargs)
        self.is_root_calc = is_root_calc
        self.device=device
        # 1. 彻底绕过 Torch，使用 MLIP 的原生 JAX 加载方式
        from mlip.models import Visnet
        from mlip.models.model_io import load_model_from_zip

        print(ckpt_path)
        # 加载 JAX ForceField (这部分不涉及 Torch)
        self.force_field = load_model_from_zip(Visnet, ckpt_path)
        self.jax_engine = None  # 先不初始化

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        """
        这个方法解决了你的 AttributeError。
        模拟器调用 self.qmcalc.calculate(...) 时会进入这里。
        """
        if self.is_root_calc:
            Calculator.calculate(self, atoms, properties, system_changes)

        if self.jax_engine is None:
            #print(f"Initializing JAX Engine with real fragment size: {len(atoms)} atoms")
            from mlip.simulation.ase.mlip_ase_calculator import MLIPForceFieldASECalculator
            self.jax_engine = MLIPForceFieldASECalculator(
                atoms=atoms,  # 使用真实的第一个碎片进行初始化
                force_field=self.force_field,
                edge_capacity_multiplier=2.0, # 稍微给点余量
                node_capacity_multiplier=2.0,
                allow_nodes_to_change=True
            )

        # 3. 核心：调用 JAX 引擎进行计算
        # 注意：这里直接把 atoms 传给内部的 jax_engine
        self.jax_engine.calculate(atoms, properties=['energy', 'forces'], system_changes=all_changes)

        # 4. 提取 JAX 结果并转回 Numpy (确保模拟器能识别)
        # JAX 的 DeviceArray 需要转成 Numpy 以免后续 Torch 逻辑报错
        self.results = {
            "energy": np.array(self.jax_engine.results["energy"]).item(),
            "forces": np.array(self.jax_engine.results["forces"]).astype(np.float64)
        }
        
    def dl_potential_loader(self, data):
        """
        利用 FragmentData 内置方法确保 JAX 逐个处理碎片
        """
        import numpy as np
        
        # 情况 A: data 包含多个碎片 (一个 List 里的 partitions)
        if hasattr(data, 'start') and len(data.start) > 1:
            total_energy = []
            all_forces = []

            for i in range(len(data)):
                # 直接调用 FragmentData 自带的 get_atoms 获得标准的 ASE Atoms
                tmp_atoms = data.get_atoms(i)
                
                # 执行 JAX 计算
                self.calculate(tmp_atoms)
                
                total_energy.append(self.results["energy"])
                all_forces.append(self.results["forces"])
            
            # 返回拼接后的能量（标量包装）和受力（长数组）
            return np.array(total_energy), np.concatenate(all_forces, axis=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("ViSNet proxy")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--socket-path", type=str, required=True)
    args = parser.parse_args()

    kwargs = {
        'model_path': args.model_path,
        'device': args.device,
    }
    calculator = ViSNetModel.from_file(**kwargs)
    client = AsyncClient(args.socket_path)
    # start serving
    try:
        while True:
            data: FragmentData = client.recv_object()
            output = calculator.dl_potential_loader(data)
            client.send_object(output)
    except Exception:
        exit(0)

ViSNetModelLike = Union[ViSNetModel, ViSNetAsyncModel, ViSNetPIMACalculator]
_local_calc: dict[str, ViSNetModel] = {}


def get_visnet_model(model_path: str, device: str):
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
            return ViSNetAsyncModel(model_path, device)
    else: # doesn't exist in master, create one
        kwargs = {
            'model_path': model_path,
            'device': device,
        }
        calc = ViSNetModel.from_file(**kwargs)
        _local_calc[signature] = calc
        return calc
