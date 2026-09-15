import numpy as np
import torch
from AIMD.protein import Protein
from ViSNet.model.visnet import load_model
from torch_geometric.data import Batch, Data
from os import path as osp

class PimaNonBondedCalculator:
    r"""
    PimaNonBondedCalculator is a non-bonded calculator based on ViSNet-PIMA.
    """

    def __init__(self, ckpt_path: str, device="cpu") -> None:
        super().__init__()
        self.ckpt_path = ckpt_path
        self.device = device
        model_path = osp.join(self.ckpt_path, f"visnet-uni-pima.ckpt")
        model = load_model(model_path, ckpt_type='pima')
        self.model = model
        self.model.eval()
        self.model.to(self.device)
    
    def set_parameters(self, prot: Protein) -> None:
        pass

    def __call__(self, prot: Protein) -> tuple[np.float32, np.ndarray]:
        z = torch.Tensor(prot.get_atomic_numbers()).long().to(self.device)
        pos = torch.FloatTensor(prot.get_positions()).to(self.device)
        x_rep = torch.FloatTensor(prot.x_rep).to(self.device)
        v_rep = torch.FloatTensor(prot.v_rep).to(self.device)
        data = Data(z = z, pos = pos, x_rep = x_rep, v_rep = v_rep)
        batch = Batch.from_data_list([data])
        prot_data = dict(z=z, pos=pos, batch=batch.batch, x_rep = x_rep, v_rep = v_rep)
        with torch.set_grad_enabled(True):
            energy, force, _, _, _ = self.model(prot_data)
        energy = energy.detach().cpu().numpy()
        force = force.detach().cpu().numpy().reshape(-1,3)


        return energy, force