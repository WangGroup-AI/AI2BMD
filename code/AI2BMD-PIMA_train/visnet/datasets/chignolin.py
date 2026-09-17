import os
import subprocess

import numpy as np
import torch
from ase.units import Bohr, Hartree, kcal, mol
from torch_geometric.data import Data, InMemoryDataset
from tqdm import trange
import scipy

class Chignolin(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None):
        
        super(Chignolin, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ['smaller_dataset.npz']

    @property
    def processed_file_names(self):
        return ["Chignolin.pt"]

    def process(self):
        
        samples = []
        
        for path in self.raw_paths:
            data_npz = np.load(path)
            concat_forces = torch.from_numpy(data_npz["F"]).float() * Hartree
        
            concat_positions = torch.from_numpy(data_npz["R"]).float()

            energies = torch.from_numpy(data_npz["E"]).float() * Hartree

            z=torch.from_numpy(data_npz["Z"]).long()

            
            for index in trange(data_npz['R'].shape[0], desc=f'Processing {os.path.basename(path)}:'):
                pos = concat_positions[index]
                y = energies[index]
                # ! NOTE: Convert Engrad to Force
                dy = concat_forces[index]
                
                data = Data(z=z, pos=pos, y=y.reshape(1, 1), dy=dy)

                if self.pre_filter is not None:
                    data = self.pre_filter(data)

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                samples.append(data)

        data, slices = self.collate(samples)
        torch.save((data, slices), self.processed_paths[0])
    
    def get_atomref(self, max_z=100):
        atomref = torch.zeros(max_z)
        atomref[[1, 6, 7, 8, 16]] = torch.tensor([-0.496675195977, -37.760690035605, -54.566616171740, -75.031346432045, -398.0025913303]) * Hartree
        return atomref.unsqueeze(1)
