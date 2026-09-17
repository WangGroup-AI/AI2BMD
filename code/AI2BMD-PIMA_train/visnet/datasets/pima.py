import os
import subprocess

import numpy as np
import torch
from ase.units import Bohr, Hartree, kcal, mol
from torch_geometric.data import Data, InMemoryDataset
from tqdm import trange
import scipy

class Pima(InMemoryDataset):

    def __init__(self, root, dataset_args = None, transform=None, pre_transform=None):
        self.dataset_arg = dataset_args
        super(Pima, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    # @property
    # def raw_file_names(self):
    #     return [file for dipeptide in self.AcL.keys() for file in self.AcL[dipeptide]]

    @property
    def processed_file_names(self):
        return [f'{self.dataset_arg}.pt'] #'chig.pt',"trp.pt",'ww.pt','abd.pt',

    def process(self):
        
        
        name_list = ['chig','trp','ww','abd','pacsin'] #'chig','trp','ww','abd',
        raw_path = self.raw_dir

        for i, name in enumerate(name_list):
            samples = []
            for file_name in os.listdir(raw_path):
                if file_name.split('_')[0] == name:
                    data_npz = np.load(os.path.join(raw_path, file_name))
                    concat_z = torch.from_numpy(data_npz["z"]).long()
                    concat_positions = torch.from_numpy(data_npz["pos"]).float()
                    prot_energies = torch.from_numpy(data_npz["prot_energy"]).float()
                    frag_energies = torch.from_numpy(data_npz["frag_energy"]).float()
                    # Hartree / Bohr to eV / A
                    concat_prot_forces = torch.from_numpy(data_npz["prot_force"]).float()
                    concat_frag_forces = torch.from_numpy(data_npz["frag_force"]).float()
                    concat_x = torch.from_numpy(data_npz["x_rep"]).float()
                    concat_v = torch.from_numpy(data_npz["v_rep"]).float()
                    for index in trange(prot_energies.shape[0]):                           #(energies.shape[0]
                        z = concat_z
                        pos = concat_positions[index]
                        y = prot_energies[index] - frag_energies[index]
                        # ! NOTE: Convert Engrad to Force
                        dy = concat_prot_forces[index] - concat_frag_forces[index]
                        x_rep = concat_x[index]
                        v_rep = concat_v[index]
                        data = Data(z=z, pos=pos, y=y.reshape(1, 1), dy=dy, x_rep = x_rep, v_rep = v_rep)

                        if self.pre_filter is not None:
                            data = self.pre_filter(data)

                        if self.pre_transform is not None:
                            data = self.pre_transform(data)
                            
                        samples.append(data)

            data, slices = self.collate(samples)
            torch.save((data, slices), self.processed_paths[i])

    
    
    def get_atomref(self, max_z=100):
        atomref = torch.zeros(max_z)
        atomref[[1, 6, 7, 8, 16]] = torch.tensor([-0.496675195977, -37.760690035605, -54.566616171740, -75.031346432045, -398.0025913303]) * Hartree
        return atomref.unsqueeze(1)


if __name__ == '__main__':
    chig_dataset = Pima(root=os.path.join(os.path.dirname(__file__), '../../../../data/Trp-cage/pima_dataset_pyg'),dataset_args ='chig')
    pacsin_dataset = Pima(root=os.path.join(os.path.dirname(__file__), '../../../../data/Trp-cage/pima_dataset_pyg'),dataset_args ='pacsin')
    chig_y_list = []
    pacsin_y_list = []
    for i in trange(len(chig_dataset)):
        chig_y = chig_dataset[i].y
        pacsin_y = pacsin_dataset[i].y
        chig_y_list.append(chig_y)
        pacsin_y_list.append(pacsin_y)
    chig_y_array = torch.tensor(chig_y_list)
    pacsin_y_array = torch.tensor(pacsin_y_list)
    print(chig_y_array.mean())
    print(pacsin_y_array.mean())
    
