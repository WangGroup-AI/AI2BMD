# active learning dataset
import os
import subprocess

import numpy as np
import torch
from ase.units import Bohr, Hartree, kcal, mol
from torch_geometric.data import Data, InMemoryDataset
from tqdm import trange


class Cumulene_train(InMemoryDataset):

    def __init__(self, root, dataset_args, train_val_test,transform=None, pre_transform=None):
        self.train_val_test = train_val_test
        self.dataset_args = dataset_args
        super(Cumulene_train, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return ["train.pt"]

    def process(self):
        
        samples = []
        path = os.path.join(self.root, f"{self.dataset_args}_n_train_1000_n_valid_1000_seed_0.npz")
        for train_val_test in ['train','valid','test']:
            '''
            npz files:
            ['TYPE', 'CODE_VERSION', 'NAME', 'THEORY', 'R_train', 'R_valid', 'R_test', 'R_mean', 'R_scale', 'Z_train', 'Z_valid', 'Z_test', 'F_train', 'F_valid', 'F_test', 'F_mean', 'F_scale', 'F_MIN', 'F_MAX', 'F_MEAN', 'F_VAR', 'R_UNIT', 'E_UNIT', 'E_train', 'E_valid', 'E_test', 'E_mean', 'E_scale', 'E_MIN', 'E_MAX', 'E_MEAN', 'E_VAR', 'MD5']
            '''
            data_npz = np.load(path)
            concat_z = torch.from_numpy(data_npz[f"Z_{train_val_test}"]).long()
            concat_positions = torch.from_numpy(data_npz[f"R_{train_val_test}"]).float()
            # Hartree to eV
            energies = torch.from_numpy(data_npz[f"E_{train_val_test}"]).float() 
            # Hartree / Bohr to eV / A
            concat_forces = torch.from_numpy(data_npz[f"F_{train_val_test}"]).float() 
            for index in trange(energies.shape[0], desc=f'Processing {os.path.basename(path)}:'):
                z = concat_z[index]
                pos = concat_positions[index, :, :]
                y = energies[index]
                dy = concat_forces[index, :, :]
                data = Data(z=z, pos=pos, y=y.reshape(1, 1), dy=dy)
                if self.pre_filter is not None:
                    data = self.pre_filter(data)
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                samples.append(data)
        data, slices = self.collate(samples)
        print(len(data))
        torch.save((data, slices), self.processed_paths[0])
    
class Cumulene(InMemoryDataset):

    def __init__(self, root, dataset_args, train_val_test, transform=None, pre_transform=None):
        self.dataset_args = dataset_args
        super(Cumulene, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return [f"test.pt"]

    def process(self):
        
        samples = []
        
        path = os.path.join(self.root, f"{self.dataset_args}_test.npz")
        '''
        npz files:
        ['TYPE', 'CODE_VERSION', 'NAME', 'THEORY', 'R', 'z', 'F','E', 'MD5']
        '''
        data_npz = np.load(path)
        concat_z = torch.from_numpy(data_npz["z"]).long()
        concat_positions = torch.from_numpy(data_npz["R"]).float()
        energies = torch.from_numpy(data_npz["E"]).float() 
        concat_forces = torch.from_numpy(data_npz[f"F"]).float() 
        for index in trange(energies.shape[0], desc=f'Processing {os.path.basename(path)}:'):
            z = concat_z
            pos = concat_positions[index, :, :]
            y = energies[index]
            dy = concat_forces[index, :, :]
            data = Data(z=z, pos=pos, y=y.reshape(1, 1), dy=dy)
            if self.pre_filter is not None:
                data = self.pre_filter(data)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            samples.append(data)
        data, slices = self.collate(samples)
        print(len(data))
        torch.save((data, slices), self.processed_paths[0])

if __name__ == '__main__':
    dataset = Cumulene_train(root=os.path.join(os.path.dirname(__file__), '../../../../data/cumulene'),dataset_args = 'cumulene8',train_val_test="train")
    #for i in trange(len(dataset)):
        #assert torch.isnan(dataset[i]['dy']).sum()==0
    print(len(dataset))
    print(dataset[0])
