from typing import Callable, Optional

import numpy as np
import os.path as osp
import torch

from torch_geometric.data import Data, InMemoryDataset, download_url


class Critical(InMemoryDataset):

    url = 'https://github.com/yandex-research/heterophilous-graphs/raw/refs/heads/main/data/' # /NAME_filtered.npz

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name.lower()
        assert self.name in ['squirrel', 'chameleon']
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name.capitalize(), 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name.capitalize(), 'processed')

    @property
    def raw_file_names(self) -> str:
        return self.name+'_filtered.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        download_url(self.url + self.raw_file_names, self.raw_dir)

    def process(self) -> None:
        print("processing data...")
        raw_data = np.load(self.raw_paths[0])
        data = Data(x=torch.from_numpy(raw_data['node_features']), edge_index=torch.t(torch.from_numpy(raw_data['edges'])), y=torch.from_numpy(raw_data['node_labels']), 
                    train_mask=torch.t(torch.from_numpy(raw_data['train_masks'])), val_mask=torch.t(torch.from_numpy(raw_data['val_masks'])), test_mask=torch.t(torch.from_numpy(raw_data['test_masks'])))
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])