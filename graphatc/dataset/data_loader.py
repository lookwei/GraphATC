import numpy as np

from torch.utils.data import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader

from graphatc.dataset.common import const
from graphatc.dataset.graph_dataset import ATCDataset


class BaseDataLoader:

    def __init__(self, dataset_instance, data_loader_class, train_kwargs,
                 val_batch_size=1, seed=42, shuffle_indices=True):
        self.dataset = dataset_instance
        self.collate = self.dataset.collate
        self.DataLoader = data_loader_class
        self.train_kwargs = train_kwargs
        self.val_batch_size = val_batch_size
        self.seed = seed
        self.shuffle_indices = shuffle_indices
        if shuffle_indices:
            np.random.seed(self.seed)

        self.dataset_size = len(self.dataset)
        self.indices = list(range(self.dataset_size))

    def indices2loader(self, train_indices, val_indices):
        train_sampler, valid_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(val_indices)
        train_loader = self.DataLoader(self.dataset, collate_fn=self.collate,
                                       sampler=train_sampler, **self.train_kwargs)
        val_loader = self.DataLoader(self.dataset, collate_fn=self.collate,
                                     sampler=valid_sampler, batch_size=self.val_batch_size)
        return train_loader, val_loader

    def indices2loader_all(self, all_indices):
        """
        for demo only, return all data as train_loader
        """
        train_sampler = SubsetRandomSampler(all_indices)
        train_loader = self.DataLoader(self.dataset, collate_fn=self.collate,
                                       sampler=train_sampler, **self.train_kwargs)
        return train_loader

    def jack_k_to_loader(self, _k: int):
        val_indices = self.indices[_k:_k + 1]
        train_indices = self.indices[0:_k] + self.indices[_k + 1:self.dataset_size]
        if self.shuffle_indices:
            np.random.shuffle(train_indices)
        return self.indices2loader(train_indices, val_indices)

    def get_data_loader(self, train_method: str, **kwargs):
        assert train_method in ['K-Fold', 'Jackknife', 'all']

        if train_method == 'K-Fold':
            assert 'K' in kwargs.keys()
            one_fold_size = int(self.dataset_size / kwargs['K'])
            val_low, val_high = 0, one_fold_size
            for k in range(kwargs['K']):
                if k == kwargs['K'] - 1:
                    val_high = self.dataset_size
                val_indices = self.indices[val_low:val_high]
                train_indices = self.indices[0:val_low] + self.indices[val_high:self.dataset_size]
                if self.shuffle_indices:
                    np.random.shuffle(train_indices)
                val_low += one_fold_size
                val_high += one_fold_size
                if k not in kwargs['K_range']:
                    # if k = 0,1,2,...K-1, K_range=list(range(1,4)) -> 1,2,3
                    continue
                yield self.indices2loader(train_indices, val_indices)
        elif train_method == 'all': # for all data training, foo validation
            all_indices = self.indices
            val_indices = self.indices[0:100]
            print(f">>> Use all data for training\n, len(all_indices): {len(all_indices)}, len(val_indices): {len(val_indices)}")
            if self.shuffle_indices:
                np.random.shuffle(all_indices)
            yield self.indices2loader(all_indices, val_indices)
        elif train_method in ['Jackknife']:
            assert int('val_indices_range' in kwargs.keys()) + int('val_did_list' in kwargs.keys()) == 1
            if 'val_indices_range' in kwargs.keys():
                for k in kwargs['val_indices_range']:
                    yield self.jack_k_to_loader(k)
            else:
                for did in kwargs['val_did_list']:
                    k = self.dataset.did_to_index(did)
                    yield self.jack_k_to_loader(k)

        else:
            raise ValueError('ERROR: split_method')


if __name__ == "__main__":
    atc_dataset = ATCDataset(level=1,
                             polymer_method=const.IMPROVE_POLYMER_DELETE_STAR_CONNECT_ATOM,
                             split_component=True)

    data_loader = BaseDataLoader(atc_dataset, GraphDataLoader,
                                 train_kwargs={'batch_size': 256, 'drop_last': False, 'shuffle': False,
                                               'num_workers': 0, 'prefetch_factor': None
                                               },
                                 val_batch_size=1)

    VAL_INDICES = range(0, 3)

    real_loader = data_loader.get_data_loader(train_method='Jackknife', val_indices_range=VAL_INDICES)

    for k, (train_loader, val_loader) in enumerate(real_loader):
        # print(k, (train_loader, val_loader))
        for i, (drug_id, drug_graph, multi_label, max_len, group_arr_list) in enumerate(train_loader):
            running_batch_size = len(multi_label)
            node_feats = [drug_graph.ndata.pop('atomic_number'),
                          drug_graph.ndata.pop('chirality_type')]
            edge_feats = [drug_graph.edata.pop('bond_type'),
                          drug_graph.edata.pop('bond_direction_type')]
            # print(running_batch_size, drug_graph, node_feats, edge_feats)
        for _val_indices in val_loader.sampler.indecies:
            print(_val_indices, atc_dataset[_val_indices][0])
