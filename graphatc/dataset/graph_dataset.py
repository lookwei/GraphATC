import dgl
import torch
import numpy as np
import functools

from graphatc.dataset.common import const
from torch.utils.data import Dataset
from graphatc.dataset.molgraph import MolGraph


class ATCDataset(Dataset, MolGraph):
    def __init__(self, level=1, polymer_method=None, cuda_idx: int = 0,
                 init_graph_map=True, split_component=True, init_smiles_map=False):
        super().__init__()
        self.level = level
        self.polymer_method = polymer_method
        self.cuda_idx = cuda_idx
        self.device = f"cuda:{cuda_idx}"
        self.split_component = split_component

        self.mol_map = self.init_atc_mol_map()
        if init_graph_map:
            self.graph_map = self.get_graph_map_by_mol_map(self.mol_map, polymer_method)
        if split_component:
            self.split_map = self.get_graph_split_map_by_graph_map(self.graph_map)
        self.graph_cuda_map = self.get_graph_cuda_map(self.graph_map, cuda_idx)
        self.drug_id_list = sorted(self.graph_cuda_map.keys())

        if init_smiles_map:
            self.smiles_map = self.get_splited_smiles_map()

    def __getitem__(self, item):
        return self.get_sample_by_id(self.drug_id_list[item])

    def __len__(self):
        return len(self.drug_id_list)

    def did_to_index(self, drug_id):
        return self.drug_id_list.index(drug_id)

    @functools.lru_cache(maxsize=None)
    def get_sample_by_id(self, drug_id: str):
        graph = self.graph_cuda_map[drug_id]
        atc_code_list = self.get_drug_atc_code_list_by_id(drug_id, self.level)
        multi_hot = self.trans_atc_code_to_multi_hot(tuple(atc_code_list), self.level)
        multi_hot_tensor = torch.tensor(multi_hot, device=self.device)
        if self.split_component:
            return drug_id, graph, multi_hot_tensor, self.split_map[drug_id]
        return drug_id, graph, multi_hot_tensor

    def collate(self, batch):
        if not self.split_component:
            drug_id, drug_graph, multi_label = zip(*batch)
            return drug_id, dgl.batch(drug_graph), torch.stack(multi_label), None, None

        drug_id, drug_graph, multi_label, group_split = zip(*batch)
        multi_label = torch.stack(multi_label)

        max_len = 0
        group_arr_list = []

        batch_size = len(batch)
        idx_base = 0
        for i in range(batch_size):
            one_g = []
            max_len = max(max_len, len(group_split[i]))
            for j in range(len(group_split[i])):
                one_g.append(np.array(group_split[i][j]) + idx_base)
            idx_base += drug_graph[i].num_nodes()
            group_arr_list.append(one_g)

        return drug_id, dgl.batch(drug_graph), multi_label, max_len, group_arr_list


if __name__ == "__main__":
    atc_dataset = ATCDataset(level=1, polymer_method=const.IMPROVE_POLYMER_DELETE_STAR_CONNECT_ATOM)
    print(atc_dataset[0])

    from graphatc.dataset.common import util
    import networkx as nx
    import matplotlib.pyplot as plt

    did = "D03251"
    did = "D03561"
    mol = atc_dataset.get_mol_by_id(did)

    util.draw(mol)
    drug_id, graph, multi_hot_tensor = atc_dataset.get_sample_by_id(did)

    nx.draw(graph.cpu().to_networkx(), with_labels=True)
    plt.show()
