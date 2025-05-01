import dgl
import torch
import numpy as np
import functools
import pandas as pd

from ast import literal_eval
from graphatc.dataset.common import const
from torch.utils.data import Dataset
from graphatc.dataset.molgraph import MolGraph
from graphatc.dataset.molsmiles import MolSmiles


class ATCDataset(Dataset, MolGraph, MolSmiles):
    def __init__(self, level=1, cuda_idx: int = 0, mode="graph",
                 init_graph_map=True, polymer_method=None, split_component=True,
                 init_smiles_map=True, smiles_seq_pad_cut_len=787, author=None, br08303_version=None,
                 *args, **kwargs):
        super().__init__(br08303_version=br08303_version, *args, **kwargs)
        self.level = level
        self.cuda_idx = cuda_idx
        self.device = f"cuda:{cuda_idx}"

        self.atc_mol_map = self.init_atc_mol_map()
        self.author = author.lower()
        self.aligned_to(author, level=level)
        self.drug_id_list = sorted(self.atc_mol_map.keys())
        self.kwargs = kwargs

        # GRAPH
        if mode in ["g", "G", "graph", "Graph", "GRAPH"]:
            self.mode = "graph"
            self.polymer_method = polymer_method
            self.split_component = split_component
            if init_graph_map:
                self.graph_map = self.get_graph_map_by_mol_map(self.atc_mol_map, polymer_method)
            if split_component:
                self.split_map = self.get_graph_split_map_by_graph_map(self.graph_map)
            self.graph_cuda_map = self.get_graph_cuda_map(self.graph_map, cuda_idx)
            # self.drug_id_list = sorted(self.graph_cuda_map.keys())
            assert len(self.atc_mol_map.keys()) == len(self.graph_cuda_map.keys())
            
            

        # SMILES
        if mode in ["s", "S", "smiles", "Smiles", "SMILES"]:
            self.mode = "smiles"
            self.smiles_seq_pad_cut_len = smiles_seq_pad_cut_len
            if init_smiles_map:
                self.smiles_map = self.get_splited_smiles_map(mol_map=self.atc_mol_map)
                self.smiles_vocab_map = self.get_vocab_dic(self.get_vocab_count(self.smiles_map))
                assert len(self.atc_mol_map.keys()) == len(self.smiles_map.keys())
            if self.smiles_seq_pad_cut_len > 0:
                self.smiles_pad_cut_map = self.get_smiles_pad_cut_map(self.smiles_map, self.smiles_vocab_map,
                                                                      pad_cut_len=self.smiles_seq_pad_cut_len)
                assert len(self.atc_mol_map.keys()) == len(self.smiles_pad_cut_map.keys())


    def __getitem__(self, item):
        if self.mode == "graph":
            return self.get_sample_by_id_mode_g(self.drug_id_list[item])
        elif self.mode == "smiles":
            return self.get_sample_by_id_mode_s(self.drug_id_list[item])
        else:
            raise Exception("dataset mode error")

    def __len__(self):
        return len(self.drug_id_list)

    def collate(self, batch):
        if self.mode == "graph":
            return self.collate_mode_g(batch)
        elif self.mode == "smiles":
            return self.collate_mode_s(batch)
        else:
            raise Exception("dataset mode error")

    def did_to_index(self, drug_id):
        return self.drug_id_list.index(drug_id)

    @functools.lru_cache(maxsize=None)
    def get_sample_by_id_mode_g(self, drug_id: str):
        graph = self.graph_cuda_map[drug_id]
        atc_code_list = self.get_drug_atc_code_list_by_id(drug_id, self.level)
        multi_hot = self.trans_atc_code_to_multi_hot(tuple(atc_code_list), self.level)
        if self.author == "chen" and self.fix_chen and drug_id in self.chen_fix_did_list:
            label_str_list = self.chen_fix_df.loc[drug_id][f"l{self.level}_label"]
            multi_hot = self.trans_atc_code_to_multi_hot(tuple(label_str_list), self.level)
        multi_hot_tensor = torch.tensor(multi_hot, device=self.device)
        # print(multi_hot, multi_hot_tensor.shape)
        if self.split_component:
            return drug_id, graph, multi_hot_tensor, self.split_map[drug_id]
        return drug_id, graph, multi_hot_tensor

    def collate_mode_g(self, batch):
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

    @functools.lru_cache(maxsize=None)
    def get_sample_by_id_mode_s(self, drug_id: str):
        smiles = self.smiles_pad_cut_map[drug_id]
        smiles_tensor = torch.tensor(smiles, device=self.device)
        atc_code_list = self.get_drug_atc_code_list_by_id(drug_id, self.level)
        multi_hot = self.trans_atc_code_to_multi_hot(tuple(atc_code_list), self.level)
        if self.author == "chen" and self.fix_chen and drug_id in self.chen_fix_did_list:
            label_str_list = self.chen_fix_df.loc[drug_id][f"l{self.level}_label"]
            multi_hot = self.trans_atc_code_to_multi_hot(tuple(label_str_list), self.level)
        multi_hot_tensor = torch.tensor(multi_hot, device=self.device)
        return drug_id, smiles_tensor, multi_hot_tensor

    def collate_mode_s(self, batch):
        drug_id, drug_smiles, multi_label = zip(*batch)
        return drug_id, torch.stack(drug_smiles), torch.stack(multi_label)
    
    def is_polymer(self, mol):
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                return True
        return False
    
    def aligned_to(self, author: str, fix_chen=True, level=1):
        if author is None or author == "tian":
            return
        if author in ["tian_del_polymer"]:
            print(f"[aligned_to:{author}] before:{len(self.atc_mol_map)} ", end="")
            atc_mol_map_dids = list(self.atc_mol_map.keys())
            for mol_key in atc_mol_map_dids:
                mol = self.atc_mol_map[mol_key]
                if self.is_polymer(mol):
                    del self.atc_mol_map[mol_key] 
            print(f"----> after:{len(self.atc_mol_map)}")
            data_did_list = self.atc_mol_map.keys()
            print(f"data_did_list len: {len(data_did_list)}")
        if author in ["cao", "CAO", "Cao"]:
            data_df = pd.read_csv(const.DATASET_PATH_ATCSMILES4545)
            data_did_list = data_df["KEGG_Drug_ID"].tolist()
        if author in ["cao3785", "3785"]:
            data_df = pd.read_csv(const.DATASET_PATH_ATCSMILES3785)
            data_did_list = data_df["KEGG_Drug_ID"].tolist()
        if author in ["chen", "CHEN", "Chen"]:
            with open(const.DATASET_PATH_CHEN3883_DRUG_ID, 'r') as file:
                data_did_list = [line.strip() for line in file.readlines()]
            self.chen_3883_df = pd.read_csv(const.DATASET_PATH_CHEN3883)
            self.chen_3883_df = self.chen_3883_df.set_index("drug_id")
            self.chen_3883_df[self.chen_3883_df.columns[0]
                              ] = self.chen_3883_df[self.chen_3883_df.columns[0]].apply(literal_eval)
        if author is not None:
            print(f"[aligned_to:{author}] before:{len(self.atc_mol_map)} ", end="")
            data_did_list = set(data_did_list).intersection(self.atc_mol_map.keys())  # 4545 case C00989
            self.atc_mol_map = {did: self.atc_mol_map[did] for did in data_did_list}
            # self.mol_map = {did: self.mol_map[did] for did in data_did_list if did if self.mol_map}
            print(f"----> after:{len(self.atc_mol_map)}")

        # fix to 3875
        if fix_chen and author in ["chen", "CHEN", "Chen"]:
            self.fix_chen = True
            before_len = len(self.atc_mol_map)
            self.chen_fix_df = pd.read_csv(const.DATASET_PATH_CHEN3883_FIX_11)
            self.chen_fix_df[f"l{self.level}_label"] = self.chen_fix_df[f"l{self.level}_label"].apply(literal_eval)
            self.chen_fix_did_list = self.chen_fix_df["drug_id"].tolist()
            self.chen_fix_df = self.chen_fix_df.set_index("drug_id")
            self.chen_fix_mol_map = self.get_mol_map(self.chen_fix_did_list)
            self.atc_mol_map.update(self.chen_fix_mol_map)
            self.mol_map["atc"].update(self.chen_fix_mol_map)
            print(f"[fix:{author}] before:{before_len} ", end="")
            print(f"----> after:{len(self.atc_mol_map)}")
