import json
import functools

from typing import Union

from graphatc.dataset.common import util
from graphatc.dataset.common.const import DATASET_PATH, ATC_LEVEL_LEN
from graphatc.dataset.kegg.keggbr import KeggBr, BrNode


class DrugATC(KeggBr):
    drug_atc_root = None
    drug_atc_node_map = {}
    drug_atc_tree_map = {}

    def __init__(self, br08303_version=None, *args, **kwargs):
        super().__init__()
        self.br08303_version = br08303_version
        self.load_br08303_json(br08303_version)

    def load_br08303_json(self, version):
        if version is None or len(version) == 0:
            version = "5724_5311"
        self.br08303_version = version
        with open(f"{DATASET_PATH}/atc/br08303_drug_atc_{version}.json", "rb") as f:
            self.build_drug_atc_tree(json.load(f))

    def build_drug_atc_tree(self, d: dict):
        self.drug_atc_root, self.drug_atc_node_map, self.drug_atc_tree_map = self.build_tree(d)

    def __len__(self):
        return len(self.drug_atc_node_map.keys())

    @functools.lru_cache(maxsize=None)
    def get_atc_node_by_id(self, did: str) -> [BrNode, None]:
        if did not in self.drug_atc_node_map.keys():
            return None
        return self.drug_atc_node_map[did]

    @functools.lru_cache(maxsize=None)
    def get_atc_id_list(self) -> list[str]:
        return list(self.drug_atc_node_map.keys())

    @functools.lru_cache(maxsize=None)
    def get_drug_atc_code_list_by_id(self, did: str, level: int = 5) -> list[str]:
        node = self.get_atc_node_by_id(did)
        if node is None:
            return []
        code_list = [fa.name[:self.get_drug_atc_code_length(level)] for fa in node.fa]
        return util.list_remove_duplicate(code_list)

    @functools.lru_cache(maxsize=None)
    def is_vaccine(self, did: str) -> bool:
        return "J07" in self.get_drug_atc_code_list_by_id(did, 2)

    @functools.lru_cache(maxsize=None)
    def is_enzyme_by_atc_prefix(self, did: str) -> bool:
        atc_enzyme_list_l4 = ["A09AA", "A09AC", "A16AB", "B01AD", "B06AA", "C04AF", "D03BA", "M09AB"]
        atc_enzyme_list_l5 = ["B02AB01", "B02AB02"]
        for code in self.get_drug_atc_code_list_by_id(did, 4):
            if code in atc_enzyme_list_l4:
                return True
        for code in self.get_drug_atc_code_list_by_id(did, 5):
            if code in atc_enzyme_list_l5:
                return True
        return False

    @functools.lru_cache(maxsize=None)
    def is_protein_by_atc_prefix(self, did: str) -> bool:
        atc_enzyme_list_l4 = ["B02BD"]
        atc_enzyme_list_l5 = ["B05AA01", "B05AA08", "B05AA09", "B05AA10", "B05BA04"]
        for code in self.get_drug_atc_code_list_by_id(did, 4):
            if code in atc_enzyme_list_l4:
                return True
        for code in self.get_drug_atc_code_list_by_id(did, 5):
            if code in atc_enzyme_list_l5:
                return True
        return False


    @staticmethod
    def get_drug_atc_code_length(level: int):
        assert level in range(1, 6)
        return ATC_LEVEL_LEN[level]

    @functools.lru_cache(maxsize=6)
    def get_all_atc_code_by_tree_on_level(self, level=5) -> list[str]:
        if 1 <= level <= 5:
            select_map = self.drug_atc_tree_map
        elif level == 6:
            select_map = self.drug_atc_node_map
        else:
            raise Exception(f"wrong level:{level}")
        node_list = self.filter_horizontal_level_node(select_map, level)
        return sorted([x.entry_id for x in node_list])

    def trans_atc_code_to_index(self, code_list: Union[list[str], tuple[str]], level) -> list[int]:
        all_code_list = self.get_all_atc_code_by_tree_on_level(level)
        return [all_code_list.index(x) for x in code_list]

    @functools.lru_cache(maxsize=None)
    def trans_atc_code_to_multi_hot(self, code_list: tuple[str], level) -> list[int]:
        all_code_index = self.trans_atc_code_to_index(code_list, level)
        max_len = len(self.get_all_atc_code_by_tree_on_level(level))
        ret = [0] * max_len
        for idx in all_code_index:
            ret[idx] = 1
        return ret


if __name__ == "__main__":
    atc = DrugATC()
    # print(atc.drug_atc_root)
    # print(len(atc.drug_atc_node_map.keys()))
    # print(len(atc))
    #
    did = "D00943"
    # node = atc.get_atc_node_by_id(did)
    # print(node.level, node.name, node.entry_id)
    #
    for l in range(1, 6):
        print(atc.get_drug_atc_code_list_by_id(did, l))

    # print(d.get_mol_by_id_from_file(did))

    # print(d.get_drug_graph_by_id(did))

    # print(d.get_drug_graph_by_id(""))

    # print(len(d.drug_graph_map))

    print(len(atc.get_atc_id_list()), atc.get_atc_id_list())

    print(len(atc.filter_horizontal_level_node(atc.drug_atc_tree_map, 0)))  # 0
    print(len(atc.filter_horizontal_level_node(atc.drug_atc_tree_map, 1)))  # 14
    print(len(atc.filter_horizontal_level_node(atc.drug_atc_tree_map, 2)))  # 93
    print(len(atc.filter_horizontal_level_node(atc.drug_atc_tree_map, 3)))  # 270  # 271
    print(len(atc.filter_horizontal_level_node(atc.drug_atc_tree_map, 4)))  # 922  # 933
    print(len(atc.filter_horizontal_level_node(atc.drug_atc_tree_map, 5)))  # 5372 # 5495
    print(len(atc.filter_horizontal_level_node(atc.drug_atc_tree_map, 6)))  # 0
    print(len(atc.filter_horizontal_level_node(atc.drug_atc_node_map, 6)))  # 5724 # 5841

    print(atc.get_all_atc_code_by_tree_on_level(1))
    print(atc.trans_atc_code_to_index(["A", "J", "M"], 1))
    print(atc.trans_atc_code_to_multi_hot(tuple(["A", "J", "M"]), 1))

    print(atc.trans_atc_code_to_index(["A01", "A16"], 2))
    print(atc.trans_atc_code_to_multi_hot(tuple(["A01", "A16"]), 2))
