import json
import os

from graphatc.dataset.common.const import DATASET_PATH
from graphatc.dataset.common.client import KeggClient
from graphatc.dataset.kegg.keggbr import KeggBr, BrNode


class DrugGroup(KeggBr):
    drug_group_node_map = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_br08330_json()

    @classmethod
    def load_br08330_json(cls):
        if not os.path.exists(f"{DATASET_PATH}/drug_group/br08330_drug_group.json"):
            cls.download_br08330_br()

        with open(f"{DATASET_PATH}/drug_group/br08330_drug_group.json", "r") as f:
            cls.build_drug_group_tree(json.load(f))

    @classmethod
    def download_br08330_br(cls):
        with open(f"{DATASET_PATH}/drug_group/br08330_drug_group.json", "w") as f:
            f.write(KeggClient.download_brite_json(cls.br_id))

    @classmethod
    def build_drug_group_tree(cls, d: dict):
        cls.drug_group_root, cls.drug_group_node_map, cls.drug_group_tree_map = cls.build_tree(d)

    @classmethod
    def get_group_node_by_did(cls, did: str) -> BrNode:
        return cls.get_node_by_id_from_map(did, cls.drug_group_node_map)

    @classmethod
    def get_group_drug_node_list_by_did(cls, did: str) -> list[BrNode]:
        node: BrNode = cls.get_group_node_by_did(did)
        if node is None:
            return []
        # assert len(node.fa) == 1
        fa = node.fa[0]
        return fa.child

    @classmethod
    def get_group_drug_id_name_list_by_did(cls, did: str) -> list[list[str]]:
        node_list = cls.get_group_drug_node_list_by_did(did)
        return sorted([[x.entry_id, x.name] for x in node_list], key=(lambda x: x[1]))

    @classmethod
    def get_base_drug_id_name_list_by_did(cls, did: str) -> list[list[str]]:
        node = cls.get_group_node_by_did(did)
        if node is None:
            return []
        group_id_name_list = cls.get_group_drug_id_name_list_by_did(did)
        base_id = group_id_name_list[0][0]
        base_node = cls.get_group_node_by_did(base_id)
        ret = [[base_node.entry_id, base_node.name]]
        if did == base_id:
            return ret
        ret.append([node.entry_id, node.name])
        return ret

    @classmethod
    def get_group_drug_id_name_diff_list_by_list(cls, id_name_list: list[list[str]]) -> list[list[str]]:
        if len(id_name_list) <= 1:
            return id_name_list
        base_name = id_name_list[0][1]
        diff_list = [[x[0], x[1].replace(base_name, "").strip()] for x in id_name_list]
        diff_list[0][1] = base_name
        return sorted(diff_list, key=(lambda x: x[1]))

    @classmethod
    def get_group_drug_id_name_diff_list_by_did(cls, did: str) -> list[list[str]]:
        return cls.get_group_drug_id_name_diff_list_by_list(
            cls.get_group_drug_id_name_list_by_did(did))

    @classmethod
    def get_base_drug_id_name_diff_list_by_did(cls, did: str) -> list[list[str]]:
        return cls.get_group_drug_id_name_diff_list_by_list(
            cls.get_base_drug_id_name_list_by_did(did))


if __name__ == "__main__":
    pass
    dg_class = DrugGroup()
    # cls = dg_class
    # did = "D01618"
    # print(dg_class.get_node_by_id("D01618"))
    # print(dg_class.get_name_postfix("D01618"))
    # print(dg_class.get_group_drug_node_list_by_did("D01618"))
    # print(dg_class.get_group_drug_id_name_list_by_did("D01618"))
    # print(dg_class.get_group_drug_id_name_diff_list_by_did("D01618"))
    # print(dg_class.get_base_drug_id_name_diff_list_by_did("D01618"))
    # print(dg_class.get_base_drug_id_name_diff_list_by_did("D08260"))
    # wrong
    print(dg_class.get_base_drug_id_name_diff_list_by_did("D06516"))
    print(dg_class.get_base_drug_id_name_diff_list_by_did("D03418"))
