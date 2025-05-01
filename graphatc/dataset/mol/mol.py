import functools
import os
import time
import logging

import numpy as np

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors

from graphatc.dataset.common.client import KeggClient, ChemicalBookClient, PubChemClient
from graphatc.dataset.common import const, util
from graphatc.dataset.common.util import is_mol_has_star, is_mol_has_R, get_mol_n_u_v, count_components

from graphatc.dataset.flat import Flat
from graphatc.dataset.atc.drugatc import DrugATC
from graphatc.dataset.drug_group.dgroup import DrugGroup


class DrugMol(Flat, DrugATC, DrugGroup):
    def __init__(self, br08303_version=None, *args, **kwargs):
        super().__init__(br08303_version=br08303_version)
        self.mol_map = {}

    @staticmethod
    def is_mol_has_star(mol):
        return is_mol_has_star(mol)

    @staticmethod
    def is_mol_has_R(mol):
        return is_mol_has_R(mol)

    @staticmethod
    def get_mol_component_num(mol) -> int:
        n, u, v = get_mol_n_u_v(mol)
        return count_components(n, u=u, v=v)

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def get_mol_file_path_by_id(entry_id: str) -> str:
        prefix = entry_id[0].upper()
        assert prefix in const.MOL_PREFIX_LIST
        return f"{const.DATASET_PATH}/mol/{const.MOL_DIT_NAME_LIST[const.MOL_PREFIX_LIST.index(prefix)]}/{entry_id}.mol"

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get_mol_by_id_from_file(cls, entry_id: str):
        try:
            return Chem.MolFromMolFile(cls.get_mol_file_path_by_id(entry_id), sanitize=False)
        except OSError:
            return None

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get_mol_by_id_from_manual(cls, entry_id: str):
        try:
            return Chem.MolFromMolFile(f"{const.DATASET_PATH}/mol/manual/{entry_id}.mol", sanitize=False)
        except OSError:
            return None

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get_mol_by_id_from_kegg(cls, entry_id: str, save=True):
        logging.info(f"send get mol req to kegg, entry_id={entry_id}")
        mol_file = KeggClient.get_mol_file_by_id(entry_id)
        if len(mol_file) == 0:
            return None
        mol = Chem.MolFromMolBlock(mol_file)
        if mol is None:
            return None
        if save:
            with open(cls.get_mol_file_path_by_id(entry_id), "w") as f:
                f.write(mol_file)
        return mol

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get_mol_by_id_from_chemical_book(cls, entry_id: str, save=True):
        cas = cls.get_cas_by_id(entry_id)
        if len(cas) == 0:
            return None

        logging.info(f"send get mol req to chemical_book, entry_id={entry_id}, cas={cas}")
        mol_file = ChemicalBookClient.get_mol_file_by_cas(cas)
        if len(mol_file) == 0:
            return None
        mol = Chem.MolFromMolBlock(mol_file)
        if mol is None:
            return None
        if save:
            Chem.MolToMolFile(mol, cls.get_mol_file_path_by_id(entry_id))
            Chem.MolToMolFile(mol, f"{const.DATASET_PATH}/mol/cas/{cas}.mol")
        return mol

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get_mol_by_id_from_pubchem(cls, entry_id: str, save_new=True) -> [const.TYPE_MOL, None]:
        drug_name = cls.get_name_list_by_id(entry_id)
        drug_name = util.list_remove_duplicate(drug_name)
        for name in drug_name:
            logging.info(f"send get mol req to pubchem, entry_id={entry_id}, name={name}")
            sdf = PubChemClient.get_sdf_by_name(name)
            if len(sdf) <= 0:
                continue
            file_path = f"{const.DATASET_PATH}/mol/pubchem/{name}.sdf"
            with open(file_path, "w") as f:
                f.write(sdf)
            mol = Chem.MolFromMolFile(file_path)
            if mol is None:
                continue
            if save_new:
                Chem.MolToMolFile(mol, cls.get_mol_file_path_by_id(entry_id))
            return mol
        return None

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get_same_as_compound_mol(cls, did: str, save=True):
        same_as_cid = cls.get_same_as_compound_id_by_id(did)
        if len(same_as_cid) == 0:
            return None

        same_mol = cls.try_get_mol_by_id_from_everywhere(same_as_cid)
        if same_mol is None:
            return None

        if save:
            Chem.MolToMolFile(same_mol, cls.get_mol_file_path_by_id(did))
        return same_mol

    @classmethod
    def get_mol_from_drug_group(cls, did: str, save_new=True) -> [const.TYPE_MOL, None]:
        if len(did) == 0 or did[0].upper() != 'D':
            return None

        if len(cls.get_chemical_structure_group_by_id(did)) == 0:
            return None

        diff_list = cls.get_base_drug_id_name_diff_list_by_did(did)
        if len(diff_list) == 0:
            return None

        base_id = diff_list[0][0]
        if base_id == did:
            return

        base_mol = cls.try_get_mol_by_id_from_everywhere(base_id)
        if len(diff_list) == 1 or base_mol is None:
            return base_mol

        diff_name = diff_list[1][1]
        diff_mol_name = diff_name.split(" ")
        merged_mol = base_mol
        for _diff_mol_name in diff_mol_name:
            diff_file_path = f"{const.DATASET_PATH}/mol/group_diff/{_diff_mol_name.lower()}.mol"
            if not os.path.exists(diff_file_path):
                logging.warning(f"not find diff_mol_name: {_diff_mol_name}")
                return None

            diff_mol = Chem.MolFromMolFile(diff_file_path, sanitize=False)
            if diff_mol is None:
                return None

            merged_mol = Chem.CombineMols(merged_mol, diff_mol)
            if merged_mol is None:
                return None

        if save_new:
            Chem.MolToMolFile(merged_mol, f"{const.DATASET_PATH}/mol/drug_mol/{did}.mol")
        return merged_mol

    @classmethod
    def get_mol_by_component(cls, did: str, save_new: bool = True) -> [const.TYPE_MOL, None]:
        component_id_list = cls.get_component_id_leaf_list_by_id(did)

        if len(component_id_list) == 0:
            return None

        mol_list = [cls.try_get_mol_by_id_from_everywhere(x, save_new) for x in component_id_list]

        for idx, _mol in enumerate(mol_list):
            if _mol is None:
                logging.warning(
                    f"func get_mol_by_component, try_get_mol_by_id_from_everywhere: mol is None, id={component_id_list[idx]}")

        mol_list_not_none = list(filter(lambda x: x is not None, mol_list))

        if len(mol_list_not_none) == 0:
            return None

        merged_mol = mol_list_not_none[0]
        for _mol in mol_list_not_none[1:]:
            if _mol is not None:
                merged_mol = Chem.CombineMols(merged_mol, _mol)

        if save_new:
            Chem.MolToMolFile(merged_mol, cls.get_mol_file_path_by_id(did))

        return merged_mol

    @classmethod
    def get_mixture_mol(cls, did: str, save_new: bool = True) -> [const.TYPE_MOL, None]:
        if not cls.is_mixture_by_id(did):
            return None
        return cls.get_mol_by_component(did, save_new)

    @classmethod
    def get_crude_mol(cls, did: str, save_new: bool = True) -> [const.TYPE_MOL, None]:
        if not cls.is_crude_by_id(did):
            return None
        return cls.get_mol_by_component(did, save_new)

    @classmethod
    def try_get_mol_by_id_from_everywhere(cls, entry_id: str, save_new=True):
        # manual
        mol = cls.get_mol_by_id_from_manual(entry_id)
        if mol is not None:
            return mol

        # local
        mol = cls.get_mol_by_id_from_file(entry_id)
        if mol is not None:
            return mol

        # kegg_api
        mol = cls.get_mol_by_id_from_kegg(entry_id, save_new)
        if mol is not None:
            return mol

        # same_as
        mol = cls.get_same_as_compound_mol(entry_id)
        if mol is not None:
            return mol

        # cas
        mol = cls.get_mol_by_id_from_chemical_book(entry_id, save_new)
        if mol is not None:
            return mol

        # pubchem
        mol = cls.get_mol_by_id_from_pubchem(entry_id, save_new)
        if mol is not None:
            return mol

        # group
        mol = cls.get_mol_from_drug_group(entry_id, save_new)
        if mol is not None:
            return mol

        # mixture
        mol = cls.get_mixture_mol(entry_id, save_new)
        if mol is not None:
            return mol

        # crude
        mol = cls.get_crude_mol(entry_id, save_new)
        if mol is not None:
            return mol

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get_mol_by_id(cls, entry_id: str, save_new=True, log_info=True):

        mol = cls.try_get_mol_by_id_from_everywhere(entry_id, save_new)
        if mol is None:
            logging.warning(f"func get_mol_by_id: mol is None, id={entry_id}")
            if log_info:
                with open(f"{const.DATASET_PATH}/log/log.get_mol_by_id.failed."
                          f"{time.strftime('%Y%m%d', time.localtime())}", 'a') as f:
                    f.write(entry_id)
                    f.write("\n")
            return None
        return mol

    @classmethod
    def get_mol_map(cls, did_list: list, check_len=True) -> dict[str, const.TYPE_MOL]:
        mol_map = {}

        skip_list = None
        with open(f"{const.DATASET_PATH}/log/log.get_mol_map.skip", 'rb') as f:
            skip_list = f.read().decode("utf-8").splitlines()

        bar = tqdm(did_list)
        for did in bar:
            bar.set_description(f"get mol: {did}")

            if len(skip_list) > 0 and did in skip_list:
                continue

            mol = cls.get_mol_by_id(did)
            if mol is None:
                continue

            mol_map[did] = mol

        # if check_len and len(did_list) != len(mol_map.keys()):
        #     logging.warning(f'func get_mol_map:  '
        #                     f'len(did_list)={len(did_list)} != len(mol_map.keys())={len(mol_map.keys())}')

        return mol_map

    def init_atc_mol_map(self):
        atc_mol_map = self.get_mol_map(self.get_atc_id_list())
        self.mol_map["atc"] = atc_mol_map
        return atc_mol_map

    @staticmethod
    def get_mol_weight_by_mol(mol: const.TYPE_MOL) -> float:
        try:
            return Chem.Descriptors.MolWt(mol)
        except RuntimeError:
            mol.UpdatePropertyCache(strict=False)
            return Chem.Descriptors.MolWt(mol)
        except:
            return -1

    @staticmethod
    def update_property_cache_on_map(mol_map: dict[str, const.TYPE_MOL]):
        for k, v in mol_map.items():
            v.UpdatePropertyCache(strict=False)

    @classmethod
    def get_mol_weight_list_by_mol_map(cls, mol_map: dict[str, const.TYPE_MOL]) -> list[float]:
        return [cls.get_mol_weight_by_mol(x) for x in mol_map.values()]
    
    @classmethod
    def get_mol_weight_map_by_mol_map(cls, mol_map: dict[str, const.TYPE_MOL]) -> list[float]:
        return {k:cls.get_mol_weight_by_mol(v) for k,v in mol_map.items()}

    @classmethod
    def get_mol_component_num_list_by_mol_map(cls, mol_map: dict[str, const.TYPE_MOL]) -> list[float]:
        return [cls.get_mol_component_num(x) for x in mol_map.values()]

    def plt_atc_mol_weight_hist(self):
        weight_list = self.get_mol_weight_list_by_mol_map(self.mol_map["atc"])
        util.plt_log_hist_fig(weight_list, bins=np.arange(0, 12000, 500))

    def plt_atc_mol_component_num_hist(self):
        component_num_list = self.get_mol_component_num_list_by_mol_map(self.mol_map["atc"])
        max_num = max(component_num_list)  # 25
        util.plt_log_hist_fig(component_num_list, bins=range(1, max_num + 1, 1), xticks=range(1, max_num + 1, 1))

    @classmethod
    def get_mol_type_map_by_map(cls, mol_map: dict[str, const.TYPE_MOL]) -> dict[str, list[str]]:
        type_list_map = {const.TYPE_SMALL_MOLECULE: [], const.TYPE_BIG_MOLECULE: [],
                         const.TYPE_NOT_POLYMER: [], const.TYPE_POLYMER: [],
                         const.TYPE_SINGLE_COMPONENT: [], const.TYPE_MULTI_COMPONENT: []}
        for k, v in mol_map.items():
            if cls.get_mol_weight_by_mol(v) < const.TYPE_SMALL_BIG_SPLIT:
                type_list_map[const.TYPE_SMALL_MOLECULE].append(k)
            else:
                type_list_map[const.TYPE_BIG_MOLECULE].append(k)

            if not cls.is_mol_has_star(v):
                type_list_map[const.TYPE_NOT_POLYMER].append(k)
            else:
                type_list_map[const.TYPE_POLYMER].append(k)

            if cls.get_mol_component_num(v) == 1:
                type_list_map[const.TYPE_SINGLE_COMPONENT].append(k)
            else:
                type_list_map[const.TYPE_MULTI_COMPONENT].append(k)
        return type_list_map


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().setLevel(logging.INFO)

    dm_class = DrugMol()

    # mol = dm_class.get_mol_by_id_from_file("D00943")
    # print(mol)
    #
    # mol = dm_class.get_mol_by_id_from_file("D06816")
    # print(mol)
    #
    # mol = dm_class.get_mol_by_id_from_file("C00002")
    # print(mol)
    #
    # mol = dm_class.get_mol_by_id_from_file("C10453")
    # print(mol)
    #
    # mol = dm_class.get_mol_by_id_from_kegg("C10453")
    # print(mol)
    #
    # print(dm_class.get_mol_by_id_from_local_and_kegg("D02106"))  # None
    # print(dm_class.get_same_as_compound_mol("D02106"))  # Mol
    #
    # mol_list = dm_class.get_component_mol_list_by_id("D06816", verbose=True)
    #
    # dm_class.get_mol_by_id("D06816")

    # print(dm_class.get_mol_by_id("D10271"))

    # print(DrugMol.get_mol_by_id_from_chemical_book("D02798"))

    # draw(dm_class.get_mol_from_drug_group("D05142"))
    # draw(dm_class.get_mol_from_drug_group("D08260"))

    # draw(dm_class.get_mol_from_drug_group("D10889", save_new=False))

    dm_class.init_atc_mol_map()
    # dm_class.try_get_mol_by_id_from_everywhere("D03065")

    # dm_class.try_get_mol_by_id_from_everywhere("D04461")

    # enzyme_by_atc_prefix_list = []
    # protein_by_atc_prefix = []
    # for k in dm_class.drug_atc_map.keys():
    # if dm_class.is_vaccine(k):
    #     print(k)
    # if dm_class.is_enzyme_by_atc_prefix(k):
    #     enzyme_by_atc_prefix_list.append(k)
    #     print(k)
    # if dm_class.is_protein_by_atc_prefix(k):
    #     protein_by_atc_prefix.append(k)
    #     print(k)

    # m = Chem.MolFromMolFile("group_diff/Defibrotide.sdf")
    # draw(m)
    # Chem.MolToMolFile(m, "group_diff/Defibrotide.mol")

    # m = Chem.MolFromMolFile("manual/D05183.sdf")
    # draw(m)
    # Chem.MolToMolFile(m, "manual/D05183.mol")

    # m = Chem.CombineMols(Chem.MolFromMolFile("group_diff/Defibrotide.mol"),
    #                      Chem.MolFromMolFile("group_diff/sodium.mol"))
    # draw(m)
    # Chem.MolToMolFile(m, "manual/D07423.mol")

    # weight_list = dm_class.get_mol_weight_list_by_mol_map(dm_class.mol_map["atc"])
    dm_class.plt_atc_mol_weight_hist()
    dm_class.plt_atc_mol_component_num_hist()
    mol_type_map = dm_class.get_mol_type_map_by_map(dm_class.mol_map["atc"])
