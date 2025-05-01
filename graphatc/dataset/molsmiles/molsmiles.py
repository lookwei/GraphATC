import re
import functools
from rdkit import Chem
from graphatc.dataset.common import const
from graphatc.dataset.mol import DrugMol


class MolSmiles(DrugMol):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def mol_to_smiles(cls, mol: const.TYPE_MOL) -> const.TYPE_SMILES:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

    @classmethod
    def split_smiles(cls, smiles: const.TYPE_SMILES) -> list:
        result = re.findall(const.PATTERN_SPLIT_SMILES, smiles)
        return list(filter(lambda x: x is not None and len(x) > 0, result))

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get_smiles_by_did(cls, entry_id: str):
        mol = cls.get_mol_by_id(entry_id)
        smiles = cls.mol_to_smiles(mol)
        return smiles

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get_splited_smiles_by_did(cls, entry_id: str):
        return cls.post_process_splited_smiles(cls.split_smiles(cls.get_smiles_by_did(entry_id)))

    @classmethod
    def post_process_splited_smiles(cls, splited_list: list) -> list:
        # return ["&"] + splited_list + ["/n"]
        return splited_list

    def get_splited_smiles_map(self, did_list: list = None, mol_map: dict = None, check_len=True) -> dict[str, list]:
        if mol_map is None:
            if did_list is None:
                did_list = self.get_atc_id_list()
            mol_map = self.get_mol_map(did_list, check_len)

        splited_smiles_map = {k: self.post_process_splited_smiles(self.split_smiles(self.mol_to_smiles(v)))
                              for k, v in mol_map.items()}
        return splited_smiles_map

    def get_smiles_pad_cut_map(self, splited_smiles_map: dict[str, list], vocab_dict: dict[str, int], trans_idx=True,
                               pad_token: str = "<PAD>", pad_cut_len=787) -> dict[str, list]:
        assert pad_token in vocab_dict
        ret = {}
        for k, v in splited_smiles_map.items():
            if len(v) >= pad_cut_len:
                cut_put = v[:pad_cut_len]
            else:
                cut_put = v + [pad_token] * (pad_cut_len - len(v))
            assert len(cut_put) == pad_cut_len

            if trans_idx:
                cut_put_idx = [vocab_dict[x] for x in cut_put]
                ret[k] = cut_put_idx  # list[int]
            else:
                ret[k] = cut_put  # list[str]
        return ret

    @classmethod
    def get_vocab_count(cls, splited_smiles_map: dict[str, list]) -> dict[str, int]:
        count_map = {}
        for _, v in splited_smiles_map.items():
            for v_item in v:
                count_map[v_item] = count_map.get(v_item, 0) + 1
        return count_map

    @classmethod
    def get_vocab_dic(cls, vocab_count: dict[str, int], pad_token: str = "<PAD>") -> dict[str, int]:
        vocab_tuple = sorted(vocab_count.items(), key=lambda x: x[1], reverse=True)
        vocab_dic = {k: c for c, (k, _) in enumerate(vocab_tuple)}
        vocab_dic[pad_token] = len(vocab_dic)
        return vocab_dic
