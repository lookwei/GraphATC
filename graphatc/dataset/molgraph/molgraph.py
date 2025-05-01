import functools
import logging

from tqdm import tqdm

from graphatc.dataset.common import const, util
from graphatc.dataset.mol import DrugMol


class MolGraph(DrugMol):
    def __init__(self, br08303_version=None, *args, **kwargs):
        super().__init__(br08303_version=br08303_version, *args, **kwargs)

    @staticmethod
    def get_graph_component_num(g) -> int:
        n, u, v = util.get_graph_n_u_v(g)
        return util.count_components(n, u=u, v=v)

    @classmethod
    def get_component_num(cls, item) -> int:
        if type(item) == const.TYPE_MOL:
            return cls.get_mol_component_num(item)
        elif type(item) == const.TYPE_GRAPH:
            return cls.get_graph_component_num(item)
        else:
            raise Exception(f"func get_component_num: item type err, get: {type(item)}")

    @classmethod
    def mol_to_graph(cls, mol, polymer_method: str = None):
        if cls.is_mol_has_star(mol):
            improved_mol = util.improve_polymer(mol, polymer_method)
            return util.mol_to_graph_v2(improved_mol)
        else:
            return util.mol_to_graph_v2(mol)

    @classmethod
    def get_drug_graph_by_mol(cls, mol: const.TYPE_MOL, polymer_method: str = None):
        if mol is None:
            return None
        return cls.mol_to_graph(mol, polymer_method=polymer_method)

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get_drug_graph_by_id(cls, did: str, polymer_method: str = None):
        mol = cls.get_mol_by_id_from_file(did)
        return cls.get_drug_graph_by_mol(mol, polymer_method=polymer_method)

    @classmethod
    def get_graph_map(cls, did_list: list, polymer_method: str) -> dict[str, const.TYPE_GRAPH]:
        graph_map = {}

        for did in did_list:
            g = cls.get_drug_graph_by_id(did, polymer_method)
            if g is None:
                logging.warning(f'func get_graph_map:  '
                                f'graph is None, drug_id={did}')
                continue
            graph_map[did] = g

        if len(did_list) != len(graph_map.keys()):
            logging.warning(f'func get_graph_map:  '
                            f'len(did_list)={len(did_list)} != len(graph_map.keys())={len(graph_map.keys())}')

        return graph_map

    @classmethod
    def get_graph_map_by_mol_map(cls, mol_map: dict[str, const.TYPE_MOL],
                                 polymer_method: str) -> dict[str, const.TYPE_GRAPH]:
        graph_map = {}

        bar = tqdm(mol_map.items())
        for entry_id, mol in bar:
            bar.set_description(f"mol->graph: {entry_id}")

            g = cls.get_drug_graph_by_mol(mol, polymer_method=polymer_method)
            if g is None:
                logging.warning(f'func get_graph_map:  '
                                f'graph is None, entry_id={entry_id}')
                continue
            graph_map[entry_id] = g

        if len(mol_map.keys()) != len(graph_map.keys()):
            logging.warning(f'func get_graph_map:  '
                            f'len(mol_map.keys())={len(mol_map.keys())} != '
                            f'len(graph_map.keys())={len(graph_map.keys())}')

        return graph_map

    @classmethod
    def get_graph_split_map_by_graph_map(cls, graph_map: dict[str, const.TYPE_GRAPH]) -> dict[str, list]:
        graph_split_map = {}

        bar = tqdm(graph_map.items())
        for entry_id, g in bar:
            bar.set_description(f"split graph: {entry_id}")
            try:
                graph_split_map[entry_id] = util.split_group(g)
            except:
                continue

        if len(graph_split_map.keys()) != len(graph_map.keys()):
            logging.warning(f'func get_graph_map:  '
                            f'len(graph_split_map.keys())={len(graph_split_map.keys())} != '
                            f'len(graph_map.keys())={len(graph_map.keys())}')

        return graph_split_map

    @staticmethod
    def get_graph_cuda_map(g_map: dict[str, const.TYPE_GRAPH], cuda_idx: int = 0) -> dict[str, const.TYPE_GRAPH]:
        return {k: v.to(f"cuda:{cuda_idx}") for k, v in g_map.items()}
