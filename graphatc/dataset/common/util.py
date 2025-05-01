import matplotlib.pyplot as plt
import numpy as np

from dgllife.utils import mol_to_bigraph
from dgllife.utils import PretrainAtomFeaturizer, PretrainBondFeaturizer
from rdkit import Chem
from rdkit.Chem import MolFromMolFile, Draw

from .featurizer import PretrainAtomFeaturizerV2, PretrainBondFeaturizerV2
from .const import *

node_featurizer, edge_featurizer = PretrainAtomFeaturizer(), PretrainBondFeaturizer()
node_featurizer_v2 = PretrainAtomFeaturizerV2()
bond_featurizer_v2 = PretrainBondFeaturizerV2()


def list_remove_duplicate(one_list: list[str]) -> list[str]:
    """
    In: list(set(["A","A"])) \n
    Out: ['A']
    """
    return list(set(one_list))


def double_list_remove_duplicate(dlist: list[list[str]]) -> list[list[str]]:
    """
    In: double_list_remove_duplicate([["A"],["A"]]) \n
    Out: [['A']]
    """
    dlist = list(set([tuple(t) for t in dlist]))
    dlist = [list(v) for v in dlist]
    return dlist


def mol_to_graph(mol):
    assert mol is not None
    try:
        graph = mol_to_bigraph(mol, add_self_loop=True,
                               node_featurizer=node_featurizer,
                               edge_featurizer=edge_featurizer,
                               canonical_atom_order=False)
        return graph
    except:
        return None


def mol_to_graph_v2(mol):
    assert mol is not None
    try:
        graph = mol_to_bigraph(mol, add_self_loop=True,
                               node_featurizer=node_featurizer_v2,
                               edge_featurizer=bond_featurizer_v2,
                               canonical_atom_order=False)
        return graph
    except:
        return None


def is_mol_has_star(mol):
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "*":
            return True
    return False


def get_mol_star_id_list(mol, verbose=False):
    id_star = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "*":
            id_star.append(atom.GetIdx())
    if verbose:
        print("find * at: ", id_star)
    return id_star


def is_mol_has_R(mol):
    for atom in mol.GetAtoms():
        if "R" in atom.GetSymbol():
            return True
    return False


def get_mol_R_id_list(mol, verbose=False):
    id_star = []
    for atom in mol.GetAtoms():
        if "R" in atom.GetSymbol():
            id_star.append(atom.GetIdx())
    if verbose:
        print("find R at: ", id_star)
    return id_star


def draw(obj, res=0, size=600):
    if type(obj) == str:
        mol = Chem.MolFromSmiles(obj)
    elif isinstance(obj, type(Chem.MolFromSmiles(''))):
        mol = obj
    else:
        raise Exception("Error obj, not smiles or mol")
    if res == 0:
        pil_fig = Draw.MolsToImage([mol], (size, size))
        plt.imshow(np.array(pil_fig), )
        plt.show()
    elif res == 1:
        Draw.ShowMol(mol, size=(size, size))
    else:
        raise Exception(f"Error res: {res}")


def rw_mol_add_bond(rwmol: Chem.RWMol, bond_type: Chem.BondType, atom_idx_list: list):
    for idx_1 in atom_idx_list:
        for idx_2 in atom_idx_list:
            if idx_1 < idx_2:
                if rwmol.GetBondBetweenAtoms(idx_1, idx_2) is None:
                    rwmol.AddBond(idx_1, idx_2, bond_type)


def rw_mol_remove_atom(rwmol: Chem.RWMol, atom_idx_list_remove: list, atom_idx_list_neighbor: list):
    for i, j in zip(atom_idx_list_remove, atom_idx_list_neighbor):
        rwmol.RemoveBond(i, j)
    for _id_star in reversed(sorted(atom_idx_list_remove)):
        rwmol.RemoveAtom(_id_star)

def get_mol_star_id_list(mol, verbose=False):
    id_star = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "*":
            id_star.append(atom.GetIdx())
    if verbose:
        print("find * at: ", id_star)
    return id_star

def get_mol_star_atom_id_list(mol, id_star, verbose=False):
    
    id_star_atom = []
    for _id_star in id_star:
        assert len(mol.GetAtomWithIdx(_id_star).GetBonds()) == 1
        _atom = mol.GetAtomWithIdx(_id_star).GetBonds()[0].GetBeginAtomIdx()
        if _atom == _id_star:
            _atom = mol.GetAtomWithIdx(_id_star).GetBonds()[0].GetEndAtomIdx()
        assert _atom != _id_star
        id_star_atom.append(_atom)
    
    if verbose:
        print("find * atom at: ", id_star_atom)

    assert len(id_star) == len(id_star_atom)
    
    return id_star_atom

def split_monomer_groups_from_mol(mol):
    mol_graph = mol_to_graph_v2(mol)
    return split_group(mol_graph)

def build_mol_from_group(mol, atom_group, is_remove_star=True):
    """ Given a molecule and a group of atom indices, build a new molecule from the group. """
    new_mol = Chem.RWMol()  # Create an editable molecule
    atom_map = {}  # To keep track of atom index mapping from old mol to new mol

    # Add atoms to the new molecule from the original molecule
    for atom_idx in atom_group:
        atom = mol.GetAtomWithIdx(atom_idx)
        new_idx = new_mol.AddAtom(atom)
        atom_map[atom_idx] = new_idx

    # Add bonds between atoms in the group
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if begin_idx in atom_group and end_idx in atom_group:
            new_mol.AddBond(atom_map[begin_idx], atom_map[end_idx], bond.GetBondType())

    if is_remove_star:
        id_star = get_mol_star_id_list(new_mol)
        id_star_atom = get_mol_star_atom_id_list(new_mol, id_star)
        rw_mol_remove_atom(new_mol, id_star, id_star_atom)
    return new_mol.GetMol()  # Return the constructed molecule

def split_monomer_molecules(mol, is_remove_star=True):
    """
    Splits a molecule into its monomer groups and returns a list of new molecules,
    each representing a monomer group.
    """
    monomer_groups = split_monomer_groups_from_mol(mol)  # Get the monomer groups
    monomer_mols = []

    for group in monomer_groups:
        new_mol = build_mol_from_group(mol, group, is_remove_star)  # Build a new molecule from each group
        monomer_mols.append(new_mol)

    return monomer_mols

def improve_polymer(mol, method):
    # REF: http://rdkit.org/docs/cppapi/classRDKit_1_1RWMol.html

    if method == IMPROVE_POLYMER_NONE:
        return mol

    id_star = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "*":
            id_star.append(atom.GetIdx())

    if len(id_star) == 0:
        raise Exception("func improve_polymer: isn't polymer")

    id_star_atom = []
    for _id_star in id_star:
        assert len(mol.GetAtomWithIdx(_id_star).GetBonds()) == 1
        _atom = mol.GetAtomWithIdx(_id_star).GetBonds()[0].GetBeginAtomIdx()
        if _atom == _id_star:
            _atom = mol.GetAtomWithIdx(_id_star).GetBonds()[0].GetEndAtomIdx()
        assert _atom != _id_star
        id_star_atom.append(_atom)

    assert len(id_star) == len(id_star_atom)

    mol_rw = Chem.RWMol(mol)

    if method == IMPROVE_POLYMER_DELETE_STAR:
        rw_mol_remove_atom(mol_rw, id_star, id_star_atom)
        return mol_rw.GetMol()

    if method == IMPROVE_POLYMER_CONNECT_STAR:
        rw_mol_add_bond(mol_rw, Chem.BondType.SINGLE, id_star)
        return mol_rw.GetMol()

    if method == IMPROVE_POLYMER_DELETE_STAR_CONNECT_ATOM:
        rw_mol_add_bond(mol_rw, Chem.BondType.SINGLE, id_star_atom)
        rw_mol_remove_atom(mol_rw, id_star, id_star_atom)
        return mol_rw.GetMol()

    raise Exception(f"func improve_polymer: wrong method: {method}")


def get_mol_n_u_v(mol):
    n = mol.GetNumAtoms()
    mol_bonds = mol.GetBonds()
    u = [mol.GetBonds()[i].GetBeginAtomIdx() for i in range(len(mol_bonds))]
    v = [mol.GetBonds()[i].GetEndAtomIdx() for i in range(len(mol_bonds))]
    return n, u, v


def get_graph_n_u_v(g):
    n = g.num_nodes()
    u = g.edges()[0].tolist()
    v = g.edges()[1].tolist()
    return n, u, v


def get_u_v_zip_list(u, v):
    return list(zip(u, v))


def count_components(n: int, edges: list[list[int]] or list[tuple[int]] = None, u: list = None, v: list = None) -> int:
    assert edges is not None or (u is not None and v is not None)
    edges = get_u_v_zip_list(u, v)

    f = {}

    def find(x):
        f.setdefault(x, x)
        if x != f[x]:
            f[x] = find(f[x])
        return f[x]

    def union(x, y):
        f[find(x)] = find(y)

    for x, y in edges:
        union(x, y)

    return len(set(find(x) for x in range(n)))


def split_group(g) -> list:
    n, u, v = get_graph_n_u_v(g)
    eij = np.zeros((n, n)) - 1
    for ui, vi in list(zip(u, v)):
        eij[ui][vi] = 1

    visited = np.zeros(n) - 1
    res = []

    def dfs_search(node_id, group_id):
        if visited[node_id] != -1:
            return
        # print("search", node_id, "group_id", group_id)
        curr_group.append(node_id)
        visited[node_id] = group_id
        for i in range(n):
            if eij[node_id, i] == 1:
                dfs_search(i, group_id)

    for i in range(n):
        curr_group = []
        dfs_search(i, i)
        if curr_group:
            res.append(curr_group)

    return res


def plt_log_hist_fig(data_list: list, bins=10, xticks=None):
    fig, ax = plt.subplots()

    counts, bins, patches = ax.hist(data_list, bins=bins, rwidth=0.8)

    for count, patch in zip(counts, patches):
        height = patch.get_height()
        if count != 0:
            text_height = height + 0.5
        else:
            text_height = height + 0.7
        ax.text(patch.get_x() + patch.get_width() / 2, text_height, int(count), ha='center')

    ax.set_yscale('log')

    if xticks is not None:
        ax.set_xticks(xticks)

    plt.savefig("./temp.png")

    plt.show()
