from dgllife.utils import PretrainAtomFeaturizer, PretrainBondFeaturizer
from rdkit import Chem
from graphatc.dataset.common import util
import numpy as np
import dgl.backend as F
from dgllife.utils import mol_to_bigraph


class PretrainAtomFeaturizerV2(PretrainAtomFeaturizer):
    def __init__(self):
        super().__init__()
        self._atomic_number_types = list(range(1, 119 + 2))  # 1-118 -> 0-120, add *, R

    def __call__(self, mol):
        if not util.is_mol_has_star(mol) and not util.is_mol_has_R(mol):
            return super().__call__(mol)

        atom_features = []
        num_atoms = mol.GetNumAtoms()
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            if atom.GetSymbol() == "*":
                atom_features.append([
                    119 - 1,
                    self._chiral_types.index(atom.GetChiralTag())
                ])
            elif "R" in atom.GetSymbol():
                atom_features.append([
                    120 - 1,
                    self._chiral_types.index(atom.GetChiralTag())
                ])
            else:
                atom_features.append([
                    self._atomic_number_types.index(atom.GetAtomicNum()),
                    self._chiral_types.index(atom.GetChiralTag())
                ])
        atom_features = np.stack(atom_features)
        atom_features = F.zerocopy_from_numpy(atom_features.astype(np.int64))

        return {
            'atomic_number': atom_features[:, 0],
            'chirality_type': atom_features[:, 1]
        }


class PretrainBondFeaturizerV2(PretrainBondFeaturizer):
    def __init__(self):
        super().__init__(bond_direction_types=[
            Chem.rdchem.BondDir.NONE,
            Chem.rdchem.BondDir.ENDUPRIGHT,
            Chem.rdchem.BondDir.ENDDOWNRIGHT,
            Chem.rdchem.BondDir.EITHERDOUBLE,  # D10259, D03418...
            Chem.rdchem.BondDir.BEGINWEDGE  # "D03561"
        ])


def try_node_featurizer_v2(mol):
    node_featurizer, edge_featurizer = PretrainAtomFeaturizer(), PretrainBondFeaturizer()
    node_featurizer_v2 = PretrainAtomFeaturizerV2()

    try:
        graph = mol_to_bigraph(mol, add_self_loop=True,
                               node_featurizer=node_featurizer,
                               edge_featurizer=edge_featurizer,
                               canonical_atom_order=False)
        print("node_featurizer success")
    except:
        print("node_featurizer failed")

    try:
        graph = mol_to_bigraph(mol, add_self_loop=True,
                               node_featurizer=node_featurizer_v2,
                               edge_featurizer=edge_featurizer,
                               canonical_atom_order=False)
        print("node_featurizer_v2 success")
    except:
        print("node_featurizer_v2 failed")


if __name__ == "__main__":
    atom_featurizer = PretrainAtomFeaturizer()
    atom_featurizer_v2 = PretrainAtomFeaturizerV2()

    mol = Chem.MolFromSmiles('CCO')
    print(atom_featurizer(mol))

    mol = Chem.MolFromSmiles('[H+]')
    print(atom_featurizer(mol))

    did = "D05545"
    mol = Chem.MolFromMolFile(f"../drug_mol/mol/{did}.mol")

    try:
        print(atom_featurizer(mol))
    except:
        print("atom_featurizer(mol) failed")
    print(atom_featurizer_v2(mol))

    util.draw(mol)

    util.get_mol_star_id_list(mol, verbose=True)  # 0, 3

    star_atom = mol.GetAtoms()[0]

    print(mol.GetAtoms()[0].GetSymbol())  # *
    print(mol.GetAtoms()[0].GetAtomicNum())  # 0

    try_node_featurizer_v2(mol)

    did = "D00857"  # R
    mol = Chem.MolFromMolFile(f"../drug_mol/mol/{did}.mol")

    try:
        print(atom_featurizer(mol))
    except:
        print("atom_featurizer(mol) failed")
    print(atom_featurizer_v2(mol))

    util.draw(mol)

    util.get_mol_R_id_list(mol, verbose=True)  # 9

    R_atom = mol.GetAtoms()[9]

    print(mol.GetAtoms()[9].GetSymbol())  # R
    print(mol.GetAtoms()[9].GetAtomicNum())  # 0

    try_node_featurizer_v2(mol)
