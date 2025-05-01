from rdkit.Chem import MolFromMolFile, Draw
from common import util


if __name__ == "__main__":

    did = "D02656"
    mol = MolFromMolFile(f"mol/{did}.mol")
    util.draw(mol)

    did = "D00060"
    mol = MolFromMolFile(f"mol/{did}.mol")
    util.draw(mol)

    print(util.mol_to_graph_v2(mol))