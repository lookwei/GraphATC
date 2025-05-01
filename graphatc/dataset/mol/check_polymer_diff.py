from rdkit.Chem import MolFromMolFile, Draw
from common import util
from common.const import *


def check_improve_polymer(did):
    mol = MolFromMolFile(f"mol/{did}.mol")
    util.draw(mol)
    # util.draw(util.improve_polymer(mol, IMPROVE_POLYMER_NONE))
    util.draw(util.improve_polymer(mol, IMPROVE_POLYMER_DELETE_STAR))
    util.draw(util.improve_polymer(mol, IMPROVE_POLYMER_CONNECT_STAR))
    util.draw(util.improve_polymer(mol, IMPROVE_POLYMER_DELETE_STAR_CONNECT_ATOM))


if __name__ == "__main__":
    # polymer 2 *
    check_improve_polymer("D00060")

    # polymer 3 *
    check_improve_polymer("D07434")

    # polymer multi
    check_improve_polymer("D02898")
    check_improve_polymer("D10474")
