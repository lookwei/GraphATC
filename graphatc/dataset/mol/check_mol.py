import os
import pandas as pd

from tqdm import tqdm
from rdkit.Chem import MolFromMolFile
from common import util

drug_id_df = pd.read_csv("drug_id_name.csv")

drug_id_list = drug_id_df['drug_id'].to_list()

bar = tqdm(drug_id_list)

empty_mol_drug_id = []

not_exist_mol_file = []

mol_is_none = []

failed_convert_graph = []

failed_convert_graph_2 = []

has_star = []

has_R = []

for did in bar:
    bar.set_description(did)
    file_path = f"mol/{did}.mol"

    if not os.path.exists(file_path):
        not_exist_mol_file.append(did)
        empty_mol_drug_id.append(did)
        continue

    with open(file_path, "rb") as f:
        if len(f.read()) == 0:
            empty_mol_drug_id.append(did)
            continue

        mol = MolFromMolFile(file_path,
                             sanitize=False,
                             strictParsing=False)

        if mol is None:
            mol_is_none.append(did)
            continue

        if util.is_mol_has_star(mol):
            has_star.append(did)

        if util.is_mol_has_R(mol):
            has_R.append(did)

        try:
            g = util.mol_to_graph(mol)
        except:
            failed_convert_graph.append(did)

        try:
            g = util.mol_to_graph_v2(mol)
        except:
            failed_convert_graph_2.append(did)

print(empty_mol_drug_id)

print(len(empty_mol_drug_id))  # 3249

with open("check_mol/not_exist_mol_file.txt", "w") as f:
    for line in not_exist_mol_file:
        f.write(line + '\n')

with open("check_mol/mol_is_none.txt", "w") as f:
    for line in mol_is_none:
        f.write(line + '\n')


with open("check_mol/empty_mol_drug_id.txt", "w") as f:
    for line in empty_mol_drug_id:
        f.write(line + '\n')

with open("check_mol/has_star.txt", "w") as f:
    for line in has_star:
        f.write(line + '\n')

with open("check_mol/has_R.txt", "w") as f:
    for line in has_R:
        f.write(line + '\n')

with open("check_mol/failed_convert_graph.txt", "w") as f:
    for line in failed_convert_graph:
        f.write(line + '\n')

with open("check_mol/failed_convert_graph_2.txt", "w") as f:
    for line in failed_convert_graph_2:
        f.write(line + '\n')