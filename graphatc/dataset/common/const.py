import os
from rdkit.Chem.rdchem import Mol
from dgl import DGLGraph

# DATASET_PATH
DATASET_PATH = os.path.abspath(os.path.dirname(__file__)) + "/.."
DATASET_PATH_ATCSMILES4545 = DATASET_PATH+"/atc-smiles-4545/ATC_SMILES.csv"
DATASET_PATH_ATCSMILES3785 = DATASET_PATH+"/atc-smiles-3785/ATC-SMILES-Aligned.csv"
DATASET_PATH_CHEN3883 = DATASET_PATH+"/chen3883/chen3883_multihot.csv"
DATASET_PATH_CHEN3883_DRUG_ID = DATASET_PATH+"/chen3883/chen3883_kegg_drug_id.txt"
DATASET_PATH_CHEN3883_FIX_11 = DATASET_PATH+"/chen3883/fix_chen_11.csv"
DATASET_PATH_CHEN3883_ERR = DATASET_PATH+"/chen3883/fix_up_to_3875_19_drug_id.txt"
DATASET_PATH_CHEN3883_FIX_L2 = DATASET_PATH+"/chen3883/del_l2_null_drug_id.txt"
# print("DATASET_PATH=", DATASET_PATH)

# WEB
WEB_UA = 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) ' \
         'Chrome/86.0.4240.75 Mobile Safari/537.36'

WEB_HEADERS = {'User-Agent': WEB_UA}

WEB_URL_KEGG_BR_JSON = "https://rest.kegg.jp/get/br:{}/json"  # fill br
WEB_URL_KEGG_GET_MOL = "https://www.genome.jp/dbget-bin/www_bget?-f+m+cpd+{}"  # fill entry id

WEB_URL_PUBCHEM_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{}/{}/{}"

WEB_URL_CHEMICAL_BOOK_GET_MOL = "https://www.chemicalbook.com/CAS/MOL/{}.mol"  # fill cas
WEB_URL_CHEMICAL_BOOK_GET_MOL_LIST = ["https://www.chemicalbook.com/CAS/MOL/{}.mol",
                                      "https://www.chemicalbook.com/CAS/20180808/MOL/{}.mol",
                                      "https://www.chemicalbook.com/CAS/20180906/MOL/{}.mol"]  # fill cas

# IMPROVE_POLYMER
IMPROVE_POLYMER_NONE = 0
IMPROVE_POLYMER_DELETE_STAR = 1
IMPROVE_POLYMER_CONNECT_STAR = 2
IMPROVE_POLYMER_DELETE_STAR_CONNECT_ATOM = 3

# TYPE
TYPE_MOL = Mol
TYPE_GRAPH = DGLGraph
TYPE_SMILES = str

TYPE_SMALL_BIG_SPLIT = 1000

TYPE_SMALL_MOLECULE = "SMALL_MOLECULE"
TYPE_BIG_MOLECULE = "BIG_MOLECULE"
TYPE_POLYMER = "POLYMER"
TYPE_NOT_POLYMER = "NOT_POLYMER"
TYPE_SINGLE_COMPONENT = "SINGLE_COMPONENT"
TYPE_MULTI_COMPONENT = "MULTI_COMPONENT"

# MOL
MOL_PREFIX_LIST = ["D", "C"]
MOL_DIT_NAME_LIST = ["drug_mol", "compound_mol"]

# FLAT
FLAT_PREFIX_LIST = MOL_PREFIX_LIST
FLAT_DIT_NAME_LIST = ["drug_flat", "compound_flat"]

FLAT_SPECIAL_FIELD_SEP = "-" * 10

FLAT_FIELD_ENTRY = "ENTRY"
FLAT_FIELD_ENTRY_MIXTURE = "Mixture"
FLAT_FIELD_ENTRY_CRUDE = "Crude"
FLAT_FIELD_ENTRY_PEPTIDE = "Peptide"
FLAT_FIELD_NAME = "NAME"
FLAT_FIELD_MONOCLONAL_ANTIBODY = "Monoclonal antibody"
FLAT_FIELD_COMPONENT = "COMPONENT"
FLAT_FIELD_TYPE = "TYPE"
FLAT_FIELD_REMARK = "REMARK"
FLAT_FIELD_DBLINKS = "DBLINKS"

FLAT_FIELD_KEYWORDS = [
    FLAT_FIELD_ENTRY,
    "NAME",
    "ABBR",
    "PRODUCT",  # conflict with brite
    "FORMULA",
    "EXACT_MASS",
    "MOL_WEIGHT",
    "COMPONENT",
    "SEQUENCE",
    "TYPE",
    "SOURCE",
    "CLASS",
    FLAT_FIELD_REMARK,
    "EFFICACY",
    "DISEASE",
    FLAT_FIELD_TYPE,
    "COMMENT",
    "TARGET",
    "NETWORK",
    "PATHWAY",
    "METABOLISM",
    "INTERACTION",
    "SYN_MAP",
    "STR_MAP",
    "OTHER_MAP",
    "BRITE",
    FLAT_FIELD_DBLINKS,
    "ATOM",
    "BOND",
    "BRACKET",
    "REACTION",
    "MODULE",
    "ENZYME",
    "REFERENCE",
    "AUTHORS",
    "TITLE",
    "JOURNAL"
]

# ATC LEVEL
ATC_LEVEL_LEN = {1: 1, 2: 3, 3: 4, 4: 5, 5: 7}

# PATTERN
PATTERN_SPLIT_SMILES = r"\[.+?\]|."
