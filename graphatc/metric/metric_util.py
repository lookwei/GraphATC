import argparse
import pandas as pd
import os
import functools
from ast import literal_eval
from graphatc.dataset import const, DrugMol
from graphatc.dataset.uni_dataset import ATCDataset
from graphatc.evaluater import Evaluator


chen3883 = pd.read_csv(const.DATASET_PATH_CHEN3883)
chen3883 = chen3883.set_index("drug_id")


def get_5_metric_by_evaluater(*args):
    return Evaluator.get_5_metric(*args)


def get_5_metric(df, y_hat_name="predict", y_name="true", did_name="drug_id",
                 level="1"):
    df[y_name] = df[y_name].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
    df[y_hat_name] = df[y_hat_name].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)

    val_pre_list = df[y_hat_name].tolist()
    val_true_list = df[y_name].tolist()
    drug_id_list = df[did_name].tolist()
    n_class = len(val_true_list[0])

    columns = [f"aiming_l{level}", f"coverage_l{level}",
               f"accuracy_l{level}", f"abs_true_l{level}", f"abs_false_l{level}"]

    val_predict_df_list = []

    for j in range(len(val_true_list)):
        val_predict = pd.DataFrame({"index": j,
                                    "drug_id": drug_id_list[j],
                                    f"predict_l{level}": str(val_pre_list[j]),
                                    f"true_l{level}": str(val_true_list[j])},
                                   index=[j])
        val_predict[columns] = val_predict.apply(lambda x: get_5_metric_by_evaluater(
            val_pre_list[j], val_true_list[j], n_class), axis=1).apply(pd.Series)
        val_predict_df_list.append(val_predict)
    df_metric = pd.concat(val_predict_df_list)
    return df_metric


def concat_log(_exp_path: str, _file_path: str, replace: bool) -> pd.DataFrame:
    if os.path.exists(_file_path) and not replace:
        raise Exception(f"file exist: {_file_path}")

    print(f"DO NOT FIND \"{_file_path}\", try concat files:")
    df_list = []
    dir_list = sorted(os.listdir(_exp_path))
    for p in dir_list:
        if p[0] == '[':
            f = _exp_path + "/" + p + "/predict.csv"
            if os.path.exists(f):
                __df = pd.read_csv(f)
                df_list.append(__df)
                print(f"append size {len(__df)}:", p)
    _df = pd.concat(df_list)
    _df = _df.sort_values(by='drug_id')
    _df = _df.drop(["Unnamed: 0"], axis=1)
    _df["index"] = range(1, len(_df) + 1)
    _df.to_csv(_file_path, index=False)
    return _df


@functools.lru_cache(maxsize=2)
def get_atc_mol_type_map(author=None):
    if author in [None, "tian"]:
        dm = DrugMol()
        dm.init_atc_mol_map()
        type_map = DrugMol.get_mol_type_map_by_map(dm.mol_map["atc"])

    if author in ["chen", "Chen"]:
        ds = ATCDataset(level=1,
                        mode='g',
                        polymer_method=3,
                        author="chen",
                        cuda_idx=0,
                        split_component=False)
        type_map = ds.get_mol_type_map_by_map(ds.mol_map["atc"])
    return type_map


def print_five_metric(_df: pd.DataFrame, level: str = "1"):
    print(f"=====Five metric [level {level}]=====")
    print(_df[[f"aiming_l{level}", f"coverage_l{level}", f"accuracy_l{level}",
          f"abs_true_l{level}", f"abs_false_l{level}"]].mean())


def get_five_metric_distinguish(_df: pd.DataFrame, level: str = "1",
                                type_map=None, author=None, return_df_map_by_type=False):
    if type_map is None:
        type_map = get_atc_mol_type_map(author=author)

    df_list_map = {k: [] for k in type_map.keys()}

    for _, row in _df.iterrows():
        did = row["drug_id"]
        row = pd.DataFrame(row).T
        for k in type_map.keys():
            if did in type_map[k]:
                df_list_map[k].append(row)

    ready_to_del_key = []
    for k in df_list_map.keys():
        if len(df_list_map[k]) == 0:
            ready_to_del_key.append(k)
    for k in ready_to_del_key:
        df_list_map.pop(k)

    df_map = {k: pd.concat(v) for k, v in df_list_map.items()}

    metric_df = []

    __df = _df[[f"aiming_l{level}", f"coverage_l{level}", f"accuracy_l{level}",
                f"abs_true_l{level}", f"abs_false_l{level}"]].mean()

    __df["type"] = "Avg"

    metric_df.append(__df.T)

    for k in type_map.keys():

        if k in ready_to_del_key:
            continue

        __df = df_map[k][[f"aiming_l{level}", f"coverage_l{level}",
                          f"accuracy_l{level}", f"abs_true_l{level}", f"abs_false_l{level}"]].mean()

        __df["type"] = k

        metric_df.append(__df.T)

    if return_df_map_by_type:
        return pd.concat(metric_df, axis=1).T, df_map

    return pd.concat(metric_df, axis=1).T


def get_diff_df_with_3883(df, level="1", label_size=14):
    diff_set = set(chen3883.index) - set(df["drug_id"])
    assert len(diff_set) + len(set(df["drug_id"])) == len(set(chen3883.index))
    print(len(diff_set))

    pad_size = label_size-1
    diff_df = pd.DataFrame(columns=df.columns).reindex([])
    for did in diff_set:
        diff_df = pd.concat([diff_df, pd.DataFrame([{'drug_id': did,
                                                     f'predict_l{level}': [1]+[0]*pad_size,
                                                     f'true_l{level}': chen3883.loc[did][chen3883.columns[0]]}])], ignore_index=True).dropna(axis=1, how='all')
    return diff_df


def align_3883_metric_df(df, level="1"):
    diff = get_diff_df_with_3883(df, level)
    df = pd.concat([df, diff])
    metric_df = get_5_metric(df, f"predict_l{level}", f"true_l{level}", level=level)
    print(len(metric_df))
    return get_five_metric_distinguish(metric_df, level=level, author="chen")


def print_five_metric_distinguish_de(_df: pd.DataFrame, level: str = "1"):
    print_five_metric(_df, level)

    type_map = get_atc_mol_type_map()

    df_list_map = {k: [] for k in type_map.keys()}

    for _, row in _df.iterrows():
        did = row["drug_id"]
        row = pd.DataFrame(row).T
        for k in type_map.keys():
            if did in type_map[k]:
                df_list_map[k].append(row)

    df_map = {k: pd.concat(v) for k, v in df_list_map.items()}

    colmuns = [f"aiming_l{level}", f"coverage_l{level}",
               f"accuracy_l{level}", f"abs_true_l{level}", f"abs_false_l{level}"]
    metric_df_each_k_list = []
    for i, k in enumerate(type_map.keys()):
        _df = df_map[k][colmuns].mean()
        _df["type"] = k

        metric_df_each_k_list.append(pd.DataFrame(_df).T)
    metric_df = pd.concat(metric_df_each_k_list)
    print(metric_df)
    return metric_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, default="1")
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--resfilepath", type=str, default=None)
    parser.add_argument("--five_metric", action='store_true', default=False)
    parser.add_argument("--replace", action='store_true', default=False)
    parser.add_argument("--concat", action='store_true', default=False)

    args = parser.parse_args()

    if args.logdir is not None and args.concat:
        concat_log(args.logdir, args.logdir+"/concat.csv", args.replace)
