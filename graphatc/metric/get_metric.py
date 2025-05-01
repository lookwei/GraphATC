import pandas as pd
import os


def load_or_concat_log(_exp_path: str, _file_path: str, replace: bool) -> pd.DataFrame:
    if not os.path.exists(_file_path) or replace:
        print(f"DO NOT FIND \"{_file_path}\", try concat files:")
        df_list = []
        for p in os.listdir(_exp_path):
            # print("exp:", p)
            if p[0] == '[':
                f = _exp_path + "/" + p + "/predict.csv"
                if os.path.exists(f):
                    df_list.append(pd.read_csv(f))
                    print("append:", p)
        _df = pd.concat(df_list)
        _df = _df.sort_values(by='drug_id')
        _df = _df.drop(["Unnamed: 0"], axis=1)
        _df["index"] = range(1, len(_df) + 1)
        _df.to_csv(_file_path, index=False)
    else:
        print(f"FIND \"{_file_path}\"")
        _df = pd.read_csv(_file_path)
    return _df


def load_final_res_log(_file_path: str) -> pd.DataFrame:
    if not os.path.exists(_file_path):
        raise Exception('f"DO NOT FIND \"{_file_path}\"!"')
    print(f"FIND \"{_file_path}\"")
    _df = pd.read_csv(_file_path)
    return _df


def print_five_metric(_df: pd.DataFrame):
    print("=====Five metric=====")
    print(_df[["aiming_l1", "coverage_l1", "accuracy_l1", "abs_true_l1", "abs_false_l1"]].mean())

def get_metric_df(final=False, name="", logdir="", replace=False) -> (pd.DataFrame | None):
    if len(name)==0 or len(logdir)==0:
        return None
    
    pre_path = os.path.join(logdir, name)
    if not final:
        exp_path = pre_path
        file_path = exp_path + f"/concat.csv"
        print("concat_path=", file_path)
        df = load_or_concat_log(exp_path, file_path, replace)
    else:
        file_path = pre_path + ".csv"
        print("file_path=", file_path)
        df = load_final_res_log(file_path)

    return df

