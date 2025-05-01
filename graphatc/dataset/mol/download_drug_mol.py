import time
from tqdm import tqdm
import requests
import pandas as pd
import sys
import os

if __name__ == "__main__":

    UA = 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) ' \
         'Chrome/86.0.4240.75 Mobile Safari/537.36'

    headers = {'User-Agent': UA}

    url = "https://www.genome.jp/dbget-bin/www_bget?-f+m+cpd+{}"

    drug_id_df = pd.read_csv("drug_id_name.csv")

    drug_id_list = drug_id_df['drug_id'].to_list()

    start_idx, end_idx = 0, len(drug_id_list)

    if len(sys.argv) <= 1:
        pass
    elif len(sys.argv) == 2:
        start_idx = sys.argv[1]
    elif len(sys.argv) == 3:
        start_idx = sys.argv[2]
        end_idx = sys.argv[1]
    else:
        raise Exception("args too much")

    print(f"start_idx={start_idx}, end_idx={end_idx}")

    bar = tqdm(drug_id_list[int(start_idx):int(end_idx)])
    for did in bar:
        bar.set_description(did)
        file_path = f"mol/{did}.mol"
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                if len(f.read()) > 0:
                    continue
        time.sleep(0.2)
        for i in range(3):
            try:
                response = requests.get(url=url.format(did), params="", headers=headers, timeout=3)
                if len(response.text) == 0:
                    raise Exception("len(response.text) == 0")
                break
            except Exception as e:
                time.sleep(1)
        with open(file_path, "w") as f:
            f.write(response.text)
