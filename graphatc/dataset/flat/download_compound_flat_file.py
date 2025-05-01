import os
import time
from tqdm import tqdm
import requests
import pandas as pd
import sys

from kegg_data import KeggData

if __name__ == "__main__":

    UA = 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) ' \
         'Chrome/86.0.4240.75 Mobile Safari/537.36'

    headers = {'User-Agent': UA}

    url = "https://rest.kegg.jp/get/{}"

    kd = KeggData()

    compound_id_list = list(kd.compounds_map.keys())

    start_idx, end_idx = 0, len(compound_id_list)

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

    bar = tqdm(compound_id_list[int(start_idx):int(end_idx)])
    for cid in bar:
        bar.set_description(cid)
        file_path = f"compound_flat/{cid}.txt"
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                if len(f.read()) > 0:
                    continue
        time.sleep(0.2)
        for i in range(3):
            try:
                response = requests.get(url=url.format(cid), params="", headers=headers, timeout=3)
                if len(response.text) == 0:
                    raise Exception("len(response.text) == 0")
                break
            except Exception as e:
                time.sleep(1)
        with open(file_path, "wb") as f:
            f.write(response.text.encode('utf-8'))
