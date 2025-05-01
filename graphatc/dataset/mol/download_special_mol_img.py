import requests
import re
import io
import os
import pandas as pd
from tqdm import tqdm
import time

if __name__ == "__main__":

    UA = 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) ' \
         'Chrome/86.0.4240.75 Mobile Safari/537.36'

    headers = {'User-Agent': UA}

    url = "https://rest.kegg.jp/get/{}/image"

    # file_name = "mol_is_none"
    # file_name = "has_R"
    # file_name = "has_star"
    # file_name = "failed_convert_graph"

    file_name_list = ["mol_is_none", "has_R", "has_star",  "failed_convert_graph"]

    for file_name in file_name_list:

        dir = f"img_{file_name}"
        if not os.path.exists(dir):
            os.makedirs(dir)

        with open(f"check_mol/{file_name}.txt", "r") as f:
            file_content = f.read()

        did_list = file_content.split("\n")[:-1]

        bar = tqdm(did_list)
        for did in bar:
            bar.set_description(did)
            file_path = f"img_{file_name}/{did}.gif"
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    if len(f.read()) > 0:
                        continue
            time.sleep(0.2)
            for i in range(3):
                try:
                    response = requests.get(url=url.format(did), params="", headers=headers, timeout=5)
                    if len(response.content) == 0:
                        raise Exception("len(response.content) == 0")
                    break
                except Exception as e:
                    time.sleep(1)
            with open(file_path, "wb") as f:
                f.write(response.content)