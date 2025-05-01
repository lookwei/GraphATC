import requests
import re
import io
import pandas as pd

if __name__ == "__main__":

    UA = 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) ' \
         'Chrome/86.0.4240.75 Mobile Safari/537.36'

    headers = {'User-Agent': UA}

    url = "https://rest.kegg.jp/list/drug"

    print("request to: ", url, "\nrequesting...", sep="")

    response = requests.get(url=url, params="", headers=headers)

    # print(response.text)

    print("drug num : ", response.text.count("\n") + 1, "\nwriting...", sep="")

    with open("drugid_list.txt", "w") as f:
        f.write(response.text)

    data = response.text
    data = re.sub(r" \(.*?\)", "", data)
    df = pd.read_csv(io.StringIO(data), lineterminator="\n", sep="\t", names=['drug_id', 'drug_name'], header=None)
    df.to_csv("drug_id_name.csv", index=False)

    print("done")
