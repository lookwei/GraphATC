import requests
import json
import re
if __name__ == "__main__":

    UA = 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Mobile Safari/537.36'

    headers = {
        'User-Agent': UA
    }

    url = "https://rest.kegg.jp/get/br:br08303/json" # atc br08301
    response = requests.get(url=url, params="", headers=headers)
    print(response.text)

    data = re.sub(r"\s*\(.*?\)|\s*&lt;.*?&gt;", "", response.text)

    with open("br08303_drug_atc_raw.json", "w") as f:
        f.write(response.text)

    with open("br08303_drug_atc.json", "w") as f:
        f.write(data)
