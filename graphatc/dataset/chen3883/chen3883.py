from common.const import *


class Chen3883:
    def __init__(self):
        self.load_data()

    @classmethod
    def load_data(cls):
        with open(f"{PROJECT_PATH}/chen3883/chen3883_kegg_drug_id.txt", "r") as f:
            cls.chen_drug_id = sorted(f.read().splitlines())

    def __getitem__(self, item):
        return self.chen_drug_id[item]


if __name__ == "__main__":
    chen_data = Chen3883()
    print(chen_data[0])
    print(chen_data[3882])
    print(chen_data[0:2])
