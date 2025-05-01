import functools
import time
import requests
import graphatc.dataset.common.const as const


class BaseClient:
    def __init__(self):
        pass

    @staticmethod
    def do_get(url, headers=const.WEB_HEADERS, timeout=3):
        return requests.get(url=url, params="", headers=headers, timeout=timeout)

    @classmethod
    def try_do_get(cls, url, headers=const.WEB_HEADERS, timeout=3, retry=3, retry_sleep=1) -> requests.Response:
        response = requests.Response()
        for try_count in range(1, retry + 1):
            try:
                response = cls.do_get(url=url, headers=headers, timeout=timeout)
                if response.status_code != 200:
                    raise Exception(f"func try_do_get failed, "
                                    f"response.status_code={response.status_code}, try_count={try_count}")
                break  # cancel try again
            except:
                time.sleep(retry_sleep)
        return response


class KeggClient(BaseClient):
    def __init__(self):
        super().__init__()

    @classmethod
    @functools.lru_cache(maxsize=None)
    def download_brite_json(cls, br: str):
        resp = cls.try_do_get(url=const.WEB_URL_KEGG_BR_JSON.format(br))
        if resp.status_code == 200 and len(resp.text) > 0:
            return resp.text

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get_mol_file_by_id(cls, entry_id: str) -> str:
        resp = cls.try_do_get(url=const.WEB_URL_KEGG_GET_MOL.format(entry_id))
        if resp.status_code == 200 and len(resp.text) > 0:
            return resp.text
        return ""


class PubChemClient(BaseClient):
    def __init__(self):
        super().__init__()

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get_synonyms_list_by_name(cls, name: str) -> list[str]:
        resp = cls.try_do_get(url=const.WEB_URL_PUBCHEM_REST.format("name", name, "synonyms/TXT"))
        if resp.status_code == 200 and len(resp.text) > 0:
            return list(filter(lambda x: x != "", resp.text.split("\n")))
        return []

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get_sdf_by_name(cls, name: str) -> str:
        resp = cls.try_do_get(url=const.WEB_URL_PUBCHEM_REST.format("name", name, "SDF"))
        if resp.status_code == 200 and len(resp.text) > 0:
            return resp.text
        return ""

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get_sdf_by_name_strict_synonyms(cls, name: str) -> str:
        synonyms_list = cls.get_synonyms_list_by_name(name)
        synonyms_list = [x.lower() for x in synonyms_list]
        if name.lower() not in synonyms_list:
            return ""
        return cls.get_sdf_by_name(name)


class ChemicalBookClient(BaseClient):
    def __init__(self):
        super().__init__()

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get_mol_file_by_cas(cls, cas: str) -> str:
        for url in const.WEB_URL_CHEMICAL_BOOK_GET_MOL_LIST:
            resp = cls.try_do_get(url=url.format(cas))
            if resp.status_code == 200 and len(resp.text) > 0:
                return resp.text
        return ""


# keggClient = KeggClient()
# chemicalBookClient = ChemicalBookClient()

if __name__ == "__main__":
    pass
    # print(KeggClient.get_mol_file_by_id("C10453"))
    # print(ChemicalBookClient.get_mol_file_by_cas("180288-69-1"))
    # print(KeggClient.download_brite_json("br08330"))
    # print(PubChemClient.get_synonyms_list_by_name("Corticotropin"))
    # print(PubChemClient.get_sdf_by_name("Corticotropin"))
    print(PubChemClient.get_sdf_by_name_strict_synonyms("Corticotropin"))
