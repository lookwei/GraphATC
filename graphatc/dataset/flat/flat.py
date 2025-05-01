import functools
import re
import io
import chardet
from graphatc.dataset.common.const import *


class Flat:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def get_flat_file_path_by_id(entry_id: str) -> str:
        prefix = entry_id[0].upper()
        assert prefix in FLAT_PREFIX_LIST
        return f"{DATASET_PATH}/flat/{FLAT_DIT_NAME_LIST[FLAT_PREFIX_LIST.index(prefix)]}/{entry_id}.txt"

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get_flat_content_by_id_from_file(cls, did: str) -> str:
        try:

            with open(cls.get_flat_file_path_by_id(did), "rb") as f:
                file_content = f.read().decode('utf-8')
                for field in FLAT_FIELD_KEYWORDS:
                    file_content = file_content.replace(field, FLAT_SPECIAL_FIELD_SEP + field)
                return file_content
        except OSError:
            return ""
        except UnicodeDecodeError:
            with open(cls.get_flat_file_path_by_id(did), "rb") as f:
                enc = chardet.detect(f.read())['encoding']
            with open(cls.get_flat_file_path_by_id(did), "rb") as f:
                file_content = f.read().decode(enc)
                for field in FLAT_FIELD_KEYWORDS:
                    file_content = file_content.replace(field, FLAT_SPECIAL_FIELD_SEP + field)
                return file_content

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get_field_by_id(cls, entry_id: str, field: str):

        flat_file_content = cls.get_flat_content_by_id_from_file(entry_id)

        start_index = flat_file_content.find(field, 0) + len(field)
        end_index = flat_file_content.find(FLAT_SPECIAL_FIELD_SEP, start_index)
        if start_index == -1:
            return ""

        field_data = flat_file_content[start_index:end_index].strip()

        lines = field_data.split("\n")

        if len(lines) == 1:
            return lines[0]

        str_io = io.StringIO()
        for idx, line in enumerate(lines):
            if field != "BRITE":
                line = line.lstrip()
            elif line[:12] == " " * 12:
                line = line[12:]
            str_io.write(line)
            if idx < len(lines) - 1:
                str_io.write("\n")
        return str_io.getvalue()

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get_entry_list_by_id(cls, entry_id: str) -> list[str]:
        entry_field = cls.get_field_by_id(entry_id, FLAT_FIELD_ENTRY)
        return re.split(r" +", entry_field)

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get_name_list_by_id(cls, entry_id: str) -> list[str]:
        name_content = cls.get_field_by_id(entry_id, FLAT_FIELD_NAME)
        name_content = re.sub(r"\s*\(.*?\)|\s*&lt;.*?&gt;", "", name_content)
        if "\r\n" in name_content:
            return re.split(r";\r\n", name_content)
        return re.split(r";\n", name_content)

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get_component_list_by_id(cls, entry_id: str) -> list[str]:
        component_field = cls.get_field_by_id(entry_id, FLAT_FIELD_COMPONENT)
        return re.split(r", ", component_field)

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get_component_id_list_by_id(cls, entry_id: str) -> list[str]:
        component_field = cls.get_field_by_id(entry_id, FLAT_FIELD_COMPONENT)
        d_list = re.findall(r'DR:([D0-9]+)', component_field)
        c_list = re.findall(r'CPD:([C0-9]+)', component_field)
        return d_list + c_list

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get_component_id_leaf_list_by_id(cls, entry_id: str) -> list[str]:
        ret = []
        queue = cls.get_component_id_list_by_id(entry_id)

        while len(queue) > 0:
            top = queue.pop(0)

            component_id_list = cls.get_component_id_list_by_id(top)

            if len(component_id_list) == 0:
                ret.append(top)
                continue

            queue.extend(component_id_list)

        return ret

    @classmethod
    @functools.lru_cache(maxsize=None)
    def is_mixture_by_id(cls, entry_id: str):
        entry_list = cls.get_entry_list_by_id(entry_id)
        return FLAT_FIELD_ENTRY_MIXTURE in entry_list

    @classmethod
    @functools.lru_cache(maxsize=None)
    def is_crude_by_id(cls, entry_id: str):
        entry_list = cls.get_entry_list_by_id(entry_id)
        return FLAT_FIELD_ENTRY_CRUDE in entry_list

    @classmethod
    @functools.lru_cache(maxsize=None)
    def is_peptide_by_id(cls, entry_id: str):
        return FLAT_FIELD_ENTRY_PEPTIDE in cls.get_field_by_id(entry_id, FLAT_FIELD_TYPE)

    @classmethod
    @functools.lru_cache(maxsize=None)
    def is_enzyme_by_id(cls, entry_id: str):
        return FLAT_FIELD_ENTRY_PEPTIDE in cls.get_field_by_id(entry_id, FLAT_FIELD_TYPE)

    @classmethod
    @functools.lru_cache(maxsize=None)
    def is_monoclonal_antibody_by_id(cls, entry_id: str):
        return FLAT_FIELD_MONOCLONAL_ANTIBODY in cls.get_field_by_id(entry_id, FLAT_FIELD_TYPE)

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get_same_as_list_by_id(cls, entry_id: str) -> list[str]:
        reamrk_field = cls.get_field_by_id(entry_id, FLAT_FIELD_REMARK)
        if len(reamrk_field) == 0:
            return []
        same_as_content: str = re.findall("Same as:(.+)", reamrk_field)
        if len(same_as_content) == 0:
            return []
        same_as_content = same_as_content[0].strip()
        same_as_list = same_as_content.split(" ")
        # same_as_list = [x.strip() for x in same_as_list]
        return same_as_list

    @classmethod
    def get_same_as_compound_id_by_id(cls, entry_id: str) -> str:
        same_as_list = cls.get_same_as_list_by_id(entry_id)
        same_as_compound_list = []
        for eid in same_as_list:
            if len(eid) != 6:
                raise Exception(f"func get_same_as_compound_by_id: get wrong entry_id:{eid}")
            if eid[0].upper() == 'C':
                same_as_compound_list.append(eid)
        if len(same_as_compound_list) == 0:
            return ""
        elif len(same_as_compound_list) == 1:
            return same_as_compound_list[0]
        else:
            raise Exception(f"func get_same_as_compound_by_id: get same_as_compound_list:{same_as_compound_list},"
                            f" please check.")

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get_chemical_structure_group_by_id(cls, entry_id: str) -> str:
        reamrk_field = cls.get_field_by_id(entry_id, FLAT_FIELD_REMARK)
        if len(reamrk_field) == 0:
            return []
        csg_content: str = re.findall("Chemical structure group:(.+)", reamrk_field)
        if len(csg_content) == 0:
            return ""
        csg_content = csg_content[0].strip()
        return csg_content

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get_other_db_links_by_id(cls, entry_id: str) -> dict[str, str]:
        entry_field = cls.get_field_by_id(entry_id, FLAT_FIELD_DBLINKS)
        db_link_list = re.split(r"\n", entry_field)
        db_link_list = list(filter(lambda x: "/" not in x, db_link_list))
        return {x.split(":")[0]: x.split(":")[1].strip() for x in db_link_list}

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get_cas_by_id(cls, entry_id: str) -> str:
        try:
            return cls.get_other_db_links_by_id(entry_id)["CAS"]
        except:
            return ""


if __name__ == "__main__":
    flat_class = Flat()
    # print(len(flat_class.get_flat_content_by_id_from_file("D06816")))
    # print(len(flat_class.get_flat_content_by_id_from_file("C00002")))
    #
    # print(flat_class.get_field_by_id("D06816", FLAT_FIELD_ENTRY))
    #
    # print(flat_class.get_field_by_id("D06816", FLAT_FIELD_COMPONENT))
    #
    # print(flat_class.get_entry_list_by_id("D06816"))
    #
    # print(flat_class.is_mixture_by_id("D06816"))
    #
    # print(flat_class.is_mixture_by_id("D00001"))
    #
    # print(flat_class.get_field_by_id("D06816", "BRITE"))
    #
    # print(flat_class.get_component_list_by_id("D06816"))
    # print(sorted(flat_class.get_component_id_list_by_id("D06816")))
    # print(sorted(flat_class.get_component_id_leaf_list_by_id("D06816")))
    #
    # print(flat_class.get_same_as_list_by_id("D00289"))  # C06938
    # print(flat_class.get_same_as_list_by_id("D00903"))  # []
    # print(flat_class.get_same_as_list_by_id("D06540"))  # C00372 G10502
    #
    # print(flat_class.get_same_as_compound_id_by_id("D00289"))
    # print(flat_class.get_same_as_compound_id_by_id("D00903"))
    # print(flat_class.get_same_as_compound_id_by_id("D06540"))

    # print(flat_class.get_flat_content_by_id_from_file("D10928"))  # utf-8

    # print(flat_class.get_cas_by_id("D02798"))
    # print(Flat.get_chemical_structure_group_by_id("D01618"))

    print(Flat.get_name_list_by_id("D03593"))
    print(Flat.get_name_list_by_id("D02852"))
