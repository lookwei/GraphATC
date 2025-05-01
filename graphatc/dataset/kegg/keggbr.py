import re


class BrNode:
    def __init__(self, t: str, name: str, level: int = None, fa: "BrNode" = None, entry_id: str = None) -> None:
        assert t in ["index", "entry"]
        self.t = t  # "index" or "entry"
        self.name = name
        self.entry_id = entry_id
        self.level = level
        self.fa = [fa]
        self.child = []
        self.field = {}


class KeggBr:
    def __init__(self):
        pass

    @classmethod
    def dfs(cls, d: dict, fa: BrNode, level: int, entry_map: dict[str, BrNode], tree_map: dict[str, BrNode]) -> None:
        match = re.search(r"^[DCH]\d{5}$", d["name"][:6])

        # leaf drug node
        if match:
            t = "entry"
            entry_id = d["name"][:6]
            if entry_id not in entry_map.keys():
                node = BrNode(t, name=d["name"][8:], level=level, fa=fa, entry_id=entry_id)
                entry_map[entry_id] = node
            else:
                node = entry_map[entry_id]
                node.fa.append(fa)
            fa.child.append(node)
            return

        # index node
        t = "index"
        entry_id = d["name"].split(" ")[0]
        node = BrNode(t, name=d["name"], level=level, fa=fa, entry_id=entry_id)
        tree_map[entry_id] = node
        fa.child.append(node)

        if "children" in d:
            for ch in d["children"]:
                cls.dfs(ch, node, level + 1, entry_map, tree_map)

    @classmethod
    def build_tree(cls, d: dict) -> tuple[BrNode, dict[str, BrNode], dict[str, BrNode]]:
        root = BrNode("index", d["name"], 0, None)
        entry_map = {}
        tree_map = {}
        for ch in d["children"]:
            cls.dfs(ch, root, 1, entry_map, tree_map)
        return root, entry_map, tree_map

    @classmethod
    def get_node_by_id_from_map(cls, entry_id: str, node_map: dict[str, BrNode]) -> [BrNode, None]:
        if entry_id not in node_map.keys():
            return None
        return node_map[entry_id]

    @staticmethod
    def filter_horizontal_level_node(node_map: dict[str, BrNode], level: int) -> list[BrNode]:
        return list(filter(lambda x: x.level == level, list(node_map.values())))
