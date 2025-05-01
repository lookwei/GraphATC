import numpy as np
import argparse
from typing import List, Tuple


def get_n_u_v(g):
    n = g.num_nodes()
    u = g.edges()[0].tolist()
    v = g.edges()[1].tolist()
    return n, u, v


def countComponents(n: int, edges: List[List[int]] or List[Tuple[int]]) -> int:
    f = {}

    def find(x):
        f.setdefault(x, x)
        if x != f[x]:
            f[x] = find(f[x])
        return f[x]

    def union(x, y):
        f[find(x)] = find(y)

    for x, y in edges:
        union(x, y)

    return len(set(find(x) for x in range(n)))


def split_group(g) -> list:
    n, u, v = get_n_u_v(g)
    eij = np.zeros(((n, n))) - 1
    for ui, vi in list(zip(u, v)):
        eij[ui][vi] = 1

    visited = np.zeros(n) - 1
    res = []

    def dfs_search(node_id, group_id):
        if visited[node_id] != -1:
            return

        curr_group.append(node_id)
        visited[node_id] = group_id
        for i in range(n):
            if eij[node_id, i] == 1:
                dfs_search(i, group_id)

    for i in range(n):
        curr_group = []
        dfs_search(i, i)
        if curr_group:
            res.append(curr_group)

    return res


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
