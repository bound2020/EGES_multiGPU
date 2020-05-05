import networkx as nx
import random
import numpy as np
from joblib import Parallel, delayed
import time
from itertools import chain
import os
import re

id_mapping = {}
side_info = []
DATA_PATH = "data"
BATCH = 100000


def partition_num(num, workers):
    if num % workers == 0:
        return [num // workers] * workers
    else:
        return [num // workers] * workers + [num % workers]


def create_alias_table(area_ratio):
    """
    :param area_ratio: sum(area_ratio)=1
    :return: accept,alias
    """
    l = len(area_ratio)
    accept, alias = [0] * l, [0] * l
    small, large = [], []

    for i, prob in enumerate(area_ratio):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = area_ratio[small_idx]
        alias[small_idx] = large_idx
        area_ratio[large_idx] = area_ratio[large_idx] - (1 - area_ratio[small_idx])
        if area_ratio[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)

    while large:
        large_idx = large.pop()
        accept[large_idx] = 1
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1

    return accept, alias



def dump_seq(walks, id_mapping):
    with open(os.path.join(DATA_PATH, "walk_seq"), "w") as f:
        for line in walks:
            f.write("{0}\n".format(" ".join(map(lambda x: str(id_mapping[x]), line))))


def get_all_pairs(walks, id_mapping, window_size):
    all_pairs = []
    cnt = 0
    side_info = []
    with open(os.path.join(DATA_PATH, "side_info_feature")) as f:
        for line in f.readlines():
            line = line.strip().split("\t")
            side_info.append(line)

    for k in range(len(walks)):
        for i in range(len(walks[k])):
            for j in range(i - window_size, i + window_size):
                if i == j or j < 0 or j >= len(walks[k]):
                    continue
                else:
                    line = [id_mapping[walks[k][i]]]
                    line.extend(side_info[id_mapping[walks[k][i]]-1])
                    line.append(id_mapping[walks[k][j]])
                    all_pairs.append(line)

                if len(all_pairs) == 0:
                    return
        if len(all_pairs) % BATCH == 0:
            with open(os.path.join(DATA_PATH, "all_pairs"), "a") as f:
                for line in all_pairs:
                    f.write("{0}\n".format("\t".join(list(map(lambda x: str(x), line)))))
            print("{0} lines done. {1}".format(BATCH * cnt,
                                               time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

            all_pairs = []
            cnt += 1


class RandomWalker:
    def __init__(self, g):
        self.g = g
        self.alias_nodes = {}
        self.alias_edges = {}

    def deepwalk(self, start, walk_len):
        walk = [start]

        while len(walk) < walk_len:
            cur = walk[-1]
            neighbor = list(self.g.neighbors(cur))
            if len(neighbor) > 0:
                walk.append(random.choice(neighbor))
            else:
                break
        return walk

    def simulate(self, num_walks, walk_length, workers=4):
        g = self.g

        nodes = list(g.nodes())
        result = Parallel(n_jobs=workers)(delayed(self._simulate)(nodes, num, walk_length)
                                          for num in partition_num(num_walks, workers))
        result = list(chain(*result))
        result = list(filter(lambda x: len(x) > 2, result))
        return result

    def _simulate(self, nodes, num_walks, walk_length):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.deepwalk(node, walk_length))
        return walks

    def get_alias_edges(self, t, v):
        g = self.g
        p, q = 1, 1
        unnormalized = []
        for x in g.neighbors(v):
            weight = g[v][x].get('weight', 1.0)
            if x == t:
                unnormalized.append(weight / p)
            elif g.has_edge(x, t):
                unnormalized.append(weight)
            else:
                unnormalized.append(weight / q)
        norm = sum(unnormalized)
        normalized = list(map(lambda x: x / norm * len(unnormalized), unnormalized))
        return create_alias_table(normalized)

    def build_trans_prob(self):
        g = self.g

        for node in g.nodes:
            unnormalized = [g[node][neighbor].get('weight', 1.0) for neighbor in g.neighbors(node)]
            norm = sum(unnormalized)
            normalized = list(map(lambda x: x / norm * len(unnormalized), unnormalized))
            self.alias_nodes[node] = create_alias_table(normalized)

        for edge in g.edges():
            self.alias_edges[edge] = self.get_alias_edges(edge[0], edge[1])


def create_graph():
    edges = []
    with open(os.path.join(DATA_PATH, "graph_node")) as f:
        for line in f.readlines():
            line = line.strip()
            in_node, out_node, weight = line.split(",")
            edges.append((in_node, out_node, float(weight)))

    di = nx.DiGraph()
    di.add_weighted_edges_from(edges)
    rand = RandomWalker(di)
    # rand.build_trans_prob()
    res = rand.simulate(15, 15)
    print('simulate finished')

    id_mapping = {}
    with open(os.path.join(DATA_PATH, "id_mapping")) as f:
        for line in f.readlines():
            line = line.strip()
            line = line.split("\t")
            id_mapping[line[0]] = int(line[1])

    dump_seq(res, id_mapping)


if __name__ == '__main__':
    create_graph()
