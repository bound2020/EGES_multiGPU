import numpy as np


def write_embedding(item_list, embedding_result, outputFileName):
    f = open(outputFileName, 'w')
    for i in range(len(embedding_result)):
        s = " ".join(str(f) for f in embedding_result[i].tolist())
        f.write("{0} {1}\n".format(item_list[i], s))
    f.close()


def graph_context_batch_iter(all_pairs, batch_size, side_info, num_features):
    while True:
        start_idx = np.random.randint(0, len(all_pairs) - batch_size)
        batch_idx = np.array(range(start_idx, start_idx + batch_size))
        batch_idx = np.random.permutation(batch_idx)
        batch = np.zeros((batch_size, num_features), dtype=np.int32)
        labels = np.zeros((batch_size, 1), dtype=np.int32)
        batch[:] = side_info[all_pairs[batch_idx, 0]]
        labels[:, 0] = all_pairs[batch_idx, 1]
        yield batch, labels


def preprocess_nxgraph(graph):
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx


def partition_dict(vertices, workers):
    batch_size = (len(vertices) - 1) // workers + 1
    part_list = []
    part = []
    count = 0
    for v1, nbs in vertices.items():
        part.append((v1, nbs))
        count += 1
        if count % batch_size == 0:
            part_list.append(part)
            part = []
    if len(part) > 0:
        part_list.append(part)
    return part_list


def partition_list(vertices, workers):
    batch_size = (len(vertices) - 1) // workers + 1
    part_list = []
    part = []
    count = 0
    for v1, nbs in enumerate(vertices):
        part.append((v1, nbs))
        count += 1
        if count % batch_size == 0:
            part_list.append(part)
            part = []
    if len(part) > 0:
        part_list.append(part)
    return part_list


def partition_num(num, workers):
    if num % workers == 0:
        return [num//workers]*workers
    else:
        return [num//workers]*workers + [num % workers]
