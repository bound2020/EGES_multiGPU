import tensorflow.compat.v1 as tf
import numpy as np
import collections
import random
import math
import datetime
import time

word_map = {}
data = []
side_info = np.loadtxt('./data/side_info_feature', dtype=int)
item_size, feature_size = side_info.shape
embedding_size = 128
n_sampled = 50
num_gpus = 2
batch_size = 256
num_steps = 200001   # data_size / batch_size * n_epoch
every_k_step = 5000
num_skips = 4       # batch_size % num_skips == 0
window_size = 4
tf.disable_eager_execution()

item_set = set()
def read_data(filename):
    global item_set
    with open(filename) as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            data.extend(line)
    item_set = set(data)
    return data


data_index = 0
def generate_batch(batch_size):
    global data_index
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    label = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * window_size + 1
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0

    buffer.extend(data[data_index: data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        tgt = window_size
        visited_tgt = [tgt]
        for j in range(num_skips):
            while tgt in visited_tgt:
                tgt = random.randint(0, span - 1)
            visited_tgt.append(tgt)
            batch[i * num_skips + j] = buffer[window_size]
            label[i * num_skips + j, 0] = buffer[tgt]
        if data_index == len(data):
            for k in range(span):
                buffer.append(k)
            data_index = span
        else:
            buffer.append(data[data_index])
        data_index += 1
    data_index = (data_index + len(data) - span) % len(data)
    return batch, label


def _variable_on_cpu(name, shape, initializer, dtype=np.float32):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def tower_loss(scope, inputs, labels):
    embedding_list = []

    for i in range(feature_size):
        embedding = _variable_on_cpu('side_info_{0}_embeddings'.format(i), [max(side_info[:, i]) + 1, embedding_size],
                                     tf.random_uniform_initializer(-1.0, 1.0))
        side_info_index = tf.nn.embedding_lookup(side_info[:, i], inputs)
        side_info_embed = tf.nn.embedding_lookup(embedding, tf.cast(side_info_index[:], dtype=tf.int32))
        embedding_list.append(side_info_embed)
        
    alpha_embedding = _variable_on_cpu('alpha_embeddings', [item_size, feature_size],
                                       tf.random_uniform_initializer(-1.0, 1.0))
    stacked_embed = tf.stack(embedding_list, axis=-1)
    alpha_index = tf.nn.embedding_lookup(side_info[:, 0], inputs)
    alpha_embed = tf.nn.embedding_lookup(alpha_embedding, alpha_index)
    alpha_embed_expand = tf.expand_dims(alpha_embed, 1)
    alpha_i_sum = tf.reduce_sum(tf.exp(alpha_embed_expand), axis=-1)
    merge_embedding = tf.reduce_sum(stacked_embed * tf.exp(alpha_embed_expand), axis=-1) / alpha_i_sum

    ''' cold start item
    stacked_embed = tf.stack(embedding_list[1:], axis=-1)
    alpha_index = tf.nn.embedding_lookup(side_info[:, 1], inputs)
    alpha_embed = tf.nn.embedding_lookup(alpha_embedding, alpha_index[:])
    alpha_embed_expand = tf.expand_dims(alpha_embed, 1)
    alpha_i_sum = tf.reduce_sum(tf.exp(alpha_embed_expand), axis=-1)
    merge_embedding = tf.reduce_sum(stacked_embed * tf.exp(alpha_embed_expand), axis=-1) / alpha_i_sum
    cold_start_embedding = tf.reduce_sum(stacked_embed * tf.exp(alpha_embed_expand), axis=-1) / alpha_i_sum
    '''
    weights = _variable_on_cpu('w', [item_size, embedding_size], tf.truncated_normal_initializer(stddev=1.0/math.sqrt(embedding_size)))
    biases = _variable_on_cpu('b', [item_size], tf.zeros_initializer())
    loss = tf.reduce_mean(tf.nn.nce_loss(
        weights=weights,
        biases=biases,
        labels=labels,
        inputs=merge_embedding,
        num_sampled=n_sampled,
        num_classes=item_size
    ))
    return loss, merge_embedding


def average_gradient(tower_grads):
    avg_grads = []
    for grads_vars in zip(*tower_grads):
        values = tf.concat([g.values / num_gpus for g, _ in grads_vars], 0)
        indices = tf.concat([g.indices for g, _ in grads_vars], 0)
        grad = tf.IndexedSlices(values, indices)

        var = grads_vars[0][1]
        cur_grad_and_var = (grad, var)
        avg_grads.append(cur_grad_and_var)
    return avg_grads


def get_final_embedding():
    cnt = item_size // batch_size
    remain = item_size % batch_size
    final_embedding = {}
    all_item = side_info[:, 0]
    all_item = np.concatenate([all_item, [0] * remain], axis=0)

    for i in range(cnt):
        eval_input = all_item[i * batch_size: (i + 1) * batch_size]
        eval_label = np.zeros((batch_size, 1))
        eval_embedding = sess.run(merged_embedding, feed_dict={train_input: eval_input, train_label: eval_label})
        # for cold start item
        # cold_start_embedding = sess.run(cold_start_embedding, feed_dict={train_input: eval_input, train_label: eval_label})
        eval_embedding = eval_embedding.tolist()
        if i == cnt - 1:
            eval_embedding = eval_embedding[:-remain]
        final_embedding.update({all_item[i*batch_size+k]: eval_embedding[k] for k in range(len(eval_embedding))})
    dump_embedding(final_embedding, 'data/item_embeddings')


def dump_embedding(embedding_result, output_file):
    with open(output_file, 'w') as f:
        for k, v in embedding_result.items():
            f.write("{0} {1}\n".format(k, " ".join(list(map(lambda x: str(x), v)))))


if __name__ == '__main__':
    d = read_data('data/walk_seq')

    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):
        train_input = tf.placeholder(tf.int32, shape=[batch_size])
        train_label = tf.placeholder(tf.int32, shape=[batch_size, 1])

        train_opt = tf.train.GradientDescentOptimizer(1.0)
        #train_opt = tf.train.AdamOptimizer(1.0)

        tower_grads = []
        batch_size_gpu = batch_size // num_gpus
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_gpus):
                with tf.device('/gpu:{0}'.format(i)):
                    with tf.name_scope('tower_{0}'.format(i)) as scope:
                        train_input_gpu = tf.slice(train_input, [i * batch_size_gpu], [batch_size_gpu])
                        train_label_gpu = tf.slice(train_label, [i * batch_size_gpu, 0], [batch_size_gpu, 1])

                        loss, merged_embedding = tower_loss(scope, train_input_gpu, train_label_gpu)
                        tf.get_variable_scope().reuse_variables()

                        grads = train_opt.compute_gradients(loss)
                        tower_grads.append(grads)

        grads = average_gradient(tower_grads)
        apply_gradient_op = train_opt.apply_gradients(grads)

        init = tf.global_variables_initializer()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(graph=graph, config=config) as sess:
        start_time = datetime.datetime.now()
        init.run()
        print('Init finished')
        saver = tf.train.Saver(max_to_keep=4)

        avg_loss = 0
        final_loss = 0
        for step in range(1, num_steps):
            batch_input, batch_label = generate_batch(batch_size)
            feed_dict = {train_input: batch_input, train_label: batch_label}
            _, loss_val, batch_res = sess.run([apply_gradient_op, loss, merged_embedding], feed_dict=feed_dict)

            avg_loss += loss_val
            final_loss += loss_val

            if step % every_k_step == 0:
                end_time = datetime.datetime.now()
                avg_loss /= every_k_step
                print("step: {0}, loss: {1}, time: {2}s".format(step, avg_loss, (end_time-start_time).seconds))
                avg_loss = 0
                start_time = datetime.datetime.now()

        get_final_embedding()
