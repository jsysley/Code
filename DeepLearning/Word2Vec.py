# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 11:01:39 2017

@author: jsysley
"""

import collections
import math
import os
import random
import zipfile

import numpy as np
import urllib
import tensorflow as tf

# Step 1: Download the data.
# 使用urllib.request.urlretrieve下载数据的压缩文件并核对文件尺寸
url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urllib.request.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('text8.zip', 31344016)


# Read the data into a list of strings.
# 解压下载的压缩文件，并使用tf.compat.as_str将数据转化成单词列表；最后数据被转为一个包含17005207个单词的列表
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words"""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data

words = read_data(filename)
print('Data size', len(words))

# Step 2: Build the dictionary and replace rare words with UNK token.
# 创建vocabulary词汇表，使用collections.Counter统计单词列表中单词的频数，然后使用most_common方法去top 50000频数的单词作为vocabulary
# 再创建一个dict，将top 50000频数的单词作为vocabulary放入dictionary中以便快速查询
# 将全部单词转为编号（以频数排序的编号），top 50000以外的单词认定其为Unknown，编号0，并统计这类词的数量
vocabulary_size = 50000

def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary

# data 转换后的编码，所有单词对应的count的index
# count 每个单词的频数统计
# dictionary 50000词汇表，value为在count的index
# reverse_dictionary 反转的词汇表
data, count, dictionary, reverse_dictionary = build_dataset(words)
del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
# 生成Word2Vec的训练样本，生成训练用的batch_size数据
# batch_size:batch的大小
# skip_window:指单词最远可以联系的距离,设为1表示只能跟紧邻的两个单词生成样本
# num_skips:对每个单词生成多少个样本，它不能大于skip_window值的两倍，并且batch_size必须是它的整数倍（确保每个batch包含了一个词汇对应的所有样本）
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0   # 约束
  assert num_skips <= 2 * skip_window  # 约束
  batch = np.ndarray(shape=(batch_size), dtype=np.int32) #初始化
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  # span为对某个单词创相关样本时会使用到的单词数量，包括目标单词本身和它前后的单词
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  # 创建一个最大容量为span的双向队列deque，在对deque使用append方法添加变量时，只会保留最后插入的span变量
  buffer = collections.deque(maxlen=span)
  # 从序号data_index开始，把span个单词顺序读入buffer作为初始值
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  # buffer是目标单词和相关单词
  # 定义target = skip_window,即第skip_window是目标单词
  # 定义生成样本时需要避免的单词列表target_to_avoid
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    # 每次循环中对一个语境单词生成样本，先产生随机数，直到随机数不在targets_to_avoid，代表可以使用的语境单词，然后产生一个样本
    # feature即目标词汇buffer[skip_window]，label则是buffer[target]
    # 使用过的语境单词添加到targets_to_avoid过滤
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

# 测试
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
      '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.
# 定义训练batch_size为128
# embedding_size为128，即将单词转为稠密向量的维度，一般是50~1000这个范围内的值，这里使用128作为词向量的维度
# skip_window即单词间最远可以联系的距离，设为1
# num_skips即对每个目标单词提取的样本数，设为2
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
# 生成验证数据valid_example，这里随机抽取一些频数高的单词，看向量空间上跟他们最近的单词是否相关性比较高
# valid_size = 16指用来抽取的验证单词数
# valid_window = 100指验证单词只从频数最高的100个单词中抽取
# num_sampled是训练时用来做负样本的噪声单词的数量
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.


graph = tf.Graph()
with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    # 随机生成所有单词的词向量embeddings
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    # 查找输入train_inputs对应的向量embed
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights = nce_weights,
                     biases = nce_biases,
                     labels = train_labels,
                     inputs = embed,
                     num_sampled = num_sampled,
                     num_classes = vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  # 计算嵌入向量embeddings的L2范数norm
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims = True))
  # 再将embeddings除以其L2范数得到标准化后的normalized_embeddings
  normalized_embeddings = embeddings / norm
  # 查询验证单词的嵌入向量
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  # 计算验证单词的嵌入向量与词汇表中所有单词的相似性
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 100001

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print("Initialized")

  average_loss = 0
  for step in range(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", step, ": ", average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in range(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8 # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log_str = "Nearest to %s:" % valid_word
        for k in range(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = "%s %s," % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()

# Step 6: Visualize the embeddings.
# 定义一个可视化Word2Vec效果的函数
# low_dim_embs是降维到2维的单词向量空间
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  #in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 200
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
  labels = [reverse_dictionary[i] for i in range(plot_only)]
  plot_with_labels(low_dim_embs, labels)

except ImportError:
  print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
