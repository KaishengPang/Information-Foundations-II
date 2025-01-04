import numpy as np

# 定义语料库
corpus = [
    "人工智能 在 各个 领域 有 很多 应用 并且 改变 了 人们 的 生活 和 工作 方式",
    "苹果 很 好吃",
    "香蕉 很 甜",
    "一些 苹果 是 酸的",
    "香蕉 一般 都 甜",
    "苹果 香蕉 都是 水果",
    "猫 喜欢吃 鱼",
    "狗 喜欢吃 骨头",
    "水果 对 身体 很好",
    "鱼 是 猫 最爱",
    "草莓 和 蓝莓 都是 美味 的 水果",
    "大象 吃 草 和 水果",
    "老虎 喜欢 吃 肉 和 鱼",
    "人类 需要 水 和 食物 才能 生存",
    "鸡 喜欢 吃 米 和 蔬菜",
    "兔子 吃 胡萝卜 和 草",
    "蔬菜 对 身体 很有益",
    "运动 和 健康 饮食 很 重要",
    "跑步 是 一种 很 好的 运动",
    "朋友 之间 应该 互相 帮助",
    "学习 新 知识 很 有趣",
    "科技 改变 了 人们 的 生活",
    "大自然 是 我们 的 家园",
    "保护 环境 很 重要",
    "人类 可以让 水果 好吃"
]

# 构建词汇表
vocab = list(set([word for sentence in corpus for word in sentence.split()]))
vocab_size = len(vocab)
print("词汇表大小:", vocab_size)

# 构建 BoW 向量
def build_bow_vector(word, vocab):
    bow = [0] * len(vocab)
    if word in vocab:
        bow[vocab.index(word)] = 1
    return np.array(bow)

# 构建滑动窗口共现矩阵
def build_windowed_cooccurrence_matrix(corpus, vocab, window_size=2):
    matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    for sentence in corpus:
        words = sentence.split()
        for i, word1 in enumerate(words):
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    word2 = words[j]
                    if word1 in vocab and word2 in vocab:
                        idx1, idx2 = vocab.index(word1), vocab.index(word2)
                        matrix[idx1, idx2] += 1
    return matrix

# 构建融合后的词嵌入
def generate_combined_word_vector(word, bow_vector, cooccurrence_matrix, vocab, alpha=0.5):
    idx = vocab.index(word)
    cooc_vector = cooccurrence_matrix[idx]
    combined_vector = alpha * bow_vector + (1 - alpha) * cooc_vector
    return combined_vector

# 获取句子的词向量矩阵
def get_sentence_matrix(sentence, vocab, cooccurrence_matrix, alpha=0.5):
    words = sentence.split()
    vectors = []
    for word in words:
        bow_vector = build_bow_vector(word, vocab)
        vector = generate_combined_word_vector(word, bow_vector, cooccurrence_matrix, vocab, alpha)
        vectors.append(vector)
    return np.array(vectors)

# 计算 Q, K, V
def compute_qkv(sentence_matrix, d_k=8):
    vocab_dim = sentence_matrix.shape[1]
    W_Q = np.random.rand(vocab_dim, d_k)
    W_K = np.random.rand(vocab_dim, d_k)
    W_V = np.random.rand(vocab_dim, d_k)
    Q = np.dot(sentence_matrix, W_Q)
    K = np.dot(sentence_matrix, W_K)
    V = np.dot(sentence_matrix, W_V)
    return Q, K, V

# 计算 Attention
def compute_attention(Q, K, V):
    d_k = Q.shape[1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    output = np.dot(attention_weights, V)
    return scores, attention_weights, output

# 构建滑动窗口共现矩阵
cooccurrence_matrix = build_windowed_cooccurrence_matrix(corpus, vocab, window_size=2)

# 测试长句子
sentence = "科技 可以让 水果 和 肉 好吃"
sentence_matrix = get_sentence_matrix(sentence, vocab, cooccurrence_matrix, alpha=1)

# 计算 Q, K, V
Q, K, V = compute_qkv(sentence_matrix, d_k=8)

# 计算 Self-Attention
scores, attention_weights, output = compute_attention(Q, K, V)

# 打印结果
print("\n=== 句子矩阵 X ===")
print(sentence_matrix)
print("\n=== Q 矩阵 ===")
print(Q)
print("\n=== K 矩阵 ===")
print(K)
print("\n=== V 矩阵 ===")
print(V)
print("\n=== 注意力分数 (Score) ===")
print(scores)
print("\n=== 注意力权重矩阵 (A) ===")
print(attention_weights)
print("\n=== Self-Attention 输出矩阵 ===")
print(output)