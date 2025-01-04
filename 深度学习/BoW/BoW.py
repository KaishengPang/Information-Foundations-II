import numpy as np

# 定义语料库
corpus = [
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
    "人工智能 在 各个 领域 有 很多 应用",
    "太阳 和 月亮 都 在 天空 中",
    "星星 在 夜晚 很 漂亮",
    "孩子 们 喜欢 听 故事",
    "小鸟 在 树上 唱歌",
    "冬天 的 雪 很 美",
    "春天 开 满了 花",
    "夏天 天气 很 热",
    "秋天 落叶 很 美丽",
    "人类 是 地球 的 一部分",
    "科学家 在 探索 宇宙 的 奥秘",
    "书 是 知识 的 源泉",
    "音乐 可以 舒缓 人 的 情绪",
    "画画 是 一种 艺术 表达",
    "编程 是 一种 技能 和 工具",
    "旅行 可以 开阔 人 的 视野",
    "朋友 一起 看 电影 很 开心",
    "水果 和 蔬菜 对 健康 很 有益",
    "动物 是 我们 的 朋友",
    "猫 和 狗 是 最 受欢迎 的 宠物"
]

# 构建词汇表（按词）
vocab = list(set([word for sentence in corpus for word in sentence.split()]))  # 将语料库按空格分词，去重后得到词汇表
vocab_size = len(vocab)  # 词汇表大小
print("词汇表:", vocab)

# 构建 BoW 向量
def build_bow_vector(word, vocab):
    """
    构建词袋（BoW）向量，使用独热编码表示每个词
    功能：将输入的词转换为词袋模型表示的独热编码向量
    参数：
    - word: 需要表示的词
    - vocab: 词汇表
    返回值：
    - BoW 向量，独热编码形式
    """
    bow = [0] * len(vocab)  # 初始化一个与词汇表长度相同的全零向量
    if word in vocab:  # 检查词是否在词汇表中
        bow[vocab.index(word)] = 1  # 将词在词汇表中的位置置为 1
    return np.array(bow)  # 返回 NumPy 数组形式的 BoW 向量

# 构建滑动窗口共现矩阵
def build_windowed_cooccurrence_matrix(corpus, vocab, window_size=2):
    """
    基于滑动窗口的共现矩阵构建
    功能：根据滑动窗口构建词汇之间的共现矩阵
    参数：
    - corpus: 语料库，包含多个句子的列表
    - vocab: 词汇表，语料库中的独立词汇
    - window_size: 滑动窗口大小，决定共现的范围
    返回值：
    - 共现矩阵，形状为 (vocab_size, vocab_size)
    """
    matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)  # 初始化共现矩阵，全为零
    for sentence in corpus:
        words = sentence.split()  # 将句子分割为单词列表
        for i, word1 in enumerate(words):
            # 确定滑动窗口的范围
            start = max(0, i - window_size)  # 滑动窗口左边界
            end = min(len(words), i + window_size + 1)  # 滑动窗口右边界
            for j in range(start, end):
                if i != j:  # 不统计自己与自己的共现
                    word2 = words[j]
                    if word1 in vocab and word2 in vocab:  # 检查两个词是否在词汇表中
                        idx1, idx2 = vocab.index(word1), vocab.index(word2)  # 找到两个词在词汇表中的索引
                        matrix[idx1, idx2] += 1  # 在共现矩阵中增加相应的共现次数
    return matrix  # 返回构建的共现矩阵

# 生成融合后的词嵌入向量
def generate_combined_word_vector(word, bow_vector, cooccurrence_matrix, vocab, alpha=0.5):
    """
    融合 BoW 向量和共现矩阵中的词向量，alpha 用于控制两种向量的加权
    功能：将 BoW 向量与共现向量融合，得到综合表示
    参数：
    - word: 输入的词
    - bow_vector: BoW 模型的词向量
    - cooccurrence_matrix: 共现矩阵
    - vocab: 词汇表
    - alpha: 权重系数，控制两种向量的加权比例
    返回值：
    - 融合后的词向量
    """
    idx = vocab.index(word)  # 获取词在词汇表中的索引
    cooc_vector = cooccurrence_matrix[idx]  # 获取该词的共现向量
    # 融合 BoW 向量与共现向量，使用 alpha 控制权重
    combined_vector = alpha * bow_vector + (1 - alpha) * cooc_vector
    return combined_vector  # 返回融合后的词向量

# 向量相加并计算最相似词
def add_combined_vectors(word1, word2, vocab, cooccurrence_matrix, alpha=0.5, top_n=3):
    """
    融合向量相加并寻找最相似的词（基于余弦相似度）
    功能：对两个词的向量进行融合和相加，并找出与结果最相似的词
    参数：
    - word1, word2: 要相加的两个词
    - vocab: 词汇表
    - cooccurrence_matrix: 共现矩阵
    - alpha: 融合向量时的权重系数
    > 当 alpha 趋向 1 时，模型更偏向于 BoW 的表示，当 alpha 趋向 0 时，模型更偏向于共现矩阵的语义信息。
    - top_n: 返回最相似词的数量
    返回值：
    - 与相加结果最相似的词及其相似度
    """
    # 构建 BoW 向量
    vector1 = build_bow_vector(word1, vocab)
    vector2 = build_bow_vector(word2, vocab)

    # 生成融合后的词向量
    combined_vector1 = generate_combined_word_vector(word1, vector1, cooccurrence_matrix, vocab, alpha)
    combined_vector2 = generate_combined_word_vector(word2, vector2, cooccurrence_matrix, vocab, alpha)

    # 相加两个融合后的词向量
    combined_vector = combined_vector1 + combined_vector2

    # 计算与词汇表中其他词的余弦相似度
    similarities = []  # 存储每个词的相似度
    for word in vocab:
        vec = generate_combined_word_vector(word, build_bow_vector(word, vocab), cooccurrence_matrix, vocab, alpha)
        norm_vec = np.linalg.norm(vec)  # 计算词向量的范数
        norm_combined = np.linalg.norm(combined_vector)  # 计算相加向量的范数
        if norm_vec == 0 or norm_combined == 0:
            similarities.append(0)  # 如果范数为 0，则相似度为 0
        else:
            similarity = np.dot(vec, combined_vector) / (norm_vec * norm_combined)  # 计算余弦相似度
            similarities.append(similarity)

    # 找到相似度最高的词语，按相似度从高到低排序，返回 top_n 个词
    most_similar_indices = np.argsort(-np.array(similarities))[:top_n]
    return [(vocab[idx], similarities[idx]) for idx in most_similar_indices]

# 构建滑动窗口共现矩阵
cooccurrence_matrix = build_windowed_cooccurrence_matrix(corpus, vocab, window_size=2)
print("滑动窗口共现矩阵:\n", cooccurrence_matrix)

# 测试“词语相加”
word1 = "苹果"
word2 = "香蕉"
result = add_combined_vectors(word1, word2, vocab, cooccurrence_matrix, alpha=0.5, top_n=3)
print(f"'{word1}' + '{word2}' 的结果是: {result}")