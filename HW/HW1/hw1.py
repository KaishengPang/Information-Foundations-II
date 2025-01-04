import re
import math
from collections import Counter, defaultdict
import pandas as pd


# 将文本进行分词，提取出所有单词
# 该函数用于将每篇文章转化为单词的列表
def tokenize(text):
    # 使用正则表达式将文本转化为小写单词列表
    return re.findall(r'\b\w+\b', text.lower())


# 加载并将联邦党人文集分割为单独的文章
# 文章以“The Federalist Papers : No.”开头来划分
def load_and_split_papers(file_path):
    # 以 utf-8 编码读取文件内容，确保文件能够正确读取
    with open(file_path, 'r', encoding='utf-8') as file:
        contents = file.read()

    # 按照 "The Federalist Papers : No." 进行切分，将内容分成 85 篇文章
    papers = contents.split("The Federalist Papers : No.")

    # 去除第一个空的内容，留下实际文章
    papers = papers[1:]
    return papers


# 根据文章内容分类，将文章分为 Hamilton (H)、Madison (M) 和未知作者 (unknown)
def categorize_papers(papers):
    H, M, unknown, unknownIndex = [], [], [], []

    for i, paper in enumerate(papers, 1):  # 遍历每篇文章并编号从1开始
        # 如果文章含有 "HAMILTON OR MADISON"，则为未知作者文章
        if "HAMILTON OR MADISON" in paper:
            unknown.append(paper)
            unknownIndex.append(i)
        # 如果含有 "HAMILTON AND MADISON"，则跳过（两者共同撰写的文章）
        elif "HAMILTON AND MADISON" in paper:
            continue
            # 如果含有 "HAMILTON"，则归类为 Hamilton 撰写的文章
        elif "HAMILTON" in paper:
            H.append(paper)
        # 如果含有 "MADISON"，则归类为 Madison 撰写的文章
        elif "MADISON" in paper:
            M.append(paper)

    return H, M, unknown, unknownIndex


# 统计每篇文章中的单词出现次数
def count_words(papers):
    word_counts = Counter()  # 初始化词频统计的计数器
    for paper in papers:
        # 将文章分词，并更新每个单词的计数
        words = tokenize(paper)
        word_counts.update(words)
    return word_counts


# 筛选出出现次数超过阈值的高频词
# 该函数用于找出所有出现次数超过 10 次的单词
def filter_high_frequency_words(word_counts, threshold=10):
    # 只保留那些出现次数超过 10 次的单词，返回一个集合
    high_freq_words = {word for word, count in word_counts.items() if count > threshold}
    return high_freq_words


# 根据高频词创建每篇文章的词向量
# 词向量是基于高频词的每篇文章中这些词出现的次数
def create_word_vector(paper, high_freq_words):
    words = tokenize(paper)  # 将文章分词
    # 初始化词向量，每个高频词的初始计数为 0
    word_vector = {word: 0 for word in high_freq_words}

    # 对文章中的每个词进行计数，如果该词是高频词，则增加其计数
    for word in words:
        if word in word_vector:
            word_vector[word] += 1
    return word_vector


# 计算词的条件概率，并使用拉普拉斯平滑避免出现概率为 0 的情况
def calculate_word_probabilities(H_counts, M_counts, high_freq_words, alpha=1):
    # 计算 Hamilton 和 Madison 文章中高频词的总出现次数
    total_H_words = sum(H_counts[word] for word in high_freq_words)
    total_M_words = sum(M_counts[word] for word in high_freq_words)

    # 高频词汇的词汇表大小
    vocab_size = len(high_freq_words)

    # 初始化词的条件概率字典，并使用拉普拉斯平滑
    H_word_probs = defaultdict(lambda: alpha / (total_H_words + alpha * vocab_size))
    M_word_probs = defaultdict(lambda: alpha / (total_M_words + alpha * vocab_size))

    # 计算每个高频词在 Hamilton 和 Madison 文章中的条件概率
    for word in high_freq_words:
        H_word_probs[word] = (H_counts[word] + alpha) / (total_H_words + alpha * vocab_size)
        M_word_probs[word] = (M_counts[word] + alpha) / (total_M_words + alpha * vocab_size)

    return H_word_probs, M_word_probs


# 使用朴素贝叶斯方法预测文章的作者是 Hamilton 还是 Madison
def predict_paper_class(paper, high_freq_words, H_word_probs, M_word_probs, H_prob, M_prob):
    # 创建该文章的词向量
    word_vector = create_word_vector(paper, high_freq_words)

    # 初始化 Hamilton 和 Madison 的对数概率（使用先验概率的对数）
    log_H_prob = math.log(H_prob)
    log_M_prob = math.log(M_prob)

    # 根据词向量中每个高频词的出现次数，累加其条件概率的对数
    for word, count in word_vector.items():
        if count > 0:
            log_H_prob += count * math.log(H_word_probs[word])
            log_M_prob += count * math.log(M_word_probs[word])

    # 返回预测结果，根据对数概率的大小判断作者
    return 'Hamilton' if log_H_prob > log_M_prob else 'Madison'


# 主函数，负责加载数据、训练模型并预测未知文章的作者
def train_and_predict(file_path):
    # 加载并分类文章
    papers = load_and_split_papers(file_path)
    H, M, unknown, unknownIndex = categorize_papers(papers)

    # 统计 Hamilton 和 Madison 文章中的单词出现次数
    H_counts = count_words(H)
    M_counts = count_words(M)

    # 筛选出高频词（出现次数超过 10 次）
    all_counts = count_words(H + M)
    high_freq_words = filter_high_frequency_words(all_counts, threshold=10)

    # 计算高频词的条件概率
    H_word_probs, M_word_probs = calculate_word_probabilities(H_counts, M_counts, high_freq_words)

    # 计算先验概率
    H_prob = len(H) / (len(H) + len(M))
    M_prob = len(M) / (len(H) + len(M))

    # 对未知作者的文章进行预测
    predictions = []
    for paper in unknown:
        prediction = predict_paper_class(paper, high_freq_words, H_word_probs, M_word_probs, H_prob, M_prob)
        predictions.append(prediction)

    # 输出预测结果
    prediction_df = pd.DataFrame({
        'Unknown Paper Index': unknownIndex,
        'Predicted Author': predictions
    })

    # 打印预测结果
    print(prediction_df)
    return prediction_df

file_path = 'federalist_papers_complete1.txt'
train_and_predict(file_path)