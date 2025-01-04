from math import sqrt
from collections import defaultdict

# 定义影评者及其评分数据
critics = {
    'Lisa Rose': {
        'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,
        'Just My Luck': 3.0, 'Superman Returns': 3.5,
        'You, Me and Dupree': 2.5, 'The Night Listener': 3.0
    },
    'Gene Seymour': {
        'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5,
        'Just My Luck': 1.5, 'Superman Returns': 5.0,
        'The Night Listener': 3.0, 'You, Me and Dupree': 3.5
    },
    'Michael Phillips': {
        'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
        'Superman Returns': 3.5, 'The Night Listener': 4.0
    },
    'Claudia Puig': {
        'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
        'The Night Listener': 4.5, 'Superman Returns': 4.0,
        'You, Me and Dupree': 2.5
    },
    'Mick LaSalle': {
        'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
        'Just My Luck': 2.0, 'Superman Returns': 3.0,
        'The Night Listener': 3.0, 'You, Me and Dupree': 2.0
    },
    'Jack Matthews': {
        'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
        'The Night Listener': 3.0, 'Superman Returns': 5.0,
        'You, Me and Dupree': 3.5
    },
    'Toby': {
        'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0,
        'Superman Returns': 4.0
    }
}

# 计算两部电影之间的皮尔逊相关系数
def sim_pearson(prefs, item1, item2):
    si = {}
    for person in prefs[item1]:
        if person in prefs[item2]:
            si[person] = 1

    n = len(si)
    if n == 0:
        return 0

    sum1 = sum([prefs[item1][it] for it in si])
    sum2 = sum([prefs[item2][it] for it in si])

    sum1Sq = sum([pow(prefs[item1][it], 2) for it in si])
    sum2Sq = sum([pow(prefs[item2][it], 2) for it in si])

    pSum = sum([prefs[item1][it] * prefs[item2][it] for it in si])

    num = pSum - (sum1 * sum2 / n)
    den = sqrt((sum1Sq - pow(sum1, 2) / n) * (sum2Sq - pow(sum2, 2) / n))
    if den == 0:
        return 0

    return num / den

# 将影评者的数据转换为物品为中心
def transform_prefs(prefs):
    result = defaultdict(dict)
    for person in prefs:
        for item in prefs[person]:
            result[item][person] = prefs[person][item]
    return result

# 计算每部电影的相似电影
def calculate_similar_items(prefs, n=10):
    result = {}
    item_prefs = transform_prefs(prefs)
    for item in item_prefs:
        scores = [(sim_pearson(item_prefs, item, other), other)
                  for other in item_prefs if other != item]
        scores.sort()
        scores.reverse()
        result[item] = scores[0:n]
    return result

# 根据物品相似性推荐电影
def get_recommended_items(prefs, item_match, user):
    user_ratings = prefs[user]
    scores = {}
    total_sim = {}

    for (item, rating) in user_ratings.items():
        for (similarity, item2) in item_match[item]:
            if item2 in user_ratings: continue
            scores.setdefault(item2, 0)
            scores[item2] += similarity * rating
            total_sim.setdefault(item2, 0)
            total_sim[item2] += similarity

    rankings = [(score / total_sim[item], item) for item, score in scores.items()]
    rankings.sort()
    rankings.reverse()
    return rankings

# 计算电影的相似度矩阵
item_match = calculate_similar_items(critics)

# 获取 Toby 的电影推荐
recommendations = get_recommended_items(critics, item_match, 'Toby')

# 输出推荐结果
print("为 Toby 推荐的电影：")
for score, movie in recommendations:
    print(f"{movie}: 预测评分 {score:.2f}")