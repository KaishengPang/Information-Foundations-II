from math import sqrt

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

# 计算两个人的皮尔逊相关系数
def sim_pearson(prefs, person1, person2):
    # 得到双方都评价过的物品列表
    si = {}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item] = 1

    n = len(si)
    # 如果两者没有共同之处，返回0
    if n == 0:
        return 0

    # 对所有偏好求和
    sum1 = sum([prefs[person1][it] for it in si])
    sum2 = sum([prefs[person2][it] for it in si])

    # 求平方和
    sum1Sq = sum([pow(prefs[person1][it], 2) for it in si])
    sum2Sq = sum([pow(prefs[person2][it], 2) for it in si])

    # 求乘积之和
    pSum = sum([prefs[person1][it] * prefs[person2][it] for it in si])

    # 计算皮尔逊评分
    num = pSum - (sum1 * sum2 / n)
    den = sqrt((sum1Sq - pow(sum1, 2) / n) * (sum2Sq - pow(sum2, 2) / n))
    if den == 0:
        return 0

    r = num / den
    return r

# 为指定用户推荐电影
def get_recommendations(prefs, person, similarity=sim_pearson):
    totals = {}
    simSums = {}
    for other in prefs:
        # 不要和自己比较
        if other == person:
            continue
        sim = similarity(prefs, person, other)

        # 忽略评分为零或小于零的情况
        if sim <= 0:
            continue

        for item in prefs[other]:
            # 只对自己还未观看的电影进行评价
            if item not in prefs[person] or prefs[person][item] == 0:
                # 相似度 * 评分
                totals.setdefault(item, 0)
                totals[item] += prefs[other][item] * sim
                # 相似度之和
                simSums.setdefault(item, 0)
                simSums[item] += sim

    # 建立一个归一化的列表
    rankings = [(total / simSums[item], item) for item, total in totals.items()]

    # 返回排序后的列表
    rankings.sort()
    rankings.reverse()
    return rankings

# 获取 Toby 的电影推荐
recommendations = get_recommendations(critics, 'Toby')

# 输出推荐结果
print("为 Toby 推荐的电影：")
for score, movie in recommendations:
    print(f"{movie}: 预测评分 {score:.2f}")