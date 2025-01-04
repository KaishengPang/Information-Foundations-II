import cv2
import numpy as np
from skimage import segmentation, color
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

# 计算两个区域之间的纹理相似度
def calculate_texture_similarity(region1, region2, image, segments):
    """使用灰度共生矩阵计算纹理相似度"""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度图像
    texture1 = gray_image[segments == region1].reshape(-1, 1)  # 区域1的纹理
    texture2 = gray_image[segments == region2].reshape(-1, 1)  # 区域2的纹理

    # 计算灰度共生矩阵
    glcm1 = graycomatrix(texture1, [1], [0], symmetric=True, normed=True)
    glcm2 = graycomatrix(texture2, [1], [0], symmetric=True, normed=True)

    # 计算对比度
    contrast1 = graycoprops(glcm1, 'contrast')[0, 0]
    contrast2 = graycoprops(glcm2, 'contrast')[0, 0]

    # 返回两个区域之间的纹理相似度
    return abs(contrast1 - contrast2)

# 计算两个区域之间的颜色相似度
def calculate_color_similarity(region1, region2, image, segments, color_spaces=['RGB', 'HSV', 'GRAY']):
    """在多个颜色空间中计算颜色相似度"""
    color_similarity = 0
    for color_space in color_spaces:
        if color_space == 'RGB':
            img = image  # 原图RGB颜色空间
        elif color_space == 'HSV':
            img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 转为HSV
        elif color_space == 'GRAY':
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度
        
        # 计算区域1和区域2在颜色空间中的均值
        color1 = np.mean(img[segments == region1], axis=0)
        color2 = np.mean(img[segments == region2], axis=0)
        
        # 累加颜色空间中的相似度
        color_similarity += np.linalg.norm(color1 - color2)

    # 返回平均相似度
    return color_similarity / len(color_spaces)

# 计算两个区域之间的尺寸相似度
def calculate_size_similarity(region1, region2, segments):
    """计算尺寸相似度"""
    size1 = np.sum(segments == region1)  # 区域1的大小
    size2 = np.sum(segments == region2)  # 区域2的大小
    return 1 - abs(size1 - size2) / (size1 + size2)  # 计算相对大小差异

# 计算两个区域之间的位置相似度
def calculate_position_similarity(region1, region2, segments):
    """计算位置相似度"""
    y1, x1 = np.where(segments == region1)  # 区域1的像素位置
    y2, x2 = np.where(segments == region2)  # 区域2的像素位置
    center1 = [np.mean(x1), np.mean(y1)]  # 区域1的中心
    center2 = [np.mean(x2), np.mean(y2)]  # 区域2的中心
    return np.linalg.norm(np.array(center1) - np.array(center2))  # 计算中心之间的距离

# 计算两个区域之间的边缘相似度
def calculate_edge_similarity(region1, region2, image, segments):
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 100, 200)  # 使用Canny边缘检测
    edge_density1 = np.sum(edges[segments == region1]) / np.sum(segments == region1)  # 区域1的边缘密度
    edge_density2 = np.sum(edges[segments == region2]) / np.sum(segments == region2)  # 区域2的边缘密度
    return abs(edge_density1 - edge_density2)  # 返回边缘密度差

# 综合计算多个相似度指标
def calculate_similarity(region1, region2, image, segments, color_spaces=['RGB', 'HSV', 'GRAY']):
    """综合计算多个相似度指标"""
    color_sim = calculate_color_similarity(region1, region2, image, segments, color_spaces)
    texture_sim = calculate_texture_similarity(region1, region2, image, segments)
    size_sim = calculate_size_similarity(region1, region2, segments)
    position_sim = calculate_position_similarity(region1, region2, segments)
    edge_sim = calculate_edge_similarity(region1, region2, image, segments)
    # 综合相似度
    similarity = 0.35 * color_sim + 0.25 * texture_sim + 0.2 * size_sim + 0.1 * position_sim + 0.1 * edge_sim

    return similarity

# 选择得分最高的top_k个区域
def select_top_regions(boxes, scores, top_k=5):
    """选择得分最高的top_k个区域"""
    sorted_indices = np.argsort(scores)[:top_k]  # 得分越小越好
    return [boxes[i] for i in sorted_indices]

# 计算两个框的IoU（交并比）
def calculate_iou(boxA, boxB):
    """计算两个框的IoU（交并比）"""
    xA = max(boxA[0], boxB[0])  # 交集左上角x
    yA = max(boxA[1], boxB[1])  # 交集左上角y
    xB = min(boxA[2], boxB[2])  # 交集右下角x
    yB = min(boxA[3], boxB[3])  # 交集右下角y

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)  # 交集面积
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)  # 框A的面积
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)  # 框B的面积
    iou = interArea / float(boxAArea + boxBArea - interArea)  # IoU计算公式
    return iou

# 评估选择性搜索的性能
def evaluate_performance(boxes, ground_truth_boxes, iou_threshold=0.5):
    """评估选择性搜索的性能，计算与ground truth的重叠度"""
    matches = 0
    for box in boxes:
        for gt_box in ground_truth_boxes:
            if calculate_iou(box, gt_box) >= iou_threshold:  # 若IoU高于阈值，计为匹配
                matches += 1
                break
    return matches / len(ground_truth_boxes)  # 返回匹配的比例

# 选择性搜索主函数
def selective_search_manual(image_path, ground_truth_boxes, target_regions=20, max_merges=100, init_segments=150, top_k=5):
    """选择性搜索主函数，包含区域打分与筛选、性能评估"""
    im = cv2.imread(image_path)  # 读取图像
    new_height = 200  # 新图像高度
    new_width = int(im.shape[1] * new_height / im.shape[0])  # 新图像宽度
    im = cv2.resize(im, (new_width, new_height))  # 缩放图像

    # 初始分割图像为多个小区域
    segments = segmentation.slic(im, n_segments=init_segments, compactness=10)  # SLIC分割
    regions = np.unique(segments)  # 获取所有区域标签

    # 初始化相似度集合
    similarities = []
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            sim = calculate_similarity(regions[i], regions[j], im, segments)  # 计算相似度
            similarities.append((sim, regions[i], regions[j]))  # 保存相似度及区域信息

    similarities = sorted(similarities, key=lambda x: x[0])  # 相似度排序
    merge_count = 0
    while similarities and len(regions) > target_regions and merge_count < max_merges:
        sim, region1, region2 = similarities.pop(0)  # 取出最相似的区域
        segments[segments == region2] = region1  # 合并区域
        merge_count += 1
        regions = np.unique(segments)  # 更新区域标签
        new_similarities = []
        for region in regions:
            if region != region1:
                sim = calculate_similarity(region1, region, im, segments)  # 计算新的相似度
                new_similarities.append((sim, region1, region))
        similarities = sorted(new_similarities, key=lambda x: x[0])

    # 提取最终的区域框并计算得分
    boxes = []
    scores = []
    for region in np.unique(segments):
        y, x = np.where(segments == region)
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        boxes.append([x_min, y_min, x_max, y_max])  # 保存框信息
        scores.append(calculate_size_similarity(region, region, segments))  # 保存得分

    # 筛选出得分最高的 top_k 个区域
    selected_boxes = select_top_regions(boxes, scores, top_k)

    # 绘制筛选后的结果
    for box in selected_boxes:
        cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1, cv2.LINE_AA)
    
    # 评估性能
    performance_score = evaluate_performance(selected_boxes, ground_truth_boxes)
    print(f"Performance Score (IoU >= 0.5): {performance_score:.2f}")

    cv2.imshow("Selective Search Manual", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 初始分割展示函数
def generate_initial_segmentation(image_path, n_segments=150, compactness=10):
    # 读取图像并转换为 RGB 格式
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 使用 SLIC 对图像进行分割
    segments = segmentation.slic(image, n_segments=n_segments, compactness=compactness, start_label=1)

    # 将每个分割区域填充为均值颜色
    segmented_image = color.label2rgb(segments, image, kind='avg')

    # 显示分割结果
    plt.figure(figsize=(10, 6))
    plt.imshow(segmented_image)
    plt.axis('off')
    plt.title("Initial Segmentation with SLIC")
    plt.show()

# 展示初始分割结果
# generate_initial_segmentation('2.png', n_segments=200, compactness=100)

# ground truth boxes（用于性能评估）
# ground_truth_boxes = [
#     [20, 47, 300, 150],  # 框
#     [16, 50, 320, 161]
# ]
ground_truth_boxes = [
    [20, 20, 140, 193],  # 框
    [46, 20, 104, 88]
]


selective_search_manual('2.png', ground_truth_boxes, target_regions=15, max_merges=100, init_segments=50, top_k=30)
