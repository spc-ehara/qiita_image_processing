import sys
import numpy as np
import cv2
import random

# 2値化
def binarize(src_img, thresh, mode):
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
    bin_img = cv2.threshold(gray_img, thresh, 255, mode)[1]
    return bin_img

# ラベルテーブルの情報を元に入力画像に色をつける
def put_color_to_objects(src_img, label_table):
    label_img = np.zeros_like(src_img)
    for label in range(label_table.max()+1):
        label_group_index = np.where(label_table == label)
        label_img[label_group_index] = random.sample(range(255), k=3)
    return label_img

# 各ラベルの座標と面積を描画する
def draw_stats(src_img, stats):
    stats_img = src_img.copy()
    for coordinate in stats[1:]:
        left_top = (coordinate[0], coordinate[1])
        right_bottom = (coordinate[0] + coordinate[2], coordinate[1] + coordinate[3])
        stats_img = cv2.rectangle(stats_img, left_top, right_bottom, (0, 0, 255), 1)
        stats_img = cv2.putText(stats_img, str(coordinate[4]), left_top, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
    return stats_img

# 物体の中心座標を描画する
def draw_centroids(src_img, centroids):
    centroids_img = src_img.copy()
    for coordinate in centroids[1:]:
        center = (int(coordinate[0]), int(coordinate[1]))
        centroids_img = cv2.circle(centroids_img, center, 1, (0, 0, 255))
    return centroids_img

if __name__ == "__main__":
    
    src_img = cv2.imread(sys.argv[1])
    bin_img = binarize(src_img, 180, cv2.THRESH_BINARY_INV)
    
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)
    stats_img = draw_stats(src_img, stats)
    centroids_img = draw_centroids(stats_img, centroids)

    cv2.imwrite("labels.png", put_color_to_objects(src_img, labels))
    cv2.imwrite("centroids_img.png", centroids_img)