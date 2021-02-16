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

if __name__ == "__main__":
    
    src_img = cv2.imread(sys.argv[1])
    bin_img = binarize(src_img, 180, cv2.THRESH_BINARY_INV)
    
    retval, labels = cv2.connectedComponents(bin_img)
    cv2.imwrite("labels.png", put_color_to_objects(src_img, labels))