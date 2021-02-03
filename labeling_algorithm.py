import sys
import numpy as np
import cv2
import random

# 2値化
def binarize(src_img, thresh, mode):
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
    bin_img = cv2.threshold(gray_img, thresh, 255, mode)[1]
    cv2.imwrite("bin_img.png", bin_img)

    return bin_img

class Labeling():

    def do_labeling(self, bin_img):
        # 入力は2値化画像のみ
        self.bin_img = bin_img

        # 走査簡素化のために画像に枠を追加
        self._border_interpolate()

        # ラベルテーブル作成
        self.label_table = np.zeros_like(self.bin_img)
        print("labeltable=" + str(self.label_table.shape))

        # ルックアップテーブル作成
        self.lookup_table = [0]

        # 走査時の近傍指定
        neighbor_shape = np.array([1,1,1,1,0,0,0,0,0])

        # 走査
        height, width = self.bin_img.shape[:2]
        for y in range(height):
            for x in range(width):
                if(self.bin_img[y][x] != 255):
                    continue

                # ラベルテーブルを取得
                label_around_pixel = self._get_label_table(y, x)

                # 注目画素周辺のラベルを参照・ルックアップテーブル更新
                labels = self._update_table(label_around_pixel, neighbor_shape)

                # ラベルテーブル更新
                self._set_label_table(y, x, labels)

        self._resolve_collision()
        self._update_label_table()

        neighbor_shape = np.array([0,0,0,0,0,1,1,1,1])

        # 逆走査
        for y in list(reversed(range(height))):
            for x in list(reversed(range(width))):
                if(self.bin_img[y][x] != 255):
                    continue
                    
                # ラベルテーブルを取得
                label_around_pixel = self._get_label_table(y, x)

                # 注目画素周辺のラベルを参照・ルックアップテーブル更新
                labels = self._update_table(label_around_pixel, neighbor_shape)

                # ラベルテーブル更新
                self._set_label_table(y, x, labels)

        self._resolve_collision()
        self._update_label_table()

        # ラベルテーブルの虫食いラベルを修正
        self._compress_table()

        # 画像に追加した枠の削除
        self._border_cutoff()
    

    # 画像の外周に1ピクセルの外枠をつける(走査時のピクセル外アクセスをなくすため)
    def _border_interpolate(self):
        # 上下の外枠を追加
        horizontal_line = np.zeros_like(self.bin_img[0])
        self.bin_img = np.insert(self.bin_img, [0, len(self.bin_img)], horizontal_line, axis=0)

        # 左右の外枠を追加
        vertical_line = np.zeros((len(self.bin_img), 1))
        self.bin_img = np.insert(self.bin_img, [0, len(self.bin_img[0])], vertical_line, axis=1)

    # 画像に追加した外枠の切り落とし
    def _border_cutoff(self):
        # 上下の外枠を削除
        self.label_table = np.delete(self.label_table, [0, len(self.label_table)-1], axis=0)

        # 左右の外枠を削除
        self.label_table = np.delete(self.label_table, [0, len(self.label_table[0])-1], axis=1)
        print("labeltable=" + str(self.label_table.shape))

    
    # 注目画素周辺のラベルを参照・ルックアップテーブル更新
    def _update_table(self, all_label, neighbor_shape):
        around_label = np.where(neighbor_shape == 1, all_label, -1)
        label_index = np.where(around_label != -1)
        focus_pixels_label = around_label[label_index]

        # 注目画素の周辺ラベル番号がすべて0か
        if np.all(focus_pixels_label == 0) == True:
            if all_label[4] == 0:
                around_label[4] = self.label_table.max() + 1
                self.lookup_table.append(int(self.label_table.max() + 1))
        
        # 注目画素の周辺ラベル番号が複数ある
        else:
            uptozero_index = np.where(around_label > 0)
            min_label = around_label[uptozero_index].min()
            around_label[uptozero_index] = min_label
            around_label[4] = min_label

            # around_labelとall_labelのuptozero_index部分の差分を見て、ルックアップテーブル更新
            for index in uptozero_index[0]:
                self.lookup_table[all_label[index]] = around_label[index]
            
        return around_label

    def _resolve_collision(self):
        print(self.lookup_table)
        for nLabel in reversed(range(len(self.lookup_table))):
            tmp = []
            array_num = nLabel
            element = self.lookup_table[nLabel]

            while element != array_num:
                tmp.append(array_num)
                array_num = element
                element = self.lookup_table[element]

            for t in tmp:
               self.lookup_table[t] = element

        print(self.lookup_table)

    # ルックアップテーブルを参照して、ラベルテーブルを更新
    def _update_label_table(self):
        for array, nLabel in enumerate(self.lookup_table):
            self.label_table = np.where(self.label_table == array, nLabel, self.label_table)

    # ラベルテーブルを取得
    def _get_label_table(self, y, x):
        return np.array([self.label_table[y-1][x-1],
                         self.label_table[y-1][x],
                         self.label_table[y-1][x+1],
                         self.label_table[y][x-1],
                         self.label_table[y][x],
                         self.label_table[y][x+1],
                         self.label_table[y+1][x-1],
                         self.label_table[y+1][x],
                         self.label_table[y+1][x+1]])

    # ラベルテーブルに書込み
    def _set_label_table(self, y, x, labels):
        self.label_table[y-1][x-1] = labels[0] if labels[0] != -1 else self.label_table[y-1][x-1]
        self.label_table[y-1][x]   = labels[1] if labels[1] != -1 else self.label_table[y-1][x]  
        self.label_table[y-1][x+1] = labels[2] if labels[2] != -1 else self.label_table[y-1][x+1]
        self.label_table[y][x-1]   = labels[3] if labels[3] != -1 else self.label_table[y][x-1]  
        self.label_table[y][x]     = labels[4] if labels[4] != -1 else self.label_table[y][x]    
        self.label_table[y-1][x-1] = labels[5] if labels[5] != -1 else self.label_table[y-1][x-1]
        self.label_table[y-1][x]   = labels[6] if labels[6] != -1 else self.label_table[y-1][x]  
        self.label_table[y-1][x+1] = labels[7] if labels[7] != -1 else self.label_table[y-1][x+1]
        self.label_table[y][x-1]   = labels[8] if labels[8] != -1 else self.label_table[y][x-1]

    # 虫食いラベルを埋める
    def _compress_table(self):
        blank_label = []
        for array in range(len(self.lookup_table)):
            if array not in self.lookup_table:
                blank_label.append(array)
            elif len(blank_label) != 0:
                self.lookup_table = np.where(self.lookup_table == array, blank_label.pop(0), self.lookup_table)
                blank_label.append(array)

        print(self.lookup_table)

    def write(self):
        np.savetxt('label_table.csv', self.label_table, fmt='%s', delimiter=',')
        
        for nlabel in range(self.label_table.max()+1):
            label_obj = np.zeros_like(self.label_table)
            label_obj[np.where(self.label_table == nlabel)] = 255
            cv2.imwrite("label_obj" + str(nlabel) + ".png", label_obj)
        
        print(self.label_table.max())

    def get_label_img(self, img):
        ex = img.copy()
        for label in range(self.label_table.max()+1):
            label_group_index = np.where(self.label_table == label)
            ex[label_group_index] = random.sample(range(255), k=3)
        cv2.imwrite("label_img.png",ex)

    def show_lookup_table(self):
        for nlabel in range(self.label_table.max() + 1):
            print(str(nlabel) + ' : ' + str(self.lookup_table[nlabel]))


if __name__ == "__main__":
    
    src_img = cv2.imread(sys.argv[1])
    bin_img = binarize(src_img, 150, cv2.THRESH_BINARY_INV)
    
    label = Labeling()
    label.do_labeling(bin_img)
    label.show_lookup_table()

    label.get_label_img(src_img)
    label.write()
