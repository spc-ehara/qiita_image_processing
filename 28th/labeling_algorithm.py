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


class Labeling():

    def do_labeling(self, bin_img):
        # 入力は2値化画像のみ
        self.bin_img = bin_img

        # 走査簡素化のために画像に枠を追加
        self._border_interpolate()

        # ラベルテーブル作成
        self.label_table = np.zeros_like(self.bin_img)

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
                label_around_pixel = self._get_neighbor_label(y, x)

                # 注目画素周辺のラベルを参照・ルックアップテーブル更新
                labels = self._update_table(label_around_pixel, neighbor_shape)

                # ラベルテーブル更新
                self._set_neighbor_label(y, x, labels)

        # ラベルの衝突を解決(ルックアップテーブル内)
        self._resolve_collision()

        # 虫食いラベルを修正(ルックアップテーブル内)
        self._compress_table()

        # 修正したルックアップテーブルを参照して、ラベルテーブルを更新
        self._update_label_table()

        # 逆走査時の近傍指定
        neighbor_shape = np.array([0,0,0,0,0,1,1,1,1])

        # 逆走査
        for y in list(reversed(range(height))):
            for x in list(reversed(range(width))):
                if(self.bin_img[y][x] != 255):
                    continue
                    
                # ラベルテーブルを取得
                label_around_pixel = self._get_neighbor_label(y, x)

                # 注目画素周辺のラベルを参照・ルックアップテーブル更新
                labels = self._update_table(label_around_pixel, neighbor_shape)

                # ラベルテーブル更新
                self._set_neighbor_label(y, x, labels)

        # ラベルの衝突を解決(ルックアップテーブル内)
        self._resolve_collision()

        # 虫食いラベルを修正(ルックアップテーブル内)
        self._compress_table()

        # 修正したルックアップテーブルを参照して、ラベルテーブルを更新
        self._update_label_table()

        # 画像に追加した枠の削除
        self._border_cutoff()

        return self.label_table
    

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

    # ラベルの衝突を解決
    def _resolve_collision(self):
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

    # ルックアップテーブルを参照して、ラベルテーブルを更新
    def _update_label_table(self):
        for array, nLabel in enumerate(self.lookup_table):
            self.label_table = np.where(self.label_table == array, nLabel, self.label_table)

    # ラベルテーブルを取得
    def _get_neighbor_label(self, y, x):
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
    def _set_neighbor_label(self, y, x, labels):
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
        new_label = []
        for array in range(len(self.lookup_table)):
            if array == self.lookup_table[array]:
                new_label.append(array)

        for l in new_label:
            self.lookup_table = np.where(self.lookup_table == l, new_label.index(l), self.lookup_table)

    # Debug
    def debug_write(self):

        # ラベルテーブルをcsv出力
        np.savetxt('label_table.csv', self.label_table, fmt='%s', delimiter=',')
        
        # ラベリングしたすべての物体を1つずつ出力
        self._write_all_object()
        
        # ルックアップテーブルを出力
        self._show_lookup_table()


    def _write_all_object(self):
        for nlabel in range(self.label_table.max()+1):
            label_obj = np.zeros_like(self.label_table)
            label_obj[np.where(self.label_table == nlabel)] = 255
            cv2.imwrite("label_obj" + str(nlabel) + ".png", label_obj)

    def _show_lookup_table(self):
        for nlabel in range(self.label_table.max() + 1):
            print(str(nlabel) + ' : ' + str(self.lookup_table[nlabel]))


if __name__ == "__main__":
    
    src_img = cv2.imread(sys.argv[1])
    bin_img = binarize(src_img, 150, cv2.THRESH_BINARY_INV)
    cv2.imwrite("bin_img.png", bin_img)
    
    label = Labeling()
    label_table = label.do_labeling(bin_img)

    label.debug_write()
    cv2.imwrite("label_img.png", put_color_to_objects(src_img, label_table))
