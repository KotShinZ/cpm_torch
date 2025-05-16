import numpy as np
import torch
import torch.nn.functional as F
from cpm_torch.CPM_Map import *

class CPM:
    def __init__(self, size, device = "cuda"):
        self.size = size
        self.dim = len(size)
        self.height = size[0]
        self.width = size[1]
        if dim == 3:
            self.depth = size[2]
        
        self.cell_newer_id_counter = 1

        self.center_index = 4  # 中央のインデックス
        self.neighbors = [1, 3, 5, 7]
        self.neighbors_len = len(self.neighbors)  # 4近傍の数
        
        # マップテンソルを作成: (高さ, 幅, チャンネル数)
        # チャンネル 0: 細胞ID
        self.map_tensor = torch.zeros((self.height, self.width, 1), dtype=torch.float32, device=device)

        return map_tensor
    
    def add_cell(self, pos):
        # マップ中央に初期細胞を配置
        center_x_slice = slice(height // 2 - 1, height // 2 + 1)  # 例: 中央2x2領域
        center_y_slice = slice(width // 2 - 1, width // 2 + 1)

        # add_cell関数で細胞を追加（map_tensorが直接変更され、次のIDが返る）
        self.map_tensor, _ = add_cell(map_tensor, center_x_slice, center_y_slice, value=100)


    def add_cell(map_tensor, x_slice, y_slice, value=100.0):
        """指定されたスライスに新しいIDと値を持つ細胞を追加する。"""
        global cell_newer_id_counter  # グローバルなIDカウンターを使用
        current_id = cell_newer_id_counter  # 現在のカウンター値を新しいIDとする

        # 指定スライスと同じ形状で、IDと値で埋められたテンソルを作成
        id_tensor = torch.full_like(map_tensor[x_slice, y_slice, 0], float(current_id))
        value_tensor = torch.full_like(map_tensor[x_slice, y_slice, 1], float(value))

        # スライシングを使ってマップテンソルにIDと値を直接代入（インプレース操作）
        map_tensor[x_slice, y_slice, 0] = id_tensor  # チャンネル0 (ID)
        map_tensor[x_slice, y_slice, 1] = value_tensor  # チャンネル1 (Value)
        map_tensor[x_slice, y_slice, 2] = id_tensor  # チャンネル2 (Previous ID) も初期化

        # 次の細胞のためにIDカウンターをインクリメント
        cell_newer_id_counter += 1
        # 変更されたマップテンソルと、次に使用するIDを返す
        return map_tensor, cell_newer_id_counter


        