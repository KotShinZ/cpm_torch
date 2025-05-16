import numpy as np
import torch
import torch.nn.functional as F
from cpm_torch.CPM_Map import *

class CPM:
    def __init__(self, size, device = "cuda"):
        self.size = size
        self.dim = len(size)
        
        self.cell_newer_id_counter = 1

        self.center_index = 4  # 中央のインデックス
        self.neighbors = [1, 3, 5, 7]
        self.neighbors_len = len(neighbors)  # 4近傍の数
        