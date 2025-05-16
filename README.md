![output](https://github.com/user-attachments/assets/53eb4346-5523-48e0-952b-90a710658d5a)
# cpm_torch
細胞のシミュレーションモデルである、CelluarPottsModelをPytorchで実装します。

## Introduction
cpm_torchはCelluar Potts ModelをPyTorchで実装したライブラリです。将来的には機械学習で学習可能にすることを目指します。

## What's new?
Celluar Potts ModelをPytorchで実装しました。
GPUを用いて並列に計算することで高速なシミュレーションを行います。

## Installation
```shell
pip install cpm_torch
```

## Quick Start
```shell
import torch
from tqdm import tqdm
import cpm_torch
from cpm_torch.CPM_Image import *
from cpm_torch.CPM import *

# === ハイパーパラメータ ===
s_0 = 100.0  # 初期細胞密度（float型を使用）

l_A = 1.0  # 面積エネルギー項の係数λ_A
l_L = 1.0  # 周囲長エネルギー項の係数λ_L
A_0 = 150.0  # 目標細胞面積 A_0
L_0 = 82.0  # 目標細胞周囲長 L_0

T = 1.0  # 温度パラメータ T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

map_tensor = torch.zeros((256, 256, 3), dtype=torch.float32, device=device)

for x in range(10):
    for y in range(10):
        cell_x = x + 122
        cell_y = y + 122
        map_tensor, _ = add_cell(
            map_tensor, slice(cell_x, cell_x + 1), slice(cell_y, cell_y + 1), s_0
        )

for i in tqdm(range(500), desc="エポック"):
  for x_offset in range(3): # x方向オフセット (0 or 1)
      for y_offset in range(3): # y方向オフセット (0 or 1)
          with torch.no_grad():
            map_tensor, _ = cpm_checkerboard_step(map_tensor, l_A, A_0, l_L, L_0, T,
                                                x_offset, y_offset)
imshow_map(map_tensor)
```
