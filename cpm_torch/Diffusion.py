import torch
from cpm_torch.CPM_Map import *

diffusion_kernel = torch.tensor([1, 2, 1, 2, 0, 2, 1, 2, 1], dtype=torch.float32) / 16.0


@torch.no_grad()  # 拡散ステップでは勾配計算を無効化
def diffusion_step(map_tensor: torch.Tensor, params=[1], dts=[0.1], percent=[1]):
    """
    密度チャンネル（チャンネル1）に対して、細胞境界を尊重した拡散を1ステップ実行する。
    ラプラシアンの近似に畳み込みを使用する。
    """
    if dts is not torch.Tensor:
        dts = torch.tensor(dts, dtype=torch.float32, device=density_diff.device)
    if percent is not torch.Tensor:
        percent = torch.tensor(percent, dtype=torch.float32, device=density_diff.device)

    # map_tensor 形状: (H, W, C) C=3 [ID, Density, PrevID]
    H, W, C = map_tensor.shape

    # 入力 (H, W, C) -> 出力 (H, W, 9, C)
    patches = extract_patches_batched_channel(map_tensor, 3)
    id_patches = patches[:, :, :, 0]  # IDパッチ (H, W, 9)
    density_patches = patches[:, :, :, params]  # 密度パッチ (H, W, 9, C)

    center_density = density_patches[:, :, 4]  # 中心ピクセルの密度 (H, W)
    center_ids = id_patches[:, :, 4]  # 中心ピクセルのID (H, W)

    # 隣接ピクセルとの密度差を計算
    density_diff = density_patches - center_density.unsqueeze(-1)  # (H, W, 9)

    # 境界マスクを作成: 隣接ピクセルが同じIDなら0、異なるなら1
    same_id_mask = (id_patches != center_ids.unsqueeze(-1)).float()  # (H, W, 9)

    weights = 1 - percent * same_id_mask

    update = torch.sum(diffusion_kernel.view(1, 1, 9) * weights * density_diff, dim=2) * dts  # (H, W)

    # --- 更新された密度をマップテンソルに反映 ---
    map_out = map_tensor.clone()  # 元のマップをコピー
    map_out[:, :, params] += update  # チャンネル1（密度）を更新
    # ID (チャンネル0) と Previous ID (チャンネル2) はこのステップでは変更しない

    return map_out
