import torch
from cpm_torch.CPM_Map import *


diffusion_kernel = (
    torch.tensor([1, 2, 1, 2, -12, 2, 1, 2, 1], dtype=torch.float32) / 16.0
)


@torch.no_grad()  # 拡散ステップでは勾配計算を無効化
def diffusion_step(map_tensor: torch.Tensor, params=[2], dts=[0.1], percent=[1]):
    """

    指定したチャンネルに対して、細胞境界を尊重した拡散を1ステップ実行する。

    Args:
        map_tensor (torch.Tensor): 入力マップテンソル (H, W, C) 形式。
        params (list): 更新するチャンネルのインデックスリスト。デフォルトは [2] (密度)。
        dts (list): 各チャンネルの時間ステップ。デフォルトは [0.1]。
        percent (list): 境界を尊重する割合。デフォルトは [1]。
    Returns:
        torch.Tensor: 更新されたマップテンソル (H, W, C) 形式。
    """

    # map_tensor 形状: (H, W, C) C=3 [ID, Density, PrevID]
    H, W, C = map_tensor.shape

    # 入力 (H, W, C) -> 出力 (H, W, 9, C)
    patches = extract_patches_batched_channel(map_tensor, 3)

    device = map_tensor.device

    id_patches = patches[:, :, :, 0]  # IDパッチ (H, W, 9)
    center_ids = id_patches[:, :, 4:5]  # 中心ピクセルのID (H, W, 1)

    density_patches = patches[:, :, :, params]  # 密度パッチ (H, W, 9, C)
    center_density = density_patches[:, :, 4:5]  # 中心ピクセルの密度 (H, W, 1, C)

    # 隣接ピクセルとの密度差を計算
    density_diff = density_patches - center_density  # (H, W, 9, C)

    if dts is not torch.Tensor:
        dts = torch.tensor(dts, dtype=torch.float32, device=device)
        dts = dts.view(1, 1, len(dts))  # (1, 1, 1, C)
    if percent is not torch.Tensor:
        percent = torch.tensor(percent, dtype=torch.float32, device=device)
        percent = percent.view(1, 1, 1, len(percent))  # (1, 1, 1, C)

    # 境界マスクを作成: 隣接ピクセルが同じIDなら0、異なるなら1
    same_id_mask = (id_patches == center_ids).float()  # (H, W, 9)

    #print(percent.shape, same_id_mask.shape)
    weights = 1 - (1 - percent) * same_id_mask.unsqueeze(-1)  # (H, W, 9, C)

    kernel = diffusion_kernel.view(1, 1, 9, 1).to(device)  # (1, 1, 9, 1)
    kernel = kernel.repeat(1, 1, 1, len(params)) # (1, 1, 9, C)
    
    #print(f"Kernel shape: {kernel.shape}, Weights shape: {weights.shape}, Density diff shape: {density_diff.shape}")

    update = (
        torch.sum(
            kernel  # カーネル
            * weights  # 細胞境界での拡散抑制
            * density_diff,  # 拡散量
            dim=2,
        )
        * dts  # 時間ステップ
    )  # (H, W, 9, C) -> (H, W, C)

    # --- 更新された密度をマップテンソルに反映 ---
    map_out = map_tensor.clone()  # 元のマップをコピー
    map_out[:, :, params] += update  # チャンネル1（密度）を更新
    # ID (チャンネル0) と Previous ID (チャンネル2) はこのステップでは変更しない

    return map_out
