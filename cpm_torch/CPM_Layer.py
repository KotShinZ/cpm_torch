import torch
import torch.nn as nn
import numpy as np
from cpm_torch.CPM import CPM, CPM_config
from cpm_torch.CPM_Map import extract_patches_manual_padding_with_offset, reconstruct_image_from_patches

# ==============================================================================
# ここに、前回同様、CPM_config, CPM クラス（修正版）、
# および cpm_torch.CPM_Map のヘルパー関数を配置します。
# CPMクラスに必要な修正は、前回の回答と同じです。
# ==============================================================================

class CPMLayer(nn.Module):
    """
    内部でdH_NNを生成し、CPMシミュレーションを実行する学習可能なニューラルネットワーク層。

    Args:
        cpm_config (CPM_config): CPMのパラメータを設定するコンフィグオブジェクト。
        in_channels (int): 入力テンソルXのチャンネル数。
        id_channel_idx (int, optional): 入力Xのうち、ID画像として使用するチャンネルのインデックス。
                                        デフォルトは0。
        mcs_steps (int, optional): 1回のforwardパスで実行するモンテカルロステップ数。デフォルトは1。
        device (str, optional): 計算に使用するデバイス。デフォルトは "cuda"。
    """
    def __init__(self, cpm_config: CPM_config, in_channels: int, id_channel_idx: int = 0, mcs_steps: int = 1, device="cuda"):
        super().__init__()
        self.mcs_steps = mcs_steps
        self.device = device
        self.id_channel_idx = id_channel_idx
        
        # CPMのロジックを保持するインスタンスを作成
        self.cpm_instance = CPM(config=cpm_config, device=device)

        # dH_NNを生成するための畳み込み層 (学習可能)
        self.conv_d_h = nn.Conv2d(
            in_channels=in_channels,
            out_channels=4,  # 4近傍のエネルギー変化に対応
            kernel_size=1
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        多チャンネル入力Xを受け取り、IDチャンネルをCPMで更新して返す。

        Args:
            X (torch.Tensor): 多チャンネルの入力テンソル。
                              PyTorchの慣例に従い、形状は (B, C, H, W) または (C, H, W) を想定。
                              B: バッチサイズ, C: チャンネル数, H: 高さ, W: 幅

        Returns:
            torch.Tensor: IDチャンネルが更新された出力テンソル。入力と同じ形状。
        """
        # --- 0. 入力形状とデバイスの正規化 ---
        X = X.to(self.device)
        is_batched = X.dim() == 4
        if not is_batched:
            # (C, H, W) -> (1, C, H, W)
            X = X.unsqueeze(0)
        
        # --- 1. dH_NN の生成 ---
        # 内部のConv2d層でXからdH_NNを計算
        # 入力: (B, C, H, W) -> 出力: (B, 4, H, W)
        dH_NN = self.conv_d_h(X)

        # --- 2. id_image の抽出 ---
        # 指定されたインデックスのチャンネルをID画像として使用
        id_image = X[:, self.id_channel_idx, :, :].clone() # 形状: (B, H, W)

        # --- 3. CPM更新ロジックの実行 ---
        # dH_NNの形状を (B, H, W, 4) に変換
        dH_NN = dH_NN.permute(0, 2, 3, 1)

        batch_size, height, width = id_image.shape

        output_ids = []
        for i in range(batch_size):
            current_id_image = id_image[i]
            current_dH_NN = dH_NN[i]

            # CPMインスタンスの状態を現在の画像で設定
            map_tensor = torch.zeros(
                (height, width, self.cpm_instance.map_tensor.shape[2]),
                dtype=torch.float32,
                device=self.device,
            )
            map_tensor[:, :, 0] = current_id_image
            self.cpm_instance.map_tensor = map_tensor
            
            if current_id_image.max() > 0:
                self.cpm_instance.cell_count = int(current_id_image.max())
            else:
                self.cpm_instance.cell_count = 0

            # MCSステップを実行 (dH_NNを渡す)
            for _ in range(self.mcs_steps):
                self.cpm_instance.cpm_mcs_step(dH_NN=current_dH_NN)

            # 結果をリストに保存
            output_ids.append(self.cpm_instance.map_tensor[:, :, 0])
        
        # --- 4. 出力テンソルの作成 ---
        updated_ids_batch = torch.stack(output_ids, dim=0)

        # 元の入力XのIDチャンネルを、更新されたID画像で置き換える
        output_X = X.clone()
        output_X[:, self.id_channel_idx, :, :] = updated_ids_batch

        if not is_batched:
            output_X = output_X.squeeze(0)

        return output_X