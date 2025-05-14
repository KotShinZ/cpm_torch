import numpy as np
import torch
import torch.nn.functional as F

# 細胞IDのカウンター（グローバル変数、シンプルなPython intとして管理）
cell_newer_id_counter = 1

def map_init(height=256, width=256, device="cuda"):
    """シミュレーション用のマップ（格子）を初期化する。"""
    global cell_newer_id_counter
    # マップテンソルを作成: (高さ, 幅, チャンネル数)
    # チャンネル 0: 細胞ID
    # チャンネル 1: 細胞密度 / スカラー値（例：面積）
    # チャンネル 2: 前ステップの細胞ID（拡散の境界条件チェック用）
    map_tensor = torch.zeros((height, width, 3), dtype=torch.float32, device=device)

    # マップ中央に初期細胞を配置
    center_x_slice = slice(height // 2 - 1, height // 2 + 1)  # 例: 中央2x2領域
    center_y_slice = slice(width // 2 - 1, width // 2 + 1)

    # add_cell関数で細胞を追加（map_tensorが直接変更され、次のIDが返る）
    map_tensor, _ = add_cell(map_tensor, center_x_slice, center_y_slice, value=100)

    # IDカウンターをリセット（初期細胞追加後に次のIDを2にする）
    cell_newer_id_counter = 2
    return map_tensor


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


# === CPM パッチ抽出 / 再構成 (PyTorchのunfold/foldを使用) ===

# 入力: (H, W, C), 出力: (パッチ数, patch_h * patch_w, C)
# 3*3のパッチに分割
def extract_patches_manual_padding_with_offset(
    image, patch_h, patch_w, slide_h, slide_w
):
    """
    指定されたオフセットに基づいて手動でパディングした後、F.unfoldを用いてパッチを抽出する。
    これはTensorFlow版の挙動（特定のオフセットから始まる非オーバーラップパッチ）を再現する。
    入力: (H, W, C), 出力: (パッチ数, patch_h * patch_w, C)
    """
    assert image.ndim == 3, "入力画像は3次元 (H, W, C) である必要があります。"
    img_h, img_w, channels = image.shape

    # オフセットが非負整数であることを確認
    slide_h, slide_w = int(slide_h), int(slide_w)
    assert slide_h >= 0 and slide_w >= 0, "オフセットは非負である必要があります。"

    # --- 1. 必要なパディング量を計算 ---
    # オフセット分のパディングを含めた実効的な高さ/幅
    effective_h = img_h + slide_h
    effective_w = img_w + slide_w

    # パディング後の目標高さ/幅 (パッチサイズで割り切れるように切り上げ)
    target_h = ((effective_h + patch_h - 1) // patch_h) * patch_h
    target_w = ((effective_w + patch_w - 1) // patch_w) * patch_w

    # F.padに必要なパディング量を計算 (左、右、上、下)
    pad_top = slide_h
    pad_left = slide_w
    pad_bottom = target_h - effective_h
    pad_right = target_w - effective_w

    # PyTorchのF.padは (左, 右, 上, 下) の順で指定。入力は(C, H, W)である必要があるため転置。
    image_chw = image.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    # 定数値0でパディング実行
    padded_image_chw = F.pad(
        image_chw, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0
    )

    # パディング後の形状を取得
    padded_c, padded_h, padded_w = padded_image_chw.shape

    # --- 2. F.unfold を用いてパッチを抽出 ---
    # F.unfold は入力 (N, C, H, W) または (C, H, W) を期待。
    # カーネルサイズ = (patch_h, patch_w), ストライド = (patch_h, patch_w) で非オーバーラップ抽出。
    patches_unfolded = F.unfold(
        padded_image_chw.unsqueeze(0),  # バッチ次元を追加 (1, C, H, W)
        kernel_size=(patch_h, patch_w),
        stride=(patch_h, patch_w),
    )
    # 出力形状: (N, C * patch_h * patch_w, パッチ数L) = (1, C * ph * pw, L)
    # L = num_patches_h * num_patches_w

    # --- 3. TensorFlow版の出力フォーマットに整形 ---
    # (1, C * ph * pw, L) -> (1, C, ph * pw, L) に変形
    patches_reshaped = patches_unfolded.view(1, channels, patch_h * patch_w, -1)

    # -> (C, ph * pw, L) -> (L, ph * pw, C) に転置 (permute)
    final_patches = patches_reshaped.squeeze(0).permute(2, 1, 0)
    # 最終形状: (総パッチ数, patch_h * patch_w, チャンネル数)

    return final_patches

# 入力: (パッチ数, patch_h * patch_w, C), 出力: (target_h, target_w, C)
# 3*3のパッチに分割された画像を再構成
def reconstruct_image_from_patches(
    patches, target_shape, patch_h, patch_w, slide_h, slide_w
):
    """
    extract_patches_manual_padding_with_offset で作成されたパッチから元の画像を再構成する。
    F.fold (unfoldの逆操作) を使用する。
    入力: (パッチ数, patch_h * patch_w, C), 出力: (target_h, target_w, C)
    """
    num_total_patches, flat_patch_size, channels = patches.shape
    target_h, target_w, target_c = target_shape
    assert channels == target_c, "チャンネル数が一致しません。"
    assert flat_patch_size == patch_h * patch_w, "パッチサイズが一致しません。"

    # --- 1. パディング後の次元を計算 (抽出時と同じロジック) ---
    slide_h, slide_w = int(slide_h), int(slide_w)
    effective_h = target_h + slide_h
    effective_w = target_w + slide_w
    padded_h = ((effective_h + patch_h - 1) // patch_h) * patch_h
    padded_w = ((effective_w + patch_w - 1) // patch_w) * patch_w
    num_patches_h = padded_h // patch_h
    num_patches_w = padded_w // patch_w
    assert (
        num_total_patches == num_patches_h * num_patches_w
    ), "パッチ数が一致しません。"

    # --- 2. F.fold のための準備 ---
    # F.fold は入力 (N, C * patch_h * patch_w, L) を期待。
    # 入力パッチの形状を変換: (L, ph*pw, C) -> (C, ph*pw, L) -> (1, C*ph*pw, L)
    patches_chw = patches.permute(2, 1, 0)  # (C, ph*pw, L)
    patches_for_fold = patches_chw.reshape(
        1, channels * patch_h * patch_w, num_total_patches
    )  # (1, C*ph*pw, L)

    # --- 3. F.fold を使用してパディングされた画像を再構成 ---
    reconstructed_padded_chw = F.fold(
        patches_for_fold,
        output_size=(padded_h, padded_w),  # 出力サイズ(パディング込み)
        kernel_size=(patch_h, patch_w),
        stride=(patch_h, patch_w),
    )
    # 出力形状: (N, C, padded_h, padded_w) = (1, C, padded_h, padded_w)

    # バッチ次元を削除し、(H, W, C) フォーマットに戻す
    reconstructed_padded_hwc = reconstructed_padded_chw.squeeze(0).permute(
        1, 2, 0
    )  # (padded_h, padded_w, C)

    # --- 4. パディングを除去して元の画像サイズに戻す ---
    pad_top = slide_h
    pad_left = slide_w
    # スライシングで必要な領域を切り出す
    reconstructed_image = reconstructed_padded_hwc[
        pad_top : pad_top + target_h, pad_left : pad_left + target_w, :
    ]

    # 最終的な形状が目標形状と一致するか確認
    assert (
        reconstructed_image.shape == target_shape
    ), f"再構成後の形状 {reconstructed_image.shape} != 目標形状 {target_shape}"

    return reconstructed_image

# 入力: (H, W, C), 出力: (H, W, patch_size*patch_size, C)
# 畳み込み成分によるパッチ抽出
def extract_patches_batched_channel(
    input_tensor: torch.Tensor, patch_size: int = 3
) -> torch.Tensor:
    """
    F.unfoldを用いて、畳み込みのように各ピクセルを中心とするパッチを抽出する。
    チャンネルは独立に扱われる（概念的に）。'SAME'パディング相当。
    入力: (H, W, C), 出力: (H, W, patch_size*patch_size, C)
    TensorFlow版の出力フォーマットに合わせる。
    """
    assert (
        input_tensor.ndim == 3
    ), "入力テンソルは3次元 (H, W, C) である必要があります。"
    H, W, C = input_tensor.shape
    num_patch_elements = patch_size * patch_size
    padding = patch_size // 2  # 'SAME'パディング相当（カーネルサイズ3ならパディング1）

    # F.unfoldのために (C, H, W) に転置
    input_chw = input_tensor.permute(2, 0, 1)

    # バッチ次元を追加: (1, C, H, W)
    input_nchw = input_chw.unsqueeze(0)

    # F.unfold でストライド1でパッチ抽出
    patches_unfolded = F.unfold(
        input_nchw, kernel_size=patch_size, padding=padding, stride=1
    )
    # 出力形状: (N, C * k * k, H * W) = (1, C * 9, H * W)

    # 目標の出力形状 (H, W, 9, C) に合わせて変形と転置
    # (1, C * 9, H * W) -> (1, C, 9, H * W)
    patches_reshaped = patches_unfolded.view(1, C, num_patch_elements, H * W)

    # -> (1, C, 9, H, W)
    patches_reshaped_hw = patches_reshaped.view(1, C, num_patch_elements, H, W)

    # バッチ次元を削除 -> (C, 9, H, W)
    patches_c9hw = patches_reshaped_hw.squeeze(0)

    # -> (H, W, 9, C) に転置
    output_tensor = patches_c9hw.permute(2, 3, 1, 0)

    return output_tensor

