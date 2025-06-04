import numpy as np
import torch
import torch.nn.functional as F

# === CPM パッチ抽出 / 再構成 (PyTorchのunfold/foldを使用) ===

# 入力: (N, H, W, C), 出力: (N, パッチ数, patch_h * patch_w, C)
def extract_patches_manual_padding_with_offset_batch(
    images, patch_h, patch_w, slide_h, slide_w
):
    """
    指定されたオフセットに基づいて手動でパディングした後、F.unfoldを用いてパッチを抽出する（バッチ対応版）。
    これはTensorFlow版の挙動（特定のオフセットから始まる非オーバーラップパッチ）を再現する。
    入力: (N, H, W, C), 出力: (N, パッチ数, patch_h * patch_w, C)
    """
    assert images.ndim == 4, "入力画像は4次元 (N, H, W, C) である必要があります。"
    batch_size, img_h, img_w, channels = images.shape

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

    # PyTorchのF.padは (左, 右, 上, 下) の順で指定。入力は(N, C, H, W)である必要があるため転置。
    images_nchw = images.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    # 定数値0でパディング実行
    padded_images_nchw = F.pad(
        images_nchw, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0
    )

    # パディング後の形状を取得
    _n, _c, padded_h, padded_w = padded_images_nchw.shape

    # --- 2. F.unfold を用いてパッチを抽出 ---
    # F.unfold は入力 (N, C, H, W) を期待。
    # カーネルサイズ = (patch_h, patch_w), ストライド = (patch_h, patch_w) で非オーバーラップ抽出。
    patches_unfolded = F.unfold(
        padded_images_nchw,
        kernel_size=(patch_h, patch_w),
        stride=(patch_h, patch_w),
    )
    # 出力形状: (N, C * patch_h * patch_w, パッチ数L)
    # L = num_patches_h * num_patches_w

    # --- 3. TensorFlow版の出力フォーマットに整形 ---
    # (N, C * ph * pw, L) -> (N, C, ph * pw, L) に変形
    patches_reshaped = patches_unfolded.view(batch_size, channels, patch_h * patch_w, -1)

    # -> (N, C, ph * pw, L) -> (N, L, ph * pw, C) に転置 (permute)
    final_patches = patches_reshaped.permute(0, 3, 2, 1)
    # 最終形状: (バッチサイズN, 総パッチ数L, patch_h * patch_w, チャンネル数C)

    return final_patches


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
    assert channels == target_c, "チャンネル数が一致しません。for target_shape={target_shape}, channels={channels}"
    assert flat_patch_size == patch_h * patch_w, "パッチサイズが一致しません。flat_patch_size={flat_patch_size}, patch_h={patch_h}, patch_w={patch_w}"

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

def reconstruct_image_from_patches_batch(
    patches, target_shape, patch_h, patch_w, slide_h, slide_w
):
    """
    バッチ対応版。
    入力: (B, パッチ数, patch_h * patch_w, C)
    出力: (B, target_h, target_w, C)
    """
    B, num_total_patches, flat_patch_size, channels = patches.shape
    target_h, target_w, target_c = target_shape
    assert channels == target_c, "チャンネル数が一致しません。"
    assert flat_patch_size == patch_h * patch_w, "パッチサイズが一致しません。"

    # --- パディング後の次元を計算 ---
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

    # --- F.fold 用に整形 ---
    # permute: (B, L, ph*pw, C) → (B, C, ph*pw, L)
    patches_chw = patches.permute(0, 3, 2, 1)
    # reshape: (B, C, ph*pw, L) → (B, C*ph*pw, L)
    patches_for_fold = patches_chw.reshape(
        B, channels * patch_h * patch_w, num_total_patches
    )

    # --- F.fold で再構成 ---
    reconstructed_padded_chw = F.fold(
        patches_for_fold,
        output_size=(padded_h, padded_w),
        kernel_size=(patch_h, patch_w),
        stride=(patch_h, patch_w),
    )  # 出力: (B, C, padded_h, padded_w)

    # permute: (B, C, H, W) → (B, H, W, C)
    reconstructed_padded_hwc = reconstructed_padded_chw.permute(0, 2, 3, 1)

    # --- パディングを除去 ---
    pad_top = slide_h
    pad_left = slide_w
    reconstructed_image = reconstructed_padded_hwc[
        :, pad_top : pad_top + target_h, pad_left : pad_left + target_w, :
    ]  # (B, target_h, target_w, C)

    # チェック
    assert (
        reconstructed_image.shape == (B, target_h, target_w, target_c)
    ), f"再構成後の形状 {reconstructed_image.shape} != {(B, target_h, target_w, target_c)}"

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

