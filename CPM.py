import numpy as np 
import torch
import torch.nn.functional as F 

# === デバイス設定 ===
# CUDA (GPU) が利用可能ならGPUを、そうでなければCPUを使用
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPUを利用します: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CPUを利用します")

# === 初期化関連 ===

# 細胞IDのカウンター（グローバル変数、シンプルなPython intとして管理）
cell_newer_id_counter = 1

def map_init(height=256, width=256):
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


# === パッチ関数テスト（コメントアウト） ===
# dummy_image_2d = torch.arange(1, 256*256*2+1, dtype=torch.float32).reshape(256, 256, 2).to(device)
# patches_extracted = extract_patches_manual_padding_with_offset(dummy_image_2d, 3, 3, 2, 2)
# print("手動オフセットパッチ形状:", patches_extracted.shape) # -> (パッチ数, 9, 2)
# map_reconstructed = reconstruct_image_from_patches(patches_extracted, dummy_image_2d.shape, 3, 3, 2, 2)
# print("再構成後の形状:", map_reconstructed.shape)
# print("再構成誤差:", torch.abs(dummy_image_2d - map_reconstructed).max()) # -> ほぼ0のはず

# patches_conv = extract_patches_batched_channel(dummy_image_2d, 3)
# print("畳み込み風パッチ形状:", patches_conv.shape) # -> (256, 256, 9, 2)


# === CPM 計算関数 ===

# 3x3パッチのフラット化されたインデックス
corner_indices = [0, 2, 6, 8]  # 角のインデックス
neighbor_indices = [1, 3, 5, 7]  # 上下左右の隣接インデックス
center_index = 4  # 中央のインデックス

# マスクを事前に計算し、適切なデバイスに配置
# TensorFlow版で使われていたコーナーマスク（用途が不明確だったが再現のため保持）
corner_mask_flat = torch.zeros(9, device=device, dtype=torch.float32)
corner_mask_flat[corner_indices] = 1.0
# 直接の隣接ピクセル（上下左右）のマスク（エネルギー計算で使用）
neighbor_mask_flat = torch.zeros(9, device=device, dtype=torch.float32)
neighbor_mask_flat[neighbor_indices] = 1.0


def calc_area_bincount(map_tensor):
    """torch.bincount を使って各細胞IDの面積（ピクセル数）を計算する。"""
    ids = map_tensor[:, :, 0].long()  # IDチャンネルをlong型で取得 (H, W)
    H, W = ids.shape
    flat_ids = ids.flatten()  # bincountのために1次元配列にフラット化

    # 各IDの出現回数（ピクセル数）をカウント
    # minlengthは、存在する最大のID+1、または既知の最大ID（cell_newer_id_counter）の大きい方を指定し、
    # 存在しないIDに対してもカウント用のスロットを確保する。
    max_id_val = int(flat_ids.max().item()) if flat_ids.numel() > 0 else 0
    min_len = max(max_id_val + 1, cell_newer_id_counter)

    # bincountはCPUで高速な場合が多いので、一時的にCPUに転送して実行し、結果を元のデバイスに戻す
    area_counts = (
        torch.bincount(flat_ids.cpu(), minlength=min_len).to(device).float()
    )  # 各IDの面積カウント (ID数,)

    # 各ピクセルに、そのピクセルが属する細胞の総面積を割り当てる
    # gather操作（インデックス参照）のためにIDを安全な範囲にクランプ
    safe_ids = torch.clamp(ids, 0, min_len - 1)
    areas_per_pixel = area_counts[
        safe_ids
    ]  # 各ピクセル位置に対応する細胞の総面積 (H, W)

    return areas_per_pixel


def calc_perimeter_patch(map_tensor):
    """各ピクセルにおける周囲長の寄与（隣接ピクセルとのID境界数）を計算する。"""
    ids = map_tensor[:, :, 0]  # IDチャンネル (H, W)

    # 各ピクセル周りの3x3パッチを抽出 (畳み込み風)
    # 入力 (H, W, 1) -> 出力 (H, W, 9, 1)
    id_patches = extract_patches_batched_channel(
        ids.unsqueeze(-1), 3
    )  # チャンネル次元を追加して抽出

    center_ids = id_patches[:, :, center_index, 0]  # 各パッチの中心ピクセルのID (H, W)
    neighbor_ids = id_patches[
        :, :, neighbor_indices, 0
    ]  # 各パッチの上下左右の隣接ピクセルのID (H, W, 4)

    # 中心ピクセルのIDと各隣接ピクセルのIDを比較
    # 形状 (H, W, 4) - IDが異なる（境界である）場合にTrue
    is_boundary = neighbor_ids != center_ids.unsqueeze(-1)

    # 各ピクセルについて、IDが異なる隣接ピクセルの数を合計する（境界の数）
    # これがそのピクセルにおける周囲長の寄与となる
    perimeter_at_pixel = torch.sum(is_boundary.float(), dim=2)  # (H, W)

    return perimeter_at_pixel


def calc_total_perimeter_bincount(map_tensor, perimeter_at_pixel):
    """各細胞IDの総周囲長を計算する。"""
    ids = map_tensor[:, :, 0].long()  # IDチャンネル (H, W)
    flat_ids = ids.flatten()  # フラット化 (H*W)
    flat_perimeter_contrib = (
        perimeter_at_pixel.flatten()
    )  # 各ピクセルの周囲長寄与をフラット化 (H*W)

    # bincount を使って、各IDごとに周囲長寄与の合計を計算する
    max_id_val = int(flat_ids.max().item()) if flat_ids.numel() > 0 else 0
    min_len = max(max_id_val + 1, cell_newer_id_counter)

    # bincountにweights引数を指定すると、各bin（ID）に対して対応するweight（周囲長寄与）の合計を計算する
    total_perimeter_counts = (
        torch.bincount(
            flat_ids.cpu(), weights=flat_perimeter_contrib.cpu(), minlength=min_len
        )
        .to(device)
        .float()
    )  # 各IDの総周囲長 (ID数,)

    # 各ピクセルに、そのピクセルが属する細胞の総周囲長を割り当てる
    safe_ids = torch.clamp(ids, 0, min_len - 1)
    total_perimeter_per_pixel = total_perimeter_counts[safe_ids]  # (H, W)

    return total_perimeter_per_pixel


import torch


def has_nan_or_inf(tensor: torch.Tensor) -> bool:
    """
    テンソル内にNaNまたは無限大が含まれているかどうかを判定します。

    Args:
      tensor: チェック対象のPyTorchテンソル。

    Returns:
      NaNまたは無限大が含まれていればTrue、そうでなければFalse。
    """
    # isfiniteは有限数ならTrue、NaN/InfならFalseを返す
    # そのため、isfiniteでないものが一つでもあればTrueを返したい
    # return not torch.isfinite(tensor).all() # こちらでも同じ
    print(
        "nanを持つかどうか", (~torch.isfinite(tensor)).any()
    )  # ~ はビット反転 (True/False反転)


def calc_cpm_probabilities(map_tensor, ids_patch, l_A, A_0, l_L, L_0, T):
    """
    CPMの状態遷移確率（ロジット）を計算する。
    入力 patches_hwc: extract_patches_manual_padding_with_offset の出力 (パッチ数, 5)
    出力 logits: 各パッチ中心が隣接状態(0-8)に遷移する対数確率 (パッチ数, 9)
    """
    # --- エネルギー変化計算に必要なグローバルな性質を計算 ---
    # マップ全体に対して計算し、各ピクセルにそのピクセルが属する細胞の性質を割り当てる
    current_areas_map = calc_area_bincount(
        map_tensor
    )  # 各ピクセル位置の細胞の総面積 (H, W)
    perimeter_contrib_map = calc_perimeter_patch(
        map_tensor
    )  # 各ピクセル位置の周囲長寄与 (H, W)
    current_perimeters_map = calc_total_perimeter_bincount(
        map_tensor, perimeter_contrib_map
    )  # 各ピクセル位置の細胞の総周囲長 (H, W)

    # --- エネルギー変化 (ΔH) の計算 ---
    # 標準的なCPMのハミルトニアン（エネルギー関数）を考える:
    # H = Sum_{<i,j>隣接} J(σ_i, σ_j) * (1 - δ(σ_i, σ_j))  (接着エネルギー)
    #     + Sum_σ [ λ_A * (A_σ - A_0)^2 + λ_L * (L_σ - L_0)^2 ] (面積・周囲長エネルギー)
    # ここで、J(σ_i, σ_j)は細胞タイプσ_iとσ_j間の境界エネルギー係数。
    # 簡単のため、J(a, b) = 1 (if a != b and a,b != 0), J(a, a) = 0, J(a, 0) = 0 とする。
    # δ(a, b) はクロネッカーのデルタ（a=bなら1、a!=bなら0）。

    # あるピクセルxの状態がターゲット細胞tからソース細胞sに変化する場合のエネルギー変化ΔHを計算する。
    # ΔH = H_new - H_old

    # --- 各パッチについてΔHを計算 ---
    # Bincountから得られた細胞ごとの面積/周囲長カウントを取得
    max_id_val = int(map_tensor[:, :, 0].max().item()) if map_tensor.numel() > 0 else 0
    min_len = max(max_id_val + 1, cell_newer_id_counter)
    flat_ids_map = map_tensor[:, :, 0].long().flatten().cpu()
    area_counts = (
        torch.bincount(flat_ids_map, minlength=min_len).to(device).float()
    )  # (ID数,)
    perimeter_counts = (
        torch.bincount(
            flat_ids_map,
            weights=perimeter_contrib_map.flatten().cpu(),
            minlength=min_len,
        )
        .to(device)
        .float()
    )  # (ID数,)

    # パッチ内の各ピクセルのIDに対応する細胞の現在の面積/周囲長を取得
    safe_ids_patch = torch.clamp(ids_patch.long(), 0, min_len - 1)
    current_areas_patch = area_counts[safe_ids_patch]  # 細胞の総面積 (N, 5)
    current_perimeters_patch = perimeter_counts[safe_ids_patch]  # 細胞の総周囲長 (N, 5)

    # パッチ中心（ターゲットセル）と、パッチ内の全セル（ソース候補）を特定
    target_id = ids_patch[:, 4:]  # ターゲットセルのID (N,)
    target_area = current_areas_patch[:, 4:]  #  (N,)
    target_perimeter = current_perimeters_patch[:, 4:]  #  (N, )
    target_is_not_empty = target_id != 0  # ターゲットセルが空（ID=0）かどうか (N,)

    source_ids = ids_patch[:, :4]  # ソース候補のID (N, 4)
    source_areas = current_areas_patch[:, :4]  # (N, 4)
    source_perimeters = current_perimeters_patch[:, :4]  # (N, 4)
    source_is_not_empty = source_ids != 0  # ソース候補が空（ID=0）かどうか (N, 4)

    # --- ΔHの各項を計算 ---

    # 1. 面積エネルギー変化 ΔH_A
    # H_A = λ_A * (A - A_0)^2
    # A_s -> A_s + 1, A_t -> A_t - 1 となるときの変化
    # ΔH_A = λ_A * [ (A_s+1 - A_0)^2 - (A_s - A_0)^2 ] + λ_A * [ (A_t-1 - A_0)^2 - (A_t - A_0)^2 ]
    # ΔH_A = λ_A * [ 2*A_s + 1 - 2*A_0 ] + λ_A * [ -2*A_t + 1 + 2*A_0 ]
    # ΔH_A = λ_A * [ 2*(A_s - A_t) + 2 ]
    delta_H_area = (
        l_A * (2.0 * source_areas + 1 - 2 * A_0) * source_is_not_empty
        + (-2.0 * target_area + 1 + 2 * A_0) * target_is_not_empty
    )  # (N, 4)

    # 2. 周囲長エネルギー変化 ΔH_L (ここでは簡単化のため無視するか、近似を使う)
    # L_s -> L_s + dL_s, L_t -> L_t + dL_t となる。dLは局所的な境界変化に依存し複雑。
    # TF版では使われていなかったので、ここでは0とするか、面積項と同様の近似を使う。
    # delta_H_perimeter = l_L * (2.0 * (source_perimeters - target_perimeter.unsqueeze(1)) + dL_local_change)
    delta_H_perimeter = torch.zeros_like(delta_H_area)  # 簡単化のため0とする

    # 3. 接着エネルギー変化 ΔH_adhesion
    # ΔH_adhesion = Sum_{y neighbor of x} [ J(s, σ_y) - J(t, σ_y) ]
    # 中心ピクセルxの隣接ピクセルyについて、境界エネルギーの変化を合計する。
    # J(a, b) = 1 (if a != b and a,b > 0), 0 otherwise.
    # delta_H_adhesion_patch = torch.zeros_like(ids_patch, dtype=torch.float32) # (N, 9)

    # --- 総エネルギー変化 ΔH ---
    delta_H = delta_H_area + delta_H_perimeter  # (N, 4)

    # --- ボルツマン確率のロジット（対数確率）を計算 ---
    # Logit = -ΔH / T
    # 無限大のΔHは -無限大のロジットになる
    logits = torch.exp(-delta_H / T)  # (N, 4)

    # 遷移確率が0になるように）
    logits = torch.where(
        source_ids != target_id, logits, torch.tensor(0.0, device=device)
    )

    return logits  # 各パッチ中心に対する遷移ロジット(N, 4)を返す


neighbors = [1, 3, 5, 7, 4]


def cpm_checkerboard_step(map_input, l_A, A_0, l_L, L_0, T, x_offset, y_offset):
    """CPMの1ステップをチェッカーボードパターンの一部に対して実行する。"""
    H, W, C = map_input.shape

    # 1. 現在のチェッカーボードオフセットに対応するパッチを抽出
    # 出力: (パッチ数, 9, C)
    map_patched = extract_patches_manual_padding_with_offset(
        map_input, 3, 3, x_offset, y_offset
    )
    num_patches = map_patched.shape[0]
    # print(map_patched.shape)
    ids_patch = map_patched[:, :, 0]

    if torch.isnan(map_input).any() or torch.isinf(map_input).any():
        print(
            f"警告: map_input に NaN/Inf があります! (offset: {x_offset}, {y_offset})"
        )
        print(
            f"NaNs in map_input ch0: {torch.isnan(map_input[:,:,0]).sum()}, ch1: {torch.isnan(map_input[:,:,1]).sum()}, ch2: {torch.isnan(map_input[:,:,2]).sum()}"
        )
        # 必要に応じて処理を中断したり、値を修正したりする

    # ids_patch = map_patched[:, :, 0] の後に追加
    if torch.isnan(ids_patch).any() or torch.isinf(ids_patch).any():
        print(
            f"警告: ids_patch (インデックス操作前) に NaN/Inf があります! (offset: {x_offset}, {y_offset})"
        )
    ids_patch = ids_patch[:, neighbors]

    # パッチの形状: (パッチ数, 5, C) = (N, 5, 3)

    # 2. 各パッチ中心に対する状態遷移のロジットを計算
    # 入力: マップ全体と抽出されたパッチ
    # 出力: (N, 4) - 隣接状態(0-8)をサンプリングするためのロジット
    logits = calc_cpm_probabilities(map_input, ids_patch, l_A, A_0, l_L, L_0, T)
    logits = torch.clip(logits, 0, 1)
    # print(logits)
    # 3. 各パッチ中心について、次に採用する状態（隣接ピクセルのインデックス）をサンプリング
    # torch.multinomialは確率または対数確率を入力とする。
    # 全てのロジットが-infの場合、multinomialはエラーを起こすため、これをハンドルする。

    rand = torch.rand_like(logits)  # 確率を生成 (N, 4)
    prob = logits / 4
    

    # selects = torch.relu(torch.sign(logits - rand))  # 0か1に(N, 4)

    # 各パッチの確率 (N, 4) - 確率の合計は1になる
    #prob = selects / (torch.sum(selects, dim=1, keepdim=True) + 1e-8)  # (N, 4)

    # 遷移しない確率を追加
    prob = torch.concat((prob, 1 - torch.sum(prob, dim=1, keepdim=True)), dim=1)
    #print(prob)
    # サンプリング (N, 1)
    sampled_indices = torch.multinomial(prob, num_samples=1)

    # 4. サンプリングされたインデックスに基づいて、採用するソース細胞のIDを取得
    # ソース候補のIDは map_patched[:, :, 0] (形状: パッチ数, 9)
    # torch.gatherを使って、sampled_indicesに基づいてIDを選択
    # gather(入力テンソル, 次元, インデックステンソル)
    # source_id_all : (N, 5)
    # sampled_indices : (N, 1)

    new_center_ids = torch.gather(ids_patch, dim=1, index=sampled_indices.long())

    # 5. パッチテンソルを更新：中心ピクセルのIDを新しいIDで、前のIDを古いIDで更新
    # map_patched_updated = map_patched.clone()  # 元のパッチテンソルをコピーして変更

    # patch_indices = torch.arange(num_patches, device=device)

    # まず、チャンネル2（前のID）を現在の中心ID（古いID）で更新
    # old_center_ids = map_patched[:, center_index, 0]  # (N,)
    # map_patched_updated[patch_indices, center_index, 2] = old_center_ids

    # 次に、チャンネル0（現在のID）をサンプリングされた新しいIDで更新
    map_patched[:, center_index, 0] = new_center_ids.squeeze(1)

    # 6. 更新されたパッチテンソルからマップ全体を再構成
    map_output = reconstruct_image_from_patches(
        map_patched, map_input.shape, 3, 3, x_offset, y_offset
    )

    return map_output, logits


# === 拡散 関数 ===


def pad_repeat(x, pad=1):
    """PyTorchのtorch.catを用いて周期的境界条件（繰り返しパディング）を実装する。"""
    # 入力形状: (N, C, H, W) を想定
    # 幅方向 (最後の次元 W) のパディング
    x = torch.cat([x[..., -pad:], x, x[..., :pad]], dim=-1)
    # 高さ方向 (最後から2番目の次元 H) のパディング
    x = torch.cat([x[..., -pad:, :], x, x[..., :pad, :]], dim=-2)
    return x


@torch.no_grad()  # 拡散ステップでは勾配計算を無効化
def diffusion_step(map_tensor: torch.Tensor, dt=0.1):
    """
    密度チャンネル（チャンネル1）に対して、細胞境界を尊重した拡散を1ステップ実行する。
    ラプラシアンの近似に畳み込みを使用する。
    """
    # map_tensor 形状: (H, W, C) C=3 [ID, Density, PrevID]
    H, W, C = map_tensor.shape
    ids = map_tensor[:, :, 0]  # ID (H, W)
    density = map_tensor[:, :, 1]  # 密度 (H, W)
    prev_ids = map_tensor[:, :, 2]  # 前ステップのID (H, W) - TF版の拡散では使われていた

    # --- TF版の拡散ロジックに近い実装 (畳み込みではなくパッチベース) ---
    # 3x3の密度パッチとIDパッチを抽出
    # 入力 (H, W, 1) -> 出力 (H, W, 9, 1)
    density_patches = extract_patches_batched_channel(density.unsqueeze(-1), 3).squeeze(
        -1
    )  # (H, W, 9)
    id_patches = extract_patches_batched_channel(ids.unsqueeze(-1), 3).squeeze(
        -1
    )  # (H, W, 9)

    center_density = density_patches[:, :, center_index]  # 中心ピクセルの密度 (H, W)
    center_ids = id_patches[:, :, center_index]  # 中心ピクセルのID (H, W)

    # 隣接ピクセルとの密度差を計算
    density_diff = density_patches - center_density.unsqueeze(-1)  # (H, W, 9)

    # 境界マスクを作成: 隣接ピクセルが同じIDなら1、異なるなら0
    same_id_mask = (id_patches == center_ids.unsqueeze(-1)).float()  # (H, W, 9)

    # 拡散カーネル（重み）を定義 (TF版のカーネルに似せる)
    # 中心は寄与しないので0、合計が16になるように正規化？
    diffusion_kernel_weights = (
        torch.tensor([1, 2, 1, 2, 0, 2, 1, 2, 1], dtype=torch.float32, device=device)
        / 16.0
    )
    diffusion_kernel_weights = diffusion_kernel_weights.view(
        1, 1, 9
    )  # ブロードキャスト用に形状変更 (1, 1, 9)

    # 密度の変化量を計算: Sum( 重み * 境界マスク * 密度差 ) * dt
    # same_id_mask により、異なるIDを持つ隣接セルからの/への流束はゼロになる
    update = (
        torch.sum(diffusion_kernel_weights * same_id_mask * density_diff, dim=2) * dt
    )  # (H, W)

    # 密度を更新
    density_final = density + update

    # --- 更新された密度をマップテンソルに反映 ---
    map_out = map_tensor.clone()  # 元のマップをコピー
    map_out[:, :, 1] = density_final  # チャンネル1（密度）を更新
    # ID (チャンネル0) と Previous ID (チャンネル2) はこのステップでは変更しない

    return map_out
