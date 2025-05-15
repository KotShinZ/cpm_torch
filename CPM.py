import numpy as np
import torch
import torch.nn.functional as F
from CPM_Map import *

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

center_index = 4  # 中央のインデックス
neighbors = [1, 3, 5, 7]


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


# === CPM 計算関数 ===


def calc_area_bincount(map_tensor):
    """torch.bincount を使って各細胞IDの面積（ピクセル数）を計算する。"""
    ids = map_tensor[:, :, 0].long()  # IDチャンネルをlong型で取得 (H, W)
    H, W = ids.shape
    flat_ids = ids.flatten()  # bincountのために1次元配列にフラット化

    # bincountはCPUで高速な場合が多いので、一時的にCPUに転送して実行し、結果を元のデバイスに戻す
    area_counts = (
        torch.bincount(flat_ids.cpu(), minlength=cell_newer_id_counter)
        .to(device)
        .float()
    )  # 各IDの面積カウント (ID数,)

    # 各ピクセルに、そのピクセルが属する細胞の総面積を割り当てる
    # gather操作（インデックス参照）のためにIDを安全な範囲にクランプ
    safe_ids = torch.clamp(ids, 0, cell_newer_id_counter - 1)
    areas_per_pixel = area_counts[
        safe_ids
    ]  # 各ピクセル位置に対応する細胞の総面積 (H, W)

    return areas_per_pixel


def calc_perimeter_patch(map_tensor: torch.Tensor) -> torch.Tensor:
    """各ピクセルにおける周囲長の寄与（隣接4ピクセルとのID境界数）を計算する。"""
    ids = map_tensor[:, :, 0]  # IDチャンネル (H, W)
    # 各ピクセル周りの3x3パッチを抽出 -> (H, W, 9, 1)
    id_patches = extract_patches_batched_channel(ids.unsqueeze(-1), 3)

    center_ids = id_patches[:, :, center_index, 0]  # 各パッチ中心のID (H, W)
    # 各パッチの上下左右の隣接ピクセルのID (H, W, 4)
    neighbor_ids_data = id_patches[:, :, [1, 3, 5, 7], 0]

    # 中心IDと各隣接IDを比較 (H, W, 4) -> 境界ならTrue
    is_boundary = neighbor_ids_data != center_ids.unsqueeze(-1)
    # 各ピクセルでIDが異なる隣接ピクセルの数を合計（周囲長への寄与）
    perimeter_at_pixel = torch.sum(is_boundary.float(), dim=2)  # (H, W)
    return perimeter_at_pixel


def calc_total_perimeter_bincount(
    map_tensor: torch.Tensor, perimeter_at_pixel: torch.Tensor
) -> torch.Tensor:
    """各細胞IDの総周囲長を計算する。"""
    ids = map_tensor[:, :, 0].long()  # IDチャンネル (H, W)
    flat_ids = ids.flatten()
    flat_perimeter_contrib = perimeter_at_pixel.flatten()  # 各ピクセルの周囲長寄与

    total_perimeter_counts = (
        torch.bincount(
            flat_ids.cpu(),
            weights=flat_perimeter_contrib.cpu(),
            minlength=cell_newer_id_counter,
        )
        .to(device)
        .float()
    )  # 各IDの総周囲長 (ID数,)
    return total_perimeter_counts


def calc_dH_area(
    source_areas, target_area, l_A, A_0, source_is_not_empty, target_is_not_empty
):
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
    return delta_H_area


def calc_dH_perimeter(
    source_perimeters,  # (N, 4)
    target_perimeter,  # (N, 1)
    source_ids,  # (N, P) ソース候補のID群。
    target_id,  # (N, 1) ターゲットセルのID
    l_L: float,  # 周囲長エネルギーの係数
    L_0: float,  # 基準周囲長
    source_is_not_empty: torch.Tensor,  # (N, 4) ソース候補が空でないかのマスク (boolean)
    target_is_not_empty: torch.Tensor,  # (N, 1) ターゲットセルが空でないかのマスク (boolean)
    device: torch.device,
) -> torch.Tensor:
    """
    周囲長エネルギー変化 ΔH_L を計算する。
    ピクセルがターゲットセルtからソースセルsに変化する状況を考える。
    エネルギー H_L_i = l_L * (L_i - L_0)^2
    ΔH_L_i = l_L * [ 2 * (L_i - L_0) * dL_i + (dL_i)^2 ]
    ここで dL_i はセルiの局所的な周囲長変化。

    Args:
        source_perimeters: 各ソース候補セルの現在の総周囲長 (N, 4)
        target_perimeter: ターゲットセルの現在の総周囲長 (N, 1)
        source_ids: ソース候補のID群。実際にはターゲットピクセルの4近傍のID (N, 4)。
                    各行 ids_patch[i, :-1] に対応。
        target_id: ターゲットセルのID (N, 1)。各行 ids_patch[i, -1:] に対応。
        l_L: 周囲長エネルギーの係数。
        L_0: 基準周囲長。
        source_is_not_empty: ソース候補が空（ID=0など）でないかどうかのブールマスク (N, 4)。
        target_is_not_empty: ターゲットセルが空でないかどうかのブールマスク (N, 1)。
        device: 計算に使用するデバイス ('cpu' or 'cuda')。

    Returns:
        delta_H_perimeter: 各ソース候補への遷移による周囲長エネルギー変化 (N, 4)。
    """

    # 1. 局所的な周囲長変化 dL_s と dL_t を計算
    # dL_s: ターゲットピクセルがソースセルsになった場合の、ソースセルsの周囲長変化。
    #       これは、各ソース候補s_k (source_ids[:,k]) ごとに計算される。
    #       dL_s = 4 - 2 * (ターゲットピクセルの4近傍のうち、s_k と同じIDを持つものの数)

    # num_s_in_target_neighbors: 各ソース候補 s_k について、ターゲットピクセルの4近傍 (source_ids) に
    #                            s_k と同じIDを持つものがいくつあるかをカウントする。
    # 結果の形状は (N, 4)。各要素 (i, k) は、i番目のパッチにおいて、
    # k番目のソース候補 (source_ids[i, k]) が、そのパッチの近傍 (source_ids[i, :]) にいくつ存在するか。

    num_s_in_target_neighbors = torch.zeros_like(
        source_ids, dtype=torch.float, device=device
    )
    for k_idx in range(
        source_ids.shape[1]
    ):  # 通常は4回ループ (0, 1, 2, 3 for 4 neighbors)
        # current_s_candidate_id: (N, 1) tensor containing the ID of the k-th neighbor for each patch
        current_s_candidate_id = source_ids[:, k_idx : k_idx + 1]
        # Check how many times this k-th neighbor's ID appears in all neighbors of that patch
        # source_ids == current_s_candidate_id broadcasts (N,1) to (N,4) for comparison
        matches = source_ids == current_s_candidate_id  # (N, 4) boolean tensor
        num_s_in_target_neighbors[:, k_idx] = torch.sum(matches, dim=1).float()

    local_delta_Ls = 4.0 - 2.0 * num_s_in_target_neighbors  # (N, 4)

    # dL_t: ターゲットピクセルがターゲットセルtでなくなった場合の、ターゲットセルtの周囲長変化
    #       dL_t = -4 + 2 * (ターゲットピクセルの4近傍のうち、t と同じIDを持つものの数)
    # num_t_in_target_neighbors: ターゲットピクセルの4近傍 (source_ids) に、
    #                            ターゲットセルID (target_id) と同じものがいくつあるか。
    num_t_in_target_neighbors = torch.sum(
        source_ids == target_id, dim=1, keepdim=True
    ).float()  # (N, 1)
    local_delta_Lt = -4.0 + 2.0 * num_t_in_target_neighbors  # (N, 1)

    # 2. エネルギー変化 ΔH_L = ΔH_L_s + ΔH_L_t を計算
    # ΔH_L_i = l_L * [ 2 * (L_i - L_0) * dL_i + (dL_i)^2 ]

    # ソースセルのエネルギー変化
    # source_perimeters: (N, 4), L_0: scalar, local_delta_Ls: (N, 4)
    # source_is_not_empty: (N, 4) boolean
    term1_s = 2.0 * (source_perimeters - L_0) * local_delta_Ls
    term2_s = local_delta_Ls.pow(2)
    # Apply mask: energy change is 0 if source cell is empty
    delta_H_perimeter_s = l_L * (term1_s + term2_s) * source_is_not_empty.float()

    # ターゲットセルのエネルギー変化
    # target_perimeter: (N, 1), L_0: scalar, local_delta_Lt: (N, 1)
    # target_is_not_empty: (N, 1) boolean
    # delta_H_perimeter_t_for_each_source_candidate will be (N,1)
    term1_t = 2.0 * (target_perimeter - L_0) * local_delta_Lt
    term2_t = local_delta_Lt.pow(2)
    # Apply mask: energy change is 0 if target cell is empty
    delta_H_perimeter_t_for_each_source_candidate = (
        l_L * (term1_t + term2_s) * target_is_not_empty.float()
    )  # (N,1)

    # 総エネルギー変化
    # delta_H_perimeter_s is (N, 4)
    # delta_H_perimeter_t_for_each_source_candidate is (N, 1) and will be broadcasted
    # during addition to (N,4)
    delta_H_perimeter = (
        delta_H_perimeter_s + delta_H_perimeter_t_for_each_source_candidate
    )

    return delta_H_perimeter


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


def calc_cpm_probabilities(map_tensor, source_ids, target_id, l_A, A_0, l_L, L_0, T):
    """
    CPMの状態遷移確率（ロジット）を計算する。

    Args:
        map_tensor: マップ全体のテンソル (H, W, 3)
        source_ids: 抽出されたパッチのID (N, P) 複数個のソースを同時に計算可能
        target_id: 抽出されたパッチのターゲットID (N, 1)
        l_A: 面積エネルギーの係数
        A_0: 基準面積
        l_L: 周囲長エネルギーの係数
        L_0: 基準周囲長
        T: 温度（ボルツマン確率計算用）
    """
    # --- エネルギー変化計算に必要なグローバルな性質を計算 ---
    # マップ全体に対して計算し、各ピクセルにそのピクセルが属する細胞の性質を割り当てる
    area_counts = calc_area_bincount(map_tensor)
    perimeter_contrib_map = calc_perimeter_patch(map_tensor)
    perimeter_counts = calc_total_perimeter_bincount(map_tensor, perimeter_contrib_map)

    # --- 各パッチについてΔHを計算 ---
    # Bincountから得られた細胞ごとの面積/周囲長カウントを取得
    flat_ids_map = map_tensor[:, :, 0].long().flatten().cpu()
    area_counts = (
        torch.bincount(flat_ids_map, minlength=cell_newer_id_counter).to(device).float()
    )
    perimeter_counts = (
        torch.bincount(
            flat_ids_map,
            weights=perimeter_contrib_map.flatten().cpu(),
            minlength=cell_newer_id_counter,
        )
        .to(device)
        .float()
    )  # (ID数,)

    # パッチ内の各ピクセルのIDに対応する細胞の現在の面積/周囲長を取得
    source_areas = area_counts[source_ids.long()]  # 細胞の総面積 (N, P)
    target_area = area_counts[target_id.long()]  # ターゲットセルの総面積 (N, 1)

    source_perimeters = perimeter_counts[source_ids.long()]  # ソース候補の総周囲長 (N, P)
    target_perimeter = perimeter_counts[target_id.long()]  # ターゲットセルの総周囲長 (N, 1)

    source_is_not_empty = source_ids != 0  # ソース候補が空（ID=0）かどうか (N, P)
    target_is_not_empty = target_id != 0  # ターゲットセルが空（ID=0）かどうか (N, 1)

    # --- ΔHの各項を計算 ---
    # 1. 面積エネルギー変化 ΔH_A
    delta_H_area = calc_dH_area(
        source_areas, target_area, l_A, A_0, source_is_not_empty, target_is_not_empty
    )

    # 2. 周囲長エネルギー変化 ΔH_L
    delta_H_perimeter = calc_dH_perimeter(
        source_perimeters,
        target_perimeter,
        source_ids,
        target_id,
        l_L,
        L_0,
        source_is_not_empty,
        target_is_not_empty,
        device,
    )
    # delta_H_perimeter = torch.zeros_like(delta_H_area, dtype=torch.float32)

    # 3. 接着エネルギー変化 ΔH_adhesion

    # --- 総エネルギー変化 ΔH ---
    delta_H = delta_H_area + delta_H_perimeter  # (N, P)

    # --- ボルツマン確率のロジット（対数確率）を計算 -- Logit = -ΔH / T
    logits = torch.exp(-delta_H / T)  # (N, P)

    # 遷移確率が0になるように）
    logits = torch.where(
        source_ids != target_id, logits, torch.tensor(0.0, device=device)
    )

    return logits  # 各パッチ中心に対する遷移ロジット(N, P-1)を返す


def cpm_checkerboard_step_single(map_input, l_A, A_0, l_L, L_0, T, x_offset, y_offset):
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
    ids_center = ids_patch[:, center_index]  # (N, 1)
    dir_rand = torch.randint(0, 4, (num_patches,), device=device)  # (N,)
    ids_patch = torch.gather(
        ids_patch[:, -1:], dim=1, index=dir_rand.unsqueeze(1).expand(-1, 5)
    )  # (N, 5)

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

    selects = torch.relu(torch.sign(logits - rand))  # 0か1に(N, 4)

    # 各パッチの確率 (N, 4) - 確率の合計は1になる
    # prob = selects / (torch.sum(selects, dim=1, keepdim=True) + 1e-8)  # (N, 4)

    prob = selects / 4

    # 遷移しない確率を追加
    prob = torch.concat((prob, 1 - torch.sum(prob, dim=1, keepdim=True)), dim=1)
    # print(prob)
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


def cpm_checkerboard_step(map_input, l_A, A_0, l_L, L_0, T, x_offset, y_offset):
    """CPMの1ステップをチェッカーボードパターンの一部に対して実行する。"""
    H, W, C = map_input.shape

    # 1. 現在のチェッカーボードオフセットに対応するパッチを抽出
    # 出力: (パッチ数, 9, C)
    map_patched = extract_patches_manual_padding_with_offset(
        map_input, 3, 3, x_offset, y_offset
    )
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

    source_ids = ids_patch[:, neighbors]  # (N, 4)
    target_id = ids_patch[:, center_index].unsqueeze(1)  # (N, 1)

    # 2. 各パッチ中心に対する状態遷移のロジットを計算
    logits = calc_cpm_probabilities(
        map_input, source_ids, target_id, l_A, A_0, l_L, L_0, T
    )
    logits = torch.clip(logits, 0, 1)
    # print(logits)

    # 3. 各パッチ中心について、次に採用する状態（隣接ピクセルのインデックス）をサンプリング
    rand = torch.rand_like(logits)  # 確率を生成 (N, 4)

    selects = torch.relu(torch.sign(logits - rand))  # 0か1に(N, 4)

    # 各パッチの確率 (N, 4) - 確率の合計は1になる
    # prob = selects / (torch.sum(selects, dim=1, keepdim=True) + 1e-8)  # (N, 4)

    prob = selects / 4

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

    ids_concat = torch.concat([source_ids, target_id], dim=1)  # (N, 5)
    new_center_ids = torch.gather(ids_concat, dim=1, index=sampled_indices.long())

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


def print_cpm_bins(map_tensor):
    print("面積")
    ids = map_tensor[:, :, 0].long()  # IDチャンネルをlong型で取得 (H, W)
    H, W = ids.shape
    flat_ids = ids.flatten()  # bincountのために1次元配列にフラット化

    # bincountはCPUで高速な場合が多いので、一時的にCPUに転送して実行し、結果を元のデバイスに戻す
    area_counts = (
        torch.bincount(flat_ids.cpu(), minlength=cell_newer_id_counter)
        .to(device)
        .float()
    )  # 各IDの面積カウント (ID数,)
    print(area_counts)
    print("周囲長")
    p = calc_perimeter_patch(map_tensor)
    print(calc_total_perimeter_bincount(map_tensor, p))


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
