import numpy as np
import torch
import torch.nn.functional as F
from cpm_torch.CPM_Map import *


class CPM_config:
    def __init__(self, **kwargs):
        self.size = kwargs.pop("size", (256, 256))
        self.dim = len(self.size)
        self.height = self.size[0]
        self.width = self.size[1]
        if self.dim == 3:
            self.depth = self.size[2]
        else:
            self.depth = 1

        self.l_A = kwargs.pop("l_A", 1.0)  # 面積エネルギーの係数
        self.l_L = kwargs.pop("l_L", 1.0)  # 周囲長エネルギーの係数
        self.A_0 = kwargs.pop("A_0", 1.0)  # 基準面積
        self.L_0 = kwargs.pop("L_0", 4.0)  # 基準周囲長
        self.T = kwargs.pop("T", 1.0)  # 温度


class CPM:
    def __init__(self, config: CPM_config, device="cuda"):
        self.config = config

        self.cell_count = 0

        self.center_index = 4  # 中央のインデックス
        self.neighbors = [1, 3, 5, 7]  # 4近傍
        self.neighbors_len = len(self.neighbors)  # 4近傍の数

        # マップテンソルを作成: (高さ, 幅, チャンネル数)
        # チャンネル 0: 細胞ID
        self.map_tensor = torch.zeros(
            (self.config.height, self.config.width, 1),
            dtype=torch.float32,
            device=device,
        )
        self.device = device

    def add_cell(self, *pos):
        x = slice(pos[0], pos[0] + 1)
        y = slice(pos[1], pos[1] + 1)
        self.add_cell_slice(x, y)

    def add_cell_slice(self, *slices):
        """指定されたスライスに新しいIDと値を持つ細胞を追加する。"""
        self.cell_count += 1

        # 指定スライスと同じ形状で、IDと値で埋められたテンソルを作成
        id_tensor = torch.full_like(
            self.map_tensor[slices[0], slices[1], 0], float(self.cell_count)
        )

        # スライシングを使ってマップテンソルにIDと値を直接代入（インプレース操作）
        self.map_tensor[slices[0], slices[1], 0] = id_tensor  # チャンネル0 (ID)

        # 次の細胞のためにIDカウンターをインクリメント

    def calc_area_bincount(self) -> torch.Tensor:
        """
        torch.bincount を使って各細胞IDの面積（ピクセル数）を計算する。

        Returns:
            torch.Tensor: 各細胞IDの面積を格納したテンソル (cell_count,)
        """
        ids = self.map_tensor[:, :, 0].long()  # IDチャンネルをlong型で取得 (H, W)
        H, W = ids.shape
        flat_ids = ids.flatten()  # bincountのために1次元配列にフラット化

        # bincountはCPUで高速な場合が多いので、一時的にCPUに転送して実行し、結果を元のデバイスに戻す
        area_counts = (
            torch.bincount(flat_ids.cpu(), minlength=self.cell_count)
            .to(self.device)
            .float()
        )  # 各IDの面積カウント (ID数,)

        return area_counts

    def calc_perimeter_patch(self) -> torch.Tensor:
        """各ピクセルにおける周囲長の寄与（隣接4ピクセルとのID境界数）を計算する。
        Returns:
            torch.Tensor: 各ピクセルの周囲長寄与を格納したテンソル (H, W)
        """
        ids = self.map_tensor[:, :, 0]  # IDチャンネル (H, W)
        # 各ピクセル周りの3x3パッチを抽出 -> (H, W, 9, 1)
        id_patches = extract_patches_batched_channel(ids.unsqueeze(-1), 3)

        center_ids = id_patches[:, :, self.center_index, 0]  # 各パッチ中心のID (H, W)
        # 各パッチの上下左右の隣接ピクセルのID (H, W, 4)
        neighbor_ids_data = id_patches[:, :, self.neighbors, 0]

        # 中心IDと各隣接IDを比較 (H, W, 4) -> 境界ならTrue
        is_boundary = neighbor_ids_data != center_ids.unsqueeze(-1)
        # 各ピクセルでIDが異なる隣接ピクセルの数を合計（周囲長への寄与）
        perimeter_at_pixel = torch.sum(is_boundary.float(), dim=2)  # (H, W)
        return perimeter_at_pixel

    def calc_total_perimeter_bincount(self) -> torch.Tensor:
        """
        各細胞IDの総周囲長を計算する。

        Returns:
            torch.Tensor: 各細胞IDの総周囲長を格納したテンソル (cell_count,)
        """
        ids = self.map_tensor[:, :, 0].long()  # IDチャンネル (H, W)
        flat_ids = ids.flatten()
        perimeter_at_pixel = self.calc_perimeter_patch()  # (H, W)
        flat_perimeter_contrib = perimeter_at_pixel.flatten()  # 各ピクセルの周囲長寄与

        total_perimeter_counts = (
            torch.bincount(
                flat_ids.cpu(),
                weights=flat_perimeter_contrib.cpu(),
                minlength=self.cell_count,
            )
            .to(self.device)
            .float()
        )  # 各IDの総周囲長 (ID数,)
        return total_perimeter_counts

    def calc_dH_area(
        self, source_areas, target_area, source_is_not_empty, target_is_not_empty
    ):
        """
        面積エネルギー変化 ΔH_A を計算する。

        Args:
            source_areas: 各ソース候補セルの現在の面積 (N, 4)
            target_area: ターゲットセルの現在の面積 (N, 1)
            source_is_not_empty: ソース候補が空でないかどうかのブールマスク (N, 4)
            target_is_not_empty: ターゲットセルが空でないかどうかのブールマスク (N, 1)

        Returns:
            delta_H_area: 各ソース候補への遷移による面積エネルギー変化 (N, 4)
        """
        # 1. 面積エネルギー変化 ΔH_A
        # H_A = λ_A * (A - A_0)^2
        # A_s -> A_s + 1, A_t -> A_t - 1 となるときの変化
        # ΔH_A = λ_A * [ (A_s+1 - A_0)^2 - (A_s - A_0)^2 ] + λ_A * [ (A_t-1 - A_0)^2 - (A_t - A_0)^2 ]
        # ΔH_A = λ_A * [ 2*A_s + 1 - 2*A_0 ] + λ_A * [ -2*A_t + 1 + 2*A_0 ]
        # ΔH_A = λ_A * [ 2*(A_s - A_t) + 2 ]
        l_A = self.config.l_A
        A_0 = self.config.A_0

        delta_H_area = (
            l_A * (2.0 * source_areas + 1 - 2 * A_0) * source_is_not_empty
            + (-2.0 * target_area + 1 + 2 * A_0) * target_is_not_empty
        )  # (N, 4)
        return delta_H_area

    def calc_dH_perimeter(
        self,
        source_perimeters,  # (N, 4)
        target_perimeter,  # (N, 1)
        source_ids,  # (N, P) ソース候補のID群。
        target_id,  # (N, 1) ターゲットセルのID
        source_is_not_empty: torch.Tensor,  # (N, 4) ソース候補が空でないかのマスク (boolean)
        target_is_not_empty: torch.Tensor,  # (N, 1) ターゲットセルが空でないかのマスク (boolean)
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

        l_L = self.config.l_L
        L_0 = self.config.L_0

        num_s_in_target_neighbors = torch.zeros_like(
            source_ids, dtype=torch.float, device=self.device
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

    def calc_cpm_probabilities(self, source_ids, target_id):
        """
        CPMの状態遷移確率（ロジット）を計算する。

        Args:
            source_ids: 抽出されたパッチのID (N, P) 複数個のソースを同時に計算可能
            target_id: 抽出されたパッチのターゲットID (N, 1)

        Returns:
            logits: 各パッチ中心に対する状態遷移のロジット (N, P)
        """
        area_counts = self.calc_area_bincount()  # (ID数,)
        perimeter_counts = self.calc_total_perimeter_bincount()  # (ID数,)

        # 細胞の総面積 (N, P)
        source_areas = area_counts[source_ids.long()]
        # ターゲットセルの総面積 (N, 1)
        target_area = area_counts[target_id.long()]

        # ソース候補の総周囲長 (N, P)
        source_perimeters = perimeter_counts[source_ids.long()]
        # ターゲットセルの総周囲長 (N, 1)
        target_perimeter = perimeter_counts[target_id.long()]

        # ソース候補が空（ID=0）かどうか (N, P)
        source_is_not_empty = source_ids != 0
        # ターゲットセルが空（ID=0）かどうか (N, 1)
        target_is_not_empty = target_id != 0

        # --- ΔHの各項を計算 ---
        # 1. 面積エネルギー変化 ΔH_A
        delta_H_area = self.calc_dH_area(
            source_areas,
            target_area,
            source_is_not_empty,
            target_is_not_empty,
        )

        # 2. 周囲長エネルギー変化 ΔH_L
        delta_H_perimeter = self.calc_dH_perimeter(
            source_perimeters,
            target_perimeter,
            source_ids,
            target_id,
            source_is_not_empty,
            target_is_not_empty,
        )
        # delta_H_perimeter = torch.zeros_like(delta_H_area, dtype=torch.float32)

        # 3. 接着エネルギー変化 ΔH_adhesion

        # --- 総エネルギー変化 ΔH ---
        delta_H = delta_H_area + delta_H_perimeter  # (N, P)

        # --- ボルツマン確率のロジット（対数確率）を計算 -- Logit = -ΔH / T
        logits = torch.exp(-delta_H / self.config.T)  # (N, P)

        # 遷移確率が0になるように）
        logits = torch.where(
            source_ids != target_id, logits, torch.tensor(0.0, device=self.device)
        )

        return logits  # 各パッチ中心に対する遷移ロジット(N, P-1)を返す

    def cpm_checkerboard_step_single(self, x_offset, y_offset):
        """CPMの1ステップをチェッカーボードパターンの一部に対して実行する。"""
        H, W, C = self.map_tensor.shape

        # 1. 現在のチェッカーボードオフセットに対応するパッチを抽出 (N, 9, C)
        map_patched = extract_patches_manual_padding_with_offset(
            self.map_tensor, 3, 3, x_offset, y_offset
        )
        # print(map_patched.shape)
        ids_patch = map_patched[:, :, 0]

        source_ids = ids_patch[:, self.neighbors]  # (N, 4)
        target_id = ids_patch[:, self.center_index].unsqueeze(1)  # (N, 1)

        source_rand_ids = torch.randint(
            0, self.neighbors_len, (source_ids.shape[0], 1), device=self.device
        )  # (N, 1)
        source_ids_one = torch.gather(
            source_ids, dim=1, index=source_rand_ids.long()
        )  # (N, 1)

        # 2. 各パッチ中心に対する状態遷移のロジットを計算
        logits = self.calc_cpm_probabilities(source_ids_one, target_id)
        # logits = torch.clip(logits, 0, 1)
        # print(logits)

        # 3. 各パッチ中心について、次に採用する状態（隣接ピクセルのインデックス）をサンプリング
        rand = torch.rand_like(logits)  # 確率を生成 (N, 1)

        new_center_ids = torch.where(logits > rand, source_ids_one, target_id)  # (N, 1)

        # 次に、チャンネル0（現在のID）をサンプリングされた新しいIDで更新
        map_patched[:, self.center_index, 0] = new_center_ids.squeeze(1)

        # 6. 更新されたパッチテンソルからマップ全体を再構成
        map_output = reconstruct_image_from_patches(
            map_patched, self.map_tensor.shape, 3, 3, x_offset, y_offset
        )

        self.map_tensor = map_output

        return logits
    
    def cpm_checkerboard_step_single2(self, x_offset, y_offset):
        """CPMの1ステップをチェッカーボードパターンの一部に対して実行する。"""
        H, W, C = self.map_tensor.shape

        # 1. 現在のチェッカーボードオフセットに対応するパッチを抽出
        # 出力: (パッチ数, 9, C)
        map_patched = extract_patches_manual_padding_with_offset(
            self.map_tensor, 3, 3, x_offset, y_offset
        )
        # print(map_patched.shape)
        ids_patch = map_patched[:, :, 0]

        source_ids = ids_patch[:, self.neighbors]  # (N, 4)
        target_id = ids_patch[:, self.center_index].unsqueeze(1)  # (N, 1)

        # 2. 各パッチ中心に対する状態遷移のロジットを計算
        logits = self.calc_cpm_probabilities(source_ids, target_id)
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
        # print(prob)
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
        map_patched[:, self.center_index, 0] = new_center_ids.squeeze(1)

        # 6. 更新されたパッチテンソルからマップ全体を再構成
        map_output = reconstruct_image_from_patches(
            map_patched, self.map_tensor.shape, 3, 3, x_offset, y_offset
        )

        self.map_tensor = map_output

        return logits

    def cpm_checkerboard_step(self, x_offset, y_offset):
        """CPMの1ステップをチェッカーボードパターンの一部に対して実行する。"""
        H, W, C = self.map_tensor.shape

        # 1. 現在のチェッカーボードオフセットに対応するパッチを抽出
        # 出力: (パッチ数, 9, C)
        map_patched = extract_patches_manual_padding_with_offset(
            self.map_tensor, 3, 3, x_offset, y_offset
        )
        # print(map_patched.shape)
        ids_patch = map_patched[:, :, 0]

        source_ids = ids_patch[:, self.neighbors]  # (N, 4)
        target_id = ids_patch[:, self.center_index].unsqueeze(1)  # (N, 1)

        # 2. 各パッチ中心に対する状態遷移のロジットを計算
        logits = self.calc_cpm_probabilities(source_ids, target_id)
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
        # print(prob)
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
        map_patched[:, self.center_index, 0] = new_center_ids.squeeze(1)

        # 6. 更新されたパッチテンソルからマップ全体を再構成
        map_output = reconstruct_image_from_patches(
            map_patched, self.map_tensor.shape, 3, 3, x_offset, y_offset
        )

        self.map_tensor = map_output

        return logits

    def cpm_mcs_step(self):
        for x_offset in range(3):  # x方向オフセット (0 or 1)
            for y_offset in range(3):  # y方向オフセット (0 or 1)
                #self.cpm_checkerboard_step(x_offset, y_offset)
                self.cpm_checkerboard_step_single(x_offset, y_offset)

    def check_map_tensor(self):
        if torch.isnan(self.map_tensor).any() or torch.isinf(self.map_tensor).any():
            print(f"警告: map_tensor に NaN/Inf があります! ")
            print(
                f"NaNs in map_tensor ch0: {torch.isnan(self.map_tensor[:,:,0]).sum()}, ch1: {torch.isnan(self.map_tensor[:,:,1]).sum()}, ch2: {torch.isnan(self.map_tensor[:,:,2]).sum()}"
            )

    def print_cpm_bins(self):
        area_counts = self.calc_area_bincount()
        perimeter_counts = self.calc_total_perimeter_bincount()

        print("面積カウント:", area_counts)
        print("周囲長カウント:", perimeter_counts)
        print("細胞数:", self.cell_count)
