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

        self.diffusion_channels = kwargs.pop(
            "diffusion_channels", [2]
        )  # 拡散チャンネル数
        self.other_channels = len(self.diffusion_channels)  # 他のチャンネル数
        self.diffusion_D = kwargs.pop("diffusion_D", [0.1])  # 拡散係数
        self.diffusion_percent = kwargs.pop(
            "diffusion_percent", [1.0]
        )  # 細胞壁での拡散を抑制する割合


class CPM:
    center_index = 4  # 中央のインデックス
    neighbors = [1, 3, 5, 7]  # 4近傍
    neighbors_len = len(neighbors)  # 4近傍の数

    def __init__(self, config: CPM_config, device="cuda"):
        self.config = config
        self.cell_count = 0
        self.device = device

    def reset(self):
        """マップテンソルを初期化する。"""
        # マップテンソルを作成: (高さ, 幅, チャンネル数)
        # チャンネル 0: 細胞ID
        # チャンネル 1: ターゲット値
        # チャンネル 2から: 他のチャンネル（拡散など）
        self.map_tensor = torch.zeros(
            (self.config.height, self.config.width, 2 + self.config.other_channels),
            dtype=torch.float32,
            device=self.device,
        )
        self.cell_count = 0

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

        self.map_tensor[slices[0], slices[1], 0] = id_tensor  # チャンネル0 (ID)

    @classmethod
    def calc_area_bincount(cls, ids, cell_count) -> torch.Tensor:
        """
        torch.bincount を使って各細胞IDの面積（ピクセル数）を計算する。

        Parameters:
            ids (torch.Tensor): 細胞IDの2Dテンソル (H, W)。
            cell_count (int): 細胞の総数。bincountの最小値を設定するために使用。

        Returns:
            torch.Tensor: 各細胞IDの面積を格納したテンソル (cell_count,)
        """
        ids = ids.long()  # IDチャンネルをlong型で取得 (H, W)
        device = ids.device  # デバイスを取得
        H, W = ids.shape
        flat_ids = ids.flatten()  # bincountのために1次元配列にフラット化

        # bincountはCPUで高速な場合が多いので、一時的にCPUに転送して実行し、結果を元のデバイスに戻す
        area_counts = (
            torch.bincount(flat_ids.cpu(), minlength=cell_count + 1).to(device).float()
        )  # 各IDの面積カウント (ID数,)

        return area_counts

    @classmethod
    def calc_perimeter_patch(cls, ids) -> torch.Tensor:
        """各ピクセルにおける周囲長の寄与（隣接4ピクセルとのID境界数）を計算する。

        Parameters:
            ids (torch.Tensor): 細胞IDの2Dテンソル (H, W)。

        Returns:
            torch.Tensor: 各ピクセルの周囲長寄与を格納したテンソル (H, W)
        """
        # 各ピクセル周りの3x3パッチを抽出 -> (H, W, 9, 1)
        id_patches = extract_patches_batched_channel(ids.unsqueeze(-1), 3)

        center_ids = id_patches[:, :, cls.center_index, 0]  # 各パッチ中心のID (H, W)

        neighbor_ids_data = id_patches[:, :, cls.neighbors, 0]  # (H, W, 4)

        is_boundary = neighbor_ids_data != center_ids.unsqueeze(-1)  # 境界ならTrue

        # 各ピクセルでIDが異なる隣接ピクセルの数を合計（周囲長への寄与）
        perimeter_at_pixel = torch.sum(is_boundary.float(), dim=2)  # (H, W)
        return perimeter_at_pixel

    @classmethod
    def calc_total_perimeter_bincount(cls, ids, cell_count) -> torch.Tensor:
        """
        各細胞IDの総周囲長を計算する。

        Parameters:
            ids (torch.Tensor): 細胞IDの2Dテンソル (H, W)。
            cell_count (int): 細胞の総数。bincountの最小値を設定するために使用。

        Returns:
            torch.Tensor: 各細胞IDの総周囲長を格納したテンソル (cell_count,)
        """
        flat_ids = ids.long().flatten()  # (H, W) -> (H*W,)
        device = ids.device  # デバイスを取得
        perimeter_at_pixel = cls.calc_perimeter_patch(ids)  # (H, W)
        flat_perimeter_contrib = perimeter_at_pixel.flatten()  # (H*W,)

        total_perimeter_counts = (
            torch.bincount(
                flat_ids.cpu(),
                weights=flat_perimeter_contrib.cpu(),
                minlength=cell_count,
            )
            .to(device)
            .float()
        )  # 各IDの総周囲長 (ID数,)
        return total_perimeter_counts

    @classmethod
    def calc_area_perimeter(cls, ids, source_ids, target_ids) -> torch.Tensor:
        """
        各細胞IDの面積と周囲長を計算する。

        Parameters:
            ids (torch.Tensor): 細胞IDの2Dテンソル ((B,) H, W)。
            source_ids (torch.Tensor): ソース候補のID ((B,) N, P)。
            target_ids (torch.Tensor): ターゲットセルのID ((B,) N, 1)。

        Returns:
            source_areas (torch.Tensor): 各ソース候補セルの面積 ((B,) N, P)。
            target_area (torch.Tensor): ターゲットセルの面積 ((B,) N, 1)。
            source_perimeters (torch.Tensor): 各ソース候補セルの総周囲長 ((B,) N, P)。
            target_perimeter (torch.Tensor): ターゲットセルの総周囲長 ((B,) N, 1).
        """
        _iter = ids.shape[0] if ids.dim() == 3 else 1

        source_areas = []
        target_area = []
        source_perimeters = []
        target_perimeter = []

        for i in range(_iter):
            if ids.dim() == 3:
                ids_i = ids[i]  # (H, W)
            else:
                ids_i = ids  # (H, W)
                source_ids = source_ids.unsqueeze(0)  # (1, N, P)
                target_ids = target_ids.unsqueeze(0)  # (1, N, 1)
            cell_count = int(ids_i.max().item() + 1)  # 細胞数

            area = cls.calc_area_bincount(ids_i, cell_count)  # (cell_count,)
            perimeter = cls.calc_total_perimeter_bincount(ids_i, cell_count)

            # print("area:", area)
            # print("perimeter:", perimeter)

            source_areas.append(area[source_ids[i].long()])
            target_area.append(area[target_ids[i].long()])
            source_perimeters.append(perimeter[source_ids[i].long()])
            target_perimeter.append(perimeter[target_ids[i].long()])

        if ids.dim() == 3:
            return (
                torch.stack(source_areas, dim=0),  # (B, N, P)
                torch.stack(target_area, dim=0),  # (B, N, 1)
                torch.stack(source_perimeters, dim=0),  # (B, N, P)
                torch.stack(target_perimeter, dim=0),  # (B, N, 1)
            )
        else:
            return (
                source_areas[0],  # (N, P)
                target_area[0],  # (N, 1)
                source_perimeters[0],  # (N, P)
                target_perimeter[0],  # (N, 1)
            )

    @classmethod
    def calc_area_perimeter_mask(cls, ids, source_ids, target_ids, batch_indices) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        各細胞IDの面積と周囲長を計算する。
        この実装は、バッチ処理を効率的に行うため、ループを排除しています。
        source_ids, target_ids, batch_indicesはバッチ間で平坦化されたテンソルを想定しています。

        Parameters:
            ids (torch.Tensor): 細胞IDのテンソル (H, W) または (B, H, W)。
            source_ids (torch.Tensor): ソース候補のID (N', P)。
            target_ids (torch.Tensor): ターゲットセルのID (N', 1)。
            batch_indices (torch.Tensor): 各要素が属するバッチのインデックス (N',)。

        Returns:
            A tuple containing:
            - source_areas (torch.Tensor): 各ソース候補セルの面積 (N', P)。
            - target_area (torch.Tensor): ターゲットセルの面積 (N', 1)。
            - source_perimeters (torch.Tensor): 各ソース候補セルの総周囲長 (N', P)。
            - target_perimeter (torch.Tensor): ターゲットセルの総周囲長 (N', 1)。
        """
        # --- 1. 入力形状の正規化 ---
        if ids.dim() != 3:
            # バッチ次元がない場合、追加する
            ids = ids.unsqueeze(0)

        B, H, W = ids.shape
        device = ids.device

        # --- 2. 面積と周囲長のバッチ一括計算 ---
        # 全バッチを通じての最大セルIDを取得
        max_id = int(ids.max().item()) + 1

        # IDにバッチごとのオフセットを追加し、全バッチを平坦化
        # これで一回のbincountで全バッチ分をまとめて計算できる
        offset = torch.arange(B, device=device).view(B, 1, 1) * max_id
        ids_offset = ids + offset

        # 面積計算 (Area Calculation)
        # 全バッチのIDを1次元に平坦化してbincount
        areas_flat = torch.bincount(ids_offset.flatten().long(), minlength=B * max_id)
        areas = areas_flat.view(B, max_id)  # (B, max_id) の形状に戻す

        # 周囲長計算 (Perimeter Calculation)
        # 水平方向の境界 (左右のIDが異なる場所)
        h_boundaries = ids_offset[:, :, :-1] != ids_offset[:, :, 1:]
        # 垂直方向の境界 (上下のIDが異なる場所)
        v_boundaries = ids_offset[:, :-1, :] != ids_offset[:, 1:, :]

        # 境界を構成する両側のセルのIDを取得
        h_left = ids_offset[:, :, :-1][h_boundaries]
        h_right = ids_offset[:, :, 1:][h_boundaries]
        v_up = ids_offset[:, :-1, :][v_boundaries]
        v_down = ids_offset[:, 1:, :][v_boundaries]

        # 全ての境界IDを一つのリストに結合
        all_boundary_ids = torch.cat([h_left, h_right, v_up, v_down])

        # 一括でbincountを計算し、各IDが境界に現れる回数を数える
        perimeters_flat = torch.bincount(all_boundary_ids.long(), minlength=B * max_id)
        perimeters = perimeters_flat.view(B, max_id)  # (B, max_id) の形状に戻す

        # --- 3. IDに対応する値を取得 (Gather) ---
        # `batch_indices`と`*_ids`を使って、対応する面積と周囲長を一度に取得
        
        # ソースセルの情報を取得
        batch_indices_expanded = batch_indices.unsqueeze(1)

        # `batch_indices_expanded` は (N', 1) のため、(N', P) の source_ids とブロードキャスト可能になる
        source_areas = areas[batch_indices_expanded, source_ids.long()]
        source_perimeters = perimeters[batch_indices_expanded, source_ids.long()]

        # target_ids (N', 1) も同様に batch_indices_expanded を使ってインデックス指定する
        target_area = areas[batch_indices_expanded, target_ids.long()]
        target_perimeter = perimeters[batch_indices_expanded, target_ids.long()]
        
        return source_areas, target_area, source_perimeters, target_perimeter


    def calc_dH_area(
        self, source_areas, target_area, source_is_not_empty, target_is_not_empty
    ):
        """
        面積エネルギー変化 ΔH_A を計算する。

        Args:
            source_areas: 各ソース候補セルの現在の面積 ((B), N, 4)
            target_area: ターゲットセルの現在の面積 ((B), N, 1)
            source_is_not_empty: ソース候補が空でないかどうかのマスク ((B), N, 4)
            target_is_not_empty: ターゲットセルが空でないかどうかのマスク ((B), N, 1)

        Returns:
            delta_H_area: 各ソース候補への遷移による面積エネルギー変化 ((B), N, 4)
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
            l_A * ((2.0 * source_areas + 1 - 2 * A_0) * source_is_not_empty
            + (-2.0 * target_area + 1 + 2 * A_0) * target_is_not_empty)
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
            # current_s_candidate_id: (N, 1) それぞれのパッチに対するk番目の隣接ピクセルのID
            current_s_candidate_id = source_ids[:, k_idx : k_idx + 1]
            # source_ids == current_s_candidate_id は (N, 4) のブールテンソル
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
            l_L * (term1_t + term2_t) * target_is_not_empty.float()
        )  # (N,1)

        # 総エネルギー変化
        # delta_H_perimeter_s is (N, 4)
        # delta_H_perimeter_t_for_each_source_candidate is (N, 1) and will be broadcasted
        # during addition to (N,4)
        delta_H_perimeter = (
            delta_H_perimeter_s + delta_H_perimeter_t_for_each_source_candidate
        )

        return delta_H_perimeter

    def calc_dH_perimeter_single(
        self,
        source_perimeters: torch.Tensor,  # (N, 1)
        target_perimeter: torch.Tensor,   # (N, 1)
        source_ids_4: torch.Tensor,       # (N, 4)
        source_ids: torch.Tensor,         # (N, 1)
        target_id: torch.Tensor,          # (N, 1)
        source_is_not_empty: torch.Tensor, # (N, 1)
        target_is_not_empty: torch.Tensor, # (N, 1)
    ) -> torch.Tensor:
        """
        【単一ソース候補用】周囲長エネルギー変化 ΔH_L を計算する。
        ピクセルがターゲットセルtから特定のソースセルsに変化する状況を考える。

        Args:
            source_perimeters:     注目するソースセルの現在の総周囲長 ((B,) N, 1)
            target_perimeter:      ターゲットセルの現在の総周囲長 ((B,) N, 1)
            source_ids_4:          ターゲットピクセルの4近傍のID群 ((B,) N, 4)
            source_ids:            注目するソースセルのID ((B,) N, 1)
            target_id:             ターゲットセルのID ((B,) N, 1)
            source_is_not_empty:  注目するソースセルが空でないかのマスク ((B,) N, 1)
            target_is_not_empty:  ターゲットセルが空でないかのマスク ((B,) N, 1)

        Returns:
            delta_H_perimeter: 選ばれたソース候補への遷移によるエネルギー変化 ((B,) N, 1)
        """
        
        # ターゲットピクセルの4近傍における、ソースセルおよびターゲットセルの数を数える
        # .float()で後の計算のために浮動小数点数に変換
        n_s = torch.sum(source_ids_4 == source_ids, dim=-1, keepdim=True).float()
        n_t = torch.sum(source_ids_4 == target_id, dim=-1, keepdim=True).float()

        # ソースセルとターゲットセルの周囲長の変化量を計算
        # ΔL_s = 4 - 2 * n_s: 4つの辺のうち、n_s個は内部境界に変わり、(4-n_s)個は新たな外部境界となる
        delta_L_s = 4.0 - 2.0 * n_s
        # ΔL_t = 2 * n_t - 4: 4つの辺のうち、n_t個は新たな外部境界となり、(4-n_t)個は消滅する
        delta_L_t = 2.0 * n_t - 4.0

        # エネルギー計算に必要なパラメータを取得 (selfから取得することを想定)
        # self.get_paramsは、セルIDに基づき λ_L と L_target を返すヘルパーメソッドと仮定
        lambda_s, target_L_s = self.config.l_L, self.config.L_0
        lambda_t, target_L_t = self.config.l_L, self.config.L_0
        
        # ソースセルのエネルギー変化を計算
        new_source_perimeter = source_perimeters + delta_L_s
        delta_H_s = lambda_s * (
            torch.pow(new_source_perimeter - target_L_s, 2)
            - torch.pow(source_perimeters - target_L_s, 2)
        )

        # ターゲットセルのエネルギー変化を計算
        new_target_perimeter = target_perimeter + delta_L_t
        delta_H_t = lambda_t * (
            torch.pow(new_target_perimeter - target_L_t, 2)
            - torch.pow(target_perimeter - target_L_t, 2)
        )

        # セルが存在しない（empty）場合は、エネルギー変化を0にする
        delta_H_s = delta_H_s * source_is_not_empty.float()
        delta_H_t = delta_H_t * target_is_not_empty.float()

        # 総エネルギー変化
        delta_H_perimeter = delta_H_s + delta_H_t

        # ソースセルとターゲットセルが同じIDの場合、状態変化は起こらないためエネルギー変化は0
        is_same_id = (source_ids == target_id)
        delta_H_perimeter[is_same_id] = 0.0
        
        return delta_H_perimeter

    def calc_cpm_probabilities(
        self, source_ids, target_id, ids, dH_NN=None, source_ids_4=None, batch_indices=None
    ):
        """遷移確率を計算する。

        Parameters:
            source_ids (torch.Tensor): ソース候補のID ((B,) N, P)。
            target_id (torch.Tensor): ターゲットセルのID ((B,) N, 1)。
            ids (torch.Tensor): マップのID ((B,) N, W)。

            dH_NN (torch.Tensor, optional): ニューラルネットワークによるエネルギー変化 ((B,) N, P)。
            source_ids_4 (torch.Tensor, optional): 周囲長計算用の4近傍のID ((B,) N, 4)。
            batch_indices (torch.Tensor, optional): バッチインデックス ((B,) N)。

        Returns:
            torch.Tensor: 各パッチ中心に対する遷移ロジット ((B,) N, P)。
        """
        # 1. 面積と周囲長を計算
        if batch_indices is not None:
            # バッチ処理用のマスクを使用して、面積と周囲長を計算
            source_areas, target_area, source_perimeters, target_perimeter = (
                self.calc_area_perimeter_mask(
                    ids, source_ids, target_id, batch_indices
                )
            )
        else:
            source_areas, target_area, source_perimeters, target_perimeter = (
                self.calc_area_perimeter(ids, source_ids, target_id)
            )  # (N, P), (N, 1), (N, P), (N, 1)

        # 2. マスクを作成
        source_is_not_empty = source_ids != 0
        target_is_not_empty = target_id != 0

        # 3. 面積エネルギー変化 ΔH_A を計算
        delta_H_area = self.calc_dH_area(
            source_areas,
            target_area,
            source_is_not_empty,
            target_is_not_empty,
        )
        # print("delta_H_area:", delta_H_area[0, :, 0])

        # 4. 周囲長エネルギー変化 ΔH_L を計算
        if source_ids.shape[-1] == 1:
            # 【ランダム選択モード】
            delta_H_perimeter = self.calc_dH_perimeter_single(
                source_perimeters,
                target_perimeter,
                source_ids_4,  # (N, 4) のIDを渡す
                source_ids,
                target_id,
                source_is_not_empty,
                target_is_not_empty,
            )

        else:
            # 【全候補計算モード]
            delta_H_perimeter = self.calc_dH_perimeter(
                source_perimeters,
                target_perimeter,
                source_ids,
                target_id,
                source_is_not_empty,
                target_is_not_empty,
            )

        delta_H = delta_H_area + delta_H_perimeter
        delta_H = delta_H.detach()  # 勾配計算を無効化

        if dH_NN is not None:
            delta_H = dH_NN + delta_H  # ニューラルネットワークによるエネルギー変化

        # print("delta_H:", delta_H[0, :, 0])
        # print("dH_area:", delta_H_area[0, :, 0])
        # print("dH_perimeter:", delta_H_perimeter[0, :, 0])

        # 5. 遷移確率を計算
        logits = torch.exp(-delta_H / self.config.T)  # (N, P)

        # 6. ソース候補がターゲットセルと同じ場合、確率を0にする
        logits = torch.where(
            source_ids != target_id, logits, torch.tensor(0.0, device=self.device)
        )

        return logits

    def cpm_checkerboard_step(self, x_offset, y_offset, dH_NN=None):
        """CPMの1ステップをチェッカーボードパターンの一部に対して実行する。"""
        H, W, C = self.map_tensor.shape

        map_patched = extract_patches_manual_padding_with_offset(
            self.map_tensor, 3, 3, x_offset, y_offset
        )  # (パッチ数, 9, C)

        # 1. 現在のチェッカーボードオフセットに対応するパッチを抽出
        # 出力: (パッチ数, 9, C)
        ids_patch = map_patched[:, :, 0]

        source_ids = ids_patch[:, CPM.neighbors]  # (N, 4)
        target_id = ids_patch[:, CPM.center_index].unsqueeze(1)  # (N, 1)

        # 2. 各パッチ中心に対する状態遷移のロジットを計算
        logits = self.calc_cpm_probabilities(
            source_ids, target_id, self.map_tensor[:, :, 0], dH_NN=dH_NN
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
        map_patched[:, CPM.center_index, 0] = new_center_ids.squeeze(1)

        # 6. 更新されたパッチテンソルからマップ全体を再構成
        map_output = reconstruct_image_from_patches(
            map_patched, self.map_tensor.shape, 3, 3, x_offset, y_offset
        )

        self.map_tensor = map_output

        return logits

    def cpm_checkerboard_step_single(self, x_offset, y_offset, dH_NN=None):
        """CPMの1ステップをチェッカーボードパターンの一部に対して実行する。"""
        H, W, C = self.map_tensor.shape

        map_patched = extract_patches_manual_padding_with_offset(
            self.map_tensor, 3, 3, x_offset, y_offset
        )  # (パッチ数, 9, C)

        N = map_patched.shape[0]  # パッチ数

        # 1. 現在のチェッカーボードオフセットに対応するパッチを抽出
        # 出力: (パッチ数, 9, C)
        ids_patch = map_patched[:, :, 0]

        source_ids = ids_patch[:, CPM.neighbors]  # (N, 4)
        source_ids_4 = source_ids.clone()

        # ランダムに1つのソースを選択　(N, 1)
        rand_dir = torch.randint(0, CPM.neighbors_len, (N, 1), device=self.device)
        source_ids = torch.gather(source_ids, 1, rand_dir)  # (N, 1)

        target_id = ids_patch[:, CPM.center_index].unsqueeze(1)  # (N, 1)

        # 2. 各パッチ中心に対する状態遷移のロジットを計算
        logits = self.calc_cpm_probabilities(
            source_ids,
            target_id,
            self.map_tensor[:, :, 0],
            dH_NN=dH_NN,
            rand_dir=rand_dir,
            source_ids_4=source_ids_4,
        )  # (N, 1)
        logits = torch.clip(logits, 0, 1)

        # 3. 各パッチ中心について、次に採用する状態（隣接ピクセルのインデックス）をサンプリング
        rand = torch.rand_like(logits)  # 確率を生成 (N, 1)

        selects = torch.relu(torch.sign(logits - rand))  # 0か1に(N, 1)

        new_center_ids = torch.where(selects > 0, source_ids, target_id)  # (N, 1)

        # 次に、チャンネル0（現在のID）をサンプリングされた新しいIDで更新
        map_patched[:, CPM.center_index, 0] = new_center_ids.squeeze(1)

        # 6. 更新されたパッチテンソルからマップ全体を再構成
        map_output = reconstruct_image_from_patches(
            map_patched, self.map_tensor.shape, 3, 3, x_offset, y_offset
        )

        self.map_tensor = map_output

        return logits

    def cpm_checkerboard_step_single_func(
        self,
        tensor,
        dH_NN_func=None,
        x_offset=-1,
        y_offset=-1,
    ):
        """CPMの1ステップをチェッカーボードパターンの一部に対して実行する。

        Parameters:
            tensor (torch.Tensor): 入力テンソル (B, H, W, C)
            x_offset (int): x方向のオフセット (0, 1, 2)
            y_offset (int): y方向のオフセット (0, 1, 2)
            dH_NN_func (callable, optional): エネルギー変化を計算する関数。
                    (sources(B, N, 1, C), targets(B, N, 1, C)) -> (B, N, 1)
        Returns:
            torch.Tensor: 更新されたマップテンソル (B, H, W, C)
        """
        B, H, W, C = tensor.shape
        if x_offset < 0 or y_offset < 0:
            x_offset = np.random.randint(0, 3)  # ランダムなオフセット
            y_offset = np.random.randint(0, 3)  # ランダムなオフセット

        map_patched = extract_patches_manual_padding_with_offset_batch(
            tensor, 3, 3, x_offset, y_offset
        )  # (B, N, 9, C)

        N = map_patched.shape[1]  # パッチ数

        sources = map_patched[:, :, CPM.neighbors]  # (B, N, 4, C)
        source_ids_4 = sources[..., 0]  # (B, N, 4)　周囲長の計算用

        # ランダムに選択する方向　(B, N, 1, 1)
        rand_dir = torch.randint(0, CPM.neighbors_len, (B, N, 1, 1), device=self.device)
        rand_dir = rand_dir.expand(-1, -1, -1, C)  # (B, N, 1, C)

        sources = torch.gather(sources, 2, rand_dir)  # (B, N, 1, C)
        targets = map_patched[:, :, CPM.center_index].unsqueeze(2)  # (B, N, 1, C)

        source_id = sources[..., 0]  # (B, N, 1)
        target_id = targets[..., 0]  # (B, N, 1)

        # ニューラルネットワークでエネルギー変化を計算
        dH_NN = dH_NN_func(sources, targets) if dH_NN_func else None

        # 2. 各パッチ中心に対する状態遷移のロジットを計算
        logits = self.calc_cpm_probabilities(
            source_id,
            target_id,
            tensor[..., 0],
            dH_NN,
            source_ids_4,
        )  # (B, N, 1)
        logits = torch.clip(logits, 0, 1)  # (B, N, 1)

        rand = torch.rand_like(logits)  # 確率を生成 (B, N, 1)

        selects = torch.relu(torch.sign(logits - rand))  # 0か1に(B, N, 1)

        new_center_ids = torch.where(selects > 0, source_id, target_id)  # (B, N, 1)

        # 次に、チャンネル0（現在のID）をサンプリングされた新しいIDで更新
        map_patched[:, :, CPM.center_index, 0] = new_center_ids.squeeze(2)

        # 6. 更新されたパッチテンソルからマップ全体を再構成
        map_output = reconstruct_image_from_patches_batch(
            map_patched, tensor.shape, 3, 3, x_offset, y_offset
        )

        return map_output

    
    def cpm_checkerboard_step_single_masked_func(
        self,
        tensor,
        dH_NN_func=None,
        x_offset=-1,
        y_offset=-1,
    ):
        """CPMの1ステップをチェッカーボードパターンの一部に対して実行する。

        Parameters:
            tensor (torch.Tensor): 入力テンソル (B, H, W, C)
            x_offset (int): x方向のオフセット (0, 1, 2)
            y_offset (int): y方向のオフセット (0, 1, 2)
            dH_NN_func (callable, optional): エネルギー変化を計算する関数。
                    (sources(B, N, 1, C), targets(B, N, 1, C)) -> (B, N, 1)
        Returns:
            torch.Tensor: 更新されたマップテンソル (B, H, W, C)
        """
        B, H, W, C = tensor.shape
        if x_offset < 0 or y_offset < 0:
            x_offset = np.random.randint(0, 3)  # ランダムなオフセット
            y_offset = np.random.randint(0, 3)  # ランダムなオフセット

        map_patched = extract_patches_manual_padding_with_offset_batch(
            tensor, 3, 3, x_offset, y_offset
        )  # (B, N, 9, C)

        N = map_patched.shape[1]  # パッチ数

        sources = map_patched[:, :, CPM.neighbors]  # (B, N, 4, C)
        source_ids_4 = sources[..., 0]  # (B, N, 4)　周囲長の計算用

        # ランダムに選択する方向　(B, N, 1, 1)
        rand_dir = torch.randint(0, CPM.neighbors_len, (B, N, 1, 1), device=self.device)
        rand_dir = rand_dir.expand(-1, -1, -1, C)  # (B, N, 1, C)

        sources = torch.gather(sources, 2, rand_dir)  # (B, N, 1, C)
        targets = map_patched[:, :, CPM.center_index].unsqueeze(2)  # (B, N, 1, C)

        source_id = sources[..., 0]  # (B, N, 1)
        target_id = targets[..., 0]  # (B, N, 1)

        # 1. マスクとインデックスの準備
        masks = source_id != target_id  # (B, N, 1)
        # nonzeroで使うために次元を1つ削除
        masks_squeezed = masks.squeeze(-1)  # (B, N)
        
        # マスクがTrueの要素の(バッチ, パッチ)インデックスを取得
        # batch_indices, patch_indices はそれぞれ1次元のテンソル (形状: (N',))
        # N' はバッチ全体でマスクがTrueの要素の総数
        batch_indices, patch_indices = torch.nonzero(masks_squeezed, as_tuple=True)
        
        # もし更新対象が一つもなければ、何もせず元のテンソルを返す
        if batch_indices.shape[0] == 0:
            return tensor

        # 2. 必要なデータだけを収集 (Gather)
        # インデックスを使って、計算に必要な小さなテンソルを作成する
        sources_masked = sources[batch_indices, patch_indices]  # (N', 1, C)
        targets_masked = targets[batch_indices, patch_indices]  # (N', 1, C)
        source_id_masked = source_id[batch_indices, patch_indices]  # (N', 1)
        target_id_masked = target_id[batch_indices, patch_indices]  # (N', 1)
        source_ids_4_masked = source_ids_4[batch_indices, patch_indices]  # (N', 4)

        # 3. 小さなテンソルで高速に計算
        dH_NN_masked = dH_NN_func(sources_masked, targets_masked) if dH_NN_func else None

        logits_masked = self.calc_cpm_probabilities(
            source_id_masked,
            target_id_masked,
            tensor[..., 0], # これは全体のIDマップなのでそのまま渡す
            dH_NN_masked,
            source_ids_4_masked,
            batch_indices
        )  # (N', 1)
        logits_masked = torch.clip(logits_masked, 0, 1)

        # 4. 計算結果を元の位置に戻す準備
        rand_masked = torch.rand_like(logits_masked)  # (N', 1)
        selects_masked = torch.relu(torch.sign(logits_masked - rand_masked))  # (N', 1)

        new_center_ids_masked = torch.where(selects_masked > 0, source_id_masked, target_id_masked)  # (N', 1)

        # 5. 元の形状のテンソルに結果を代入 (Scatter)
        # まず、更新用のテンソルを元のIDで初期化しておく
        updated_center_ids = target_id.clone()
        # マスクされた位置にだけ、計算結果を書き戻す
        updated_center_ids[batch_indices, patch_indices] = new_center_ids_masked
        
        # パッチの中心IDを更新
        map_patched[:, :, CPM.center_index, 0] = updated_center_ids.squeeze(2)

        # --- 修正ここまで ---

        # 6. 更新されたパッチテンソルからマップ全体を再構成
        map_output = reconstruct_image_from_patches_batch(
            map_patched, tensor.shape, 3, 3, x_offset, y_offset
        )


        # 6. 更新されたパッチテンソルからマップ全体を再構成
        map_output = reconstruct_image_from_patches_batch(
            map_patched, tensor.shape, 3, 3, x_offset, y_offset
        )

        return map_output


    def cpm_mcs_step(self):
        for x_offset in range(3):  # x方向オフセット (0 or 1)
            for y_offset in range(3):  # y方向オフセット (0 or 1)
                self.cpm_checkerboard_step_single(x_offset, y_offset)
                # self.cpm_checkerboard_step_single2(x_offset, y_offset)

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
