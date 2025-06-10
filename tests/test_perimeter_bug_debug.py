#!/usr/bin/env python3
"""
周囲長計算バグの詳細デバッグ
なぜ局所的周囲長変化が理論値と異なるのかを特定
"""

import torch
import sys
import numpy as np
sys.path.append('/app')

from cpm_torch.CPM import CPM, CPM_config

def debug_perimeter_calculation_step_by_step():
    """周囲長計算を段階的にデバッグ"""
    print("=== 周囲長計算の段階的デバッグ ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 簡単な4×4正方形を作成
    test_map = torch.zeros((8, 8), device=device)
    test_map[2:6, 2:6] = 1.0  # 中央に4×4正方形
    
    print("テストマップ (4×4正方形):")
    print(test_map.cpu().numpy().astype(int))
    
    # 1. 各ピクセルの周囲長寄与を計算
    perimeter_contrib = CPM.calc_perimeter_patch(test_map)
    
    print(f"\n各ピクセルの周囲長寄与:")
    print(perimeter_contrib.cpu().numpy())
    
    # 2. 総周囲長を計算
    total_perimeter = CPM.calc_total_perimeter_bincount(test_map, 2)
    print(f"\n総周囲長: ID=0: {total_perimeter[0].item()}, ID=1: {total_perimeter[1].item()}")
    
    # 3. 手動で境界ピクセルを確認
    print(f"\n手動境界ピクセル確認:")
    boundary_pixels = []
    for i in range(8):
        for j in range(8):
            if test_map[i, j] == 1:
                # 4近傍をチェック
                neighbors = []
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < 8 and 0 <= nj < 8:
                        neighbors.append(test_map[ni, nj].item())
                    else:
                        neighbors.append(0)  # 境界外は0
                
                boundary_count = sum(1 for n in neighbors if n != 1)
                if boundary_count > 0:
                    boundary_pixels.append((i, j, boundary_count))
                    print(f"  ピクセル({i},{j}): 近傍{neighbors}, 境界数={boundary_count}")
    
    manual_perimeter = sum(count for _, _, count in boundary_pixels)
    print(f"手動計算の総周囲長: {manual_perimeter}")

def test_pixel_addition_detailed():
    """1ピクセル追加の詳細テスト"""
    print("\n=== 1ピクセル追加の詳細テスト ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 基準4×4正方形
    base_map = torch.zeros((8, 8), device=device)
    base_map[2:6, 2:6] = 1.0
    
    base_perimeter = CPM.calc_total_perimeter_bincount(base_map, 2)[1].item()
    print(f"基準4×4正方形の周囲長: {base_perimeter}")
    
    # テスト位置: 上の境界
    test_position = (1, 3)  # 4×4正方形の上に1ピクセル追加
    
    print(f"\n位置{test_position}に1ピクセル追加:")
    
    # 追加前の状況
    print("追加前:")
    print(base_map[0:7, 1:7].cpu().numpy().astype(int))
    
    # 1ピクセル追加
    modified_map = base_map.clone()
    modified_map[test_position[0], test_position[1]] = 1.0
    
    print("追加後:")
    print(modified_map[0:7, 1:7].cpu().numpy().astype(int))
    
    # 追加後の周囲長計算
    new_perimeter = CPM.calc_total_perimeter_bincount(modified_map, 2)[1].item()
    change = new_perimeter - base_perimeter
    
    print(f"新周囲長: {new_perimeter}")
    print(f"変化: {change}")
    
    # 各ピクセルの寄与変化を詳細確認
    base_contrib = CPM.calc_perimeter_patch(base_map)
    new_contrib = CPM.calc_perimeter_patch(modified_map)
    contrib_diff = new_contrib - base_contrib
    
    print(f"\n周囲長寄与の変化:")
    print(contrib_diff[0:7, 1:7].cpu().numpy())
    
    # 影響を受けたピクセルを特定
    print(f"\n影響を受けたピクセル:")
    for i in range(8):
        for j in range(8):
            diff = contrib_diff[i, j].item()
            if abs(diff) > 0.1:
                print(f"  位置({i},{j}): 変化{diff:+.1f}")

def test_corner_vs_edge_addition():
    """角と辺での追加の違いをテスト"""
    print("\n=== 角と辺での追加の違いテスト ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 基準3×3正方形（小さくして詳細観察）
    base_map = torch.zeros((7, 7), device=device)
    base_map[2:5, 2:5] = 1.0
    
    base_perimeter = CPM.calc_total_perimeter_bincount(base_map, 2)[1].item()
    print(f"基準3×3正方形の周囲長: {base_perimeter}")
    
    # テスト位置
    test_cases = [
        ("上辺中央", (1, 3)),
        ("右辺中央", (3, 5)), 
        ("下辺中央", (5, 3)),
        ("左辺中央", (3, 1)),
        ("左上角", (1, 2)),
        ("右上角", (1, 4)),
        ("右下角", (5, 4)),
        ("左下角", (5, 2)),
    ]
    
    print(f"\n各位置での追加効果:")
    print("位置 | 新周囲長 | 変化 | 期待値 | 差")
    print("-" * 45)
    
    for name, pos in test_cases:
        modified_map = base_map.clone()
        modified_map[pos[0], pos[1]] = 1.0
        
        new_perimeter = CPM.calc_total_perimeter_bincount(modified_map, 2)[1].item()
        change = new_perimeter - base_perimeter
        
        # 理論的期待値
        if "角" in name:
            expected_change = 2  # 角に追加なら+2
        else:
            expected_change = 2  # 辺に追加でも+2
        
        diff = change - expected_change
        
        print(f"{name:8s} | {new_perimeter:7.1f} | {change:+4.0f} | {expected_change:+6d} | {diff:+3.0f}")

def analyze_patch_extraction():
    """パッチ抽出処理の詳細分析"""
    print("\n=== パッチ抽出処理の詳細分析 ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 小さなテストケース
    test_map = torch.zeros((5, 5), device=device)
    test_map[1:4, 1:4] = 1.0  # 3×3正方形
    
    print("テストマップ (3×3正方形):")
    print(test_map.cpu().numpy().astype(int))
    
    # パッチ抽出（実際の関数を使用）
    try:
        from cpm_torch.CPM_func import extract_patches_batched_channel
        
        # 3×3パッチを抽出
        patches = extract_patches_batched_channel(test_map.unsqueeze(-1), 3)
        print(f"\nパッチ形状: {patches.shape}")
        
        # 中央ピクセル（2,2）のパッチを確認
        center_patch = patches[2, 2, :, 0]
        print(f"中央ピクセル(2,2)のパッチ: {center_patch.cpu().numpy()}")
        
        # neighbors のインデックスを確認
        neighbors_indices = CPM.neighbors
        print(f"近傍インデックス: {neighbors_indices}")
        
        # 近傍の値
        neighbor_values = center_patch[neighbors_indices]
        print(f"近傍の値: {neighbor_values.cpu().numpy()}")
        
        # 中心の値
        center_value = center_patch[CPM.center_index]
        print(f"中心の値: {center_value.item()}")
        
        # 境界チェック
        is_boundary = neighbor_values != center_value
        print(f"境界チェック: {is_boundary.cpu().numpy()}")
        print(f"境界数: {torch.sum(is_boundary).item()}")
        
    except ImportError as e:
        print(f"パッチ抽出関数のインポートエラー: {e}")

def run_perimeter_bug_debug():
    """周囲長バグのデバッグを実行"""
    print("周囲長計算バグの詳細デバッグ\n")
    
    try:
        # 1. 周囲長計算の段階的デバッグ
        debug_perimeter_calculation_step_by_step()
        
        # 2. 1ピクセル追加の詳細テスト
        test_pixel_addition_detailed()
        
        # 3. 角と辺での追加の違い
        test_corner_vs_edge_addition()
        
        # 4. パッチ抽出処理の分析
        analyze_patch_extraction()
        
        print("\n" + "="*70)
        print("周囲長バグデバッグの結論")
        print("="*70)
        print("重要な発見:")
        print("1. 周囲長計算のロジック確認")
        print("2. パッチ抽出での境界処理")
        print("3. 局所的変化の正確性")
        print("4. 期待値との差異の原因")
        
        return True
        
    except Exception as e:
        print(f"デバッグエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_perimeter_bug_debug()
    sys.exit(0 if success else 1)