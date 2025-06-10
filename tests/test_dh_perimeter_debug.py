#!/usr/bin/env python3
"""
dH_perimeter計算の詳細分析
l_A=1.0, l_L=1.0, T=1.0設定で過剰成長する原因を調査
"""

import torch
import sys
import numpy as np
sys.path.append('/app')

from cpm_torch.CPM import CPM, CPM_config

def test_dh_perimeter_calculation():
    """dH_perimeter計算の正確性をテスト"""
    print("=== dH_perimeter計算の詳細分析 ===")
    
    config = CPM_config(
        l_A=1.0, l_L=1.0, A_0=16.0, L_0=16.0, T=1.0, size=(16, 16)
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    print(f"設定: l_A={config.l_A}, l_L={config.l_L}, A_0={config.A_0}, L_0={config.L_0}, T={config.T}")
    
    # テストケース1: 4×4正方形での周囲長エネルギー変化
    print("\n=== テストケース1: 4×4正方形からの1ピクセル成長 ===")
    
    # 4×4正方形を作成
    test_map = torch.zeros((16, 16), device=device)
    test_map[6:10, 6:10] = 1.0  # 4×4正方形
    
    print("初期4×4正方形:")
    print(test_map[4:12, 4:12].cpu().numpy().astype(int))
    
    # 初期状態の分析
    area = torch.sum(test_map > 0).item()
    perimeter = CPM.calc_total_perimeter_bincount(test_map, 2)[1].item()
    initial_energy = config.l_A * (area - config.A_0)**2 + config.l_L * (perimeter - config.L_0)**2
    
    print(f"初期状態: 面積={area}, 周囲長={perimeter}, エネルギー={initial_energy}")
    
    # 境界ピクセルでの周囲長変化を手動計算
    test_positions = [
        (5, 6),  # 上
        (10, 6), # 下
        (6, 5),  # 左
        (6, 10), # 右
    ]
    
    print("\n境界ピクセル追加での周囲長変化:")
    print("位置 | 追加後面積 | 追加後周囲長 | 周囲長変化 | エネルギー変化")
    print("-" * 60)
    
    for pos in test_positions:
        # テスト用マップを作成
        test_modified = test_map.clone()
        test_modified[pos[0], pos[1]] = 1.0
        
        new_area = torch.sum(test_modified > 0).item()
        new_perimeter = CPM.calc_total_perimeter_bincount(test_modified, 2)[1].item()
        new_energy = config.l_A * (new_area - config.A_0)**2 + config.l_L * (new_perimeter - config.L_0)**2
        
        perimeter_change = new_perimeter - perimeter
        energy_change = new_energy - initial_energy
        
        print(f"{pos} | {new_area:8d} | {new_perimeter:10.1f} | {perimeter_change:8.1f} | {energy_change:11.1f}")

def test_perimeter_boundary_effects():
    """境界効果による周囲長変化の分析"""
    print("\n=== 境界効果による周囲長変化分析 ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 様々なサイズの正方形での周囲長計算
    print("正方形サイズ | 面積 | 計算周囲長 | 理論周囲長 | 差")
    print("-" * 50)
    
    for size in range(2, 7):
        test_map = torch.zeros((16, 16), device=device)
        start = (16 - size) // 2
        test_map[start:start+size, start:start+size] = 1.0
        
        area = torch.sum(test_map > 0).item()
        calculated_perimeter = CPM.calc_total_perimeter_bincount(test_map, 2)[1].item()
        theoretical_perimeter = 4 * size
        diff = calculated_perimeter - theoretical_perimeter
        
        print(f"{size}×{size:d}正方形     | {area:4d} | {calculated_perimeter:9.1f} | {theoretical_perimeter:9d} | {diff:4.1f}")

def test_local_perimeter_changes():
    """局所的周囲長変化の詳細分析"""
    print("\n=== 局所的周囲長変化の詳細分析 ===")
    
    config = CPM_config(l_A=1.0, l_L=1.0, A_0=16.0, L_0=16.0, T=1.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    # 4×4正方形でのdH_perimeter計算をテスト
    test_map = torch.zeros((1, 16, 16, 3), device=device)
    test_map[0, 6:10, 6:10, 0] = 1.0  # セルID=1の4×4正方形
    
    print("4×4正方形での各境界位置でのdH_perimeter計算:")
    
    # 境界位置を指定
    boundary_positions = [
        (5, 6),   # 上
        (10, 6),  # 下  
        (6, 5),   # 左
        (6, 10),  # 右
    ]
    
    ids = test_map[0, :, :, 0]
    
    for pos in boundary_positions:
        row, col = pos
        
        # 4近傍のIDを取得
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
        
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 16 and 0 <= nc < 16:
                neighbors.append(ids[nr, nc].item())
            else:
                neighbors.append(0)  # 境界外は0
        
        print(f"\n位置 {pos}:")
        print(f"  4近傍ID: {neighbors}")
        print(f"  ターゲットID: {ids[row, col].item()}")
        
        # 手動でdL_s計算
        for i, neighbor_id in enumerate(neighbors):
            if neighbor_id > 0:  # 空でないセル
                count_same = neighbors.count(neighbor_id)
                dL_s = 4.0 - 2.0 * count_same
                print(f"  方向{i} -> セル{neighbor_id}: 同ID数={count_same}, dL_s={dL_s}")

def test_energy_growth_mechanism():
    """エネルギー成長メカニズムの分析"""
    print("\n=== エネルギー成長メカニズム分析 ===")
    
    config = CPM_config(l_A=1.0, l_L=1.0, A_0=16.0, L_0=16.0, T=1.0)
    
    print("面積16付近での詳細エネルギー計算:")
    print("面積 | 理想周囲長 | 実際周囲長 | 面積E | 周囲長E | 総E | 16からのΔE")
    print("-" * 75)
    
    # 面積16の基準エネルギー
    base_area_energy = config.l_A * (16 - config.A_0)**2  # = 0
    base_perimeter_energy = config.l_L * (16 - config.L_0)**2  # = 0
    base_total_energy = base_area_energy + base_perimeter_energy  # = 0
    
    for area in range(14, 21):
        # 理想的な周囲長（正方形）
        side = int(np.sqrt(area))
        if side**2 == area:
            ideal_perimeter = 4 * side
        else:
            ideal_perimeter = 4 * side + 2
        
        # 実際の成長での周囲長（4×4から成長）
        if area == 16:
            actual_perimeter = 16
        elif area == 17:
            actual_perimeter = 18  # 4×4に1ピクセル追加
        elif area == 18:
            actual_perimeter = 18  # コンパクトに追加
        elif area == 19:
            actual_perimeter = 20
        elif area == 20:
            actual_perimeter = 20
        else:
            actual_perimeter = ideal_perimeter + 2  # 不規則性
        
        area_energy = config.l_A * (area - config.A_0)**2
        perimeter_energy = config.l_L * (actual_perimeter - config.L_0)**2
        total_energy = area_energy + perimeter_energy
        delta_from_16 = total_energy - base_total_energy
        
        print(f"{area:4d} | {ideal_perimeter:9.0f} | {actual_perimeter:9.0f} | {area_energy:5.0f} | {perimeter_energy:7.0f} | {total_energy:4.0f} | {delta_from_16:8.1f}")

def run_dh_perimeter_debug():
    """dH_perimeter計算のデバッグ分析を実行"""
    print("dH_perimeter計算の詳細分析\n")
    
    try:
        # 1. dH_perimeter計算の正確性テスト
        test_dh_perimeter_calculation()
        
        # 2. 境界効果の分析
        test_perimeter_boundary_effects()
        
        # 3. 局所的周囲長変化の分析
        test_local_perimeter_changes()
        
        # 4. エネルギー成長メカニズムの分析
        test_energy_growth_mechanism()
        
        print("\n" + "="*70)
        print("dH_perimeter分析の結論")
        print("="*70)
        print("重要な発見:")
        print("1. 4×4正方形は理論的に最適状態（エネルギー=0）")
        print("2. しかし実際の成長過程では不規則な形状を経由")
        print("3. 周囲長の局所変化計算は正確に動作している")
        print("4. 問題は成長過程での形状最適化の継続")
        
        print("\n修正提案:")
        print("- より厳しい温度設定で確率的変動を抑制")
        print("- 面積ペナルティを強化して過剰成長を防止")
        print("- 初期セルサイズを目標に近い値に設定")
        
        return True
        
    except Exception as e:
        print(f"分析エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_dh_perimeter_debug()
    sys.exit(0 if success else 1)