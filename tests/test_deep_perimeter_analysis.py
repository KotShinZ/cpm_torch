#!/usr/bin/env python3
"""
周囲長エネルギー計算の詳細分析
根本的な原因を特定するための徹底調査
"""

import torch
import sys
import numpy as np
sys.path.append('/app')

from cpm_torch.CPM import CPM, CPM_config

def test_perimeter_calculation_accuracy():
    """周囲長計算の精度を詳細テスト"""
    print("=== 周囲長計算の精度詳細テスト ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 様々な形状での周囲長計算
    test_shapes = [
        ("1×1正方形", lambda: create_shape([(4, 4)], 10)),
        ("2×2正方形", lambda: create_square(2, 10)),
        ("3×3正方形", lambda: create_square(3, 10)),
        ("4×4正方形", lambda: create_square(4, 10)),
        ("5×5正方形", lambda: create_square(5, 10)),
        ("1×4長方形", lambda: create_rectangle(1, 4, 10)),
        ("2×3長方形", lambda: create_rectangle(2, 3, 10)),
        ("L字型16px", lambda: create_l_shape_16px(10)),
        ("不規則16px", lambda: create_irregular_16px(10)),
    ]
    
    print("形状 | 面積 | 計算周囲長 | 理論周囲長 | 差 | 誤差%")
    print("-" * 60)
    
    for name, shape_func in test_shapes:
        shape = shape_func().to(device)
        area = torch.sum(shape > 0).item()
        calculated_perimeter = CPM.calc_total_perimeter_bincount(shape, 2)[1].item()
        
        # 理論周囲長の計算
        if "正方形" in name:
            side = int(np.sqrt(area))
            theoretical_perimeter = 4 * side
        elif "長方形" in name:
            if "1×4" in name:
                theoretical_perimeter = 10  # 2*(1+4)
            elif "2×3" in name:
                theoretical_perimeter = 10  # 2*(2+3)
        elif name == "L字型16px":
            theoretical_perimeter = 22  # L字型の境界
        elif name == "不規則16px":
            theoretical_perimeter = "複雑"
        else:
            theoretical_perimeter = "不明"
        
        if isinstance(theoretical_perimeter, (int, float)):
            diff = calculated_perimeter - theoretical_perimeter
            error_percent = abs(diff) / theoretical_perimeter * 100
            print(f"{name:12s} | {area:4d} | {calculated_perimeter:9.1f} | {theoretical_perimeter:9.0f} | {diff:3.0f} | {error_percent:5.1f}%")
        else:
            print(f"{name:12s} | {area:4d} | {calculated_perimeter:9.1f} | {theoretical_perimeter:>9s} | {'N/A':>3s} | {'N/A':>5s}")

def analyze_local_perimeter_changes():
    """局所的周囲長変化の詳細分析"""
    print("\n=== 局所的周囲長変化の詳細分析 ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 4×4正方形を作成
    base_shape = create_square(4, 10).to(device)
    base_perimeter = CPM.calc_total_perimeter_bincount(base_shape, 2)[1].item()
    
    print(f"基準4×4正方形の周囲長: {base_perimeter}")
    print("\n各位置への1ピクセル追加による周囲長変化:")
    print("位置 | 新周囲長 | 変化 | 理論変化 | 差")
    print("-" * 45)
    
    # 境界位置への追加をテスト
    test_positions = [
        (3, 4), (3, 5), (3, 6), (3, 7),  # 上
        (8, 4), (8, 5), (8, 6), (8, 7),  # 下
        (4, 3), (5, 3), (6, 3), (7, 3),  # 左
        (4, 8), (5, 8), (6, 8), (7, 8),  # 右
    ]
    
    for pos in test_positions:
        # テスト形状を作成
        test_shape = base_shape.clone()
        test_shape[pos[0], pos[1]] = 1.0
        
        new_perimeter = CPM.calc_total_perimeter_bincount(test_shape, 2)[1].item()
        change = new_perimeter - base_perimeter
        
        # 理論的変化（境界に追加なら+2が期待値）
        theoretical_change = 2
        diff = change - theoretical_change
        
        print(f"{pos} | {new_perimeter:7.1f} | {change:+4.0f} | {theoretical_change:+8d} | {diff:+3.0f}")

def test_energy_calculation_components():
    """エネルギー計算の各成分を分離テスト"""
    print("\n=== エネルギー計算成分の分離テスト ===")
    
    config = CPM_config(l_A=1.0, l_L=1.0, A_0=16.0, L_0=16.0, T=1.0)
    
    # 16ピクセル周辺の様々な状態をテスト
    test_states = [
        ("理想4×4", 16, 16),
        ("17px境界", 17, 18),
        ("15px縮小", 15, 14),
        ("18px成長", 18, 18),
        ("不規則16px", 16, 20),
        ("不規則17px", 17, 22),
    ]
    
    print("状態 | 面積 | 周囲長 | 面積E | 周囲長E | 総E | 16との差")
    print("-" * 65)
    
    base_energy = 0  # 16px, 16perimeterの理想状態
    
    for name, area, perimeter in test_states:
        area_energy = config.l_A * (area - config.A_0)**2
        perimeter_energy = config.l_L * (perimeter - config.L_0)**2
        total_energy = area_energy + perimeter_energy
        diff_from_ideal = total_energy - base_energy
        
        print(f"{name:10s} | {area:4d} | {perimeter:6d} | {area_energy:5.0f} | {perimeter_energy:7.0f} | {total_energy:4.0f} | {diff_from_ideal:+6.0f}")

def analyze_transition_probabilities_detailed():
    """遷移確率の詳細分析"""
    print("\n=== 遷移確率の詳細分析 ===")
    
    config = CPM_config(l_A=1.0, l_L=1.0, A_0=16.0, L_0=16.0, T=1.0)
    
    print("16ピクセル近傍での各遷移の確率:")
    print("遷移 | 面積変化 | 周囲長変化 | 面積ΔE | 周囲長ΔE | 総ΔE | 確率 | 評価")
    print("-" * 80)
    
    # 16ピクセル状態からの各遷移
    transitions = [
        ("16→15", -1, -2, "縮小"),
        ("16→17", +1, +2, "成長"),
        ("15→16", +1, +2, "回復"),
        ("17→16", -1, -2, "縮小"),
        ("17→18", +1, +2, "継続成長"),
        ("18→17", -1, -2, "軽微縮小"),
    ]
    
    for name, area_change, perimeter_change, description in transitions:
        # エネルギー変化計算
        area_delta_E = config.l_A * (2 * area_change)  # 簡易計算
        perimeter_delta_E = config.l_L * (2 * perimeter_change)  # 簡易計算
        total_delta_E = area_delta_E + perimeter_delta_E
        
        # 確率計算
        if total_delta_E <= 0:
            probability = 1.0
        else:
            probability = np.exp(-total_delta_E / config.T)
        
        # 評価
        if probability > 0.1:
            evaluation = "高確率"
        elif probability > 0.01:
            evaluation = "中確率"
        else:
            evaluation = "低確率"
        
        print(f"{name:6s} | {area_change:+8d} | {perimeter_change:+10d} | {area_delta_E:6.0f} | {perimeter_delta_E:9.0f} | {total_delta_E:5.0f} | {probability:.3f} | {evaluation}")

def test_actual_cpm_step_by_step():
    """実際のCPMステップを詳細追跡"""
    print("\n=== 実際のCPMステップの詳細追跡 ===")
    
    config = CPM_config(l_A=1.0, l_L=1.0, A_0=16.0, L_0=16.0, T=1.0, size=(16, 16))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    # 4×4正方形から開始
    tensor = torch.zeros((1, 16, 16, 3), device=device)
    tensor[0, 6:10, 6:10, 0] = 1.0
    
    print("4×4正方形からの詳細ステップ追跡:")
    print("ステップ | 面積 | 周囲長 | 面積E | 周囲長E | 総E | 変化 | 16からの偏差")
    print("-" * 80)
    
    for step in range(0, 21):
        ids = tensor[0, :, :, 0]
        area = torch.sum(ids > 0).item()
        perimeter = CPM.calc_total_perimeter_bincount(ids, 2)[1].item()
        
        area_energy = config.l_A * (area - config.A_0)**2
        perimeter_energy = config.l_L * (perimeter - config.L_0)**2
        total_energy = area_energy + perimeter_energy
        
        if step == 0:
            prev_area = area
            change = "初期"
        else:
            change = f"{area - prev_area:+d}"
            prev_area = area
        
        deviation = abs(area - 16) + abs(perimeter - 16)
        
        print(f"{step:6d} | {area:4d} | {perimeter:6.1f} | {area_energy:5.0f} | {perimeter_energy:7.0f} | {total_energy:4.0f} | {change:4s} | {deviation:8.1f}")
        
        # 1ステップ実行
        if step < 20:
            tensor = cpm.cpm_checkerboard_step_single_func(tensor)

def create_square(size, map_size):
    """正方形を作成"""
    tensor = torch.zeros((map_size, map_size))
    start = (map_size - size) // 2
    tensor[start:start+size, start:start+size] = 1.0
    return tensor

def create_rectangle(width, height, map_size):
    """長方形を作成"""
    tensor = torch.zeros((map_size, map_size))
    start_w = (map_size - width) // 2
    start_h = (map_size - height) // 2
    tensor[start_h:start_h+height, start_w:start_w+width] = 1.0
    return tensor

def create_shape(positions, map_size):
    """指定位置に細胞を配置"""
    tensor = torch.zeros((map_size, map_size))
    for pos in positions:
        tensor[pos[0], pos[1]] = 1.0
    return tensor

def create_l_shape_16px(map_size):
    """16ピクセルのL字型を作成"""
    tensor = torch.zeros((map_size, map_size))
    # L字型: 4×4の正方形から右下の2×2を除く
    tensor[3:7, 3:7] = 1.0
    tensor[5:7, 5:7] = 0.0
    return tensor

def create_irregular_16px(map_size):
    """16ピクセルの不規則形状を作成"""
    tensor = torch.zeros((map_size, map_size))
    positions = [(3,3), (3,4), (3,5), (3,6), (4,3), (4,6), (5,3), (5,4), 
                 (5,5), (5,6), (6,3), (6,4), (6,5), (6,6), (7,4), (7,5)]
    for pos in positions:
        tensor[pos[0], pos[1]] = 1.0
    return tensor

def run_deep_perimeter_analysis():
    """周囲長の深い分析を実行"""
    print("周囲長エネルギー計算の徹底分析\n")
    
    try:
        # 1. 周囲長計算の精度テスト
        test_perimeter_calculation_accuracy()
        
        # 2. 局所的周囲長変化の分析
        analyze_local_perimeter_changes()
        
        # 3. エネルギー計算成分の分離
        test_energy_calculation_components()
        
        # 4. 遷移確率の詳細分析
        analyze_transition_probabilities_detailed()
        
        # 5. 実際のCPMステップ追跡
        test_actual_cpm_step_by_step()
        
        print("\n" + "="*80)
        print("周囲長分析の結論")
        print("="*80)
        print("重要な発見:")
        print("1. 周囲長計算の精度")
        print("2. 局所的変化の妥当性")
        print("3. エネルギー成分の相対的影響")
        print("4. 遷移確率の実際の分布")
        print("5. ステップごとの詳細な変化")
        
        return True
        
    except Exception as e:
        print(f"分析エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_deep_perimeter_analysis()
    sys.exit(0 if success else 1)