#!/usr/bin/env python3
"""
細胞成長問題のデバッグテスト
cpm_one_cell_test.ipynbで観察される異常な細胞成長の原因を調査
"""

import torch
import sys
import os
sys.path.append('/app')

from cpm_torch.CPM import CPM, CPM_config
from cpm_torch.CPM_Image import *
import matplotlib.pyplot as plt

def test_single_cell_growth():
    """1細胞の成長パターンを詳細に調査"""
    print("=== 単一細胞成長テスト ===")
    
    # 同じ設定を使用
    config = CPM_config(
        l_A=1.0,   # 面積エネルギー係数
        l_L=1.0,   # 周囲長エネルギー係数
        A_0=15.0,  # 目標面積
        L_0=15.0,  # 目標周囲長
        T=1.0,     # 温度
        size=(16, 16)
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    # 初期状態: 1ピクセルの細胞
    tensor = torch.zeros((1, 16, 16, 3), device=device)
    tensor[0, 8, 8, 0] = 1.0  # 中央に1ピクセル細胞
    
    print(f"初期設定:")
    print(f"  目標面積 A_0: {config.A_0}")
    print(f"  目標周囲長 L_0: {config.L_0}")
    print(f"  温度 T: {config.T}")
    print(f"  面積係数 l_A: {config.l_A}")
    print(f"  周囲長係数 l_L: {config.l_L}")
    
    # 成長過程を段階的に観察
    growth_data = []
    steps_to_check = [0, 1, 5, 10, 20, 50, 100]
    
    for step in range(101):
        # 現在の細胞面積と周囲長を計算
        ids = tensor[0, :, :, 0]
        cell_area = torch.sum(ids > 0).item()
        
        # 周囲長計算
        if cell_area > 0:
            perimeter = CPM.calc_total_perimeter_bincount(ids, 2)[1].item()
        else:
            perimeter = 0
        
        # データ記録
        if step in steps_to_check:
            growth_data.append({
                'step': step,
                'area': cell_area,
                'perimeter': perimeter,
                'area_energy': config.l_A * (cell_area - config.A_0)**2,
                'perimeter_energy': config.l_L * (perimeter - config.L_0)**2
            })
            print(f"ステップ {step:3d}: 面積={cell_area:2d}, 周囲長={perimeter:4.1f}, "
                  f"面積エネルギー={config.l_A * (cell_area - config.A_0)**2:6.1f}, "
                  f"周囲長エネルギー={config.l_L * (perimeter - config.L_0)**2:6.1f}")
        
        # 1ステップ実行
        if step < 100:
            tensor = cpm.cpm_checkerboard_step_single_func(tensor)
    
    return growth_data

def test_energy_analysis():
    """エネルギー関数の分析"""
    print("\n=== エネルギー関数分析 ===")
    
    config = CPM_config(l_A=1.0, l_L=1.0, A_0=15.0, L_0=15.0, T=1.0)
    
    # 異なる面積でのエネルギーを計算
    areas = range(1, 30)
    area_energies = []
    
    for area in areas:
        # 正方形に近い形状の周囲長を推定
        if area == 1:
            perimeter = 4  # 1ピクセルの周囲長
        else:
            # 正方形の場合の周囲長の近似
            side = int(area**0.5)
            perimeter = 4 * side
        
        area_energy = config.l_A * (area - config.A_0)**2
        perimeter_energy = config.l_L * (perimeter - config.L_0)**2
        total_energy = area_energy + perimeter_energy
        
        area_energies.append({
            'area': area,
            'perimeter': perimeter,
            'area_energy': area_energy,
            'perimeter_energy': perimeter_energy,
            'total_energy': total_energy
        })
    
    # 最小エネルギーを見つける
    min_energy = min(area_energies, key=lambda x: x['total_energy'])
    print(f"最小エネルギー状態:")
    print(f"  面積: {min_energy['area']}")
    print(f"  推定周囲長: {min_energy['perimeter']}")
    print(f"  総エネルギー: {min_energy['total_energy']:.1f}")
    
    # 面積1(初期状態)のエネルギー
    initial_energy = area_energies[0]
    print(f"\n初期状態 (面積=1):")
    print(f"  面積エネルギー: {initial_energy['area_energy']:.1f}")
    print(f"  周囲長エネルギー: {initial_energy['perimeter_energy']:.1f}")
    print(f"  総エネルギー: {initial_energy['total_energy']:.1f}")
    
    # 目標面積でのエネルギー
    target_idx = 14  # A_0=15, index=14
    if target_idx < len(area_energies):
        target_energy = area_energies[target_idx]
        print(f"\n目標状態 (面積=15):")
        print(f"  面積エネルギー: {target_energy['area_energy']:.1f}")
        print(f"  周囲長エネルギー: {target_energy['perimeter_energy']:.1f}")
        print(f"  総エネルギー: {target_energy['total_energy']:.1f}")
    
    return area_energies

def test_transition_probabilities():
    """遷移確率の分析"""
    print("\n=== 遷移確率分析 ===")
    
    config = CPM_config(l_A=1.0, l_L=1.0, A_0=15.0, L_0=15.0, T=1.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    # 1ピクセル細胞の場合の成長確率を計算
    tensor = torch.zeros((1, 8, 8, 3), device=device)
    tensor[0, 4, 4, 0] = 1.0  # 中央に1ピクセル細胞
    
    # 1ステップでの変化を観察
    print("1ピクセル細胞からの成長確率:")
    
    # 現在のエネルギー状態
    current_area = 1
    current_perimeter = 4
    current_energy = config.l_A * (current_area - config.A_0)**2 + config.l_L * (current_perimeter - config.L_0)**2
    
    # 1ピクセル成長した場合のエネルギー変化
    new_area = 2
    new_perimeter = 6  # L字型の場合
    new_energy = config.l_A * (new_area - config.A_0)**2 + config.l_L * (new_perimeter - config.L_0)**2
    
    delta_E = new_energy - current_energy
    probability = torch.exp(torch.tensor(-delta_E / config.T)).item()
    
    print(f"  現在のエネルギー: {current_energy:.1f}")
    print(f"  成長後のエネルギー: {new_energy:.1f}")
    print(f"  エネルギー変化 ΔE: {delta_E:.1f}")
    print(f"  成長確率 exp(-ΔE/T): {probability:.3f}")
    
    if delta_E < 0:
        print("  → エネルギーが下がるため、成長が非常に起こりやすい")
    else:
        print("  → エネルギーが上がるため、成長が抑制される")
    
    return probability

def test_parameter_sensitivity():
    """パラメータ感度分析"""
    print("\n=== パラメータ感度分析 ===")
    
    # 異なるパラメータでの挙動を比較
    test_configs = [
        {"name": "現在設定", "l_A": 1.0, "l_L": 1.0, "A_0": 15.0, "L_0": 15.0, "T": 1.0},
        {"name": "高い面積ペナルティ", "l_A": 10.0, "l_L": 1.0, "A_0": 15.0, "L_0": 15.0, "T": 1.0},
        {"name": "低い温度", "l_A": 1.0, "l_L": 1.0, "A_0": 15.0, "L_0": 15.0, "T": 0.1},
        {"name": "小さい目標面積", "l_A": 1.0, "l_L": 1.0, "A_0": 5.0, "L_0": 9.0, "T": 1.0},
    ]
    
    for test_config in test_configs:
        name = test_config.pop("name")
        config = CPM_config(**test_config, size=(16, 16))
        
        # 1→2ピクセルの成長エネルギー変化を計算
        current_energy = config.l_A * (1 - config.A_0)**2 + config.l_L * (4 - config.L_0)**2
        new_energy = config.l_A * (2 - config.A_0)**2 + config.l_L * (6 - config.L_0)**2
        delta_E = new_energy - current_energy
        probability = torch.exp(torch.tensor(-delta_E / config.T)).item()
        
        print(f"{name}: ΔE={delta_E:6.1f}, P={probability:.3f}")

def run_growth_debug_tests():
    """細胞成長デバッグテストを実行"""
    print("CPM 細胞成長問題デバッグテスト開始\n")
    
    try:
        # 1. 実際の成長パターンを観察
        growth_data = test_single_cell_growth()
        
        # 2. エネルギー関数の分析
        energy_data = test_energy_analysis()
        
        # 3. 遷移確率の分析
        probability = test_transition_probabilities()
        
        # 4. パラメータ感度分析
        test_parameter_sensitivity()
        
        print("\n=== 診断結果 ===")
        
        # 成長傾向の判定
        initial_area = growth_data[0]['area']
        final_area = growth_data[-1]['area']
        growth_rate = final_area / initial_area
        
        print(f"成長率: {growth_rate:.1f}倍 ({initial_area} → {final_area})")
        
        if growth_rate > 5:
            print("❌ 異常な成長が確認されました")
            
            # 原因分析
            if probability > 0.5:
                print("原因: 成長のエネルギー変化が負またはゼロに近く、成長が非常に起こりやすい")
                print("推奨: 目標面積A_0を小さくするか、面積係数l_Aを大きくする")
            else:
                print("原因: 温度Tが高すぎて、エネルギー的に不利な遷移も頻繁に起こる")
                print("推奨: 温度Tを下げる")
        else:
            print("✅ 成長は適切な範囲内です")
        
        return True
        
    except Exception as e:
        print(f"テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_growth_debug_tests()
    sys.exit(0 if success else 1)