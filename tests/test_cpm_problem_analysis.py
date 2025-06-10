#!/usr/bin/env python3
"""
CPM細胞成長問題の根本原因分析
なぜ細胞が無制限に成長するのかを詳細に分析し、解決策を提示
"""

import torch
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('/app')

from cpm_torch.CPM import CPM, CPM_config

def analyze_energy_landscape():
    """エネルギーランドスケープの詳細分析"""
    print("=== エネルギーランドスケープ分析 ===")
    
    # 現在の問題のあるパラメータ
    config = CPM_config(l_A=1.0, l_L=1.0, A_0=15.0, L_0=15.0, T=1.0)
    
    areas = np.arange(1, 50)
    energies = []
    
    print("面積別エネルギー計算:")
    print("面積 | 周囲長(推定) | 面積エネルギー | 周囲長エネルギー | 総エネルギー")
    print("-" * 70)
    
    for area in areas:
        # 正方形に近い形状での周囲長推定
        if area == 1:
            perimeter = 4
        elif area <= 4:
            perimeter = 4 + (area - 1) * 2  # L字型やT字型
        else:
            # 正方形の場合
            side = int(np.sqrt(area))
            remainder = area - side**2
            if remainder == 0:
                perimeter = 4 * side
            else:
                # 不完全な正方形の場合の近似
                perimeter = 4 * side + 2
        
        area_energy = config.l_A * (area - config.A_0)**2
        perimeter_energy = config.l_L * (perimeter - config.L_0)**2
        total_energy = area_energy + perimeter_energy
        
        energies.append(total_energy)
        
        if area <= 20 or area % 5 == 0:
            print(f"{area:4d} | {perimeter:11.0f} | {area_energy:12.1f} | {perimeter_energy:14.1f} | {total_energy:10.1f}")
    
    # 最小エネルギーを見つける
    min_idx = np.argmin(energies)
    min_area = areas[min_idx]
    min_energy = energies[min_idx]
    
    print(f"\n最小エネルギー: 面積={min_area}, エネルギー={min_energy:.1f}")
    
    # 1ピクセルから目標面積への経路
    print(f"\n成長の動機:")
    initial_energy = energies[0]  # 面積=1
    target_energy = energies[14]  # 面積=15
    print(f"  初期状態(面積=1): エネルギー={initial_energy:.1f}")
    print(f"  目標状態(面積=15): エネルギー={target_energy:.1f}")
    print(f"  エネルギー差: {target_energy - initial_energy:.1f}")
    
    if target_energy < initial_energy:
        print("  → 成長によってエネルギーが大幅に下がるため、成長が必然的に発生")
    
    return areas, energies

def analyze_step_by_step_growth():
    """ステップバイステップでの成長エネルギー変化"""
    print("\n=== ステップ別成長分析 ===")
    
    config = CPM_config(l_A=1.0, l_L=1.0, A_0=15.0, L_0=15.0, T=1.0)
    
    # 各ステップでのエネルギー変化を計算
    print("成長ステップ | 現在面積 | 次面積 | ΔE | 成長確率")
    print("-" * 55)
    
    for current_area in range(1, 21):
        next_area = current_area + 1
        
        # 現在のエネルギー
        if current_area == 1:
            current_perimeter = 4
        else:
            current_perimeter = 4 * int(np.sqrt(current_area))
        
        current_energy = config.l_A * (current_area - config.A_0)**2 + config.l_L * (current_perimeter - config.L_0)**2
        
        # 成長後のエネルギー
        if next_area <= 4:
            next_perimeter = current_perimeter + 2
        else:
            next_perimeter = 4 * int(np.sqrt(next_area))
        
        next_energy = config.l_A * (next_area - config.A_0)**2 + config.l_L * (next_perimeter - config.L_0)**2
        
        delta_E = next_energy - current_energy
        probability = min(1.0, np.exp(-delta_E / config.T))
        
        print(f"{current_area:11d} | {current_area:7d} | {next_area:6d} | {delta_E:6.1f} | {probability:8.3f}")
        
        if current_area == 15:  # 目標面積に到達
            break

def propose_solutions():
    """問題の解決策を提案"""
    print("\n=== 解決策の提案 ===")
    
    solutions = [
        {
            "name": "解決策1: 小さい目標面積",
            "l_A": 1.0, "l_L": 1.0, "A_0": 4.0, "L_0": 8.0, "T": 1.0
        },
        {
            "name": "解決策2: 高い面積ペナルティ",
            "l_A": 5.0, "l_L": 1.0, "A_0": 15.0, "L_0": 15.0, "T": 1.0
        },
        {
            "name": "解決策3: 低い温度",
            "l_A": 1.0, "l_L": 1.0, "A_0": 15.0, "L_0": 15.0, "T": 0.1
        },
        {
            "name": "解決策4: バランス調整",
            "l_A": 2.0, "l_L": 1.0, "A_0": 9.0, "L_0": 12.0, "T": 0.5
        }
    ]
    
    for solution in solutions:
        name = solution.pop("name")
        config = CPM_config(**solution)
        
        # 1→2ピクセルの成長エネルギー変化
        current_energy = config.l_A * (1 - config.A_0)**2 + config.l_L * (4 - config.L_0)**2
        new_energy = config.l_A * (2 - config.A_0)**2 + config.l_L * (6 - config.L_0)**2
        delta_E = new_energy - current_energy
        probability = min(1.0, np.exp(-delta_E / config.T))
        
        print(f"\n{name}:")
        print(f"  パラメータ: l_A={config.l_A}, l_L={config.l_L}, A_0={config.A_0}, L_0={config.L_0}, T={config.T}")
        print(f"  成長エネルギー変化: ΔE={delta_E:.1f}")
        print(f"  成長確率: {probability:.3f}")
        
        if delta_E > 0 and probability < 0.5:
            print("  ✅ 成長が適切に抑制される")
        else:
            print("  ❌ まだ成長しやすい")

def test_corrected_simulation():
    """修正されたパラメータでのシミュレーションテスト"""
    print("\n=== 修正パラメータでのテスト ===")
    
    # 推奨パラメータ
    config = CPM_config(
        l_A=2.0,      # 面積ペナルティを強化
        l_L=1.0,      # 周囲長係数
        A_0=4.0,      # より小さい目標面積
        L_0=8.0,      # 対応する目標周囲長
        T=0.5,        # 低い温度
        size=(16, 16)
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    # シミュレーション実行
    tensor = torch.zeros((1, 16, 16, 3), device=device)
    tensor[0, 8, 8, 0] = 1.0
    
    print("修正パラメータでの成長テスト:")
    print("ステップ | 面積 | 周囲長 | 総エネルギー")
    print("-" * 40)
    
    for step in range(0, 51, 10):
        ids = tensor[0, :, :, 0]
        area = torch.sum(ids > 0).item()
        
        if area > 0:
            perimeter = CPM.calc_total_perimeter_bincount(ids, 2)[1].item()
        else:
            perimeter = 0
        
        total_energy = config.l_A * (area - config.A_0)**2 + config.l_L * (perimeter - config.L_0)**2
        print(f"{step:6d} | {area:4d} | {perimeter:6.1f} | {total_energy:10.1f}")
        
        # 5ステップ実行
        if step < 50:
            for _ in range(10):
                tensor = cpm.cpm_checkerboard_step_single_func(tensor)
    
    final_area = torch.sum(tensor[0, :, :, 0] > 0).item()
    growth_ratio = final_area / 1
    
    print(f"\n最終成長率: {growth_ratio:.1f}倍")
    if growth_ratio < 10:
        print("✅ 成長が適切に制御されています")
    else:
        print("❌ まだ過剰成長が発生しています")
    
    return final_area

def run_problem_analysis():
    """問題分析を実行"""
    print("CPM 細胞成長問題の根本原因分析\n")
    
    try:
        # 1. エネルギーランドスケープ分析
        areas, energies = analyze_energy_landscape()
        
        # 2. ステップ別成長分析
        analyze_step_by_step_growth()
        
        # 3. 解決策の提案
        propose_solutions()
        
        # 4. 修正パラメータでのテスト
        final_area = test_corrected_simulation()
        
        print("\n" + "="*60)
        print("結論:")
        print("="*60)
        print("問題の根本原因:")
        print("1. 目標面積A_0=15が初期面積1に対して大きすぎる")
        print("2. 1ピクセル→15ピクセルの成長で大幅なエネルギー低下が発生")
        print("3. エネルギー差が-280以上で、成長確率がほぼ1.0になる")
        print("4. 温度T=1.0では確率的制御が効かない")
        print("")
        print("推奨修正:")
        print("- A_0を4.0程度に下げる")
        print("- l_Aを2.0以上に上げる")  
        print("- Tを0.5以下に下げる")
        print("- または初期セルを目標サイズに近い形で配置する")
        
        return True
        
    except Exception as e:
        print(f"分析エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_problem_analysis()
    sys.exit(0 if success else 1)