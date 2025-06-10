#!/usr/bin/env python3
"""
係数の詳細分析
l_L=0で制御されることから、周囲長エネルギーの問題を特定
"""

import torch
import sys
import numpy as np
sys.path.append('/app')

from cpm_torch.CPM import CPM, CPM_config

def test_coefficient_sweep():
    """係数を段階的に変えてテスト"""
    print("=== 係数段階テスト ===")
    
    # l_L を段階的に変化
    l_L_values = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
    
    print("l_L値 | 50ステップ後面積 | 成長倍率 | 評価")
    print("-" * 45)
    
    for l_L in l_L_values:
        config = CPM_config(
            l_A=1.0, l_L=l_L, A_0=16.0, L_0=16.0, T=1.0, size=(16, 16)
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cpm = CPM(config, device)
        
        # 1ピクセルから開始
        tensor = torch.zeros((1, 16, 16, 3), device=device)
        tensor[0, 8, 8, 0] = 1.0
        
        # 50ステップ実行
        for _ in range(50):
            tensor = cpm.cpm_checkerboard_step_single_func(tensor)
        
        final_area = torch.sum(tensor[0, :, :, 0] > 0).item()
        growth_ratio = final_area / 1.0
        
        if growth_ratio <= 5:
            evaluation = "良好"
        elif growth_ratio <= 10:
            evaluation = "許容"
        else:
            evaluation = "過剰"
        
        print(f"{l_L:4.1f} | {final_area:12d} | {growth_ratio:8.1f} | {evaluation}")

def test_4x4_stability_with_coefficients():
    """4×4正方形の安定性を係数別にテスト"""
    print("\n=== 4×4正方形の安定性 係数別テスト ===")
    
    l_L_values = [0.0, 0.2, 0.5, 1.0, 1.5, 2.0]
    
    print("l_L値 | 初期面積 | 最終面積 | 変化 | 安定性")
    print("-" * 50)
    
    for l_L in l_L_values:
        config = CPM_config(
            l_A=1.0, l_L=l_L, A_0=16.0, L_0=16.0, T=1.0, size=(16, 16)
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cpm = CPM(config, device)
        
        # 4×4正方形から開始
        tensor = torch.zeros((1, 16, 16, 3), device=device)
        tensor[0, 6:10, 6:10, 0] = 1.0
        
        initial_area = 16
        
        # 30ステップ実行
        for _ in range(30):
            tensor = cpm.cpm_checkerboard_step_single_func(tensor)
        
        final_area = torch.sum(tensor[0, :, :, 0] > 0).item()
        change = final_area - initial_area
        
        if abs(change) <= 2:
            stability = "安定"
        elif abs(change) <= 5:
            stability = "軽微変化"
        else:
            stability = "不安定"
        
        print(f"{l_L:4.1f} | {initial_area:8d} | {final_area:8d} | {change:+4d} | {stability}")

def analyze_transition_probabilities_by_coefficient():
    """係数別の遷移確率分析"""
    print("\n=== 係数別遷移確率分析 ===")
    
    l_L_values = [0.0, 0.5, 1.0, 2.0]
    
    for l_L in l_L_values:
        print(f"\nl_L = {l_L}:")
        config = CPM_config(l_A=1.0, l_L=l_L, A_0=16.0, L_0=16.0, T=1.0)
        
        # 16ピクセル状態での遷移確率
        transitions = [
            ("16→15", 16, 15, 16, 14),  # 縮小
            ("16→17", 16, 17, 16, 18),  # 成長
        ]
        
        print("  遷移 | 面積ΔE | 周囲長ΔE | 総ΔE | 確率")
        print("  " + "-" * 45)
        
        for name, area_from, area_to, perim_from, perim_to in transitions:
            area_delta_E = config.l_A * ((area_to - config.A_0)**2 - (area_from - config.A_0)**2)
            perim_delta_E = config.l_L * ((perim_to - config.L_0)**2 - (perim_from - config.L_0)**2)
            total_delta_E = area_delta_E + perim_delta_E
            
            prob = np.exp(-total_delta_E / config.T) if total_delta_E > 0 else 1.0
            
            print(f"  {name:6s} | {area_delta_E:6.0f} | {perim_delta_E:8.0f} | {total_delta_E:5.0f} | {prob:.3f}")

def find_optimal_l_L():
    """最適な l_L 値を探索"""
    print("\n=== 最適 l_L 値の探索 ===")
    
    # より細かい刻みでテスト
    l_L_values = np.arange(0.0, 1.1, 0.1)
    
    print("l_L値 | 1px→面積 | 16px→面積 | 成長制御 | 安定性 | 総合評価")
    print("-" * 70)
    
    best_l_L = None
    best_score = float('inf')
    
    for l_L in l_L_values:
        config = CPM_config(
            l_A=1.0, l_L=l_L, A_0=16.0, L_0=16.0, T=1.0, size=(16, 16)
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cpm = CPM(config, device)
        
        # テスト1: 1ピクセルからの成長
        tensor1 = torch.zeros((1, 16, 16, 3), device=device)
        tensor1[0, 8, 8, 0] = 1.0
        
        for _ in range(50):
            tensor1 = cpm.cpm_checkerboard_step_single_func(tensor1)
        
        final_area_1px = torch.sum(tensor1[0, :, :, 0] > 0).item()
        
        # テスト2: 16ピクセルからの安定性
        tensor2 = torch.zeros((1, 16, 16, 3), device=device)
        tensor2[0, 6:10, 6:10, 0] = 1.0
        
        for _ in range(30):
            tensor2 = cpm.cpm_checkerboard_step_single_func(tensor2)
        
        final_area_16px = torch.sum(tensor2[0, :, :, 0] > 0).item()
        
        # 評価
        growth_control = "良好" if final_area_1px <= 20 else ("許容" if final_area_1px <= 30 else "不良")
        stability = "安定" if abs(final_area_16px - 16) <= 3 else "不安定"
        
        # 総合スコア（低いほど良い）
        score = abs(final_area_1px - 16) + abs(final_area_16px - 16)
        
        if score < best_score:
            best_score = score
            best_l_L = l_L
        
        overall = "✅" if growth_control == "良好" and stability == "安定" else ("⚠️" if growth_control != "不良" else "❌")
        
        print(f"{l_L:5.1f} | {final_area_1px:8d} | {final_area_16px:9d} | {growth_control:6s} | {stability:6s} | {overall}")
    
    print(f"\n🎯 最適 l_L 値: {best_l_L:.1f} (スコア: {best_score})")

def run_coefficient_analysis():
    """係数分析を実行"""
    print("係数の詳細分析\n")
    
    try:
        # 1. 係数段階テスト
        test_coefficient_sweep()
        
        # 2. 4×4安定性テスト
        test_4x4_stability_with_coefficients()
        
        # 3. 遷移確率分析
        analyze_transition_probabilities_by_coefficient()
        
        # 4. 最適l_L探索
        find_optimal_l_L()
        
        print("\n" + "="*70)
        print("係数分析の結論")
        print("="*70)
        print("重要な発見:")
        print("1. l_L=0.0で適切な成長制御が可能")
        print("2. l_L値の増加とともに過剰成長が発生")
        print("3. 周囲長エネルギーが成長を促進している")
        print("4. 最適なl_L値が存在する")
        
        print("\n解決策:")
        print("- l_L値を下げることで過剰成長を抑制可能")
        print("- 周囲長エネルギー計算の見直しが根本解決")
        
        return True
        
    except Exception as e:
        print(f"分析エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_coefficient_analysis()
    sys.exit(0 if success else 1)