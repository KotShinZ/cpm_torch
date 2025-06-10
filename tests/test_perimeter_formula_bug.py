#!/usr/bin/env python3
"""
周囲長エネルギー公式のバグ調査
不規則→規則的遷移が過度に有利になる原因を特定
"""

import torch
import sys
import numpy as np
sys.path.append('/app')

from cpm_torch.CPM import CPM, CPM_config

def analyze_perimeter_energy_formula():
    """周囲長エネルギー公式の詳細分析"""
    print("=== 周囲長エネルギー公式の詳細分析 ===")
    
    # 16ピクセル、周囲長16の理想状態
    L_current = 16
    L_0 = 16
    l_L = 1.0
    
    print("現在の公式: ΔH_L = l_L * [2 * (L - L_0) * dL + (dL)^2]")
    print(f"理想状態: L = {L_current}, L_0 = {L_0}")
    
    # 様々なdL値での計算
    dL_values = [-4, -2, -1, 0, 1, 2, 4]
    
    print("\ndL | term1: 2*(L-L_0)*dL | term2: dL^2 | 総ΔH_L | 解釈")
    print("-" * 65)
    
    for dL in dL_values:
        term1 = 2.0 * (L_current - L_0) * dL  # = 0 * dL = 0
        term2 = dL**2
        total_delta_H = l_L * (term1 + term2)
        
        if dL < 0:
            interpretation = "境界減少"
        elif dL > 0:
            interpretation = "境界増加"
        else:
            interpretation = "変化なし"
        
        print(f"{dL:2d} | {term1:15.0f} | {term2:7.0f} | {total_delta_H:7.0f} | {interpretation}")
    
    print("\n❌ 問題発見: dL^2項により、すべてのdL≠0でエネルギーが増加")
    print("→ 理想状態からの任意の変化が不利になり、実質的に動けなくなる")

def test_non_ideal_state_transitions():
    """非理想状態での遷移を分析"""
    print("\n=== 非理想状態での遷移分析 ===")
    
    # 不規則な17ピクセル状態（周囲長22）
    L_current = 22  # 不規則形状
    L_0 = 16
    l_L = 1.0
    
    print(f"不規則17ピクセル状態: L = {L_current}, L_0 = {L_0}")
    print(f"L - L_0 = {L_current - L_0}")
    
    dL_values = [-4, -2, -1, 0, 1, 2, 4]
    
    print("\ndL | term1: 2*(L-L_0)*dL | term2: dL^2 | 総ΔH_L | エネルギー方向")
    print("-" * 70)
    
    for dL in dL_values:
        term1 = 2.0 * (L_current - L_0) * dL  # = 2 * 6 * dL = 12 * dL
        term2 = dL**2
        total_delta_H = l_L * (term1 + term2)
        
        if total_delta_H < 0:
            energy_direction = "エネルギー減少（有利）"
        elif total_delta_H > 0:
            energy_direction = "エネルギー増加（不利）"
        else:
            energy_direction = "変化なし"
        
        print(f"{dL:2d} | {term1:15.0f} | {term2:7.0f} | {total_delta_H:7.0f} | {energy_direction}")
    
    print("\n✅ 発見: dL < 0（境界減少）で大幅にエネルギーが減少")
    print("→ 不規則形状からの「形状改善」が過度に有利")

def analyze_dL_calculation_bug():
    """dL計算のバグを分析"""
    print("\n=== dL計算のバグ分析 ===")
    
    print("現在の実装:")
    print("dL_s = 4 - 2 * (同じIDの近傍数)")
    print("dL_t = -4 + 2 * (同じIDの近傍数)")
    
    # シミュレーション: 4×4正方形の境界での遷移
    print("\n4×4正方形境界での遷移シミュレーション:")
    
    # ケース1: 上端中央への成長 (細胞外→細胞内)
    print("\nケース1: 上端中央への成長")
    print("近傍ID: [0, 1, 0, 0] (上, 下, 左, 右)")
    print("遷移: 0 → 1")
    
    # dL_s (成長するセル=1)
    neighbors_target = [0, 1, 0, 0]  # ターゲット位置の近傍
    same_id_count_s = sum(1 for nid in neighbors_target if nid == 1)
    dL_s = 4 - 2 * same_id_count_s
    print(f"dL_s = 4 - 2 * {same_id_count_s} = {dL_s}")
    
    # dL_t (元の空セル=0)
    same_id_count_t = sum(1 for nid in neighbors_target if nid == 0)
    dL_t = -4 + 2 * same_id_count_t
    print(f"dL_t = -4 + 2 * {same_id_count_t} = {dL_t}")
    
    # ケース2: 内部から外部への縮小 (細胞内→細胞外)
    print("\nケース2: 内部境界での縮小")
    print("近傍ID: [0, 1, 1, 1] (境界セル)")
    print("遷移: 1 → 0")
    
    neighbors_target2 = [0, 1, 1, 1]
    same_id_count_s2 = sum(1 for nid in neighbors_target2 if nid == 0)
    dL_s2 = 4 - 2 * same_id_count_s2
    print(f"dL_s = 4 - 2 * {same_id_count_s2} = {dL_s2}")
    
    same_id_count_t2 = sum(1 for nid in neighbors_target2 if nid == 1)
    dL_t2 = -4 + 2 * same_id_count_t2
    print(f"dL_t = -4 + 2 * {same_id_count_t2} = {dL_t2}")
    
    print("\n🔍 重要な観察:")
    print("- 成長時: dL_s > 0, dL_t < 0")
    print("- 縮小時: dL_s > 0, dL_t < 0")
    print("- 両方向とも同様のdL値 → エネルギー計算で差が生じる原因は他にある")

def test_actual_perimeter_calculation():
    """実際の周囲長計算をステップ別にテスト"""
    print("\n=== 実際の周囲長計算のステップ別テスト ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 4×4正方形
    base_map = torch.zeros((8, 8), device=device)
    base_map[2:6, 2:6] = 1.0
    
    base_perimeter = CPM.calc_total_perimeter_bincount(base_map, 2)[1].item()
    print(f"4×4正方形の周囲長: {base_perimeter}")
    
    # 1ピクセル追加のテスト
    test_positions = [
        ((1, 3), "上端中央"),
        ((1, 2), "上端角"),
        ((2, 1), "左端中央"),
    ]
    
    for pos, desc in test_positions:
        modified_map = base_map.clone()
        modified_map[pos[0], pos[1]] = 1.0
        
        new_perimeter = CPM.calc_total_perimeter_bincount(modified_map, 2)[1].item()
        change = new_perimeter - base_perimeter
        
        print(f"\n{desc} {pos}:")
        print(f"  新周囲長: {new_perimeter}")
        print(f"  変化: {change:+.0f}")
        
        # 理論値との比較
        expected_change = 2  # 1ピクセル追加で通常+2
        diff = change - expected_change
        print(f"  理論値: +{expected_change}")
        print(f"  差異: {diff:+.0f}")
        
        if abs(diff) > 0.1:
            print(f"  ❌ 異常: 理論値との差が大きい")
        else:
            print(f"  ✅ 正常: 理論値と一致")

def debug_energy_formula_fix():
    """エネルギー公式の修正案をテスト"""
    print("\n=== エネルギー公式の修正案テスト ===")
    
    print("問題のある現在の公式:")
    print("ΔH_L = l_L * [2 * (L - L_0) * dL + (dL)^2]")
    
    print("\n修正案1: dL^2項を除去")
    print("ΔH_L = l_L * [2 * (L - L_0) * dL]")
    
    print("\n修正案2: 絶対値を使用")
    print("ΔH_L = l_L * [2 * |L - L_0| * |dL|]")
    
    print("\n修正案3: 符号を考慮した重み")
    print("ΔH_L = l_L * [2 * (L - L_0) * dL] (dL^2項なし)")
    
    # 各修正案での計算例
    L_values = [16, 18, 22]  # 理想、軽微不規則、大幅不規則
    L_0 = 16
    l_L = 1.0
    dL = -2  # 境界改善方向
    
    print(f"\n例: dL = {dL} (境界改善)")
    print("L | 現在の公式 | 修正案1 | 修正案2 | 修正案3")
    print("-" * 50)
    
    for L in L_values:
        current_formula = l_L * (2 * (L - L_0) * dL + dL**2)
        fix1 = l_L * (2 * (L - L_0) * dL)
        fix2 = l_L * (2 * abs(L - L_0) * abs(dL))
        fix3 = l_L * (2 * (L - L_0) * dL)  # 修正案1と同じ
        
        print(f"{L:2d} | {current_formula:10.0f} | {fix1:7.0f} | {fix2:7.0f} | {fix3:7.0f}")
    
    print("\n🔍 修正案1の効果:")
    print("- L=16 (理想): ΔH_L = 0 (変化なし)")
    print("- L>16 (不規則): dL<0でΔH_L<0 (改善有利)")
    print("- 二次項除去により過度な安定化を防ぐ")

def run_perimeter_formula_bug_analysis():
    """周囲長公式バグの分析を実行"""
    print("周囲長エネルギー公式のバグ調査\n")
    
    try:
        analyze_perimeter_energy_formula()
        test_non_ideal_state_transitions()
        analyze_dL_calculation_bug()
        test_actual_perimeter_calculation()
        debug_energy_formula_fix()
        
        print("\n" + "="*70)
        print("🎯 根本原因の特定")
        print("="*70)
        print("問題: ΔH_L = l_L * [2 * (L - L_0) * dL + (dL)^2]")
        print("     dL^2項により理想状態が過度に安定化")
        print("     不規則状態からの改善が異常に有利")
        print()
        print("解決策: dL^2項を除去")
        print("修正後: ΔH_L = l_L * [2 * (L - L_0) * dL]")
        print()
        print("効果:")
        print("- 理想状態(L=L_0)では ΔH_L = 0")
        print("- 改善方向の遷移確率が適正化")
        print("- 過剰成長が抑制される")
        
        return True
        
    except Exception as e:
        print(f"分析エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_perimeter_formula_bug_analysis()
    sys.exit(0 if success else 1)