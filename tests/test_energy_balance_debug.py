#!/usr/bin/env python3
"""
エネルギーバランスのデバッグ
修正後に細胞が消失する原因を調査
"""

import torch
import sys
import numpy as np
sys.path.append('/app')

from cpm_torch.CPM import CPM, CPM_config

def analyze_energy_balance():
    """エネルギーバランスの分析"""
    print("=== エネルギーバランスの分析 ===")
    
    config = CPM_config(l_A=1.0, l_L=1.0, A_0=16.0, L_0=16.0, T=1.0)
    
    print("修正後の公式: ΔH_L = l_L * [2 * (L - L_0) * dL]")
    
    # 1ピクセル状態での遷移
    print("\n1ピクセル状態での遷移分析:")
    area_1 = 1
    perimeter_1 = 4
    energy_1 = config.l_A * (area_1 - config.A_0)**2 + config.l_L * (perimeter_1 - config.L_0)**2
    
    print(f"1ピクセル状態: 面積={area_1}, 周囲長={perimeter_1}, エネルギー={energy_1}")
    
    # 1ピクセル→0ピクセル (消失)
    area_0 = 0
    perimeter_0 = 0
    
    # 面積エネルギー変化 (空セルは除外されるので、細胞部分のみ)
    area_delta_E = config.l_A * ((area_0 - config.A_0)**2 - (area_1 - config.A_0)**2)
    
    # 周囲長エネルギー変化 (修正後の公式)
    # dL = -4 (1ピクセルが消失)
    dL = -4
    perimeter_delta_E = config.l_L * (2 * (perimeter_1 - config.L_0) * dL)
    
    total_delta_E = area_delta_E + perimeter_delta_E
    
    print(f"\n1→0遷移:")
    print(f"  面積ΔE: {area_delta_E}")
    print(f"  周囲長ΔE: {perimeter_delta_E} (dL={dL})")
    print(f"  総ΔE: {total_delta_E}")
    print(f"  消失確率: {np.exp(-total_delta_E / config.T):.6f}")
    
    if total_delta_E < 0:
        print("  ❌ 消失が有利 → 細胞が消える")
    else:
        print("  ✅ 消失が不利 → 細胞が維持")

def test_small_configurations():
    """小さな設定での動作テスト"""
    print("\n=== 小さな設定での動作テスト ===")
    
    # より小さな目標設定をテスト
    test_configs = [
        {"A_0": 1.0, "L_0": 4.0, "name": "1ピクセル目標"},
        {"A_0": 4.0, "L_0": 8.0, "name": "2×2目標"},
        {"A_0": 9.0, "L_0": 12.0, "name": "3×3目標"},
    ]
    
    for config_params in test_configs:
        print(f"\n--- {config_params['name']} ---")
        
        config = CPM_config(
            l_A=1.0, l_L=1.0, 
            A_0=config_params["A_0"], L_0=config_params["L_0"], 
            T=1.0, size=(16, 16)
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cpm = CPM(config, device)
        
        # 1ピクセルから開始
        tensor = torch.zeros((1, 16, 16, 3), device=device)
        tensor[0, 8, 8, 0] = 1.0
        
        # 20ステップ実行
        for _ in range(20):
            tensor = cpm.cpm_checkerboard_step_single_func(tensor)
        
        final_area = torch.sum(tensor[0, :, :, 0] > 0).item()
        
        print(f"  20ステップ後: {final_area}ピクセル")
        if final_area == 0:
            print("  ❌ 細胞消失")
        elif final_area <= config_params["A_0"] * 1.5:
            print("  ✅ 適切な制御")
        else:
            print("  ⚠️ 過剰成長")

def analyze_perimeter_term_impact():
    """周囲長項の影響を分析"""
    print("\n=== 周囲長項の影響分析 ===")
    
    config = CPM_config(l_A=1.0, l_L=1.0, A_0=16.0, L_0=16.0, T=1.0)
    
    # 1ピクセル状態での詳細分析
    print("1ピクセル状態 (L=4, L_0=16):")
    L_current = 4
    L_0 = 16
    
    dL_values = [-4, -2, -1, 0, 1, 2, 4]
    
    print("dL | 周囲長項 | 解釈")
    print("-" * 30)
    
    for dL in dL_values:
        perimeter_term = config.l_L * (2 * (L_current - L_0) * dL)
        
        if dL == -4:
            interpretation = "完全消失"
        elif dL < 0:
            interpretation = "周囲長減少"
        elif dL > 0:
            interpretation = "周囲長増加"
        else:
            interpretation = "変化なし"
        
        print(f"{dL:2d} | {perimeter_term:8.0f} | {interpretation}")
    
    print(f"\n🔍 問題発見:")
    print(f"L=4, L_0=16の場合、L-L_0={L_current-L_0}")
    print(f"dL=-4 (消失) で周囲長項 = {config.l_L * (2 * (L_current - L_0) * (-4))}")
    print(f"これは大幅にエネルギーを下げ、消失を促進する")

def test_coefficient_rebalancing():
    """係数の再バランステスト"""
    print("\n=== 係数の再バランステスト ===")
    
    # l_L を小さくしてテスト
    l_L_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    print("l_L値 | 20ステップ後面積 | 状態")
    print("-" * 35)
    
    for l_L in l_L_values:
        config = CPM_config(
            l_A=1.0, l_L=l_L, A_0=16.0, L_0=16.0, T=1.0, size=(16, 16)
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cpm = CPM(config, device)
        
        # 1ピクセルから開始
        tensor = torch.zeros((1, 16, 16, 3), device=device)
        tensor[0, 8, 8, 0] = 1.0
        
        # 20ステップ実行
        for _ in range(20):
            tensor = cpm.cpm_checkerboard_step_single_func(tensor)
        
        final_area = torch.sum(tensor[0, :, :, 0] > 0).item()
        
        if final_area == 0:
            status = "消失"
        elif final_area <= 5:
            status = "安定"
        elif final_area <= 20:
            status = "制御"
        else:
            status = "過剰"
        
        print(f"{l_L:4.1f} | {final_area:12d} | {status}")

def run_energy_balance_debug():
    """エネルギーバランスデバッグを実行"""
    print("エネルギーバランスのデバッグ\n")
    
    try:
        analyze_energy_balance()
        test_small_configurations()
        analyze_perimeter_term_impact()
        test_coefficient_rebalancing()
        
        print("\n" + "="*70)
        print("🔍 問題の特定")
        print("="*70)
        print("修正後の問題:")
        print("1. dL^2項除去により、周囲長項が過度に強くなった")
        print("2. 小さなセル (L << L_0) で消失方向が異常に有利")
        print("3. L - L_0 の符号により、過度な不均衡が発生")
        
        print("\n解決策:")
        print("1. l_L 係数を小さくして周囲長項の影響を調整")
        print("2. または周囲長項の計算方法を再考")
        print("3. 絶対値を使用するなど、符号問題を解決")
        
        return True
        
    except Exception as e:
        print(f"デバッグエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_energy_balance_debug()
    sys.exit(0 if success else 1)