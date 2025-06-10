#!/usr/bin/env python3
"""
A_0=16, L_0=16設定で適切に動作させるための具体的解決策テスト
目標: 16ピクセル（4×4正方形）で安定する設定を見つける
"""

import torch
import sys
import numpy as np
sys.path.append('/app')

from cpm_torch.CPM import CPM, CPM_config

def test_current_notebook_settings():
    """現在のnotebook設定での詳細分析"""
    print("=== 現在のnotebook設定での分析 ===")
    
    # notebook設定
    config = CPM_config(
        l_A=2.0, l_L=1.0, A_0=16.0, L_0=16.0, T=0.5, size=(16, 16)
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    print(f"設定: l_A={config.l_A}, l_L={config.l_L}, A_0={config.A_0}, L_0={config.L_0}, T={config.T}")
    
    # 初期状態
    tensor = torch.zeros((1, 16, 16, 3), device=device)
    tensor[0, 8, 8, 0] = 1.0
    
    print("\n成長過程の詳細追跡:")
    print("ステップ | 面積 | 周囲長 | 面積E | 周囲長E | 総E | 1→2成長確率")
    print("-" * 70)
    
    growth_data = []
    
    for step in range(0, 201, 20):
        ids = tensor[0, :, :, 0]
        area = torch.sum(ids > 0).item()
        
        if area > 0:
            perimeter = CPM.calc_total_perimeter_bincount(ids, 2)[1].item()
        else:
            perimeter = 0
        
        area_energy = config.l_A * (area - config.A_0)**2
        perimeter_energy = config.l_L * (perimeter - config.L_0)**2
        total_energy = area_energy + perimeter_energy
        
        # 次のステップ（面積+1）でのエネルギーを推定
        next_area = area + 1
        estimated_next_perimeter = perimeter + 2  # 推定
        next_area_energy = config.l_A * (next_area - config.A_0)**2
        next_perimeter_energy = config.l_L * (estimated_next_perimeter - config.L_0)**2
        next_total_energy = next_area_energy + next_perimeter_energy
        
        delta_E = next_total_energy - total_energy
        growth_prob = min(1.0, np.exp(-delta_E / config.T))
        
        growth_data.append({
            'step': step, 'area': area, 'perimeter': perimeter,
            'total_energy': total_energy, 'growth_prob': growth_prob
        })
        
        print(f"{step:6d} | {area:4d} | {perimeter:6.1f} | {area_energy:5.0f} | {perimeter_energy:7.0f} | {total_energy:5.0f} | {growth_prob:8.3f}")
        
        # 20ステップ実行
        if step < 200:
            for _ in range(20):
                tensor = cpm.cpm_checkerboard_step_single_func(tensor)
    
    final_area = growth_data[-1]['area']
    print(f"\n最終結果: 面積={final_area} (目標16の{final_area/16:.1f}倍)")
    
    return growth_data

def find_optimal_parameters_for_16_16():
    """A_0=16, L_0=16で最適なパラメータを探索"""
    print("\n=== A_0=16, L_0=16での最適パラメータ探索 ===")
    
    test_configs = [
        {"name": "現在設定", "l_A": 2.0, "l_L": 1.0, "T": 0.5},
        {"name": "高面積ペナルティ1", "l_A": 5.0, "l_L": 1.0, "T": 0.5},
        {"name": "高面積ペナルティ2", "l_A": 10.0, "l_L": 1.0, "T": 0.5},
        {"name": "超低温度", "l_A": 2.0, "l_L": 1.0, "T": 0.1},
        {"name": "極低温度", "l_A": 2.0, "l_L": 1.0, "T": 0.01},
        {"name": "組合せ1", "l_A": 5.0, "l_L": 1.0, "T": 0.1},
        {"name": "組合せ2", "l_A": 10.0, "l_L": 1.0, "T": 0.1},
        {"name": "強力設定", "l_A": 20.0, "l_L": 1.0, "T": 0.01},
    ]
    
    print("設定 | l_A | l_L | T | 1→2成長確率 | 16→17成長確率 | 評価")
    print("-" * 75)
    
    best_config = None
    best_score = float('inf')
    
    for config_params in test_configs:
        name = config_params.pop("name")
        config = CPM_config(A_0=16.0, L_0=16.0, **config_params)
        
        # 1→2ピクセル成長の確率
        current_energy_1 = config.l_A * (1 - 16)**2 + config.l_L * (4 - 16)**2
        next_energy_1 = config.l_A * (2 - 16)**2 + config.l_L * (6 - 16)**2
        delta_E_1 = next_energy_1 - current_energy_1
        prob_1_2 = min(1.0, np.exp(-delta_E_1 / config.T))
        
        # 16→17ピクセル成長の確率
        current_energy_16 = config.l_A * (16 - 16)**2 + config.l_L * (16 - 16)**2  # = 0
        next_energy_16 = config.l_A * (17 - 16)**2 + config.l_L * (18 - 16)**2  # 推定周囲長18
        delta_E_16 = next_energy_16 - current_energy_16
        prob_16_17 = min(1.0, np.exp(-delta_E_16 / config.T))
        
        # 評価スコア（16→17の成長確率が低いほど良い）
        score = prob_16_17
        
        if score < best_score:
            best_score = score
            best_config = {"name": name, **config_params}
        
        evaluation = "✅ 良好" if prob_16_17 < 0.01 else "⚠️ 要改善" if prob_16_17 < 0.1 else "❌ 問題あり"
        
        print(f"{name:12s} | {config.l_A:3.0f} | {config.l_L:3.0f} | {config.T:4.2f} | {prob_1_2:10.3f} | {prob_16_17:12.3f} | {evaluation}")
    
    print(f"\n推奨設定: {best_config['name']}")
    print(f"パラメータ: l_A={best_config['l_A']}, l_L={best_config['l_L']}, T={best_config['T']}")
    
    return best_config

def test_recommended_settings():
    """推奨設定での実際のテスト"""
    print("\n=== 推奨設定での実際のテスト ===")
    
    # 推奨設定
    config = CPM_config(
        l_A=10.0,   # 高い面積ペナルティ
        l_L=1.0,    # 標準の周囲長係数
        A_0=16.0,   # 目標面積
        L_0=16.0,   # 目標周囲長
        T=0.1,      # 低い温度
        size=(16, 16)
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    print(f"推奨設定: l_A={config.l_A}, l_L={config.l_L}, T={config.T}")
    
    # シミュレーション実行
    tensor = torch.zeros((1, 16, 16, 3), device=device)
    tensor[0, 8, 8, 0] = 1.0
    
    print("\n推奨設定での成長テスト:")
    print("ステップ | 面積 | 周囲長 | 総エネルギー | 状態")
    print("-" * 50)
    
    for step in range(0, 201, 25):
        ids = tensor[0, :, :, 0]
        area = torch.sum(ids > 0).item()
        
        if area > 0:
            perimeter = CPM.calc_total_perimeter_bincount(ids, 2)[1].item()
        else:
            perimeter = 0
        
        total_energy = config.l_A * (area - config.A_0)**2 + config.l_L * (perimeter - config.L_0)**2
        
        if area <= 12:
            status = "成長中"
        elif area <= 20:
            status = "目標付近"
        else:
            status = "過剰成長"
        
        print(f"{step:6d} | {area:4d} | {perimeter:6.1f} | {total_energy:10.1f} | {status}")
        
        # 25ステップ実行
        if step < 200:
            for _ in range(25):
                tensor = cpm.cpm_checkerboard_step_single_func(tensor)
    
    final_area = torch.sum(tensor[0, :, :, 0] > 0).item()
    
    print(f"\n最終結果:")
    print(f"  目標面積: 16")
    print(f"  実際面積: {final_area}")
    print(f"  成長倍率: {final_area/16:.1f}倍")
    
    if final_area <= 20:
        print("  ✅ 成長が適切に制御されました！")
        success = True
    elif final_area <= 30:
        print("  ⚠️ 軽微な過剰成長")
        success = False
    else:
        print("  ❌ 依然として過剰成長")
        success = False
    
    return success, final_area

def test_alternative_initialization():
    """初期化方法を変更したテスト"""
    print("\n=== 初期化方法の改善テスト ===")
    
    config = CPM_config(
        l_A=5.0, l_L=1.0, A_0=16.0, L_0=16.0, T=0.1, size=(16, 16)
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    # 方法1: 2×2初期セル
    print("方法1: 2×2初期セル")
    tensor1 = torch.zeros((1, 16, 16, 3), device=device)
    tensor1[0, 7:9, 7:9, 0] = 1.0  # 2×2セル
    
    for _ in range(100):
        tensor1 = cpm.cpm_checkerboard_step_single_func(tensor1)
    
    final_area_1 = torch.sum(tensor1[0, :, :, 0] > 0).item()
    print(f"  結果: {final_area_1}ピクセル ({final_area_1/16:.1f}倍)")
    
    # 方法2: 3×3初期セル
    print("方法2: 3×3初期セル")
    tensor2 = torch.zeros((1, 16, 16, 3), device=device)
    tensor2[0, 7:10, 7:10, 0] = 1.0  # 3×3セル
    
    for _ in range(100):
        tensor2 = cpm.cpm_checkerboard_step_single_func(tensor2)
    
    final_area_2 = torch.sum(tensor2[0, :, :, 0] > 0).item()
    print(f"  結果: {final_area_2}ピクセル ({final_area_2/16:.1f}倍)")
    
    # 方法3: 4×4初期セル（目標サイズ）
    print("方法3: 4×4初期セル（目標サイズ）")
    tensor3 = torch.zeros((1, 16, 16, 3), device=device)
    tensor3[0, 6:10, 6:10, 0] = 1.0  # 4×4セル
    
    for _ in range(100):
        tensor3 = cpm.cpm_checkerboard_step_single_func(tensor3)
    
    final_area_3 = torch.sum(tensor3[0, :, :, 0] > 0).item()
    print(f"  結果: {final_area_3}ピクセル ({final_area_3/16:.1f}倍)")
    
    print(f"\n最適な初期化方法: ", end="")
    if final_area_3 <= 20:
        print("4×4初期セル ✅")
    elif final_area_2 <= 20:
        print("3×3初期セル ✅")
    elif final_area_1 <= 20:
        print("2×2初期セル ✅")
    else:
        print("更なる調整が必要")

def run_specific_16_16_solution():
    """A_0=16, L_0=16での具体的解決策テスト"""
    print("A_0=16, L_0=16設定での具体的解決策テスト\n")
    
    try:
        # 1. 現在設定の分析
        current_data = test_current_notebook_settings()
        
        # 2. 最適パラメータの探索
        best_config = find_optimal_parameters_for_16_16()
        
        # 3. 推奨設定のテスト
        success, final_area = test_recommended_settings()
        
        # 4. 初期化方法の改善
        test_alternative_initialization()
        
        print("\n" + "="*60)
        print("A_0=16, L_0=16での最終推奨案")
        print("="*60)
        
        if success:
            print("✅ 成功: 以下の設定で適切な制御を確認")
            print(f"   l_A=10.0, l_L=1.0, T=0.1")
            print(f"   結果: {final_area}ピクセル (目標16の{final_area/16:.1f}倍)")
        else:
            print("⚠️ 更なる調整が必要:")
            print("   l_A=20.0, l_L=1.0, T=0.01 (より厳格な設定)")
            print("   または初期セルサイズを3×3以上に")
        
        print("\nnotebook用の推奨コード:")
        print("```python")
        print("config = CPM_config(")
        print("    l_A=10.0,   # 高い面積ペナルティ")
        print("    l_L=1.0,    # 周囲長係数")
        print("    A_0=16.0,   # 目標面積")
        print("    L_0=16.0,   # 目標周囲長")
        print("    T=0.1,      # 低い温度")
        print("    size=(16, 16)")
        print(")")
        print("```")
        
        return True
        
    except Exception as e:
        print(f"テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_specific_16_16_solution()
    sys.exit(0 if success else 1)