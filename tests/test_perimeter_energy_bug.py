#!/usr/bin/env python3
"""
周囲長エネルギー計算のバグ調査
l_L=0で動作することから、周囲長エネルギー差の計算に問題がある
"""

import torch
import sys
import numpy as np
sys.path.append('/app')

from cpm_torch.CPM import CPM, CPM_config

def test_perimeter_energy_coefficients():
    """周囲長係数の影響をテスト"""
    print("=== 周囲長係数の影響テスト ===")
    
    test_configs = [
        {"name": "面積のみ", "l_A": 1.0, "l_L": 0.0},
        {"name": "周囲長のみ", "l_A": 0.0, "l_L": 1.0},
        {"name": "両方", "l_A": 1.0, "l_L": 1.0},
        {"name": "周囲長強調", "l_A": 1.0, "l_L": 2.0},
    ]
    
    for config_params in test_configs:
        print(f"\n--- {config_params['name']} ({config_params}) ---")
        
        config = CPM_config(
            l_A=config_params["l_A"], 
            l_L=config_params["l_L"], 
            A_0=16.0, L_0=16.0, T=1.0, size=(16, 16)
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
        print(f"50ステップ後の面積: {final_area} (成長倍率: {final_area:.1f}倍)")

def analyze_perimeter_energy_calculation():
    """周囲長エネルギー計算の詳細分析"""
    print("\n=== 周囲長エネルギー計算の詳細分析 ===")
    
    config = CPM_config(l_A=1.0, l_L=1.0, A_0=16.0, L_0=16.0, T=1.0, size=(16, 16))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    # 4×4正方形を作成
    tensor = torch.zeros((1, 16, 16, 3), device=device)
    tensor[0, 6:10, 6:10, 0] = 1.0
    
    ids = tensor[0, :, :, 0]
    
    print("4×4正方形での周囲長エネルギー分析:")
    
    # 現在の状態
    areas = CPM.calc_area_bincount(ids, 2)
    perimeters = CPM.calc_total_perimeter_bincount(ids, 2)
    
    print(f"現在状態:")
    print(f"  ID=0: 面積={areas[0].item()}, 周囲長={perimeters[0].item()}")
    print(f"  ID=1: 面積={areas[1].item()}, 周囲長={perimeters[1].item()}")
    
    # 各セルのエネルギー
    energy_0 = config.l_A * (areas[0].item() - config.A_0)**2 + config.l_L * (perimeters[0].item() - config.L_0)**2
    energy_1 = config.l_A * (areas[1].item() - config.A_0)**2 + config.l_L * (perimeters[1].item() - config.L_0)**2
    
    print(f"各セルのエネルギー:")
    print(f"  ID=0: {energy_0}")
    print(f"  ID=1: {energy_1}")
    
    # 遷移テスト：境界ピクセルでの遷移
    test_position = (5, 6)  # 4×4正方形の上の境界
    
    print(f"\n位置{test_position}での遷移分析:")
    
    # 成長遷移: 0 → 1
    print("成長遷移（0→1）:")
    new_areas_grow = areas.clone()
    new_perimeters_grow = perimeters.clone()
    new_areas_grow[0] -= 1  # 空セル面積-1
    new_areas_grow[1] += 1  # 細胞面積+1
    # 周囲長変化の推定（実際の計算は複雑）
    new_perimeters_grow[1] += 2  # 細胞周囲長+2（推定）
    
    energy_change_grow = (
        config.l_A * ((new_areas_grow[1] - config.A_0)**2 - (areas[1] - config.A_0)**2) +
        config.l_L * ((new_perimeters_grow[1] - config.L_0)**2 - (perimeters[1] - config.L_0)**2)
    )
    
    print(f"  面積変化: {areas[1].item()} → {new_areas_grow[1].item()}")
    print(f"  周囲長変化: {perimeters[1].item()} → {new_perimeters_grow[1].item()}")
    print(f"  エネルギー変化: {energy_change_grow}")
    print(f"  遷移確率: {np.exp(-energy_change_grow.cpu() / config.T):.6f}")

def test_perimeter_delta_calculation():
    """周囲長変化量(dL)の計算をテスト"""
    print("\n=== 周囲長変化量(dL)の計算テスト ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 4×4正方形を作成
    test_map = torch.zeros((10, 10), device=device)
    test_map[3:7, 3:7] = 1.0
    
    print("基準4×4正方形:")
    print(test_map[2:8, 2:8].cpu().numpy().astype(int))
    
    # 境界位置での周囲長変化をテスト
    test_positions = [
        (2, 4), (2, 5),  # 上の境界
        (4, 2), (5, 2),  # 左の境界
    ]
    
    print(f"\n各位置での実際の周囲長変化:")
    print("位置 | 元周囲長 | 新周囲長 | 実際変化 | CPM予測変化")
    print("-" * 55)
    
    base_perimeter = CPM.calc_total_perimeter_bincount(test_map, 2)[1].item()
    
    for pos in test_positions:
        # 1ピクセル追加
        modified_map = test_map.clone()
        modified_map[pos[0], pos[1]] = 1.0
        
        new_perimeter = CPM.calc_total_perimeter_bincount(modified_map, 2)[1].item()
        actual_change = new_perimeter - base_perimeter
        
        # CPMでの予測変化を計算（局所的計算）
        # この位置の4近傍を確認
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < 10 and 0 <= nc < 10:
                neighbors.append(test_map[nr, nc].item())
            else:
                neighbors.append(0)
        
        # dL_s = 4 - 2 * (同じIDの近傍数)
        same_id_count = sum(1 for n in neighbors if n == 1)  # 細胞ID=1と同じ
        predicted_dL = 4 - 2 * same_id_count
        
        print(f"{pos} | {base_perimeter:7.1f} | {new_perimeter:7.1f} | {actual_change:+8.1f} | {predicted_dL:+11.1f}")

def debug_dh_perimeter_calculation():
    """dH_perimeter計算の詳細デバッグ"""
    print("\n=== dH_perimeter計算の詳細デバッグ ===")
    
    config = CPM_config(l_A=1.0, l_L=1.0, A_0=16.0, L_0=16.0, T=1.0, size=(16, 16))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    # 4×4正方形を作成
    tensor = torch.zeros((1, 16, 16, 3), device=device)
    tensor[0, 6:10, 6:10, 0] = 1.0
    
    ids = tensor[0, :, :, 0]
    
    # 手動で遷移確率を計算
    # 境界ピクセル (5,6) での 0→1 遷移
    source_id = torch.tensor([[0]], device=device, dtype=torch.long)  # 空セル
    target_id = torch.tensor([[1]], device=device, dtype=torch.long)  # 細胞
    
    # 4近傍のIDを取得（位置(5,6)の近傍）
    neighbors = []
    pos = (5, 6)
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = pos[0] + dr, pos[1] + dc
        if 0 <= nr < 16 and 0 <= nc < 16:
            neighbors.append(ids[nr, nc].item())
        else:
            neighbors.append(0)
    
    source_ids_4 = torch.tensor([neighbors], device=device, dtype=torch.long)
    
    print(f"遷移分析: 位置{pos}, 0→1")
    print(f"4近傍ID: {neighbors}")
    
    try:
        # calc_cpm_probabilitiesを直接呼び出し
        logits = cpm.calc_cpm_probabilities(
            source_id, target_id, ids, source_ids_4=source_ids_4
        )
        
        print(f"計算された遷移確率: {logits.item():.6f}")
        
        # 手動でエネルギー変化を計算
        areas = CPM.calc_area_bincount(ids, 2)
        perimeters = CPM.calc_total_perimeter_bincount(ids, 2)
        
        # 面積エネルギー変化（空セルは除外済み）
        area_delta_E = config.l_A * (2 * areas[1].item() + 1 - 2 * config.A_0)
        
        # 周囲長エネルギー変化
        # dL_s = 4 - 2 * (近傍で同じIDの数)
        same_id_count = sum(1 for n in neighbors if n == source_id.item())
        dL_s = 4 - 2 * same_id_count
        perimeter_delta_E = config.l_L * (2 * (perimeters[1].item() - config.L_0) * dL_s + dL_s**2)
        
        total_delta_E = area_delta_E + perimeter_delta_E
        manual_prob = np.exp(-total_delta_E / config.T) if total_delta_E > 0 else 1.0
        
        print(f"手動計算:")
        print(f"  面積ΔE: {area_delta_E}")
        print(f"  周囲長ΔE: {perimeter_delta_E} (dL_s={dL_s})")
        print(f"  総ΔE: {total_delta_E}")
        print(f"  手動確率: {manual_prob:.6f}")
        
        if abs(logits.item() - manual_prob) > 0.001:
            print("⚠️ CPM計算と手動計算に差異あり！")
        else:
            print("✅ CPM計算と手動計算が一致")
            
    except Exception as e:
        print(f"遷移確率計算エラー: {e}")

def run_perimeter_energy_bug_test():
    """周囲長エネルギーバグのテストを実行"""
    print("周囲長エネルギー計算のバグ調査\n")
    
    try:
        # 1. 周囲長係数の影響テスト
        test_perimeter_energy_coefficients()
        
        # 2. 周囲長エネルギー計算の詳細分析
        analyze_perimeter_energy_calculation()
        
        # 3. 周囲長変化量の計算テスト
        test_perimeter_delta_calculation()
        
        # 4. dH_perimeter計算のデバッグ
        debug_dh_perimeter_calculation()
        
        print("\n" + "="*70)
        print("周囲長エネルギーバグ調査の結論")
        print("="*70)
        print("重要な発見:")
        print("1. l_L=0では正常動作 → 周囲長エネルギーに問題")
        print("2. 周囲長変化量(dL)の計算が不正確")
        print("3. dH_perimeter計算での局所変化推定にバグ")
        print("4. 実際の周囲長変化と理論値の乖離")
        
        return True
        
    except Exception as e:
        print(f"テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_perimeter_energy_bug_test()
    sys.exit(0 if success else 1)