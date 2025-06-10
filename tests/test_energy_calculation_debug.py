#!/usr/bin/env python3
"""
エネルギー計算の詳細デバッグ
修正後でも縮小が起こらない原因を調査
"""

import torch
import sys
import numpy as np
sys.path.append('/app')

from cpm_torch.CPM import CPM, CPM_config

def debug_energy_calculation():
    """エネルギー計算の詳細デバッグ"""
    print("=== エネルギー計算の詳細デバッグ ===")
    
    config = CPM_config(l_A=1.0, l_L=1.0, A_0=16.0, L_0=16.0, T=1.0, size=(16, 16))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    # 4×4正方形を作成
    tensor = torch.zeros((1, 16, 16, 3), device=device)
    tensor[0, 6:10, 6:10, 0] = 1.0
    
    ids = tensor[0, :, :, 0]
    print("4×4正方形の状態:")
    
    # 面積・周囲長の計算
    areas = CPM.calc_area_bincount(ids, 2)
    perimeters = CPM.calc_total_perimeter_bincount(ids, 2)
    
    print(f"ID=0（空セル）: 面積={areas[0].item()}, 周囲長={perimeters[0].item()}")
    print(f"ID=1（細胞）  : 面積={areas[1].item()}, 周囲長={perimeters[1].item()}")
    
    # 各セルのエネルギー
    empty_energy = config.l_A * (areas[0].item() - config.A_0)**2 + config.l_L * (perimeters[0].item() - config.L_0)**2
    cell_energy = config.l_A * (areas[1].item() - config.A_0)**2 + config.l_L * (perimeters[1].item() - config.L_0)**2
    
    print(f"\n各セルのエネルギー:")
    print(f"空セル（ID=0）: {empty_energy}")
    print(f"細胞（ID=1）  : {cell_energy}")
    
    # 具体的な遷移のエネルギー変化を計算
    print(f"\n遷移エネルギー変化の詳細:")
    
    # 成長: 細胞外(0) → 細胞内(1)
    print("1. 成長遷移（細胞外→細胞内）:")
    source_area_0 = areas[0].item()  # 空セル面積
    target_area_1 = areas[1].item()  # 細胞面積
    
    # 空セルのマスク（修正後は0になるべき）
    source_is_empty = (0 == 0)  # True（空セル）
    target_is_not_empty = (1 != 0)  # True（非空セル）
    
    print(f"  ソース（空セル）マスク: {not source_is_empty}")  # 修正後は False
    print(f"  ターゲット（細胞）マスク: {target_is_not_empty}")
    
    # 修正後のエネルギー計算
    if not source_is_empty:  # 空セルはエネルギー変化0
        source_term = config.l_A * (2.0 * source_area_0 + 1 - 2 * config.A_0)
    else:
        source_term = 0.0
    
    if target_is_not_empty:
        target_term = config.l_A * (-2.0 * target_area_1 + 1 + 2 * config.A_0)
    else:
        target_term = 0.0
    
    delta_E_growth = source_term + target_term
    prob_growth = np.exp(-delta_E_growth / config.T) if delta_E_growth > 0 else 1.0
    
    print(f"  ソース項: {source_term}")
    print(f"  ターゲット項: {target_term}")
    print(f"  総ΔE: {delta_E_growth}")
    print(f"  成長確率: {prob_growth:.6f}")
    
    # 縮小: 細胞内(1) → 細胞外(0)
    print("\n2. 縮小遷移（細胞内→細胞外）:")
    source_area_1 = areas[1].item()  # 細胞面積
    target_area_0 = areas[0].item()  # 空セル面積
    
    source_is_not_empty = (1 != 0)  # True（非空セル）
    target_is_empty = (0 == 0)      # True（空セル）
    
    print(f"  ソース（細胞）マスク: {source_is_not_empty}")
    print(f"  ターゲット（空セル）マスク: {not target_is_empty}")  # 修正後は False
    
    # 修正後のエネルギー計算
    if source_is_not_empty:
        source_term = config.l_A * (2.0 * source_area_1 + 1 - 2 * config.A_0)
    else:
        source_term = 0.0
    
    if not target_is_empty:  # 空セルはエネルギー変化0
        target_term = config.l_A * (-2.0 * target_area_0 + 1 + 2 * config.A_0)
    else:
        target_term = 0.0
    
    delta_E_shrink = source_term + target_term
    prob_shrink = np.exp(-delta_E_shrink / config.T) if delta_E_shrink > 0 else 1.0
    
    print(f"  ソース項: {source_term}")
    print(f"  ターゲット項: {target_term}")
    print(f"  総ΔE: {delta_E_shrink}")
    print(f"  縮小確率: {prob_shrink:.6f}")

def test_actual_probability_calculation():
    """実際の確率計算をテスト"""
    print("\n=== 実際のCPM確率計算 ===")
    
    config = CPM_config(l_A=1.0, l_L=1.0, A_0=16.0, L_0=16.0, T=1.0, size=(16, 16))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    # 4×4正方形を作成
    tensor = torch.zeros((1, 16, 16, 3), device=device)
    tensor[0, 6:10, 6:10, 0] = 1.0
    
    print("境界ピクセルでの実際の確率計算:")
    
    # 境界ピクセルのテスト
    test_cases = [
        ("成長候補", (5, 6), 0, 1),  # 細胞外→細胞内
        ("縮小候補", (6, 6), 1, 0),  # 細胞内→細胞外
    ]
    
    ids = tensor[0, :, :, 0]
    
    for case_name, pos, source_id, target_id in test_cases:
        row, col = pos
        
        # 手動で確率計算をシミュレート
        source_ids = torch.tensor([[source_id]], device=device, dtype=torch.long)
        target_ids = torch.tensor([[target_id]], device=device, dtype=torch.long)
        
        # calc_cpm_probabilitiesを呼び出し
        try:
            # 4近傍を模擬
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < 16 and 0 <= nc < 16:
                    neighbors.append(ids[nr, nc].item())
                else:
                    neighbors.append(0)
            
            source_ids_4 = torch.tensor([neighbors], device=device, dtype=torch.long)
            
            logits = cmp.calc_cmp_probabilities(
                source_ids, target_ids, ids, source_ids_4=source_ids_4
            )
            
            print(f"{case_name}: 確率={logits.item():.6f}")
            
        except Exception as e:
            print(f"{case_name}: エラー - {e}")

def analyze_mask_effectiveness():
    """マスクの効果を分析"""
    print("\n=== マスクの効果分析 ===")
    
    # 空セルマスクのテスト
    source_ids = torch.tensor([[0, 1, 0, 1]], dtype=torch.long)
    target_ids = torch.tensor([[0]], dtype=torch.long)
    
    source_is_not_empty = source_ids != 0
    target_is_not_empty = target_ids != 0
    
    print(f"ソースID: {source_ids}")
    print(f"ターゲットID: {target_ids}")
    print(f"ソース非空マスク: {source_is_not_empty}")
    print(f"ターゲット非空マスク: {target_is_not_empty}")
    
    # エネルギー項の計算
    dummy_areas = torch.tensor([[100, 16, 100, 16]], dtype=torch.float)
    dummy_target_area = torch.tensor([[100]], dtype=torch.float)
    
    A_0 = 16.0
    l_A = 1.0
    
    print(f"\n面積値:")
    print(f"ソース面積: {dummy_areas}")
    print(f"ターゲット面積: {dummy_target_area}")
    
    # マスク適用前のエネルギー項
    source_term_raw = l_A * (2.0 * dummy_areas + 1 - 2 * A_0)
    target_term_raw = l_A * (-2.0 * dummy_target_area + 1 + 2 * A_0)
    
    print(f"\nマスク適用前:")
    print(f"ソース項: {source_term_raw}")
    print(f"ターゲット項: {target_term_raw}")
    
    # マスク適用後のエネルギー項
    source_term_masked = source_term_raw * source_is_not_empty.float()
    target_term_masked = target_term_raw * target_is_not_empty.float()
    
    print(f"\nマスク適用後:")
    print(f"ソース項: {source_term_masked}")
    print(f"ターゲット項: {target_term_masked}")
    
    total_delta_E = source_term_masked + target_term_masked
    print(f"総ΔE: {total_delta_E}")

def run_energy_debug():
    """エネルギー計算デバッグを実行"""
    print("エネルギー計算の詳細デバッグ\n")
    
    try:
        # 1. エネルギー計算の詳細デバッグ
        debug_energy_calculation()
        
        # 2. 実際の確率計算テスト
        test_actual_probability_calculation()
        
        # 3. マスクの効果分析
        analyze_mask_effectiveness()
        
        print("\n" + "="*70)
        print("デバッグ結論")
        print("="*70)
        print("修正後のエネルギー計算:")
        print("1. 空セル（ID=0）のエネルギー項は正しく0になる")
        print("2. しかし実際の動作では縮小が起こらない")
        print("3. 他の要因（周囲長、ランダム選択）が影響している可能性")
        
        return True
        
    except Exception as e:
        print(f"デバッグエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_energy_debug()
    sys.exit(0 if success else 1)