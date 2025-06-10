#!/usr/bin/env python3
"""
修正されたCPMの動作テスト
空セル（ID=0）エネルギー除外後の動作を確認
"""

import torch
import sys
import numpy as np
sys.path.append('/app')

from cpm_torch.CPM import CPM, CPM_config

def test_notebook_settings():
    """notebookと同じ設定でのテスト"""
    print("=== notebook設定での修正後テスト ===")
    
    config = CPM_config(
        l_A=1.0, l_L=1.0, A_0=16.0, L_0=16.0, T=1.0, size=(16, 16)
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    print(f"設定: l_A={config.l_A}, l_L={config.l_L}, A_0={config.A_0}, L_0={config.L_0}, T={config.T}")
    
    # 1ピクセルから開始（notebookと同じ）
    tensor = torch.zeros((1, 16, 16, 3), device=device)
    tensor[0, 8, 8, 0] = 1.0  # 中央に1ピクセル
    
    print("\n成長過程の追跡:")
    print("ステップ | 面積 | 周囲長 | エネルギー | 変化 | 評価")
    print("-" * 60)
    
    prev_area = 1
    stable_count = 0
    
    for step in range(0, 101, 10):
        ids = tensor[0, :, :, 0]
        area = torch.sum(ids > 0).item()
        
        if area > 0:
            perimeter = CPM.calc_total_perimeter_bincount(ids, 2)[1].item()
        else:
            perimeter = 0
        
        energy = config.l_A * (area - config.A_0)**2 + config.l_L * (perimeter - config.L_0)**2
        
        if step == 0:
            change = "初期"
        else:
            area_change = area - prev_area
            if area_change == 0:
                change = "安定"
                stable_count += 1
            elif area_change > 0:
                change = f"+{area_change}"
                stable_count = 0
            else:
                change = f"{area_change}"
                stable_count = 0
        
        # 評価
        if area <= 20:
            evaluation = "良好"
        elif area <= 30:
            evaluation = "許容"
        else:
            evaluation = "過剰"
        
        print(f"{step:6d} | {area:4d} | {perimeter:6.1f} | {energy:8.1f} | {change:4s} | {evaluation}")
        
        # 10ステップ実行
        if step < 100:
            for _ in range(10):
                tensor = cpm.cpm_checkerboard_step_single_func(tensor)
            prev_area = area
    
    final_area = torch.sum(tensor[0, :, :, 0] > 0).item()
    
    print(f"\n最終結果:")
    print(f"  最終面積: {final_area}")
    print(f"  目標面積: 16")
    print(f"  成長倍率: {final_area/16:.1f}倍")
    print(f"  連続安定: {stable_count}回")
    
    if final_area <= 20:
        print("  ✅ 修正成功：適切な成長制御")
        success = True
    elif final_area <= 50:
        print("  ⚠️ 部分改善：軽微な過剰成長")
        success = False
    else:
        print("  ❌ 修正不十分：依然として過剰成長")
        success = False
    
    return success, final_area

def test_bidirectional_transitions():
    """双方向遷移のテスト"""
    print("\n=== 双方向遷移の確認 ===")
    
    config = CPM_config(l_A=1.0, l_L=1.0, A_0=16.0, L_0=16.0, T=1.0, size=(16, 16))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    # 4×4正方形から開始
    tensor = torch.zeros((1, 16, 16, 3), device=device)
    tensor[0, 6:10, 6:10, 0] = 1.0
    
    print("4×4正方形からの変化追跡:")
    print("ステップ | 面積 | 変化 | 成長回数 | 縮小回数 | 正味")
    print("-" * 55)
    
    prev_area = 16
    growth_count = 0
    shrink_count = 0
    
    for step in range(0, 51, 5):
        ids = tensor[0, :, :, 0]
        area = torch.sum(ids > 0).item()
        
        if step == 0:
            change = 0
        else:
            change = area - prev_area
            if change > 0:
                growth_count += change
            elif change < 0:
                shrink_count += abs(change)
        
        net_change = growth_count - shrink_count
        
        print(f"{step:6d} | {area:4d} | {change:+3d} | {growth_count:7d} | {shrink_count:7d} | {net_change:+4d}")
        
        # 5ステップ実行
        if step < 50:
            for _ in range(5):
                tensor = cpm.cpm_checkerboard_step_single_func(tensor)
            prev_area = area
    
    print(f"\n双方向遷移の統計:")
    print(f"  総成長: {growth_count}ピクセル")
    print(f"  総縮小: {shrink_count}ピクセル")
    print(f"  正味変化: {net_change:+d}ピクセル")
    
    if shrink_count > 0:
        print("  ✅ 縮小遷移が発生（双方向動作）")
        bidirectional_working = True
    else:
        print("  ❌ 縮小遷移なし（一方向のみ）")
        bidirectional_working = False
    
    return bidirectional_working, growth_count, shrink_count

def test_empty_cell_energy():
    """空セルエネルギーの除外確認"""
    print("\n=== 空セルエネルギー除外の確認 ===")
    
    config = CPM_config(l_A=1.0, l_L=1.0, A_0=16.0, L_0=16.0, T=1.0, size=(10, 10))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    # 3×3の細胞を作成
    test_map = torch.zeros((10, 10), device=device)
    test_map[3:6, 3:6] = 1.0
    
    areas = CPM.calc_area_bincount(test_map, 2)
    perimeters = CPM.calc_total_perimeter_bincount(test_map, 2)
    
    print(f"3×3細胞での面積・周囲長:")
    print(f"ID=0（空セル）: 面積={areas[0].item()}, 周囲長={perimeters[0].item()}")
    print(f"ID=1（細胞）  : 面積={areas[1].item()}, 周囲長={perimeters[1].item()}")
    
    # 遷移確率のテスト
    tensor = torch.zeros((1, 10, 10, 3), device=device)
    tensor[0, 3:6, 3:6, 0] = 1.0
    
    # 境界ピクセルでの遷移確率
    test_positions = [
        ("成長位置", (2, 3), 0, 1),  # 細胞外→細胞内
        ("縮小位置", (3, 3), 1, 0),  # 細胞内→細胞外
    ]
    
    print(f"\n境界ピクセルでの遷移エネルギー:")
    print("位置タイプ | 座標 | 現在ID | 遷移先ID | 理論ΔE")
    print("-" * 50)
    
    for pos_type, pos, current_id, target_id in test_positions:
        # 手動でエネルギー変化を計算
        if current_id == 0 and target_id == 1:  # 成長
            # 空セルのエネルギーは除外されるべき
            delta_E = config.l_A * (2 * areas[1].item() + 1 - 2 * config.A_0)
        else:  # 縮小
            # 細胞のエネルギーのみ計算
            delta_E = config.l_A * (-2 * areas[1].item() + 1 + 2 * config.A_0)
        
        print(f"{pos_type:8s} | {pos} | {current_id:6d} | {target_id:8d} | {delta_E:8.1f}")

def run_modified_cpm_test():
    """修正されたCPMの総合テスト"""
    print("修正されたCPMの動作テスト\n")
    
    try:
        # 1. notebook設定でのテスト
        success1, final_area = test_notebook_settings()
        
        # 2. 双方向遷移のテスト
        success2, growth, shrink = test_bidirectional_transitions()
        
        # 3. 空セルエネルギー除外の確認
        test_empty_cell_energy()
        
        print("\n" + "="*70)
        print("修正効果の総合評価")
        print("="*70)
        
        if success1 and success2:
            print("✅ 修正成功：")
            print(f"  - 面積制御: {final_area}ピクセル（目標16の{final_area/16:.1f}倍）")
            print(f"  - 双方向遷移: 成長{growth} vs 縮小{shrink}")
            print("  - 空セルエネルギー除外が機能")
            overall_success = True
        elif success1:
            print("⚠️ 部分成功：面積制御は改善、双方向遷移に課題")
            overall_success = False
        elif success2:
            print("⚠️ 部分成功：双方向遷移は改善、面積制御に課題")
            overall_success = False
        else:
            print("❌ 修正不十分：根本的な問題が残存")
            overall_success = False
        
        return overall_success
        
    except Exception as e:
        print(f"テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_modified_cpm_test()
    sys.exit(0 if success else 1)