#!/usr/bin/env python3
"""
エネルギー計算のエッジケースとコーナーケースのテスト
dH_areaとdH_perimeterが境界条件で正しく動作するかを検証
"""

import torch
import sys
sys.path.append('/app')

from cpm_torch.CPM import CPM, CPM_config

def test_zero_energy_cases():
    """エネルギー変化が0になるケースのテスト"""
    print("=== ゼロエネルギー変化テスト ===")
    
    config = CPM_config(l_A=1.0, A_0=4.0, l_L=1.0, L_0=8.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    # ケース1: 面積が目標面積と等しい場合
    print("\n--- ケース1: 目標面積での変化 ---")
    source_areas = torch.tensor([[4.0, 3.0]], device=device)  # A_s = 4(目標), 3
    target_area = torch.tensor([[5.0]], device=device)       # A_t = 5
    source_is_not_empty = torch.tensor([[True, True]], device=device)
    target_is_not_empty = torch.tensor([[True]], device=device)
    
    delta_H_area = cpm.calc_dH_area(source_areas, target_area, source_is_not_empty, target_is_not_empty)
    print(f"面積エネルギー変化: {delta_H_area}")
    
    # A_s=4の場合: (5-4)^2 - (4-4)^2 = 1 - 0 = 1
    # A_t=5の場合: (4-4)^2 - (5-4)^2 = 0 - 1 = -1
    # 合計: 1 + (-1) = 0
    expected_1 = 1.0 * ((4+1-4)**2 - (4-4)**2 + (5-1-4)**2 - (5-4)**2)
    print(f"A_s=4の理論値: {expected_1}, 実際値: {delta_H_area[0,0].item()}")
    
    assert abs(delta_H_area[0,0].item() - expected_1) < 1e-5, "目標面積でのエネルギー変化が不正確"
    
    print("✓ ゼロエネルギー変化テスト完了")
    return True

def test_empty_cell_handling():
    """空セル（ID=0）の処理テスト"""
    print("\n=== 空セル処理テスト ===")
    
    config = CPM_config(l_A=2.0, A_0=5.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    # 空セルを含むケース
    source_areas = torch.tensor([[3.0, 0.0, 7.0]], device=device)  # 通常, 空, 通常
    target_area = torch.tensor([[4.0]], device=device)
    source_is_not_empty = torch.tensor([[True, False, True]], device=device)  # 空セルをマスク
    target_is_not_empty = torch.tensor([[True]], device=device)
    
    delta_H_area = cpm.calc_dH_area(source_areas, target_area, source_is_not_empty, target_is_not_empty)
    print(f"空セル含む面積エネルギー変化: {delta_H_area}")
    
    # 空セル（インデックス1）のエネルギー変化は0になるべき
    assert delta_H_area[0,1].item() == 0.0, "空セルのエネルギー変化が0でない"
    
    # 非空セルは正常に計算されるべき
    l_A, A_0 = 2.0, 5.0
    expected_0 = l_A * ((2*3 + 1 - 2*A_0) + (-2*4 + 1 + 2*A_0))  # source=3, target=4
    expected_2 = l_A * ((2*7 + 1 - 2*A_0) + (-2*4 + 1 + 2*A_0))  # source=7, target=4
    
    assert abs(delta_H_area[0,0].item() - expected_0) < 1e-5, "非空セル0のエネルギー変化が不正確"
    assert abs(delta_H_area[0,2].item() - expected_2) < 1e-5, "非空セル2のエネルギー変化が不正確"
    
    print("✓ 空セル処理テスト完了")
    return True

def test_perimeter_edge_cases():
    """周囲長計算のエッジケーステスト"""
    print("\n=== 周囲長エッジケーステスト ===")
    
    config = CPM_config(l_L=1.0, L_0=6.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    print("\n--- ケース1: 全て同じIDの近傍 ---")
    # 全て同じIDの場合、局所的周囲長変化は最大になるはず
    source_perimeters = torch.tensor([[8.0]], device=device)
    target_perimeter = torch.tensor([[8.0]], device=device)
    source_ids = torch.tensor([[1, 1, 1, 1]], device=device)  # 全て同じ
    target_id = torch.tensor([[0]], device=device)
    source_is_not_empty = torch.tensor([[True, True, True, True]], device=device)
    target_is_not_empty = torch.tensor([[True]], device=device)
    
    delta_H_perimeter = cpm.calc_dH_perimeter(
        source_perimeters, target_perimeter, source_ids, target_id,
        source_is_not_empty, target_is_not_empty
    )
    print(f"全て同じID近傍の周囲長エネルギー変化: {delta_H_perimeter}")
    
    # 手動計算: 
    # ID=1が4個あるので、dL_s = 4 - 2*4 = -4
    # ID=0が0個なので、dL_t = -4 + 2*0 = -4
    l_L, L_0 = 1.0, 6.0
    dL_s, dL_t = -4, -4
    expected_s = l_L * (2*(8.0 - L_0)*dL_s + dL_s**2)
    expected_t = l_L * (2*(8.0 - L_0)*dL_t + dL_t**2)
    expected_total = expected_s + expected_t
    
    print(f"理論値: {expected_total}, 実際値: {delta_H_perimeter[0,0].item()}")
    
    print("\n--- ケース2: 全て異なるIDの近傍 ---")
    source_ids_diff = torch.tensor([[1, 2, 3, 4]], device=device)  # 全て異なる
    
    delta_H_perimeter_diff = cpm.calc_dH_perimeter(
        source_perimeters, target_perimeter, source_ids_diff, target_id,
        source_is_not_empty, target_is_not_empty
    )
    print(f"全て異なるID近傍の周囲長エネルギー変化: {delta_H_perimeter_diff}")
    
    # 各IDが1個ずつなので、dL_s = 4 - 2*1 = 2 (全て同じ)
    # dL_t = -4 + 2*0 = -4 (target_id=0は近傍にない)
    
    print("✓ 周囲長エッジケーステスト完了")
    return True

def test_large_values():
    """大きな値でのエネルギー計算テスト"""
    print("\n=== 大きな値テスト ===")
    
    config = CPM_config(l_A=0.1, A_0=1000.0, l_L=0.01, L_0=100.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    # 大きな面積値
    large_areas = torch.tensor([[1000.0, 5000.0]], device=device)
    large_target = torch.tensor([[2000.0]], device=device)
    source_is_not_empty = torch.tensor([[True, True]], device=device)
    target_is_not_empty = torch.tensor([[True]], device=device)
    
    delta_H_area_large = cpm.calc_dH_area(
        large_areas, large_target, source_is_not_empty, target_is_not_empty
    )
    
    print(f"大きな値での面積エネルギー変化: {delta_H_area_large}")
    
    # 有限値であることを確認
    assert torch.all(torch.isfinite(delta_H_area_large)), "大きな値でエネルギー計算が発散"
    
    # 相対的な大きさが正しいことを確認
    # source_areas[1] > source_areas[0] なので、delta_H_area[1] > delta_H_area[0] のはず
    assert delta_H_area_large[0,1] > delta_H_area_large[0,0], "大きな値での相対的エネルギー変化が不正確"
    
    print("✓ 大きな値テスト完了")
    return True

def run_edge_energy_tests():
    """エネルギー計算のエッジケーステストを実行"""
    print("CPM.py エネルギー計算エッジケーステスト開始\n")
    
    test_functions = [
        test_zero_energy_cases,
        test_empty_cell_handling,
        test_perimeter_edge_cases,
        test_large_values,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_func.__name__} 成功")
            else:
                failed += 1
                print(f"❌ {test_func.__name__} 失敗")
        except Exception as e:
            failed += 1
            print(f"❌ {test_func.__name__} でエラー: {e}")
            import traceback
            traceback.print_exc()
        print()
    
    print(f"\nエッジケーステスト結果: {passed}成功, {failed}失敗")
    return failed == 0

if __name__ == "__main__":
    success = run_edge_energy_tests()
    sys.exit(0 if success else 1)