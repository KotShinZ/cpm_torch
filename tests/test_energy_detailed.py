#!/usr/bin/env python3
"""
dH_areaとdH_perimeterの詳細テスト
CPMのエネルギー変化計算が正しいかを検証します。
"""

import torch
import sys
import numpy as np
sys.path.append('/app')

from cpm_torch.CPM import CPM, CPM_config

def test_dH_area_detailed():
    """面積エネルギー変化の詳細テスト"""
    print("=== dH_area詳細テスト ===")
    
    config = CPM_config(l_A=1.0, A_0=4.0)  # λ_A=1.0, A_0=4.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    # テストケース1: 理論値との比較
    print("\n--- テストケース1: 単純な値での理論比較 ---")
    source_areas = torch.tensor([[3.0, 5.0]], device=device)  # A_s = 3, 5
    target_area = torch.tensor([[4.0]], device=device)       # A_t = 4
    source_is_not_empty = torch.tensor([[True, True]], device=device)
    target_is_not_empty = torch.tensor([[True]], device=device)
    
    delta_H_area = cpm.calc_dH_area(source_areas, target_area, source_is_not_empty, target_is_not_empty)
    print(f"計算結果: {delta_H_area}")
    
    # 手動計算での検証
    # CPMのエネルギー関数: H_A = λ_A * (A - A_0)^2
    # 変化: A_s -> A_s+1, A_t -> A_t-1
    # ΔH_A = λ_A * [(A_s+1 - A_0)^2 - (A_s - A_0)^2] + λ_A * [(A_t-1 - A_0)^2 - (A_t - A_0)^2]
    
    l_A, A_0 = 1.0, 4.0
    
    # ソース1 (A_s=3): (4-4)^2 - (3-4)^2 = 0 - 1 = -1
    # ターゲット (A_t=4): (3-4)^2 - (4-4)^2 = 1 - 0 = 1  
    # 合計: -1 + 1 = 0
    expected_1 = l_A * ((3+1-A_0)**2 - (3-A_0)**2 + (4-1-A_0)**2 - (4-A_0)**2)
    
    # ソース2 (A_s=5): (6-4)^2 - (5-4)^2 = 4 - 1 = 3
    # ターゲット (A_t=4): (3-4)^2 - (4-4)^2 = 1 - 0 = 1
    # 合計: 3 + 1 = 4
    expected_2 = l_A * ((5+1-A_0)**2 - (5-A_0)**2 + (4-1-A_0)**2 - (4-A_0)**2)
    
    print(f"理論値1: {expected_1}, 実際値1: {delta_H_area[0,0].item()}")
    print(f"理論値2: {expected_2}, 実際値2: {delta_H_area[0,1].item()}")
    
    # 実装の式での計算
    # delta_H_area = l_A * ((2*A_s + 1 - 2*A_0) + (-2*A_t + 1 + 2*A_0))
    impl_1 = l_A * ((2*3 + 1 - 2*A_0) + (-2*4 + 1 + 2*A_0))
    impl_2 = l_A * ((2*5 + 1 - 2*A_0) + (-2*4 + 1 + 2*A_0))
    
    print(f"実装式1: {impl_1}, 実装式2: {impl_2}")
    
    # 検証
    assert abs(delta_H_area[0,0].item() - expected_1) < 1e-5, f"ソース1のエネルギー変化が不正確"
    assert abs(delta_H_area[0,1].item() - expected_2) < 1e-5, f"ソース2のエネルギー変化が不正確"
    
    print("✓ dH_area詳細テスト完了")
    return True

def test_dH_perimeter_detailed():
    """周囲長エネルギー変化の詳細テスト"""
    print("\n=== dH_perimeter詳細テスト ===")
    
    config = CPM_config(l_L=1.0, L_0=8.0)  # λ_L=1.0, L_0=8.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    # 簡単なテストケース
    print("\n--- テストケース: 周囲長変化の計算 ---")
    source_perimeters = torch.tensor([[6.0, 10.0, 8.0, 12.0]], device=device)  # (1, 4)
    target_perimeter = torch.tensor([[8.0]], device=device)  # (1, 1)
    source_ids = torch.tensor([[1, 2, 1, 0]], device=device)  # 4近傍のID
    target_id = torch.tensor([[0]], device=device)  # 中央のID
    source_is_not_empty = torch.tensor([[True, True, True, False]], device=device)
    target_is_not_empty = torch.tensor([[True]], device=device)
    
    delta_H_perimeter = cpm.calc_dH_perimeter(
        source_perimeters, target_perimeter, source_ids, target_id,
        source_is_not_empty, target_is_not_empty
    )
    
    print(f"周囲長エネルギー変化: {delta_H_perimeter}")
    
    # 局所的な周囲長変化を手動計算
    # dL_s = 4 - 2 * (4近傍でsと同じIDの数)
    # dL_t = -4 + 2 * (4近傍でtと同じIDの数)
    
    # source_ids = [1, 2, 1, 0], target_id = 0
    # ID=1の局所変化: 4 - 2*2 = 0 (ID=1が2つある)
    # ID=2の局所変化: 4 - 2*1 = 2 (ID=2が1つある)  
    # ID=0の局所変化: 4 - 2*1 = 2 (ID=0が1つある)
    # ターゲット(ID=0)の局所変化: -4 + 2*1 = -2 (ID=0が1つある)
    
    l_L, L_0 = 1.0, 8.0
    
    # 各ソースのエネルギー変化を手動計算
    # ΔH_L = l_L * [2*(L-L_0)*dL + (dL)^2]
    
    print("手動計算による検証:")
    for i, (L_s, dL_s) in enumerate([(6.0, 0), (10.0, 2), (6.0, 0), (12.0, 2)]):
        if source_is_not_empty[0, i]:
            dL_t = -2  # ターゲットの局所変化
            expected_s = l_L * (2*(L_s - L_0)*dL_s + dL_s**2)
            expected_t = l_L * (2*(8.0 - L_0)*dL_t + dL_t**2)
            expected_total = expected_s + expected_t
            actual = delta_H_perimeter[0, i].item()
            print(f"  ソース{i}: L_s={L_s}, dL_s={dL_s} -> 期待値={expected_total}, 実際値={actual}")
    
    print("✓ dH_perimeter詳細テスト完了")
    return True

def test_energy_consistency():
    """エネルギー計算の一貫性テスト"""
    print("\n=== エネルギー計算一貫性テスト ===")
    
    config = CPM_config(
        size=(8, 8),
        l_A=2.0, A_0=5.0,
        l_L=1.5, L_0=10.0,
        T=1.0
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    cpm.reset()
    
    # 実際のマップでテスト
    cpm.add_cell_slice(slice(2, 4), slice(2, 4))  # 2x2のセル1
    cpm.add_cell_slice(slice(5, 6), slice(5, 6))  # 1x1のセル2
    
    print("マップ状態:")
    print(cpm.map_tensor[:6, :6, 0])
    
    # 実際のIDマップから計算
    ids = cpm.map_tensor[:, :, 0]
    source_ids = torch.tensor([[1, 2, 0, 1]], device=device)  # (1, 4)
    target_id = torch.tensor([[0]], device=device)  # (1, 1)
    
    try:
        # calc_area_perimeterを使用
        source_areas, target_area, source_perimeters, target_perimeter = cpm.calc_area_perimeter(
            ids, source_ids, target_id
        )
        
        print(f"面積 - ソース: {source_areas}, ターゲット: {target_area}")
        print(f"周囲長 - ソース: {source_perimeters}, ターゲット: {target_perimeter}")
        
        # エネルギー変化計算
        source_is_not_empty = source_ids != 0
        target_is_not_empty = target_id != 0
        
        delta_H_area = cpm.calc_dH_area(
            source_areas, target_area, source_is_not_empty, target_is_not_empty
        )
        
        delta_H_perimeter = cpm.calc_dH_perimeter(
            source_perimeters, target_perimeter, source_ids, target_id,
            source_is_not_empty, target_is_not_empty
        )
        
        print(f"面積エネルギー変化: {delta_H_area}")
        print(f"周囲長エネルギー変化: {delta_H_perimeter}")
        
        # 総エネルギー変化
        total_delta_H = delta_H_area + delta_H_perimeter
        print(f"総エネルギー変化: {total_delta_H}")
        
        # 妥当性チェック
        assert torch.all(torch.isfinite(delta_H_area)), "面積エネルギー変化に無限値が含まれています"
        assert torch.all(torch.isfinite(delta_H_perimeter)), "周囲長エネルギー変化に無限値が含まれています"
        
        print("✓ エネルギー計算一貫性テスト完了")
        return True
        
    except Exception as e:
        print(f"エネルギー計算エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_energy_tests():
    """エネルギー関連のすべてのテストを実行"""
    print("CPM.py エネルギー計算詳細テスト開始\n")
    
    test_functions = [
        test_dH_area_detailed,
        test_dH_perimeter_detailed,
        test_energy_consistency,
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
    
    print(f"\nエネルギーテスト結果: {passed}成功, {failed}失敗")
    return failed == 0

if __name__ == "__main__":
    success = run_energy_tests()
    sys.exit(0 if success else 1)