#!/usr/bin/env python3
"""
修正されたcpm_torch/CPM.pyの最終テストスイート
実際の実装に合わせて動作することを確認します。
"""

import torch
import sys
import traceback
sys.path.append('/app')

from cpm_torch.CPM import CPM, CPM_config

def test_basic_functionality():
    """CPMクラスの基本機能テスト"""
    print("=== CPM基本機能テスト ===")
    
    config = CPM_config(size=(32, 32), l_A=1.0, l_L=1.0, A_0=10.0, L_0=20.0, T=1.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    # リセットテスト
    cpm.reset()
    assert cpm.cell_count == 0, "初期セル数は0である必要があります"
    assert torch.all(cpm.map_tensor[:, :, 0] == 0), "初期マップはすべて0である必要があります"
    
    # セル追加テスト
    cpm.add_cell(15, 15)
    assert cpm.cell_count == 1, "セル追加後のセル数は1である必要があります"
    assert cpm.map_tensor[15, 15, 0] == 1.0, "セル1は正しい位置に配置される必要があります"
    
    # 複数セル追加
    cpm.add_cell(10, 10)
    assert cpm.cell_count == 2, "2つ目のセル追加後のセル数は2である必要があります"
    
    # スライスでのセル追加テスト
    cpm.add_cell_slice(slice(5, 7), slice(5, 7))
    assert cpm.cell_count == 3, "スライス追加後のセル数は3である必要があります"
    
    print("✓ 基本機能テスト完了")
    return True

def test_area_and_perimeter():
    """面積と周囲長の統合テスト"""
    print("=== 面積・周囲長統合テスト ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 複数の形状をテスト
    ids = torch.zeros((10, 10), device=device)
    ids[2:4, 2:4] = 1.0    # 2x2正方形 (面積4, 周囲長8)
    ids[6:9, 6:7] = 2.0    # 3x1長方形 (面積3, 周囲長8)
    ids[1, 1] = 3.0        # 1x1点 (面積1, 周囲長4)
    
    cell_count = 4
    areas = CPM.calc_area_bincount(ids, cell_count)
    perimeters = CPM.calc_total_perimeter_bincount(ids, cell_count)
    
    print(f"面積: {areas}")
    print(f"周囲長: {perimeters}")
    
    # 期待値チェック
    assert areas[1] == 4.0, f"2x2正方形の面積: 期待値4, 実際{areas[1]}"
    assert areas[2] == 3.0, f"3x1長方形の面積: 期待値3, 実際{areas[2]}"
    assert areas[3] == 1.0, f"1x1点の面積: 期待値1, 実際{areas[3]}"
    
    assert perimeters[1] == 8.0, f"2x2正方形の周囲長: 期待値8, 実際{perimeters[1]}"
    assert perimeters[3] == 4.0, f"1x1点の周囲長: 期待値4, 実際{perimeters[3]}"
    
    print("✓ 面積・周囲長統合テスト完了")
    return True

def test_energy_calculations():
    """修正されたエネルギー計算テスト"""
    print("=== エネルギー計算テスト ===")
    
    config = CPM_config(l_A=2.0, l_L=1.0, A_0=4.0, L_0=8.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    # テスト用データ
    source_areas = torch.tensor([[4.0, 6.0]], device=device)  # (1, 2)
    target_area = torch.tensor([[5.0]], device=device)  # (1, 1)
    source_is_not_empty = torch.tensor([[True, True]], device=device)
    target_is_not_empty = torch.tensor([[True]], device=device)
    
    # 面積エネルギー変化計算
    delta_H_area = cpm.calc_dH_area(
        source_areas, target_area, source_is_not_empty, target_is_not_empty
    )
    
    print(f"面積エネルギー変化: {delta_H_area}")
    
    # 修正された式での手動計算
    # delta_H_area = l_A * ((2*A_s + 1 - 2*A_0) * source_is_not_empty + (-2*A_t + 1 + 2*A_0) * target_is_not_empty)
    l_A, A_0 = 2.0, 4.0
    A_s1, A_s2, A_t = 4.0, 6.0, 5.0
    
    expected_1 = l_A * ((2*A_s1 + 1 - 2*A_0) + (-2*A_t + 1 + 2*A_0))
    expected_2 = l_A * ((2*A_s2 + 1 - 2*A_0) + (-2*A_t + 1 + 2*A_0))
    
    assert abs(delta_H_area[0, 0].item() - expected_1) < 1e-5, f"第1エネルギー変化: 期待値{expected_1}, 実際{delta_H_area[0, 0].item()}"
    assert abs(delta_H_area[0, 1].item() - expected_2) < 1e-5, f"第2エネルギー変化: 期待値{expected_2}, 実際{delta_H_area[0, 1].item()}"
    
    print("✓ エネルギー計算テスト完了")
    return True

def test_simple_probability():
    """シンプルな確率計算テスト"""
    print("=== シンプル確率計算テスト ===")
    
    config = CPM_config(size=(8, 8), T=1.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    cpm.reset()
    
    # 簡単なセル配置
    cpm.add_cell_slice(slice(3, 5), slice(3, 5))  # 2x2のセル1
    
    try:
        # 単純な4近傍ケースをテスト
        source_ids = torch.tensor([[1, 0, 0, 1]], device=device)  # (1, 4)
        target_id = torch.tensor([[0]], device=device)  # (1, 1)
        ids = cpm.map_tensor[:, :, 0]
        
        # 確率計算実行
        logits = cpm.calc_cpm_probabilities(source_ids, target_id, ids)
        
        assert torch.all(torch.isfinite(logits)), "確率は有限値である必要があります"
        assert torch.all(logits >= 0), "確率は非負である必要があります"
        print(f"計算された確率: {logits.shape} = {logits}")
        
        print("✓ シンプル確率計算テスト完了")
        return True
        
    except Exception as e:
        print(f"確率計算エラー: {e}")
        traceback.print_exc()
        return False

def test_checkerboard_basic():
    """基本的なチェッカーボードステップテスト"""
    print("=== 基本チェッカーボードテスト ===")
    
    config = CPM_config(size=(12, 12), T=10.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    cpm.reset()
    
    # セルを配置
    cpm.add_cell_slice(slice(5, 7), slice(5, 7))  # 2x2のセル
    
    initial_map = cpm.map_tensor.clone()
    
    try:
        # 通常のチェッカーボードステップ（全候補版）を試す
        logits = cpm.cpm_checkerboard_step(0, 0)
        
        print(f"チェッカーボードステップ成功: logits shape = {logits.shape}")
        
        # マップの変更を確認
        diff = torch.sum(torch.abs(cpm.map_tensor - initial_map))
        print(f"マップ変更量: {diff.item()}")
        
        return True
        
    except Exception as e:
        print(f"チェッカーボードステップエラー: {e}")
        traceback.print_exc()
        return False

def test_edge_cases():
    """エッジケースのテスト"""
    print("=== エッジケーステスト ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 空のマップ
    ids_empty = torch.zeros((5, 5), device=device)
    areas_empty = CPM.calc_area_bincount(ids_empty, 1)
    assert areas_empty[0] == 25, f"空マップの面積: 期待値25, 実際{areas_empty[0]}"
    
    # 全て同じIDのマップ
    ids_full = torch.ones((4, 4), device=device)
    areas_full = CPM.calc_area_bincount(ids_full, 2)
    perimeters_full = CPM.calc_total_perimeter_bincount(ids_full, 2)
    assert areas_full[1] == 16, f"全面セルの面積: 期待値16, 実際{areas_full[1]}"
    assert perimeters_full[1] == 0, f"全面セルの周囲長: 期待値0, 実際{perimeters_full[1]}"
    
    # 境界セル
    config = CPM_config(size=(6, 6))
    cpm = CPM(config, device)
    cpm.reset()
    cpm.add_cell(0, 0)  # 左上角
    cpm.add_cell(5, 5)  # 右下角
    assert cpm.cell_count == 2, "境界セル追加成功"
    
    print("✓ エッジケーステスト完了")
    return True

def run_all_tests():
    """すべてのテストを実行"""
    print("CPM.py最終テスト開始\n")
    
    test_functions = [
        test_basic_functionality,
        test_area_and_perimeter,
        test_energy_calculations,
        test_simple_probability,
        test_checkerboard_basic,
        test_edge_cases,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            print(f"\n--- {test_func.__name__} ---")
            if test_func():
                passed += 1
                print(f"✅ {test_func.__name__} 成功")
            else:
                failed += 1
                print(f"❌ {test_func.__name__} 失敗")
        except Exception as e:
            failed += 1
            print(f"❌ {test_func.__name__} でエラー: {e}")
            traceback.print_exc()
        print()
    
    print(f"\n最終結果: {passed}成功, {failed}失敗")
    print(f"成功率: {passed}/{passed+failed} ({100*passed/(passed+failed):.1f}%)")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)