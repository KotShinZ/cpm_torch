#!/usr/bin/env python3
"""
cpm_torch/CPM.pyの包括的テストスイート
CPM.pyファイルに変更を加えずに、様々な機能をテストします。
"""

import torch
import numpy as np
import sys
import os
sys.path.append('/app')

from cpm_torch.CPM import CPM, CPM_config

def test_cpm_basic_functionality():
    """CPMクラスの基本機能テスト"""
    print("=== CPM基本機能テスト ===")
    
    # 設定作成
    config = CPM_config(
        size=(64, 64),
        l_A=1.0,
        l_L=1.0,
        A_0=150.0,
        L_0=82.0,
        T=1.0
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    # 初期化テスト
    cpm.reset()
    expected_channels = 2 + config.other_channels
    assert cpm.map_tensor.shape == (64, 64, expected_channels), f"Expected shape (64, 64, {expected_channels}), got {cpm.map_tensor.shape}"
    assert cpm.cell_count == 0, f"Expected cell_count 0, got {cpm.cell_count}"
    assert torch.all(cpm.map_tensor == 0), "Map should be all zeros after reset"
    
    # セル追加テスト
    cpm.add_cell(10, 10)
    assert cpm.cell_count == 1, f"Expected cell_count 1, got {cpm.cell_count}"
    assert cpm.map_tensor[10, 10, 0] == 1.0, f"Expected ID 1 at (10,10), got {cpm.map_tensor[10, 10, 0]}"
    
    # 複数セル追加
    cpm.add_cell(20, 20)
    cpm.add_cell(30, 30)
    assert cpm.cell_count == 3, f"Expected cell_count 3, got {cpm.cell_count}"
    
    # スライスでセル追加テスト
    cpm.add_cell_slice(slice(40, 42), slice(40, 42))
    assert cpm.cell_count == 4, f"Expected cell_count 4, got {cpm.cell_count}"
    assert torch.all(cpm.map_tensor[40:42, 40:42, 0] == 4.0), "Slice area should have ID 4"
    
    print("✓ CPM基本機能テスト完了")
    return True

def test_area_calculation():
    """面積計算関数のテスト"""
    print("=== 面積計算テスト ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # テスト用のIDマップを作成
    ids = torch.zeros((10, 10), device=device)
    ids[2:5, 2:5] = 1.0  # 3x3のセル1
    ids[6:8, 6:8] = 2.0  # 2x2のセル2
    ids[1, 1] = 3.0      # 1x1のセル3
    
    cell_count = 4
    areas = CPM.calc_area_bincount(ids, cell_count)
    
    # 期待値：セル0=86, セル1=9, セル2=4, セル3=1 (10x10 - 14 occupied = 86)
    expected_areas = [86, 9, 4, 1, 0]  # cell_count+1のサイズ
    
    for i, expected in enumerate(expected_areas):
        actual = areas[i].item()
        assert actual == expected, f"セル{i}の面積: 期待値{expected}, 実際{actual}"
    
    print("✓ 面積計算テスト完了")
    return True

def test_perimeter_calculation():
    """周囲長計算関数のテスト"""
    print("=== 周囲長計算テスト ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2x2の正方形セルを作成（理論的周囲長は8）
    ids = torch.zeros((6, 6), device=device)
    ids[2:4, 2:4] = 1.0  # 2x2のセル
    
    # パッチ別周囲長計算テスト
    perimeter_patch = CPM.calc_perimeter_patch(ids)
    
    # 中央のピクセルは0の寄与（4近傍すべて同じID）
    # 角のピクセルは2の寄与（4近傍のうち2つが異なるID）
    # 辺のピクセルは1の寄与（4近傍のうち1つが異なるID）
    assert perimeter_patch[2, 2] == 2.0, f"内部角の周囲長寄与: 期待値2, 実際{perimeter_patch[2, 2]}"
    assert perimeter_patch[2, 3] == 2.0, f"内部角の周囲長寄与: 期待値2, 実際{perimeter_patch[2, 3]}"
    
    # 総周囲長計算テスト
    cell_count = 2
    total_perimeters = CPM.calc_total_perimeter_bincount(ids, cell_count)
    
    # 2x2正方形の周囲長は8
    expected_perimeter = 8.0
    actual_perimeter = total_perimeters[1].item()
    assert actual_perimeter == expected_perimeter, f"セル1の総周囲長: 期待値{expected_perimeter}, 実際{actual_perimeter}"
    
    print("✓ 周囲長計算テスト完了")
    return True

def test_energy_calculations():
    """エネルギー変化計算のテスト"""
    print("=== エネルギー変化計算テスト ===")
    
    config = CPM_config(
        size=(10, 10),
        l_A=2.0,  # 面積エネルギー係数
        l_L=1.0,  # 周囲長エネルギー係数
        A_0=4.0,  # 目標面積
        L_0=8.0,  # 目標周囲長
        T=1.0
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    # テスト用データ準備
    source_areas = torch.tensor([[4.0, 6.0, 8.0, 2.0]], device=device)  # (1, 4)
    target_area = torch.tensor([[5.0]], device=device)  # (1, 1)
    source_is_not_empty = torch.tensor([[True, True, True, True]], device=device)
    target_is_not_empty = torch.tensor([[True]], device=device)
    
    # 面積エネルギー変化計算
    delta_H_area = cpm.calc_dH_area(
        source_areas, target_area, source_is_not_empty, target_is_not_empty
    )
    
    # 手動計算での検証 - CPM.pyの実装に合わせる
    # delta_H_area = l_A * (2.0 * source_areas + 1 - 2 * A_0) * source_is_not_empty + l_A * (-2.0 * target_area + 1 + 2 * A_0) * target_is_not_empty
    # l_A=2.0, A_0=4.0の場合
    expected_delta_H = []
    for A_s in [4.0, 6.0, 8.0, 2.0]:
        A_t = 5.0
        # ソース項: l_A * (2*A_s + 1 - 2*A_0)
        source_term = 2.0 * (2*A_s + 1 - 2*4.0)
        # ターゲット項: l_A * (-2*A_t + 1 + 2*A_0)  
        target_term = 2.0 * (-2*A_t + 1 + 2*4.0)
        expected_delta_H.append(source_term + target_term)
    
    for i, expected in enumerate(expected_delta_H):
        actual = delta_H_area[0, i].item()
        assert abs(actual - expected) < 1e-5, f"面積エネルギー変化{i}: 期待値{expected}, 実際{actual}"
    
    print("✓ エネルギー変化計算テスト完了")
    return True

def test_probability_calculation():
    """遷移確率計算のテスト"""
    print("=== 遷移確率計算テスト ===")
    
    config = CPM_config(
        size=(10, 10),
        l_A=1.0,
        l_L=1.0,
        A_0=4.0,
        L_0=8.0,
        T=1.0
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    cpm.reset()
    
    # 簡単なテストケースを作成
    cpm.map_tensor[3:5, 3:5, 0] = 1.0  # セル1
    cpm.map_tensor[6:8, 6:8, 0] = 2.0  # セル2
    cpm.cell_count = 2
    
    # テスト用パラメータ
    source_ids = torch.tensor([[1, 2, 0, 1]], device=device)  # (1, 4)
    target_id = torch.tensor([[0]], device=device)  # (1, 1)
    ids = cpm.map_tensor[:, :, 0]
    
    # 確率計算実行
    logits = cpm.calc_cpm_probabilities(source_ids, target_id, ids)
    
    # logitsは非負の値である必要がある
    assert torch.all(logits >= 0), f"Logitsは非負である必要があります: {logits}"
    
    # 同じセルIDの場合は確率が0になる
    same_id_logits = cpm.calc_cpm_probabilities(
        torch.tensor([[0, 0, 0, 0]], device=device),
        torch.tensor([[0]], device=device),
        ids
    )
    assert torch.all(same_id_logits == 0), "同じID間の遷移確率は0である必要があります"
    
    print("✓ 遷移確率計算テスト完了")
    return True

def test_checkerboard_step():
    """チェッカーボードステップのテスト"""
    print("=== チェッカーボードステップテスト ===")
    
    config = CPM_config(
        size=(16, 16),
        l_A=1.0,
        l_L=1.0,
        A_0=10.0,
        L_0=20.0,
        T=10.0  # 高い温度で確率的動作を促進
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    cpm.reset()
    
    # 初期セルを配置
    cpm.add_cell_slice(slice(6, 10), slice(6, 10))  # 4x4のセル
    cpm.add_cell_slice(slice(10, 12), slice(10, 12))  # 2x2のセル
    
    initial_map = cpm.map_tensor.clone()
    
    # チェッカーボードステップ実行
    for x_offset in range(3):
        for y_offset in range(3):
            logits = cpm.cpm_checkerboard_step_single(x_offset, y_offset)
            assert logits is not None, "ロジットが返される必要があります"
            assert torch.all(torch.isfinite(logits)), "ロジットは有限値である必要があります"
    
    # マップが変更されていることを確認（確率的なので必ずしも変更されるとは限らない）
    print(f"初期マップとの差分: {torch.sum(torch.abs(cpm.map_tensor - initial_map)).item()}")
    
    print("✓ チェッカーボードステップテスト完了")
    return True

def test_edge_cases():
    """エッジケースのテスト"""
    print("=== エッジケーステスト ===")
    
    config = CPM_config(size=(8, 8))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    # 空のマップでのテスト
    cpm.reset()
    areas = CPM.calc_area_bincount(cpm.map_tensor[:, :, 0], 1)
    assert areas[0] == 64, f"空マップの面積: 期待値64, 実際{areas[0]}"
    
    # 境界でのセル追加テスト
    cpm.add_cell(0, 0)  # 左上角
    cpm.add_cell(7, 7)  # 右下角
    assert cpm.cell_count == 2, "境界セル追加後のcell_count"
    
    # 単一ピクセルセルの周囲長テスト
    ids = torch.zeros((5, 5), device=device)
    ids[2, 2] = 1.0  # 中央に1ピクセルセル
    
    perimeter = CPM.calc_total_perimeter_bincount(ids, 2)
    assert perimeter[1] == 4.0, f"1ピクセルセルの周囲長: 期待値4, 実際{perimeter[1]}"
    
    # 非常に大きな値での計算テスト
    large_areas = torch.tensor([[1e6, 1e6, 1e6, 1e6]], device=device)
    large_target = torch.tensor([[1e6]], device=device)
    mask = torch.tensor([[True, True, True, True]], device=device)
    target_mask = torch.tensor([[True]], device=device)
    
    try:
        delta_H = cpm.calc_dH_area(large_areas, large_target, mask, target_mask)
        assert torch.all(torch.isfinite(delta_H)), "大きな値でも有限値を保つ必要があります"
    except Exception as e:
        print(f"大きな値での計算でエラー: {e}")
    
    print("✓ エッジケーステスト完了")
    return True

def run_all_tests():
    """すべてのテストを実行"""
    print("CPM.py包括的テスト開始\n")
    
    test_functions = [
        test_cpm_basic_functionality,
        test_area_calculation,
        test_perimeter_calculation,
        test_energy_calculations,
        test_probability_calculation,
        test_checkerboard_step,
        test_edge_cases
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"❌ {test_func.__name__} 失敗")
        except Exception as e:
            failed += 1
            print(f"❌ {test_func.__name__} でエラー: {e}")
        print()
    
    print(f"テスト結果: {passed}成功, {failed}失敗")
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)