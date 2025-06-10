#!/usr/bin/env python3
"""
cpm_torch/CPM.pyの実際に動作するテストスイート
CPM.pyの現在の実装に合わせてテストを作成します。
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
    
    cpm.add_cell(10, 10)
    assert cpm.cell_count == 2, "2つ目のセル追加後のセル数は2である必要があります"
    assert cpm.map_tensor[10, 10, 0] == 2.0, "セル2は正しい位置に配置される必要があります"
    
    # スライスでのセル追加テスト
    cpm.add_cell_slice(slice(5, 7), slice(5, 7))
    assert cpm.cell_count == 3, "スライス追加後のセル数は3である必要があります"
    assert torch.all(cpm.map_tensor[5:7, 5:7, 0] == 3.0), "スライス領域はすべて同じIDである必要があります"
    
    print("✓ 基本機能テスト完了")
    return True

def test_area_calculation():
    """面積計算のテスト"""
    print("=== 面積計算テスト ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 明確なテストケース作成
    ids = torch.zeros((8, 8), device=device)
    ids[2:4, 2:4] = 1.0  # 2x2のセル1 (面積4)
    ids[5:7, 5:7] = 2.0  # 2x2のセル2 (面積4) 
    ids[1, 1] = 3.0      # 1x1のセル3 (面積1)
    
    cell_count = 4
    areas = CPM.calc_area_bincount(ids, cell_count)
    
    print(f"計算された面積: {areas}")
    
    # 各セルの面積をチェック
    expected_area_0 = 8*8 - 4 - 4 - 1  # 背景領域 (55)
    assert areas[0] == expected_area_0, f"セル0の面積: 期待値{expected_area_0}, 実際{areas[0]}"
    assert areas[1] == 4.0, f"セル1の面積: 期待値4, 実際{areas[1]}"
    assert areas[2] == 4.0, f"セル2の面積: 期待値4, 実際{areas[2]}"
    assert areas[3] == 1.0, f"セル3の面積: 期待値1, 実際{areas[3]}"
    
    print("✓ 面積計算テスト完了")
    return True

def test_perimeter_calculation():
    """周囲長計算のテスト"""
    print("=== 周囲長計算テスト ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 単純な2x2正方形
    ids = torch.zeros((6, 6), device=device)
    ids[2:4, 2:4] = 1.0
    
    cell_count = 2
    perimeters = CPM.calc_total_perimeter_bincount(ids, cell_count)
    
    print(f"計算された周囲長: {perimeters}")
    
    # 2x2正方形の周囲長は8
    assert perimeters[1] == 8.0, f"2x2セルの周囲長: 期待値8, 実際{perimeters[1]}"
    
    # より複雑なケース: L字型
    ids2 = torch.zeros((6, 6), device=device)
    ids2[2:4, 2:3] = 1.0  # 縦2ピクセル
    ids2[3:4, 3:4] = 1.0  # 右下1ピクセル
    
    perimeters2 = CPM.calc_total_perimeter_bincount(ids2, 2)
    print(f"L字型の周囲長: {perimeters2}")
    
    print("✓ 周囲長計算テスト完了")
    return True

def test_energy_calculation():
    """エネルギー計算のテスト"""
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
    
    # 手動計算での検証
    # ソース項: l_A * (2*A_s + 1 - 2*A_0) = 2.0 * (2*4 + 1 - 2*4) = 2.0 * 1 = 2.0
    # ターゲット項: l_A * (-2*A_t + 1 + 2*A_0) = 2.0 * (-2*5 + 1 + 2*4) = 2.0 * (-1) = -2.0
    # 合計: 2.0 + (-2.0) = 0.0
    expected_first = 2.0 * (2*4 + 1 - 2*4.0) + 2.0 * (-2*5 + 1 + 2*4.0)
    
    assert abs(delta_H_area[0, 0].item() - expected_first) < 1e-5, f"最初のエネルギー変化: 期待値{expected_first}, 実際{delta_H_area[0, 0].item()}"
    
    print("✓ エネルギー計算テスト完了")
    return True

def test_probability_calculation():
    """確率計算のテスト（エラーなしバージョン）"""
    print("=== 確率計算テスト ===")
    
    config = CPM_config(size=(8, 8), T=1.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    cpm.reset()
    
    # 簡単なセル配置
    cpm.add_cell_slice(slice(3, 5), slice(3, 5))  # 2x2のセル
    
    # 確率計算に必要なパラメータを手動で準備
    source_ids = torch.tensor([[1, 0, 0, 1]], device=device)  # (1, 4)
    target_id = torch.tensor([[0]], device=device)  # (1, 1)
    ids = cpm.map_tensor[:, :, 0]
    
    try:
        # dH_NNなしで確率計算
        logits = cpm.calc_cpm_probabilities(source_ids, target_id, ids)
        assert torch.all(torch.isfinite(logits)), "確率は有限値である必要があります"
        assert torch.all(logits >= 0), "確率は非負である必要があります"
        print(f"計算された確率: {logits}")
        
        print("✓ 確率計算テスト完了")
        return True
    except Exception as e:
        print(f"確率計算エラー: {e}")
        # エラーが発生しても、他のテストは続行
        return False

def test_checkerboard_alternatives():
    """チェッカーボード以外のステップ関数をテスト"""
    print("=== 代替ステップ関数テスト ===")
    
    config = CPM_config(size=(16, 16), T=10.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    cpm.reset()
    
    # セルを配置
    cpm.add_cell_slice(slice(6, 8), slice(6, 8))
    
    initial_map = cpm.map_tensor.clone()
    
    try:
        # MCUステップを試す
        cpm.cpm_mcs_step()
        print("MCSステップ実行成功")
        
        # 変更量を確認
        diff = torch.sum(torch.abs(cpm.map_tensor - initial_map))
        print(f"マップ変更量: {diff.item()}")
        
        return True
    except Exception as e:
        print(f"代替ステップ関数エラー: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """すべてのテストを実行"""
    print("CPM.py動作テスト開始\n")
    
    test_functions = [
        test_basic_functionality,
        test_area_calculation,
        test_perimeter_calculation,
        test_energy_calculation,
        test_probability_calculation,
        test_checkerboard_alternatives,
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
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)