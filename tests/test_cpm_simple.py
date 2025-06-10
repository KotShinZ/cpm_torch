#!/usr/bin/env python3
"""
cpm_torch/CPM.pyの簡潔なテストスイート
CPM.pyファイルに変更を加えずに、基本的な機能をテストします。
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
    print(f"マップ形状: {cpm.map_tensor.shape}")
    print(f"初期セル数: {cpm.cell_count}")
    
    # セル追加テスト
    cpm.add_cell(15, 15)
    cpm.add_cell(10, 10)
    print(f"セル追加後のセル数: {cpm.cell_count}")
    
    # 面積計算テスト
    areas = CPM.calc_area_bincount(cpm.map_tensor[:, :, 0], cpm.cell_count)
    print(f"面積: {areas}")
    
    # 周囲長計算テスト
    perimeters = CPM.calc_total_perimeter_bincount(cpm.map_tensor[:, :, 0], cpm.cell_count)
    print(f"周囲長: {perimeters}")
    
    print("✓ 基本機能テスト完了")
    return True

def test_area_calculation():
    """面積計算のテスト"""
    print("=== 面積計算テスト ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 簡単なテストケース
    ids = torch.zeros((6, 6), device=device)
    ids[2:4, 2:4] = 1.0  # 2x2のセル
    ids[4, 4] = 2.0     # 1x1のセル
    
    areas = CPM.calc_area_bincount(ids, 3)
    print(f"計算された面積: {areas}")
    
    # セル1は4ピクセル、セル2は1ピクセル
    assert areas[1] == 4.0, f"セル1の面積: 期待値4, 実際{areas[1]}"
    assert areas[2] == 1.0, f"セル2の面積: 期待値1, 実際{areas[2]}"
    
    print("✓ 面積計算テスト完了")
    return True

def test_perimeter_calculation():
    """周囲長計算のテスト"""
    print("=== 周囲長計算テスト ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2x2の正方形
    ids = torch.zeros((6, 6), device=device)
    ids[2:4, 2:4] = 1.0
    
    perimeters = CPM.calc_total_perimeter_bincount(ids, 2)
    print(f"計算された周囲長: {perimeters}")
    
    # 2x2正方形の周囲長は8
    assert perimeters[1] == 8.0, f"2x2セルの周囲長: 期待値8, 実際{perimeters[1]}"
    
    print("✓ 周囲長計算テスト完了")
    return True

def test_step_function():
    """ステップ関数のテスト"""
    print("=== ステップ関数テスト ===")
    
    config = CPM_config(size=(16, 16), T=10.0)  # 高い温度
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    cpm.reset()
    
    # セルを配置
    cpm.add_cell_slice(slice(6, 8), slice(6, 8))  # 2x2のセル
    
    initial_ids = cpm.map_tensor[:, :, 0].clone()
    
    # 1ステップ実行
    try:
        logits = cpm.cpm_checkerboard_step_single(0, 0)
        print(f"ステップ実行成功: logits shape = {logits.shape}")
        
        # マップが変更されたかチェック
        diff = torch.sum(torch.abs(cpm.map_tensor[:, :, 0] - initial_ids))
        print(f"マップ変更量: {diff.item()}")
        
    except Exception as e:
        print(f"ステップ実行エラー: {e}")
        return False
    
    print("✓ ステップ関数テスト完了")
    return True

def run_all_tests():
    """すべてのテストを実行"""
    print("CPM.py簡潔テスト開始\n")
    
    test_functions = [
        test_basic_functionality,
        test_area_calculation,
        test_perimeter_calculation,
        test_step_function,
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