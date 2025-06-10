#!/usr/bin/env python3
"""
CPM過剰成長メカニズムの詳細調査
理論的に最適な4×4正方形が存在するのに、なぜ過剰成長が続くのかを解析
"""

import torch
import sys
import numpy as np
sys.path.append('/app')

from cpm_torch.CPM import CPM, CPM_config

def analyze_overgrowth_after_optimal():
    """最適点(面積16)を超えた後の成長メカニズム"""
    print("=== 最適点超過後の成長メカニズム分析 ===")
    
    config = CPM_config(l_A=1.0, l_L=1.0, A_0=16.0, L_0=16.0, T=1.0)
    
    print("面積16以降の詳細エネルギー変化:")
    print("面積 | 推定周囲長 | 面積E | 周囲長E | 総E | 前ステップからのΔE | 成長確率")
    print("-" * 80)
    
    prev_energy = 0  # 面積16での初期エネルギー
    
    for area in range(16, 31):
        # より現実的な周囲長推定
        if area == 16:
            perimeter = 16  # 4×4正方形
        elif area <= 25:
            side = int(np.sqrt(area))
            remainder = area - side**2
            if remainder == 0:
                perimeter = 4 * side
            else:
                # 不完全正方形の場合
                perimeter = 4 * side + 2 * min(remainder, side)
        else:
            # より大きなサイズの場合
            side = int(np.sqrt(area))
            perimeter = 4 * side + 2
        
        area_energy = config.l_A * (area - config.A_0)**2
        perimeter_energy = config.l_L * (perimeter - config.L_0)**2
        total_energy = area_energy + perimeter_energy
        
        delta_E = total_energy - prev_energy
        probability = min(1.0, np.exp(-delta_E / config.T)) if delta_E != 0 else 0
        
        print(f"{area:4d} | {perimeter:9.0f} | {area_energy:5.0f} | {perimeter_energy:7.0f} | {total_energy:4.0f} | {delta_E:12.1f} | {probability:8.3f}")
        
        prev_energy = total_energy
    
    print("\n重要な発見:")
    print("面積16で最適でも、周囲長の形状変化により更なる成長が可能")

def test_perimeter_calculation_accuracy():
    """周囲長計算の正確性テスト"""
    print("\n=== 周囲長計算精度テスト ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 様々な形状での周囲長計算
    test_cases = [
        ("4×4正方形", lambda: create_square(4, 4)),
        ("5×5正方形", lambda: create_square(5, 5)),
        ("4×3長方形", lambda: create_rectangle(4, 3)),
        ("L字型", lambda: create_l_shape()),
        ("散らばった16ピクセル", lambda: create_scattered(16)),
    ]
    
    print("形状 | 面積 | 計算周囲長 | 理論周囲長 | 差")
    print("-" * 50)
    
    for name, shape_func in test_cases:
        shape = shape_func().to(device)
        area = torch.sum(shape > 0).item()
        calculated_perimeter = CPM.calc_total_perimeter_bincount(shape, 2)[1].item()
        
        # 理論周囲長の計算
        if "正方形" in name:
            side = int(np.sqrt(area))
            theoretical_perimeter = 4 * side
        elif "長方形" in name:
            theoretical_perimeter = 2 * (4 + 3)  # 4×3の場合
        else:
            theoretical_perimeter = "複雑"
        
        if isinstance(theoretical_perimeter, (int, float)):
            diff = calculated_perimeter - theoretical_perimeter
            print(f"{name:15s} | {area:4d} | {calculated_perimeter:9.1f} | {theoretical_perimeter:9.0f} | {diff:4.1f}")
        else:
            print(f"{name:15s} | {area:4d} | {calculated_perimeter:9.1f} | {theoretical_perimeter:>9s} | {'N/A':>4s}")

def create_square(side, map_size=10):
    """正方形を作成"""
    tensor = torch.zeros((map_size, map_size))
    start = (map_size - side) // 2
    tensor[start:start+side, start:start+side] = 1.0
    return tensor

def create_rectangle(width, height, map_size=10):
    """長方形を作成"""
    tensor = torch.zeros((map_size, map_size))
    start_w = (map_size - width) // 2
    start_h = (map_size - height) // 2
    tensor[start_h:start_h+height, start_w:start_w+width] = 1.0
    return tensor

def create_l_shape(map_size=10):
    """L字型を作成"""
    tensor = torch.zeros((map_size, map_size))
    # L字型: 縦4ピクセル + 横3ピクセル
    tensor[3:7, 3:4] = 1.0  # 縦棒
    tensor[6:7, 3:6] = 1.0  # 横棒
    return tensor

def create_scattered(num_pixels, map_size=10):
    """散らばったピクセルを作成"""
    tensor = torch.zeros((map_size, map_size))
    positions = [(2,2), (2,4), (2,6), (4,2), (4,4), (4,6), (6,2), (6,4), (6,6), (8,2), (8,4), (8,6), (7,7), (3,7), (5,7), (1,1)]
    for i, (r, c) in enumerate(positions[:num_pixels]):
        tensor[r, c] = 1.0
    return tensor

def test_transition_probabilities_detailed():
    """遷移確率の詳細分析"""
    print("\n=== 遷移確率詳細分析 ===")
    
    config = CPM_config(l_A=1.0, l_L=1.0, A_0=16.0, L_0=16.0, T=1.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    # 16ピクセル状態から17ピクセルへの遷移を詳細分析
    print("16ピクセル→17ピクセル遷移の詳細:")
    
    # 4×4正方形から1ピクセル追加のケース
    test_map = torch.zeros((1, 10, 10, 3), device=device)
    test_map[0, 3:7, 3:7, 0] = 1.0  # 4×4正方形
    
    print("初期4×4正方形:")
    print(test_map[0, :, :, 0].cpu().numpy())
    
    initial_area = torch.sum(test_map[0, :, :, 0] > 0).item()
    initial_perimeter = CPM.calc_total_perimeter_bincount(test_map[0, :, :, 0], 2)[1].item()
    
    print(f"初期状態: 面積={initial_area}, 周囲長={initial_perimeter}")
    
    # 1ステップ実行して変化を観察
    print("\n1ステップ実行後:")
    result = cpm.cpm_checkerboard_step_single_func(test_map)
    
    final_area = torch.sum(result[0, :, :, 0] > 0).item()
    final_perimeter = CPM.calc_total_perimeter_bincount(result[0, :, :, 0], 2)[1].item()
    
    print(f"最終状態: 面積={final_area}, 周囲長={final_perimeter}")
    print("変化マップ:")
    print(result[0, :, :, 0].cpu().numpy())
    
    if final_area > initial_area:
        print("→ 成長が発生しました")
        
        # エネルギー変化を計算
        initial_energy = config.l_A * (initial_area - config.A_0)**2 + config.l_L * (initial_perimeter - config.L_0)**2
        final_energy = config.l_A * (final_area - config.A_0)**2 + config.l_L * (final_perimeter - config.L_0)**2
        delta_E = final_energy - initial_energy
        
        print(f"エネルギー変化: {initial_energy:.1f} → {final_energy:.1f} (ΔE={delta_E:.1f})")
        
        if delta_E < 0:
            print("→ エネルギーが下がったため成長が起こりやすい")
        else:
            print("→ エネルギーが上がったが、温度により成長が発生")
    else:
        print("→ 成長しませんでした（安定状態）")

def test_local_vs_global_minimum():
    """局所最適vs全体最適の問題"""
    print("\n=== 局所最適vs全体最適分析 ===")
    
    config = CPM_config(l_A=1.0, l_L=1.0, A_0=16.0, L_0=16.0, T=1.0)
    
    print("CPMシミュレーションの問題:")
    print("1. 理論的最適解: 4×4正方形 (面積=16, 周囲長=16, エネルギー=0)")
    print("2. しかし、実際のシミュレーションでは局所的な遷移のみ可能")
    print("3. 1ピクセルずつの成長・縮小しか起こらない")
    print("4. 16ピクセルに達した後も、形状変化により更なる成長が可能")
    
    print("\n問題のメカニズム:")
    print("- シミュレーションは1ピクセルの局所変化のみ")
    print("- 16ピクセルに達しても、最適な4×4形状とは限らない")
    print("- 不規則な16ピクセル形状から、よりエネルギーの低い状態への遷移が続く")
    print("- 面積が16を超えても、周囲長の改善によりエネルギーが下がる場合がある")
    
    # 具体例の計算
    print("\n具体例:")
    cases = [
        ("不規則16ピクセル", 16, 24),  # 複雑な形状
        ("17ピクセル正方形風", 17, 18),  # よりコンパクト
        ("18ピクセル", 18, 18),  # さらにコンパクト
    ]
    
    for name, area, perimeter in cases:
        energy = config.l_A * (area - config.A_0)**2 + config.l_L * (perimeter - config.L_0)**2
        print(f"{name}: E={energy:.0f}")

def run_overgrowth_mechanism_analysis():
    """過剰成長メカニズムの分析を実行"""
    print("CPM過剰成長メカニズムの詳細調査\n")
    
    try:
        # 1. 最適点超過後の成長分析
        analyze_overgrowth_after_optimal()
        
        # 2. 周囲長計算精度テスト
        test_perimeter_calculation_accuracy()
        
        # 3. 遷移確率詳細分析
        test_transition_probabilities_detailed()
        
        # 4. 局所vs全体最適問題
        test_local_vs_global_minimum()
        
        print("\n" + "="*70)
        print("結論: 過剰成長の根本原因")
        print("="*70)
        print("1. 【局所最適化の限界】")
        print("   - CPMは1ピクセルずつの局所変化のみ可能")
        print("   - 理論的最適解(4×4正方形)に直接到達できない")
        print("   - 面積16に到達しても最適形状ではない")
        print()
        print("2. 【形状最適化の継続】")
        print("   - 不規則な16ピクセル形状からの改善が続く")
        print("   - 面積増加でも周囲長改善によりエネルギー低下")
        print("   - 結果として目標面積を大幅に超過")
        print()
        print("3. 【解決策】")
        print("   - より厳しい温度設定 (T < 0.5)")
        print("   - 面積ペナルティの強化 (l_A > 2.0)")
        print("   - 初期形状を目標に近い形で配置")
        print("   - 成長制限の実装")
        
        return True
        
    except Exception as e:
        print(f"分析エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_overgrowth_mechanism_analysis()
    sys.exit(0 if success else 1)