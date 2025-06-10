#!/usr/bin/env python3
"""
A_0=16, L_0=16設定での異常成長の分析
4x4正方形（面積16, 周囲長16）を目標とするが、実際は過剰成長する問題を調査
"""

import torch
import sys
import numpy as np
sys.path.append('/app')

from cpm_torch.CPM import CPM, CPM_config

def analyze_16x16_target_energy():
    """16x16目標での理論的エネルギー分析"""
    print("=== A_0=16, L_0=16でのエネルギー分析 ===")
    
    config = CPM_config(l_A=1.0, l_L=1.0, A_0=16.0, L_0=16.0, T=1.0)
    
    print("4×4正方形の理論値:")
    print(f"  面積: 16ピクセル")
    print(f"  周囲長: 16 (4×4の境界)")
    print(f"  面積エネルギー: {config.l_A * (16 - config.A_0)**2} (理想的には0)")
    print(f"  周囲長エネルギー: {config.l_L * (16 - config.L_0)**2} (理想的には0)")
    
    # 実際の形状での周囲長計算
    print("\n実際の形状パターン別周囲長:")
    
    patterns = [
        ("4×4正方形", 16, 16),
        ("不規則な16ピクセル", 16, 20),  # 不規則な形状
        ("L字型16ピクセル", 16, 22),
        ("散らばった16ピクセル", 16, 30),
        ("20ピクセル正方形風", 20, 18),
        ("25ピクセル", 25, 20),
        ("30ピクセル", 30, 22),
        ("40ピクセル", 40, 26),
    ]
    
    print("形状 | 面積 | 周囲長 | 面積E | 周囲長E | 総E")
    print("-" * 55)
    
    min_energy = float('inf')
    best_pattern = None
    
    for name, area, perimeter in patterns:
        area_energy = config.l_A * (area - config.A_0)**2
        perimeter_energy = config.l_L * (perimeter - config.L_0)**2
        total_energy = area_energy + perimeter_energy
        
        print(f"{name:15s} | {area:4d} | {perimeter:6d} | {area_energy:5.0f} | {perimeter_energy:7.0f} | {total_energy:5.0f}")
        
        if total_energy < min_energy:
            min_energy = total_energy
            best_pattern = (name, area, perimeter)
    
    print(f"\n最小エネルギー状態: {best_pattern[0]} (面積={best_pattern[1]}, 周囲長={best_pattern[2]}, E={min_energy:.0f})")
    
    return best_pattern

def test_actual_simulation_analysis():
    """実際のシミュレーションを段階的に分析"""
    print("\n=== 実際のシミュレーション分析 ===")
    
    config = CPM_config(
        l_A=1.0, l_L=1.0, A_0=16.0, L_0=16.0, T=1.0, size=(16, 16)
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    # 初期状態
    tensor = torch.zeros((1, 16, 16, 3), device=device)
    tensor[0, 8, 8, 0] = 1.0
    
    print("シミュレーション進行:")
    print("ステップ | 面積 | 周囲長 | 面積E | 周囲長E | 総E | 成長傾向")
    print("-" * 70)
    
    growth_data = []
    
    for step in range(0, 201, 20):
        # 現在の状態を分析
        ids = tensor[0, :, :, 0]
        area = torch.sum(ids > 0).item()
        
        if area > 0:
            perimeter = CPM.calc_total_perimeter_bincount(ids, 2)[1].item()
        else:
            perimeter = 0
        
        area_energy = config.l_A * (area - config.A_0)**2
        perimeter_energy = config.l_L * (perimeter - config.L_0)**2
        total_energy = area_energy + perimeter_energy
        
        # 成長傾向分析
        if step == 0:
            growth_trend = "初期"
        else:
            prev_area = growth_data[-1]['area']
            if area > prev_area:
                growth_trend = "成長中"
            elif area == prev_area:
                growth_trend = "安定"
            else:
                growth_trend = "縮小"
        
        growth_data.append({
            'step': step,
            'area': area,
            'perimeter': perimeter,
            'area_energy': area_energy,
            'perimeter_energy': perimeter_energy,
            'total_energy': total_energy
        })
        
        print(f"{step:6d} | {area:4d} | {perimeter:6.1f} | {area_energy:5.0f} | {perimeter_energy:7.0f} | {total_energy:5.0f} | {growth_trend}")
        
        # 20ステップ実行
        if step < 200:
            for _ in range(20):
                tensor = cpm.cpm_checkerboard_step_single_func(tensor)
    
    # 最終分析
    final_data = growth_data[-1]
    target_area = 16
    actual_area = final_data['area']
    
    print(f"\n最終結果:")
    print(f"  目標面積: {target_area}")
    print(f"  実際面積: {actual_area}")
    print(f"  面積比: {actual_area/target_area:.1f}倍")
    print(f"  最終エネルギー: {final_data['total_energy']:.0f}")
    
    if actual_area > target_area * 1.5:
        print("  ❌ 明らかな過剰成長")
    elif actual_area > target_area * 1.2:
        print("  ⚠️ 軽微な過剰成長")
    else:
        print("  ✅ 適切な成長")
    
    return growth_data

def analyze_growth_driving_force():
    """成長の駆動力を詳細分析"""
    print("\n=== 成長駆動力の分析 ===")
    
    config = CPM_config(l_A=1.0, l_L=1.0, A_0=16.0, L_0=16.0, T=1.0)
    
    print("面積別エネルギー変化の分析:")
    print("現在面積 | 次面積 | 現在E | 次E | ΔE | 成長確率")
    print("-" * 60)
    
    for current_area in [1, 5, 10, 15, 16, 17, 20, 25, 30]:
        next_area = current_area + 1
        
        # 周囲長の推定（正方形に近い形状を仮定）
        def estimate_perimeter(area):
            if area == 1:
                return 4
            else:
                side = int(np.sqrt(area))
                if side**2 == area:
                    return 4 * side  # 完全な正方形
                else:
                    return 4 * side + 2  # 不完全な正方形
        
        current_perim = estimate_perimeter(current_area)
        next_perim = estimate_perimeter(next_area)
        
        current_energy = config.l_A * (current_area - config.A_0)**2 + config.l_L * (current_perim - config.L_0)**2
        next_energy = config.l_A * (next_area - config.A_0)**2 + config.l_L * (next_perim - config.L_0)**2
        
        delta_E = next_energy - current_energy
        probability = min(1.0, np.exp(-delta_E / config.T))
        
        print(f"{current_area:8d} | {next_area:6d} | {current_energy:6.0f} | {next_energy:4.0f} | {delta_E:6.1f} | {probability:8.3f}")
    
    print("\n重要な観察:")
    print("- 面積16付近での成長確率を確認")
    print("- 面積16で本当に安定するかを検証")

def test_ideal_4x4_square():
    """理想的な4×4正方形でのエネルギー計算"""
    print("\n=== 理想的4×4正方形の検証 ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 4×4正方形を作成
    test_map = torch.zeros((8, 8), device=device)
    test_map[2:6, 2:6] = 1.0  # 中央に4×4正方形
    
    print("4×4正方形の実際の計算:")
    print("マップ:")
    print(test_map.cpu().numpy())
    
    # 面積計算
    area = torch.sum(test_map > 0).item()
    perimeter = CPM.calc_total_perimeter_bincount(test_map, 2)[1].item()
    
    print(f"実際の面積: {area}")
    print(f"実際の周囲長: {perimeter}")
    
    # エネルギー計算
    config = CPM_config(l_A=1.0, l_L=1.0, A_0=16.0, L_0=16.0, T=1.0)
    area_energy = config.l_A * (area - config.A_0)**2
    perimeter_energy = config.l_L * (perimeter - config.L_0)**2
    total_energy = area_energy + perimeter_energy
    
    print(f"面積エネルギー: {area_energy}")
    print(f"周囲長エネルギー: {perimeter_energy}")
    print(f"総エネルギー: {total_energy}")
    
    if total_energy == 0:
        print("✅ 4×4正方形は理論的に最適")
    else:
        print(f"❌ 4×4正方形でもエネルギー={total_energy}")
        print("理由: 実際の周囲長が目標値16と異なる")

def run_16x16_analysis():
    """16×16目標設定の分析を実行"""
    print("A_0=16, L_0=16設定での異常成長分析\n")
    
    try:
        # 1. 理論的エネルギー分析
        best_pattern = analyze_16x16_target_energy()
        
        # 2. 実際のシミュレーション分析
        growth_data = test_actual_simulation_analysis()
        
        # 3. 成長駆動力の分析
        analyze_growth_driving_force()
        
        # 4. 理想的4×4正方形の検証
        test_ideal_4x4_square()
        
        print("\n" + "="*60)
        print("診断結果:")
        print("="*60)
        
        final_area = growth_data[-1]['area']
        final_energy = growth_data[-1]['total_energy']
        
        if final_area > 20:
            print("❌ 過剰成長が確認されました")
            print("可能な原因:")
            print("1. 目標周囲長L_0=16が4×4正方形の実際の周囲長と異なる")
            print("2. 初期位置(1ピクセル)から目標(16ピクセル)への経路でエネルギーが下がり続ける")
            print("3. 周囲長計算方法が期待値と異なる")
            
            # 最適エネルギー状態と比較
            if best_pattern:
                print(f"4. 実際の最適状態は: {best_pattern[0]} (面積={best_pattern[1]})")
        else:
            print("✅ 成長は適切な範囲内")
        
        return True
        
    except Exception as e:
        print(f"分析エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_16x16_analysis()
    sys.exit(0 if success else 1)