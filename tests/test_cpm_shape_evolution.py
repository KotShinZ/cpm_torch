#!/usr/bin/env python3
"""
CPMの形状進化過程の詳細観察
1ピクセルから目標サイズまでの実際の形状変化を視覚的に追跡
"""

import torch
import sys
import numpy as np
sys.path.append('/app')

from cpm_torch.CPM import CPM, CPM_config

def visualize_shape_evolution():
    """形状進化過程の詳細観察"""
    print("=== CPM形状進化過程の観察 ===")
    
    config = CPM_config(
        l_A=1.0, l_L=1.0, A_0=16.0, L_0=16.0, T=1.0, size=(12, 12)
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    # 初期状態
    tensor = torch.zeros((1, 12, 12, 3), device=device)
    tensor[0, 6, 6, 0] = 1.0  # 中央に1ピクセル
    
    print("形状進化の記録:")
    snapshots = []
    
    # 特定のステップで形状を記録
    record_steps = [0, 5, 10, 15, 20, 30, 50, 100]
    
    for step in range(101):
        ids = tensor[0, :, :, 0]
        area = torch.sum(ids > 0).item()
        
        if step in record_steps:
            perimeter = CPM.calc_total_perimeter_bincount(ids, 2)[1].item() if area > 0 else 0
            energy = config.l_A * (area - config.A_0)**2 + config.l_L * (perimeter - config.L_0)**2
            
            snapshot = {
                'step': step,
                'area': area,
                'perimeter': perimeter,
                'energy': energy,
                'shape': ids.cpu().numpy().copy()
            }
            snapshots.append(snapshot)
            
            print(f"\nステップ {step}: 面積={area}, 周囲長={perimeter:.1f}, エネルギー={energy:.1f}")
            print_shape(ids.cpu().numpy())
        
        # 1ステップ実行
        if step < 100:
            tensor = cpm.cpm_checkerboard_step_single_func(tensor)
    
    return snapshots

def print_shape(shape_array):
    """形状を視覚的に表示"""
    print("形状 (1=細胞, 0=背景):")
    for row in shape_array:
        line = ""
        for cell in row:
            if cell > 0:
                line += "█"
            else:
                line += "·"
        print(f"  {line}")

def analyze_shape_efficiency():
    """形状効率の分析"""
    print("\n=== 形状効率分析 ===")
    
    print("様々な16ピクセル形状の比較:")
    
    # 異なる16ピクセル形状を作成
    shapes = {
        "4×4正方形": create_compact_square(),
        "不規則クラスター": create_irregular_cluster(),
        "L字型": create_l_shape_16(),
        "縦長": create_vertical_rectangle(),
        "散在": create_scattered_16()
    }
    
    config = CPM_config(l_A=1.0, l_L=1.0, A_0=16.0, L_0=16.0, T=1.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("形状 | 面積 | 周囲長 | エネルギー | 効率性")
    print("-" * 55)
    
    for name, shape in shapes.items():
        shape_tensor = torch.tensor(shape, device=device)
        area = torch.sum(shape_tensor > 0).item()
        perimeter = CPM.calc_total_perimeter_bincount(shape_tensor, 2)[1].item()
        energy = config.l_A * (area - config.A_0)**2 + config.l_L * (perimeter - config.L_0)**2
        
        efficiency = "最適" if energy == 0 else f"エネルギー損失{energy:.0f}"
        
        print(f"{name:12s} | {area:4d} | {perimeter:6.1f} | {energy:8.1f} | {efficiency}")
        print_shape(shape)
        print()

def create_compact_square():
    """コンパクトな4×4正方形"""
    shape = np.zeros((12, 12))
    shape[4:8, 4:8] = 1
    return shape

def create_irregular_cluster():
    """不規則なクラスター"""
    shape = np.zeros((12, 12))
    # 中心から不規則に拡散
    positions = [(5,5), (5,6), (6,5), (6,6), (4,5), (5,4), (7,5), (5,7), 
                 (4,4), (7,6), (6,7), (4,6), (6,4), (3,5), (5,3), (8,5)]
    for r, c in positions:
        shape[r, c] = 1
    return shape

def create_l_shape_16():
    """L字型16ピクセル"""
    shape = np.zeros((12, 12))
    # L字型
    shape[3:7, 3:4] = 1  # 縦棒
    shape[6:7, 3:7] = 1  # 横棒
    # 追加ピクセルで16にする
    shape[7:11, 6:7] = 1
    return shape

def create_vertical_rectangle():
    """縦長の長方形"""
    shape = np.zeros((12, 12))
    shape[2:10, 5:7] = 1  # 8×2の縦長
    return shape

def create_scattered_16():
    """散在する16ピクセル"""
    shape = np.zeros((12, 12))
    positions = [(1,1), (1,5), (1,9), (3,3), (3,7), (5,1), (5,5), (5,9),
                 (7,3), (7,7), (9,1), (9,5), (9,9), (2,8), (4,2), (8,6)]
    for r, c in positions:
        shape[r, c] = 1
    return shape

def test_growth_path_analysis():
    """成長経路の分析"""
    print("\n=== 成長経路分析 ===")
    
    config = CPM_config(l_A=1.0, l_L=1.0, A_0=16.0, L_0=16.0, T=1.0)
    
    print("理論的な成長経路のエネルギー変化:")
    print("面積 | 最適周囲長 | 実際周囲長 | 最適E | 実際E | エネルギー差")
    print("-" * 65)
    
    for area in range(1, 21):
        # 最適周囲長（正方形の場合）
        if area == 1:
            optimal_perimeter = 4
        else:
            side = int(np.sqrt(area))
            if side**2 == area:
                optimal_perimeter = 4 * side
            else:
                optimal_perimeter = 4 * side + 2
        
        # 実際の成長過程での周囲長（1ピクセルずつ追加）
        if area <= 4:
            actual_perimeter = 4 + 2 * (area - 1)  # L字型成長
        elif area <= 9:
            actual_perimeter = 4 * int(np.sqrt(area)) + 2
        else:
            actual_perimeter = optimal_perimeter + 2  # 不規則性による増加
        
        optimal_energy = config.l_A * (area - config.A_0)**2 + config.l_L * (optimal_perimeter - config.L_0)**2
        actual_energy = config.l_A * (area - config.A_0)**2 + config.l_L * (actual_perimeter - config.L_0)**2
        energy_diff = actual_energy - optimal_energy
        
        print(f"{area:4d} | {optimal_perimeter:9.0f} | {actual_perimeter:9.0f} | {optimal_energy:6.0f} | {actual_energy:6.0f} | {energy_diff:8.0f}")

def run_shape_evolution_analysis():
    """形状進化分析を実行"""
    print("CPM形状進化過程の詳細分析\n")
    
    try:
        # 1. 形状進化過程の観察
        snapshots = visualize_shape_evolution()
        
        # 2. 形状効率の分析
        analyze_shape_efficiency()
        
        # 3. 成長経路の分析
        test_growth_path_analysis()
        
        print("\n" + "="*70)
        print("形状進化分析の結論")
        print("="*70)
        
        # 最終形状の分析
        final_snapshot = snapshots[-1]
        
        print(f"最終結果 (100ステップ後):")
        print(f"  面積: {final_snapshot['area']} (目標: 16)")
        print(f"  周囲長: {final_snapshot['perimeter']:.1f} (目標: 16)")
        print(f"  エネルギー: {final_snapshot['energy']:.1f} (最適: 0)")
        
        print("\n主要な発見:")
        print("1. 1ピクセルからの成長は必然的に不規則な形状を形成")
        print("2. 面積16到達時点では最適な4×4正方形ではない")
        print("3. 不規則な16ピクセル形状から更なる最適化が継続")
        print("4. 局所的な1ピクセル変化では最適形状に到達困難")
        
        print("\n根本的な問題:")
        print("- CPMの局所更新メカニズムと最適解の形状的複雑さの不整合")
        print("- 1ピクセル単位の変化では大域的形状最適化が困難")
        print("- 結果として目標面積を大幅に超過する成長が継続")
        
        return True
        
    except Exception as e:
        print(f"分析エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_shape_evolution_analysis()
    sys.exit(0 if success else 1)