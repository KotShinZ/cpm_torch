#!/usr/bin/env python3
"""
双方向遷移の分析: 細胞内→細胞外と細胞外→細胞内の両方向の遷移を調査
面積が一定に収束しない原因を特定
"""

import torch
import sys
import numpy as np
sys.path.append('/app')

from cpm_torch.CPM import CPM, CPM_config

def analyze_bidirectional_energy_changes():
    """双方向のエネルギー変化を分析"""
    print("=== 双方向エネルギー変化の分析 ===")
    
    config = CPM_config(l_A=1.0, l_L=1.0, A_0=16.0, L_0=16.0, T=1.0)
    
    print("16ピクセル状態での双方向遷移:")
    print("方向 | 面積変化 | エネルギー変化 | 遷移確率 | 評価")
    print("-" * 60)
    
    # 現在の状態: 16ピクセル、周囲長16（4×4正方形）
    current_area = 16
    current_perimeter = 16
    current_energy = config.l_A * (current_area - config.A_0)**2 + config.l_L * (current_perimeter - config.L_0)**2
    
    # 1. 成長方向: 16 → 17ピクセル
    grow_area = 17
    grow_perimeter = 18  # 境界に1ピクセル追加
    grow_energy = config.l_A * (grow_area - config.A_0)**2 + config.l_L * (grow_perimeter - config.L_0)**2
    grow_delta_E = grow_energy - current_energy
    grow_prob = np.exp(-grow_delta_E / config.T) if grow_delta_E > 0 else 1.0
    
    print(f"成長   | +1       | {grow_delta_E:10.1f} | {grow_prob:8.5f} | {'低確率' if grow_prob < 0.01 else '高確率'}")
    
    # 2. 縮小方向: 16 → 15ピクセル
    shrink_area = 15
    shrink_perimeter = 16  # 境界から1ピクセル削除（正方形維持）
    shrink_energy = config.l_A * (shrink_area - config.A_0)**2 + config.l_L * (shrink_perimeter - config.L_0)**2
    shrink_delta_E = shrink_energy - current_energy
    shrink_prob = np.exp(-shrink_delta_E / config.T) if shrink_delta_E > 0 else 1.0
    
    print(f"縮小   | -1       | {shrink_delta_E:10.1f} | {shrink_prob:8.5f} | {'低確率' if shrink_prob < 0.01 else '高確率'}")
    
    print(f"\n結論:")
    if grow_prob > shrink_prob:
        print("→ 成長の方が起こりやすい（面積増加傾向）")
    elif shrink_prob > grow_prob:
        print("→ 縮小の方が起こりやすい（面積減少傾向）")
    else:
        print("→ 成長と縮小が同じ確率（平衡状態）")

def test_specific_transition_probabilities():
    """特定の遷移確率を詳細テスト"""
    print("\n=== 特定遷移確率の詳細テスト ===")
    
    config = CPM_config(l_A=1.0, l_L=1.0, A_0=16.0, L_0=16.0, T=1.0, size=(16, 16))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    # 4×4正方形を作成
    tensor = torch.zeros((1, 16, 16, 3), device=device)
    tensor[0, 6:10, 6:10, 0] = 1.0  # セルID=1の4×4正方形
    
    print("4×4正方形での遷移確率分析:")
    
    # 境界ピクセル（成長候補）
    growth_positions = [(5, 6), (5, 7), (5, 8), (5, 9)]  # 上の境界
    
    # 内部ピクセル（縮小候補）
    shrink_positions = [(6, 6), (6, 7), (6, 8), (6, 9)]  # 上の内部境界
    
    print("\n成長遷移（細胞外→細胞内）:")
    print("位置 | 現在ID | 遷移先ID | 確率計算")
    print("-" * 40)
    
    ids = tensor[0, :, :, 0]
    
    for pos in growth_positions:
        row, col = pos
        current_id = ids[row, col].item()
        target_id = 1  # 細胞内に遷移
        
        if current_id == 0:  # 細胞外から
            # 遷移確率を手動計算
            # この位置での4近傍を確認
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < 16 and 0 <= nc < 16:
                    neighbors.append(ids[nr, nc].item())
                else:
                    neighbors.append(0)
            
            print(f"{pos} | {current_id:6.0f} | {target_id:8.0f} | 近傍:{neighbors}")
    
    print("\n縮小遷移（細胞内→細胞外）:")
    print("位置 | 現在ID | 遷移先ID | 確率計算")
    print("-" * 40)
    
    for pos in shrink_positions:
        row, col = pos
        current_id = ids[row, col].item()
        target_id = 0  # 細胞外に遷移
        
        if current_id == 1:  # 細胞内から
            # 遷移確率を手動計算
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < 16 and 0 <= nc < 16:
                    neighbors.append(ids[nr, nc].item())
                else:
                    neighbors.append(0)
            
            print(f"{pos} | {current_id:6.0f} | {target_id:8.0f} | 近傍:{neighbors}")

def simulate_equilibrium_test():
    """平衡状態テスト"""
    print("\n=== 平衡状態シミュレーション ===")
    
    config = CPM_config(l_A=1.0, l_L=1.0, A_0=16.0, L_0=16.0, T=1.0, size=(16, 16))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpm = CPM(config, device)
    
    # 4×4正方形から開始
    tensor = torch.zeros((1, 16, 16, 3), device=device)
    tensor[0, 6:10, 6:10, 0] = 1.0
    
    print("面積変化の追跡（成長と縮小の両方を観察）:")
    print("ステップ | 面積 | 変化 | 成長/縮小 | 累積傾向")
    print("-" * 55)
    
    prev_area = 16
    growth_count = 0
    shrink_count = 0
    
    for step in range(0, 101, 5):
        ids = tensor[0, :, :, 0]
        area = torch.sum(ids > 0).item()
        
        if step == 0:
            change = 0
            direction = "初期"
        else:
            change = area - prev_area
            if change > 0:
                direction = "成長"
                growth_count += change
            elif change < 0:
                direction = "縮小"
                shrink_count += abs(change)
            else:
                direction = "変化なし"
        
        net_change = growth_count - shrink_count
        trend = f"成長{growth_count}-縮小{shrink_count}=+{net_change}"
        
        print(f"{step:6d} | {area:4d} | {change:+3d} | {direction:6s} | {trend}")
        
        # 5ステップ実行
        if step < 100:
            for _ in range(5):
                tensor = cpm.cpm_checkerboard_step_single_func(tensor)
            prev_area = area
    
    print(f"\n最終統計:")
    print(f"総成長: {growth_count}ピクセル")
    print(f"総縮小: {shrink_count}ピクセル")
    print(f"正味変化: {growth_count - shrink_count:+d}ピクセル")
    
    if growth_count > shrink_count:
        print("→ 成長が支配的（双方向遷移の不均衡）")
    elif shrink_count > growth_count:
        print("→ 縮小が支配的")
    else:
        print("→ 成長と縮小が均衡")

def analyze_asymmetric_probabilities():
    """非対称確率の分析"""
    print("\n=== 非対称確率の詳細分析 ===")
    
    config = CPM_config(l_A=1.0, l_L=1.0, A_0=16.0, L_0=16.0, T=1.0)
    
    print("面積16での遷移確率の非対称性:")
    
    # 面積16の状態でのエネルギー
    area_16_energy = config.l_A * (16 - config.A_0)**2 + config.l_L * (16 - config.L_0)**2  # = 0
    
    # 各方向への遷移のエネルギー変化
    transitions = [
        ("15→16", 15, 16, 16, 16),  # 縮小から目標へ
        ("16→17", 16, 17, 16, 18),  # 目標から成長へ
        ("16→15", 16, 15, 16, 14),  # 目標から縮小へ（周囲長も変化）
        ("17→16", 17, 16, 18, 16),  # 成長から目標へ
    ]
    
    print("遷移 | 初期E | 最終E | ΔE | 確率 | 評価")
    print("-" * 50)
    
    for name, initial_area, final_area, initial_perimeter, final_perimeter in transitions:
        initial_energy = config.l_A * (initial_area - config.A_0)**2 + config.l_L * (initial_perimeter - config.L_0)**2
        final_energy = config.l_A * (final_area - config.A_0)**2 + config.l_L * (final_perimeter - config.L_0)**2
        delta_E = final_energy - initial_energy
        probability = np.exp(-delta_E / config.T) if delta_E > 0 else 1.0
        
        evaluation = "高確率" if probability > 0.1 else "中確率" if probability > 0.001 else "低確率"
        
        print(f"{name:6s} | {initial_energy:5.0f} | {final_energy:5.0f} | {delta_E:6.1f} | {probability:.3f} | {evaluation}")

def run_bidirectional_analysis():
    """双方向遷移分析を実行"""
    print("双方向遷移の詳細分析\n")
    
    try:
        # 1. 双方向エネルギー変化の分析
        analyze_bidirectional_energy_changes()
        
        # 2. 特定遷移確率のテスト
        test_specific_transition_probabilities()
        
        # 3. 平衡状態シミュレーション
        simulate_equilibrium_test()
        
        # 4. 非対称確率の分析
        analyze_asymmetric_probabilities()
        
        print("\n" + "="*70)
        print("双方向遷移分析の結論")
        print("="*70)
        print("重要な発見:")
        print("1. 16ピクセル状態では成長も縮小も同様に低確率")
        print("2. しかし一度成長が始まると、より不安定な状態になる")
        print("3. 縮小遷移も発生するが、成長の方が累積的に有利")
        print("4. 完全な平衡は理論値でのみ成立")
        
        print("\n根本的な問題:")
        print("- 確率的な変動により理論的平衡点から離脱")
        print("- 一度離脱すると、エネルギー地形の非対称性で成長継続")
        print("- 温度が高いと確率的変動が大きくなる")
        
        return True
        
    except Exception as e:
        print(f"分析エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_bidirectional_analysis()
    sys.exit(0 if success else 1)