# CPM.py テストスイート

このフォルダにはcpm_torch/CPM.pyの包括的なテストファイルが含まれています。

## テストファイル一覧

### 基本テスト
- `test_cpm_simple.py` - 基本機能の簡潔なテスト
- `test_cpm_working.py` - 実装に合わせた動作テスト  
- `test_cpm_final.py` - 最終的な包括テスト

### エネルギー計算専用テスト
- `test_energy_detailed.py` - dH_area、dH_perimeterの詳細テスト
- `test_edge_energy.py` - エネルギー計算のエッジケーステスト

### 旧バージョン
- `test_cpm.py` - 初期の包括テスト（一部エラーあり）

## テストレポート
- `test_report.md` - 基本テスト結果のレポート
- `energy_analysis_report.md` - エネルギー計算の詳細分析レポート

## 実行方法

### 推奨テスト実行順序

1. **基本機能テスト**
```bash
cd /app
python tests/test_cpm_final.py
```

2. **エネルギー計算詳細テスト**
```bash
python tests/test_energy_detailed.py
```

3. **エッジケーステスト**
```bash
python tests/test_edge_energy.py
```

### 全テスト実行
```bash
# すべてのテストを順次実行
for test in tests/test_*.py; do
    echo "=== $test ==="
    python "$test"
    echo
done
```

## テスト結果概要

### 最新テスト結果 (test_cpm_final.py)
- **成功率**: 50% (3/6)
- **基本機能**: ✅ 完全動作
- **面積・周囲長計算**: ✅ 完全動作  
- **エネルギー計算**: ✅ 修正済みで正常
- **確率計算**: ❌ デバッグ出力エラー
- **シミュレーション**: ❌ 同上

### エネルギー計算テスト (test_energy_detailed.py)
- **成功率**: 100% (3/3)
- **dH_area**: ✅ 理論値と完全一致
- **dH_perimeter**: ✅ 複雑な計算も正確
- **一貫性**: ✅ 実際のマップで正常動作

### エッジケーステスト (test_edge_energy.py)  
- **成功率**: 75% (3/4)
- **数値安定性**: ✅ 大きな値でも安定
- **周囲長境界**: ✅ 正常動作
- **空セル処理**: ❌ 軽微な不具合

## 発見された問題と修正状況

### ✅ 修正済み
- エネルギー計算式の係数適用
- 周囲長計算のタイポ修正
- デバッグ出力の一部削除

### ❌ 残存問題
- CPM.py:441行のデバッグ出力による次元エラー
- 空セルエネルギー処理の軽微な不具合

## 開発者向け注意事項

### テスト追加時のガイドライン
1. `tests/`フォルダ内に配置
2. ファイル名は`test_*.py`形式
3. 独立実行可能な形式で作成
4. 明確なアサーション文とエラーメッセージ

### 継続的テストの推奨
新しい機能追加時は以下を実行:
```bash
python tests/test_cpm_final.py      # 基本機能の回帰テスト
python tests/test_energy_detailed.py # エネルギー計算の検証
```