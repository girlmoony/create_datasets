import pandas as pd
import re

# 例: df = pd.read_excel("input.xlsx") などで読み込み
# 列名は実データに合わせて調整してください
# 想定列: '商品番号','クラス分類','新ルールクラス名'

# 結合セルの欠損を上方向に埋める
df['クラス分類'] = df['クラス分類'].ffill()
df['新ルールクラス名'] = df['新ルールクラス名'].ffill()

# まとめクラスだけに絞る
m = df['クラス分類'].eq('まとめクラス')

# 新ルールクラス名から基準の商品番号を抽出
# パターン: _2002_ または _2002m_
pat = re.compile(r'_(\d+)(?:m)?_')
def extract_anchor(s):
    m = pat.search(str(s))
    return m.group(1) if m else None

df_loc = df.loc[m].copy()
df_loc['基準商品番号'] = df_loc['新ルールクラス名'].map(extract_anchor).astype('Int64')

# ペアの作成（タプルや文字列など好みで）
pairs = [(int(row['商品番号']), int(row['基準商品番号'])) for _, row in df_loc.dropna(subset=['基準商品番号']).iterrows()]

# 文字列表現にする場合
pair_strings = [f'{{{p},{a}}}' for p, a in pairs]

print(pairs)        # [(2002, 2002), (2006, 2002), (2037, 2002)]
print(pair_strings) # ['{2002, 2002}', '{2006, 2002}', '{2037, 2002}']
