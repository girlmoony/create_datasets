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



import re

# --- ユーティリティ: 新ルールクラス名を正規化 ---
# 1) _2002_ / _2002m_ のどちらでも _2002m_ に統一
def force_m(name: str) -> str:
    name = str(name)
    return re.sub(r'_(\d+)(?:m)?_', lambda m: f'_{m.group(1)}m_', name, count=1)

# 2) 先頭のプレフィックス（最初のアンダースコアまで）を new_prefix に置換
def replace_prefix(name: str, new_prefix: str) -> str:
    name = str(name)
    return re.sub(r'^[^_]+', new_prefix, name, count=1)

# --- ここまでユーティリティ ---

# df_loc は「まとめクラス」で '基準商品番号' が入っている DataFrame（前メッセージの続き）
# 代表の新ルールクラス名（同じアンカーで同一のはず）を1つ拾い、_m_ を強制
rep_rule_per_anchor = (
    df_loc.dropna(subset=['基準商品番号'])
         .groupby('基準商品番号')['新ルールクラス名']
         .agg(lambda s: next((x for x in s if pd.notna(x) and str(x).strip()), None))
         .map(force_m)
)

# 各アンカーに属する商品番号リスト
prods_per_anchor = (
    df_loc.dropna(subset=['基準商品番号'])
         .groupby('基準商品番号')['商品番号']
         .apply(list)
)

# ------- LABEL_DATA 形式の出力 -------
for anchor, prods in prods_per_anchor.items():
    rule_m = rep_rule_per_anchor.get(anchor, "")
    # 1行で: {アンカー,{LABEL_DATA{index,"ルール名_m_化"}}}, を列挙
    line = ", ".join(f'{{{int(anchor)},{{LABEL_DATA{{index,"{rule_m}"}}}}}}}' for _p in sorted(set(map(int, prods))))
    print(line)
    print("......")  # 区切りが必要なら。不要なら消してください

# ------- E_Class 形式の出力 -------
for anchor, prods in prods_per_anchor.items():
    rule_m = rep_rule_per_anchor.get(anchor, "")
    # 先頭プレフィックスを "AAAA" に置換した版（例の形に合わせる）
    rule_m_AAAA = replace_prefix(rule_m, "AAAA")
    # 1行で: {index,{E_Class{アンカー, "AAAA", "AAAA_アンカーm_XXXX"}}}, を列挙
    line = ", ".join(f'{{index,{{E_Class{{{int(anchor)}, "AAAA", "{rule_m_AAAA}"}}}}}}}' for _p in sorted(set(map(int, prods))))
    print(line)
    print("......")
