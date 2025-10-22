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



import pandas as pd
import re

# === 1) Excel読み込み ===
# 例: 列名は ['index','商品番号','クラス分類','新ルールクラス名'] を想定
#     必要に応じて列名を合わせてください。
df = pd.read_excel("input.xlsx")

# 'index' 列が無ければ、行番号(1始まり)で作る
if 'index' not in df.columns:
    df['index'] = df.index + 1

# === 2) 結合セルの埋め & まとめクラス抽出 ===
df['クラス分類'] = df['クラス分類'].ffill()
df['新ルールクラス名'] = df['新ルールクラス名'].ffill()

is_matome = df['クラス分類'].eq('まとめクラス')
df_loc = df.loc[is_matome].copy()

# === 3) 基準商品番号(アンカー)を抽出:  _2002_ / _2002m_ 両対応 ===
pat_anchor = re.compile(r'_(\d+)(?:m)?_')

def extract_anchor(s: str):
    m = pat_anchor.search(str(s))
    return int(m.group(1)) if m else None

df_loc['基準商品番号'] = df_loc['新ルールクラス名'].map(extract_anchor)

# === 4) 文字列を必ず _<番号>m_ に正規化 ===
def force_m(name: str) -> str:
    name = str(name)
    return re.sub(r'_(\d+)(?:m)?_', lambda m: f'_{m.group(1)}m_', name, count=1)

# === 5) 先頭プレフィックス（XXXX）を抽出 ===
def extract_prefix(name: str):
    m = re.match(r'^([^_]+)', str(name))
    return m.group(1) if m else ""

# === 6) アンカー(基準商品番号)ごとに代表のルール名・prefix・indexを拾う ===
# 代表の「新ルールクラス名」（同アンカー内で最初の非空）を取り、_m_ を強制
rep_rule_per_anchor = (
    df_loc.dropna(subset=['基準商品番号'])
         .groupby('基準商品番号')['新ルールクラス名']
         .agg(lambda s: next((x for x in s if pd.notna(x) and str(x).strip()), None))
         .map(force_m)
)

# 代表の index（同アンカー内で最初の非NaN）
rep_index_per_anchor = (
    df_loc.dropna(subset=['基準商品番号'])
         .groupby('基準商品番号')['index']
         .agg(lambda s: next((int(x) for x in s if pd.notna(x)), None))
)

# 代表の prefix（ルール名の先頭 'XXXX' を抽出）
rep_prefix_per_anchor = rep_rule_per_anchor.map(extract_prefix)

# === 7) 出力 ===
# 7-1) LABEL_DATA 形式:
#      仕様: {アンカー,{LABEL_DATA{index,"XXXX_アンカーm_XXXX"}}},
print("----- LABEL_DATA -----")
for anchor, rule_m in rep_rule_per_anchor.items():
    idx_val = rep_index_per_anchor.get(anchor, 'index')  # 見つからない時は 'index' を文字で
    line = f'{{{anchor},{{LABEL_DATA{{{idx_val},"{rule_m}"}}}}}},'
    print(line)

# 7-2) E_Class 形式:
#      仕様: {index,{E_Class{アンカー, "XXXX", "XXXX_アンカーm_XXXX"}}},
print("----- E_Class -----")
for anchor, rule_m in rep_rule_per_anchor.items():
    idx_val = rep_index_per_anchor.get(anchor, 'index')
    prefix = rep_prefix_per_anchor.get(anchor, "")
    line = f'{{{idx_val},{{E_Class{{{anchor}, "{prefix}", "{rule_m}"}}}}}},'
    print(line)
