# [マーケティングサイエンスの理論と実践](https://gihyo.jp/book/2026/978-4-297-16025-8) サポートページ

本リポジトリは、「[マーケティングサイエンスの理論と実践](https://gihyo.jp/book/2026/978-4-297-16025-8)」のサポートページです。

1. 環境構築方法を含むコード
2. 本誌では省略した証明

を管理するためのものです。

# 環境構築

このリポジトリでは、Python環境管理に `uv` を利用する。

- `uv` 公式サイト: https://docs.astral.sh/uv/
- `uv` インストール手順: https://docs.astral.sh/uv/getting-started/installation/

## Python 環境

全章で Python 3.14.4 と共通の環境を使います。
環境の設定ファイルはリポジトリのルートに置き、Python のバージョンは `.python-version`、ライブラリのバージョンは `uv.lock` で指定しています。

### 初回の準備（共通）

`uv` をインストールした後、リポジトリを取得します。

```bash
git clone https://github.com/gihyo-book/marketing_science_book.git
cd marketing_science_book
```

第6章のモデル図には Graphviz 本体（`dot` コマンド）が必要です。
macOS では、第7章で使う LightGBM の実行用に `libomp` も入れます。
Linux では PyMC の計算用に C++ コンパイラ、LightGBM などの実行用に `libgomp1` も入れます。

```bash
# macOS
brew install graphviz libomp
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install g++ graphviz libgomp1
dot -V
```

リポジトリのルートで、共通の Python 環境を構築します。
`uv sync --locked` により、リポジトリのルートに仮想環境の `.venv/` ディレクトリが作成され、必要なライブラリがインストールされます。


```bash
uv sync --locked
```

`--locked` は、設定と `uv.lock` が合わない場合にエラーにし、`uv.lock` を自動で書き換えないための指定です。

環境構築後は、VS Code または JupyterLab のいずれかでノートブックを実行できます。

### VS Code を利用する場合

1. VS Code に Microsoft の **Python** 拡張機能と **Jupyter** 拡張機能をインストールします。
2. リポジトリのルートフォルダーを VS Code で開きます。
3. `.ipynb` ファイルを開き、右上のカーネル選択から **Python Environments** を選び、ルートの `.venv/` の Python を指定します。全ノートブックでこの環境を選んでください。

Python ファイルを編集・実行する場合も、コマンドパレットの `Python: Select Interpreter` でルートの `.venv/` の Python を選びます。
`.venv/` を直接選ぶ場合、カーネルの登録は不要です。VS Code で実行するときは、`jupyter lab` の起動も不要です。

カーネルを **Marketing Science (Python 3.14)** という名前で選びたい場合は、リポジトリのルートで次のコマンドを一度実行します。

```bash
uv run --locked python -m ipykernel install --user --name marketing-science-src --display-name "Marketing Science (Python 3.14)"
```

登録後は、右上のカーネル選択で **Select Another Kernel... → Jupyter Kernels → Marketing Science (Python 3.14)** の順に選びます。
この方法では **Jupyter Kernels** から登録した名前を選び、**Python Environments** から選ぶ場合は引き続き `.venv/` の Python を指定します。

### JupyterLab を利用する場合

リポジトリのルートで、カーネルを登録して JupyterLab を起動します。カーネルの登録は初回のみ必要です。

```bash
uv run --locked python -m ipykernel install --user --name marketing-science-src --display-name "Marketing Science (Python 3.14)"
uv run --locked jupyter lab
```

ブラウザーで開いた JupyterLab で、全ノートブックのカーネルに **Marketing Science (Python 3.14)**（登録名: `marketing-science-src`）を選んでください。

どちらの場合も、[main_314.ipynb](main_314.ipynb) で Python とライブラリのバージョンを確認できます。
各章で使う外部データの準備方法は、[利用するデータセット](#利用するデータセット)に記載しています。

## 共通モジュール

`common/notebook_style.py`には、複数章のノートブックから使う表示設定とデータパス解決処理をまとめています。各章のノートブックを実行するために必要なため、`common/`は削除しないでください。

---

# 利用するデータセット

各章で使う外部データの入手方法と保存先を以下にまとめます。

## 第2・11・12章で使うデータ

Online Retailのデータは、このリポジトリに含まれていません。ライセンスは[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)です。[UCIの配布ページ](https://archive.ics.uci.edu/dataset/352/online%2Bretail)から
`Online Retail.xlsx`を取得し、CSV形式の`online_retail.csv.gz`に変換してください。
第2・11・12章で使うため、同じ内容のファイルを`02/data/`、`11/data/`、`12/data/`に保存します。

取得したファイルをリポジトリのルートに置き、次のコマンドで変換できます。

```bash
mkdir -p 02/data 11/data 12/data
uv run --locked python - <<'PY'
import pandas as pd

frame = pd.read_excel("Online Retail.xlsx")
frame["CustomerID"] = frame["CustomerID"].astype("Int64")
for chapter in ("02", "11", "12"):
    frame.to_csv(f"{chapter}/data/online_retail.csv.gz", index=False)
PY
```

## 第7章で使うデータ

第7章のデータは、このリポジトリに含まれていません。次の手順で用意してください。
配布元の説明では、サンプルコードは[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)で公開されています。CSVのライセンスは別には書かれていません。

1. [IBMの配布ページ](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv)を開き、`Download raw file`からCSVファイルをダウンロードする。
2. `Telco-Customer-Churn.csv`を`07/data/`に保存する。

## 第8章で使うデータ

第8章のデータは、このリポジトリに含まれていません。次の手順で用意してください。
Kaggleの配布ページでは、ライセンスは[CC0 1.0（Public Domain）](https://creativecommons.org/publicdomain/zero/1.0/)、データはVersion 2と表示されています（2026年9月13日確認）。

1. [Kaggleの配布ページ](https://www.kaggle.com/datasets/rashikrahmanpritom/groceries-dataset-for-market-basket-analysismba)を開き、ZIPファイルをダウンロードする。
2. ZIPファイルを展開する。
3. `Groceries data.csv`を`08/data/`に保存する。

`basket.csv`は本書のコードでは使いません。

---

# ノートブック一覧

## 環境確認

| ノートブック | 内容 |
|---|---|
| [main_314.ipynb](main_314.ipynb) | 全章共通のPython 3.14環境の主要ライブラリ（PyMC-Marketing・implicitを含む）のバージョン確認 |

## 第0章：はじめに

| ノートブック | 内容 |
|---|---|
| [00/00_probability_distributions.ipynb](00/00_probability_distributions.ipynb) | 本書に登場する確率分布（ポアソン、ガンマ、ベータ、ディリクレ、正規分布など）を、数式・統計的性質・実務での役割・分布間の関係の観点で網羅的に整理・可視化 |

## 第2章：顧客を分類する

| ノートブック | 内容 |
|---|---|
| [02/02_main.ipynb](02/02_main.ipynb) | UCI Online Retail データセットから顧客単位の特徴量を作成・標準化し、外れ値への対処を比較。エルボー法と K-means による顧客分類、および AIC・BIC で成分数を選ぶ GMM による確率的クラスタリングと所属確率の可視化を実施 |

## 第3章：新製品の普及を予測する

| ノートブック | 内容 |
|---|---|
| [03/03_main.ipynb](03/03_main.ipynb) | Bass モデルの差分方程式 $n(t) = p(M-N_{t-1}) + q \frac{N_{t-1}}{M}(M-N_{t-1})$ を用いて、ノイズを付加した月次販売台数（24ヶ月分）をシミュレーション生成。パラメータ推定と予測精度を検証 |
| [03/03_bass_model_closed_form_derivation.ipynb](03/03_bass_model_closed_form_derivation.ipynb) | 付録：Bass モデルの微分方程式を市場比率で正規化し、変数分離・部分分数分解・初期条件の適用を通して累積採用率の解析解を導出 |

## 第4章：選ばれる要因を推定する

| ノートブック | 内容 |
|---|---|
| [04/04_main.ipynb](04/04_main.ipynb) | スマートフォンの選択実験データを生成・前処理し、statsmodels で多項ロジット（MNL）を推定。係数・信頼区間・$p$ 値・オッズ比・支払意思額（WTP）を算出し、回答者単位の不確かさと条件変更時の選択確率を確認 |
| [04/04_MNL.ipynb](04/04_MNL.ipynb) | 付録：離散選択理論における多項ロジット（MNL）の確率関数が、ランダム効用最大化仮説とガンベル分布の仮定からどのように数学的に導かれるかを段階的に証明 |
| [04/04_gumbel_difference_logistic_derivation.ipynb](04/04_gumbel_difference_logistic_derivation.ipynb) | 付録：2つのガンベル分布に従う誤差の差がロジスティック分布に従うことを確認 |
| [04/04_multinomial_probit_derivation.ipynb](04/04_multinomial_probit_derivation.ipynb) | 付録：多項プロビットの選択確率を数値計算で確認 |

## 第5章：消費財の需要を解析する

| ノートブック | 内容 |
|---|---|
| [05/05_main.ipynb](05/05_main.ipynb) | 消費者のカテゴリー購買データをシミュレーション生成し、最尤法で NBD（Negative Binomial Distribution）のパラメータを推定。実測分布との比較とカイ二乗検定で適合度を確認し、ブランド別の購買頻度からダブルジョパディを検証 |
| [05/05_double_jeopardy.ipynb](05/05_double_jeopardy.ipynb) | 付録：ディリクレ多項分布の性質から、市場シェアと購買集中度の関係を示すダブルジョパディ法則が数学的に導出される過程を詳説・可視化 |
| [05/05_joint_distribution.ipynb](05/05_joint_distribution.ipynb) | 付録：複数ブランドの購買数量の同時分布がディリクレ多項分布となること、および周辺分布が一変量 NBD になることを、確率生成関数を用いて数学的に導出 |

## 第6章：消費者の異質性を捉える

| ノートブック | 内容 |
|---|---|
| [06/06_main.ipynb](06/06_main.ipynb) | 消費者ごとに異なる価格感応度を持つデータを生成し、完全プール型ロジスティック回帰と階層ベイズモデルを比較。事前・事後予測、MCMC の収束診断、消費者別パラメータの不確かさと縮小推定を確認 |
| [06/06_bayes.ipynb](06/06_bayes.ipynb) | 付録：正規－正規モデルの事後分布から縮小推定が生じる仕組みを導出し、観測数や分散による縮小の強さの違いを数値シミュレーションで確認 |

## 第7章：離反を予測する

| ノートブック | 内容 |
|---|---|
| [07/07_main.ipynb](07/07_main.ipynb) | IBM Telco Customer Churn データセットを用いて、Kaplan–Meier 推定と Cox 回帰による生存時間解析を実施。LightGBM で解約スコアを学習し、適合率・再現率、施策対象リスト、リフト曲線・累積ゲイン曲線でモデルを評価 |
| [07/07_hazard.ipynb](07/07_hazard.ipynb) | 付録：生存関数 $S(t) = P(T > t)$ とハザード関数 $h(t) = \frac{f(t)}{S(t)}$ の統計的意味、相互関係、および微積分による導出を詳述・可視化 |

## 第8章：併売パターンを発見する

| ノートブック | 内容 |
|---|---|
| [08/08_main.ipynb](08/08_main.ipynb) | Kaggle Groceries データセットを買い物単位のトランザクション形式に整形・ワンホットエンコーディングし、Apriori で頻出アイテムセットを抽出。支持度・確信度・リフトによるアソシエーションルールの作成と評価を実施 |
| [08/08_apriori_eclat.ipynb](08/08_apriori_eclat.ipynb) | 付録：Apriori と Eclat アルゴリズムをスクラッチ実装し、頻出アイテムセット探索の仕組みを詳説。TIDリスト（ハッシュテーブル）を用いた効率的な支持度計算と Apriori 性質による枝刈り（プルーニング）の最適化技法を実装 |

## 第9章：トレンドと状態を読む

| ノートブック | 内容 |
|---|---|
| [09/09_main.ipynb](09/09_main.ipynb) | トレンド（線形増加）・季節性（12ヶ月周期）・ノイズを含む 3 年分（1095 日）の日次売上データを生成。Facebook Prophet による時系列予測（加法分解モデル）と、statsmodels の構造時系列モデル（Local Linear Trend + 週次季節性）を実装。Prophet では 365 日先の予測と成分分解プロットを実施し、状態空間モデルではカルマンフィルタによる平滑化・予測区間の導出・残差診断（Ljung-Box 検定）まで実行 |

## 第10章：優良顧客を見極める

| ノートブック | 内容 |
|---|---|
| [10/10_main.ipynb](10/10_main.ipynb) | トランザクションログから RFM サマリーを作成し、BG/NBD で生存確率と将来購買回数、ガンマ–ガンマモデルで平均購買金額を予測。予測 LTV を算出し、VIPランキング・顧客セグメント・LTV分布を可視化。PyMC-Marketingでも同じRFMから12か月のLTVと94%信用区間を計算し、学習に使っていない期間の購買予測を確認 |
| [10/10_bg_nbd.ipynb](10/10_bg_nbd.ipynb) | 付録：BG/NBD モデルの尤度関数における生存項（購買継続確率）と離反項（購買終了確率）の導出と統計的解釈を、確率微積分を用いて段階的に説明 |
| [10/10_marginalization_derivation.ipynb](10/10_marginalization_derivation.ipynb) | 付録：BG/NBD モデルの尤度導出で生じる 2 つの積分（購買レート $\lambda_i$ のガンマ分布による周辺化と、離反確率 $\theta_i$ のベータ分布による周辺化）を、ガンマ積分・ベータ関数の公式を用いてステップごとに導出 |
| [10/10_bgnbd_expected_purchases_derivation.ipynb](10/10_bgnbd_expected_purchases_derivation.ipynb) | 付録：Fader et al. (2005) 付録の BG/NBD モデルにおける将来購買回数の期待値 $\mathbb{E}\{Y(t) \mid X=x,t_x,T,r,\alpha,a,b\}$ の導出を再現。生存確率・購買確率の各項を分解し、超幾何関数 ${}_2F_1$ を用いた最終的な閉形式表現に至るまでをステップごとに説明 |
| [10/10_pareto.ipynb](10/10_pareto.ipynb) | 付録：Pareto/NBD モデルの尤度関数にガウスの超幾何関数 ${}_2F_1(a,b;c;z)$ が現れるメカニズムを、確率の総和条件と積分変換の観点から数学的に説明 |

## 第11章：あなたへのおすすめを作る

| ノートブック | 内容 |
|---|---|
| [11/11_main.ipynb](11/11_main.ipynb) | Online Retail の購入履歴から暗黙的フィードバック行列を構築し、ALS・BPRを学習。特定ユーザーおよび全ユーザーの上位 N 件の推薦を生成・可視化し、Precision@K・Recall@K・nDCG@Kでランキング性能を評価 |

Online Retailのデータは、[利用するデータセット](#利用するデータセット)の説明に従って用意します。

## 第12章：つながりを活かして推薦する

| ノートブック | 内容 |
|---|---|
| [12/12_main.ipynb](12/12_main.ipynb) | Online Retail の顧客と商品のグラフを構築し、BPR・NGCF・LightGCN を PyTorch で実装・学習。購入履歴から推薦スコアを計算し、Recall@K・Precision@K・nDCG@K などで評価 |

通常の実行では、各モデルが学習データを最大300回くり返して学習します（300エポック）。

### 12/src — モデル実装モジュール

`12/12_main.ipynb` からインポートして使用するモデル・学習・ユーティリティの実装を `12/src/` に分離しています。

```
12/src/
├── config/                        # YAML 設定ファイル
│   ├── common.yaml                # 共通ハイパーパラメータ（エポック数・topk・乱数シードなど）
│   ├── dataset/
│   │   └── online-retail.yaml      # Online Retail の前処理・分割設定
│   └── model/
│       ├── BPR.yaml               # BPR モデル固有パラメータ
│       ├── LightGCN.yaml          # LightGCN モデル固有パラメータ
│       └── NGCF.yaml              # NGCF モデル固有パラメータ
├── dataset/
│   └── dataset.py                 # データ読み込み・train/val/test 分割・疎行列構築・DataLoader 生成
├── models/
│   ├── Model.py                   # 全モデル共通の抽象基底クラス（nn.Module）
│   ├── LightGCN.py                # LightGCN
│   ├── NGCF.py                    # NGCF
│   ├── BPR.py                     # BPR
│   └── __init__.py
├── trainer/
│   └── Trainer.py                 # 学習ループ・評価（Recall / Precision / NDCG / MRR / MAP）・Early Stopping・チェックポイント保存
├── util/
│   ├── loss.py                    # BPRLoss（ペアワイズ損失）・EmbLoss（L2 正則化損失）
│   ├── util.py                    # Xavier 初期化・ディレクトリ作成・学習停止判定などユーティリティ関数
│   ├── logger.py                  # loguru ベースのログ初期化（stdout + ファイル出力）
│   └── seed.py                    # 乱数シード固定（Python / NumPy / PyTorch / CUDA）
├── info.py                        # ベースディレクトリ定数・DatasetSpec / ModelSpec データクラス定義
└── main.py                        # CLI エントリーポイント（argparse で model / dataset を受け取り学習実行）
```
