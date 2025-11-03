# === 单元格 1: 导入、设置与路径定义 ===

import pandas as pd
import numpy as np
import os
import torch
import gc
import joblib 
import lightgbm as lgb
from scipy.sparse import hstack
import warnings

# 特征工程
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# 评估与验证
from sklearn.model_selection import StratifiedKFold # <-- 导入 K-Fold
from sklearn.metrics import log_loss

# 禁用不必要的警告
warnings.filterwarnings('ignore')
os.environ["WANDB_MODE"] = "disabled"

# 内存清理函数
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("🚀 (纯净版 5-Fold CV, 仅模型A) 管线启动！")

# --- 1. 竞赛数据路径 ---
COMP_DIR = "/kaggle/input/llm-classification-finetuning"
TRAIN_FILE = os.path.join(COMP_DIR, "train.csv")
TEST_FILE = os.path.join(COMP_DIR, "test.csv")
SAMPLE_FILE = os.path.join(COMP_DIR, "sample_submission.csv")

# --- 2. 公共模型路径 ---
# (根据你的截图 image_8be7c0.png 和 finalv3 (1).ipynb)
print("⏳ 正在定义公共模型路径...")
BASE_MINILM_PATH = "/kaggle/input/sentencetransformersallminilml6v2"
print(f"  ...MiniLM 路径: {BASE_MINILM_PATH}")

print("✅ 所有路径定义完毕。")

# === 单元格 2: 一站式特征工程 (已修正) ===

print(f"⏳ 正在加载 train.csv 和 test.csv...")
try:
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    sample_df = pd.read_csv(SAMPLE_FILE) # sample_df 在最后单元格会用到
    print(f"  训练集: {train_df.shape}, 测试集: {test_df.shape}")
except FileNotFoundError as e:
    print(f"❌ 数据加载失败! {e}")
    raise

# --- 2.1 基础特征 (len, lexical) ---
print("⏳ 正在创建基础特征 (len_diff, lexical_diff)...")
def create_base_features(df):
    df['text_a'] = df['prompt'] + " " + df['response_a']
    df['text_b'] = df['prompt'] + " " + df['response_b']
    df['combined_for_embedding'] = df['text_a'] + " [SEP] " + df['text_b']
    df["resp_a_len"] = df["response_a"].str.len()
    df["resp_b_len"] = df["response_b"].str.len()
    df["len_diff"] = df["resp_a_len"] - df["resp_b_len"]
    df["len_ratio"] = df["resp_a_len"] / (df["resp_b_len"] + 1e-6)
    df["lexical_a"] = df["response_a"].apply(lambda x: len(set(str(x).split())) / (len(str(x).split()) + 1e-6))
    df["lexical_b"] = df["response_b"].apply(lambda x: len(set(str(x).split())) / (len(str(x).split()) + 1e-6))
    df["lexical_diff"] = df["lexical_a"] - df["lexical_b"]
    return df

train_df = create_base_features(train_df)
test_df = create_base_features(test_df)
train_df['label'] = train_df[['winner_model_a', 'winner_model_b', 'winner_tie']].values.argmax(axis=1)
y_true_full = train_df['label'] # 准备好所有标签 (57k)

# --- 2.2 SBERT 嵌入 (MiniLM) ---
print("⏳ 正在生成 MiniLM 嵌入 (用于 A 和 相似度)...")
model_minilm = SentenceTransformer(BASE_MINILM_PATH, device='cuda') 
train_emb_minilm = model_minilm.encode(train_df['combined_for_embedding'].tolist(), show_progress_bar=True, batch_size=128)
test_emb_minilm = model_minilm.encode(test_df['combined_for_embedding'].tolist(), show_progress_bar=True, batch_size=128)

# --- 2.3 相似度特征 (来自 MiniLM) ---
print("⏳ 正在创建 相似度 特征...")
resp_a_emb_train = model_minilm.encode(train_df['response_a'].tolist(), show_progress_bar=True, batch_size=128)
resp_b_emb_train = model_minilm.encode(train_df['response_b'].tolist(), show_progress_bar=True, batch_size=128)
train_df['cosine_similarity'] = np.array([cosine_similarity(resp_a_emb_train[i].reshape(1, -1), resp_b_emb_train[i].reshape(1, -1))[0][0] for i in range(len(resp_a_emb_train))])

resp_a_emb_test = model_minilm.encode(test_df['response_a'].tolist(), show_progress_bar=True, batch_size=128)
resp_b_emb_test = model_minilm.encode(test_df['response_b'].tolist(), show_progress_bar=True, batch_size=128)
test_df['cosine_similarity'] = np.array([cosine_similarity(resp_a_emb_test[i].reshape(1, -1), resp_b_emb_test[i].reshape(1, -1))[0][0] for i in range(len(resp_a_emb_test))])

del model_minilm, resp_a_emb_train, resp_b_emb_train, resp_a_emb_test, resp_b_emb_test
clear_memory()

# --- 2.4 N-gram 特征 (即时训练) ---
print("⏳ 正在 (即时) 训练 N-gram Vectorizer 并创建特征...")
corpus = pd.concat([train_df['response_a'], train_df['response_b']]).astype(str).unique()

vectorizer = CountVectorizer(
    max_features=2000,
    ngram_range=(1, 2), # 包含 1-grams 和 2-grams
    stop_words='english',
    dtype=np.float32 
)
print("  ...正在 fit Vectorizer...")
vectorizer.fit(corpus)
del corpus
clear_memory()

print("  ...正在 transform 训练集/测试集...")
# 【【【 逻辑修复 】】】: 正确创建差异特征
train_ngram_a = vectorizer.transform(train_df['response_a'].astype(str))
train_ngram_b = vectorizer.transform(train_df['response_b'].astype(str))
train_ngram_diff = (train_ngram_a - train_ngram_b)

test_ngram_a = vectorizer.transform(test_df['response_a'].astype(str))
test_ngram_b = vectorizer.transform(test_df['response_b'].astype(str))
test_ngram_diff = (test_ngram_a - test_ngram_b)

del vectorizer, train_ngram_a, train_ngram_b, test_ngram_a, test_ngram_b
clear_memory()

# --- 2.5 【【【 NameError 修复 】】】 ---
# 在删除 train_df/test_df 之前，提取 4 个偏置特征
print("⏳ 正在提取 4 个偏置特征...")
all_4_features_train = train_df[["len_diff", "len_ratio", "lexical_diff", "cosine_similarity"]].fillna(0).values
all_4_features_test = test_df[["len_diff", "len_ratio", "lexical_diff", "cosine_similarity"]].fillna(0).values

# --- 2.6 现在可以安全删除 ---
del train_df, test_df
clear_memory()

print("✅ 所有特征工程完毕。")

# === 单元格 3: 准备最终特征矩阵 ===
print("\n--- 正在准备最终的训练/测试特征矩阵 ---")
# (所有变量 train_emb_minilm, all_4_features_train, train_ngram_diff 等都已在内存中)

# --- 堆叠 A 套餐 (MiniLM + 4 Feat + Ngram) ---
print(f"  ...堆叠 A 套餐 (MiniLM + 4 Feat + Ngram, 共 {384 + 4 + 2000} 特征)")
X_A_full = hstack([train_emb_minilm, all_4_features_train, train_ngram_diff]).tocsr()
X_test_A_ngram = hstack([test_emb_minilm, all_4_features_test, test_ngram_diff]).tocsr()

print(f"✅ 特征矩阵准备完毕。 训练集: {X_A_full.shape}, 测试集: {X_test_A_ngram.shape}")

# --- 释放内存 ---
del train_emb_minilm, test_emb_minilm, all_4_features_train, all_4_features_test, train_ngram_diff, test_ngram_diff
clear_memory()

# === 单元格 4: 定义 5-Fold 交叉验证 ===
print("\n--- 正在定义 5-Fold CV ---")

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# (y_true_full 来自 单元格 2)

# 定义 LGBM 参数
lgbm_params = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'num_leaves': 64,
    'random_state': 42,
    'device': 'gpu',
    'n_jobs': -1,
    'verbose': -1
}

# --- 初始化存储器 ---
oof_preds_A = np.zeros((len(y_true_full), 3))
test_preds_A_list = []

print(f"✅ K-Fold (n_splits={N_SPLITS}) 已准备就绪。")

# === 单元格 5: 执行 5-Fold CV 训练循环 (仅模型A) ===
print("\n--- 启动 5-Fold 交叉验证训练 (仅模型A) ---")

for fold, (train_indices, val_indices) in enumerate(skf.split(X_A_full, y_true_full)):
    print(f"\n--- 正在训练 Fold {fold+1}/{N_SPLITS} ---")
    
    # --- 准备该 Fold 的数据 ---
    y_train_fold = y_true_full.iloc[train_indices]
    y_val_fold = y_true_full.iloc[val_indices]
    
    X_train_A_fold = X_A_full[train_indices]
    X_val_A_fold = X_A_full[val_indices]

    # --- 训练模型 A (Fold {fold+1}) ---
    print(f"  ⏳ 训练 LGBM-A (Fold {fold+1})...")
    model_A_fold = lgb.LGBMClassifier(**lgbm_params)
    model_A_fold.fit(X_train_A_fold, y_train_fold,
                     eval_set=[(X_val_A_fold, y_val_fold)],
                     eval_metric='logloss',
                     callbacks=[lgb.early_stopping(15, verbose=False)])
    
    # 预测验证集 (用于 OOF)
    oof_preds_A[val_indices] = model_A_fold.predict_proba(X_val_A_fold)
    # 预测测试集
    test_preds_A_list.append(model_A_fold.predict_proba(X_test_A_ngram))
    
    print(f"  ✅ Fold {fold+1} 完成。")
    del model_A_fold, X_train_A_fold, X_val_A_fold
    clear_memory()

print("\n🎉 5-Fold CV 训练全部完成！")

# === 单元格 6: OOF 验证 与 提交 ===

print("\n--- 正在计算完整的 OOF 验证分数 ---")
# (oof_preds_A, y_true_full 均已准备好)

assert not np.any(np.sum(oof_preds_A, axis=1) == 0), "OOF A 中有未预测的行！"
oof_logloss_A = log_loss(y_true_full, oof_preds_A)
print(f"🎯 最佳 [完整 OOF] 模型 A-Ngram LogLoss: {oof_logloss_A:.5f}")

# --- 聚合测试集预测 ---
print("\n--- 正在聚合 5-Fold 的测试集预测 ---")
final_preds = np.mean(test_preds_A_list, axis=0)
print(f"  ...测试集预测已平均。 形状: {final_preds.shape}")

# --- 生成最终的 submission.csv ---
print("\n--- 正在生成最终提交文件 ---")
final_preds = final_preds / final_preds.sum(axis=1, keepdims=True)
final_preds = np.clip(final_preds, 1e-7, 1 - 1e-7)

submission_final = pd.DataFrame(final_preds, columns=sample_df.columns[1:])
submission_final.insert(0, "id", sample_df["id"])

assert len(submission_final) == len(sample_df), "提交文件行数不匹配!"
submission_final.to_csv("submission.csv", index=False) 

print("\n🎉🎉🎉 最终的 (5-Fold CV, 仅模型A-Ngram) submission.csv 已生成！🎉🎉🎉")
print(submission_final.head())

