# === 1: 导入、设置与内存清理 ===
import pandas as pd
import numpy as np
import os
import shutil
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import joblib # 用于保存 LGBM 和 Sklearn 模型

# 特征工程
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# 模型与评估
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit # 采纳分层抽样
from sklearn.metrics import log_loss, confusion_matrix
from lightgbm import LGBMClassifier
from scipy.optimize import minimize

# 深度学习
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, Trainer, 
    TrainingArguments, EarlyStoppingCallback # 采纳早停
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
from sklearn.metrics import accuracy_score

# 禁用 wandb
os.environ["WANDB_MODE"] = "disabled"

# 采纳你的内存清理函数
def clear_memory():
    """清理GPU和CPU内存"""
    print("\n--- 正在清理内存 ---")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("--- 内存清理完毕 ---")

print("🚀 Team 8 本地训练管线启动！")

import os

print(f"--- 你的 Notebook 正在这里运行 ---")
current_directory = os.getcwd()
print(f"当前工作目录 (CWD): {current_directory}")

print(f"\n--- 正在检查此目录下的文件夹 ---")

# 我们检查 'data' 和 'model' 文件夹是否存在于当前目录
data_folder_path = os.path.join(current_directory, "data")
model_folder_path = os.path.join(current_directory, "model")

if os.path.exists(data_folder_path):
    print(f"✅ 成功找到 'data' 文件夹！ 路径: {data_folder_path}")
else:
    print(f"❌ 警告: 未找到 'data' 文件夹。")

if os.path.exists(model_folder_path):
    print(f"✅ 成功找到 'model' 文件夹！ 路径: {model_folder_path}")
else:
    print(f"❌ 警告: 未找到 'model' 文件夹。")

print("\n--- 检查完毕 ---")

# === 2: 定义本地路径并加载数据集 [已修复 Windows 路径] ===

import pandas as pd
import os
import numpy as np # 提前导入 numpy

print("--- 正在设置所有本地文件路径 ---")

# 1. 基础路径 (根据你的输出)
# 【修复】在字符串前添加 'r' 来创建“原始字符串”，防止 \U 错误
BASE_DIR = r"C:\Users\f1285\Desktop\ML_Project"
DATA_DIR = r"C:\Users\f1285\Desktop\ML_Project\data"
MODELS_DIR = r"C:\Users\f1285\Desktop\ML_Project\model"
OUTPUT_DIR = r"C:\Users\f1285\Desktop\ML_Project\output"

# 2. 数据文件路径 (os.path.join 会自动处理斜杠)
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
SAMPLE_FILE = os.path.join(DATA_DIR, "sample_submission.csv")

# 3. 原始模型路径
MINILM_PATH = os.path.join(MODELS_DIR, "sentencetransformersallminilml6v2")
E5_PATH = os.path.join(MODELS_DIR, "e5-small-v2")
DEBERTA_PATH = os.path.join(MODELS_DIR, "deberta-v3-small")
ROBERTA_PATH = os.path.join(MODELS_DIR, "roberta-transformers-pytorch")

# 4. 检查/创建输出目录
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"✅ 已创建输出目录: {OUTPUT_DIR}")
else:
    print(f"✅ 输出目录已找到: {OUTPUT_DIR}")

print("\n--- 正在加载数据集 ---")

try:
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    sample_df = pd.read_csv(SAMPLE_FILE)
    print("✅ 数据集加载成功!")
    print(f"  训练集大小: {train_df.shape}")
except FileNotFoundError as e:
    print(f"❌ 数据加载失败! 错误: {e}")
    print(f"   请再次确认你的 TRAIN_FILE 路径是否正确: {TRAIN_FILE}")

# === 4: 基础特征工程 (所有模型共用) [已修复] ===
print("\n--- 正在执行：基础特征工程 ---")

def create_base_features(df):
    df['text_a'] = df['prompt'] + " " + df['response_a']
    df['text_b'] = df['prompt'] + " " + df['response_b']
    df['combined_for_embedding'] = df['text_a'] + " [SEP] " + df['text_b']
    
    # 长度特征
    df["resp_a_len"] = df["response_a"].str.len()
    df["resp_b_len"] = df["response_b"].str.len()
    df["len_diff"] = df["resp_a_len"] - df["resp_b_len"]
    df["len_ratio"] = df["resp_a_len"] / (df["resp_b_len"] + 1e-6)
    
    # 词汇度特征
    df["lexical_a"] = df["response_a"].apply(lambda x: len(set(str(x).split())) / (len(str(x).split()) + 1e-6))
    df["lexical_b"] = df["response_b"].apply(lambda x: len(set(str(x).split())) / (len(str(x).split()) + 1e-6))
    df["lexical_diff"] = df["lexical_a"] - df["lexical_b"]
    return df

train_df = create_base_features(train_df)
test_df = create_base_features(test_df)

# --- 【修复 Bug】---
# 1. 先在 DataFrame 中创建 'label' 列
train_df['label'] = train_df[['winner_model_a', 'winner_model_b', 'winner_tie']].values.argmax(axis=1)
# 2. 然后, 将 'label' 这一列 (Pandas Series) 赋值给 y_true_full
y_true_full = train_df['label']
# --- 【修复完毕】---

print("✅ 基础特征工程完成 (已创建 3 个偏置特征)。")

# === 5: 嵌入生成 (MiniLM, E5) 与 相似度特征 ===
print("\n--- 正在加载 MiniLM 模型 ---")
model_minilm = SentenceTransformer(MINILM_PATH, device='cuda')

print("⏳ 正在为训练集生成 MiniLM 嵌入...")
train_emb_minilm = model_minilm.encode(
    train_df['combined_for_embedding'].tolist(), 
    show_progress_bar=True, batch_size=128, convert_to_numpy=True
)
print("⏳ 正在为测试集生成 MiniLM 嵌入...")
test_emb_minilm = model_minilm.encode(
    test_df['combined_for_embedding'].tolist(), 
    show_progress_bar=True, batch_size=128, convert_to_numpy=True
)

print("⏳ 正在生成相似度特征 (Train)...")
resp_a_emb_train = model_minilm.encode(train_df['response_a'].tolist(), batch_size=128)
resp_b_emb_train = model_minilm.encode(train_df['response_b'].tolist(), batch_size=128)
print("⏳ 正在生成相似度特征 (Test)...")
resp_a_emb_test = model_minilm.encode(test_df['response_a'].tolist(), batch_size=128)
resp_b_emb_test = model_minilm.encode(test_df['response_b'].tolist(), batch_size=128)

del model_minilm
clear_memory()
print("✅ MiniLM 模型已释放")

print("⏳ 正在计算相似度特征...")
train_df['cosine_similarity'] = np.array([
    cosine_similarity(resp_a_emb_train[i].reshape(1, -1), resp_b_emb_train[i].reshape(1, -1))[0][0] 
    for i in range(len(resp_a_emb_train))
])
test_df['cosine_similarity'] = np.array([
    cosine_similarity(resp_a_emb_test[i].reshape(1, -1), resp_b_emb_test[i].reshape(1, -1))[0][0] 
    for i in range(len(resp_a_emb_test))
])

# --- E5 嵌入 (用于模型 C) ---
print("\n--- 正在加载 E5 模型 ---")
model_e5 = SentenceTransformer(E5_PATH, device='cuda')
print("⏳ 正在为训练集生成 E5 嵌入...")
train_emb_e5 = model_e5.encode(
    train_df["combined_for_embedding"].tolist(), 
    batch_size=128, show_progress_bar=True, convert_to_numpy=True
)
print("⏳ 正在为测试集生成 E5 嵌入...")
test_emb_e5 = model_e5.encode(
    test_df["combined_for_embedding"].tolist(), 
    batch_size=128, show_progress_bar=True, convert_to_numpy=True
)

del model_e5
clear_memory()
print("✅ E5 模型已释放")

# --- 保存所有中间文件 ---
print("⏳ 正在保存所有嵌入和特征到 .npy 文件...")
np.save(os.path.join(OUTPUT_DIR, 'train_emb_minilm.npy'), train_emb_minilm)
np.save(os.path.join(OUTPUT_DIR, 'test_emb_minilm.npy'), test_emb_minilm)
np.save(os.path.join(OUTPUT_DIR, 'train_emb_e5.npy'), train_emb_e5)
np.save(os.path.join(OUTPUT_DIR, 'test_emb_e5.npy'), test_emb_e5)

all_4_features_train = train_df[["len_diff", "len_ratio", "lexical_diff", "cosine_similarity"]].fillna(0).values
all_4_features_test = test_df[["len_diff", "len_ratio", "lexical_diff", "cosine_similarity"]].fillna(0).values
np.save(os.path.join(OUTPUT_DIR, 'train_features_4.npy'), all_4_features_train)
np.save(os.path.join(OUTPUT_DIR, 'test_features_4.npy'), all_4_features_test)

print("✅ 所有嵌入和特征已保存。")

# === 7: 准备最终数据集 (使用分层抽样) ===
print("\n--- 正在准备最终数据集 ---")

# 1. 加载所有特征
train_emb_minilm = np.load(os.path.join(OUTPUT_DIR, 'train_emb_minilm.npy'))
train_emb_e5 = np.load(os.path.join(OUTPUT_DIR, 'train_emb_e5.npy'))
all_4_features_train = np.load(os.path.join(OUTPUT_DIR, 'train_features_4.npy'))

# 2. 准备模型 A 和 C 的特征集
X_A_full = np.hstack([train_emb_minilm, all_4_features_train])
X_C_full = np.hstack([train_emb_e5, all_4_features_train])

# 3. 使用分层抽样
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_indices, val_indices = next(sss.split(train_df, y_true_full))

# 4. 准备 LGBM 模型的训练/验证集
X_train_A = X_A_full[train_indices]
X_val_A = X_A_full[val_indices]
X_train_C = X_C_full[train_indices]
X_val_C = X_C_full[val_indices]

y_train = y_true_full.iloc[train_indices]
y_val = y_true_full.iloc[val_indices]

# 5. 准备 LoRA 模型的训练/验证集
train_df_lora = train_df.iloc[train_indices]
val_df_lora = train_df.iloc[val_indices]

print(f"✅ 分层抽样数据准备完毕。")
print(f"   训练集大小: {len(y_train)} | 验证集大小: {len(y_val)}")

# === 7: 训练并保存 [模型 A (LGBM + MiniLM)] ===
print("\n--- 正在训练 [模型 A] ---")

lgbm_model_A = LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=64, random_state=42,
    device='gpu'
)

print("⏳ 正在训练 LGBM (MiniLM + 4 特征)...")
lgbm_model_A.fit(X_train_A, y_train)

# 评估
val_preds_A = lgbm_model_A.predict_proba(X_val_A)
logloss_A = log_loss(y_val, val_preds_A)
print(f"🎯 [模型 A] Validation LogLoss: {logloss_A:.5f}")

# 保存模型
lgbm_model_A.booster_.save_model(os.path.join(OUTPUT_DIR, 'model_A_lgbm.txt'))
print("✅ [模型 A] 已保存为 'model_A_lgbm.txt'")

# === 8: 训练并保存 [模型 C (LGBM + E5)] ===
print("\n--- 正在训练 [模型 C] ---")

lgbm_model_C = LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=64, random_state=42,
    device='gpu'
)

print("⏳ 正在训练 LGBM (E5 + 4 特征)...")
lgbm_model_C.fit(X_train_C, y_train)

# 评估
val_preds_C = lgbm_model_C.predict_proba(X_val_C)
logloss_C = log_loss(y_val, val_preds_C)
print(f"🎯 [模型 C] Validation LogLoss: {logloss_C:.5f}")

# 保存模型
lgbm_model_C.booster_.save_model(os.path.join(OUTPUT_DIR, 'model_C_lgbm.txt'))
print("✅ [模型 C] 已保存为 'model_C_lgbm.txt'")

# === 9: 训练并保存 [模型 B (LoRA DeBERTa-small)] [已修复 num_proc] ===
print("\n--- 正在训练 [模型 B] ---")

# 1. 加载模型
local_model_path = "./deberta-small-local"
if not os.path.exists(local_model_path):
    shutil.copytree(DEBERTA_PATH, local_model_path)
tokenizer_B = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
base_model_B = AutoModelForSequenceClassification.from_pretrained(local_model_path, num_labels=3, local_files_only=True)

# 2. 配置 LoRA
peft_config_B = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.1, bias="none")
model_B = get_peft_model(base_model_B, peft_config_B)

# 3. 数据处理
def preprocess_function_B(examples):
    texts = [f"问题: {p} [SEP] A: {a} [SEP] B: {b}" for p, a, b in zip(examples["prompt"], examples["response_a"], examples["response_b"])]
    return tokenizer_B(texts, truncation=True, padding="max_length", max_length=256)

train_dataset = Dataset.from_pandas(train_df_lora)
val_dataset = Dataset.from_pandas(val_df_lora)

# --- 【最终修复】完全删除 num_proc 参数以禁用多进程 ---
print("⏳ 正在 (单进程) 映射训练集 (DeBERTa)...")
tokenized_train_B = train_dataset.map(preprocess_function_B, batched=True) 
print("⏳ 正在 (单进程) 映射验证集 (DeBERTa)...")
tokenized_val_B = val_dataset.map(preprocess_function_B, batched=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return {"accuracy": accuracy_score(labels, np.argmax(logits, axis=-1))}

# 5. 改进的 LoRA 训练配置
training_args_B = TrainingArguments(
    output_dir="./ft_results_deberta",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=3e-4,
    save_total_limit=2,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    dataloader_num_workers=0,  # 保持为 0
    logging_steps=200,
    evaluation_strategy="epoch",
    fp16=True,
    fp16_full_eval=True,
    report_to=[]
)

trainer_B = Trainer(
    model=model_B,
    args=training_args_B,
    train_dataset=tokenized_train_B,
    eval_dataset=tokenized_val_B,
    tokenizer=tokenizer_B,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# 6. 训练
print("⏳ 开始 LoRA 微调 (DeBERTa-small)...")
trainer_B.train()
print("✅ 微调完成。")

# 7. 评估
val_logits_B = trainer_B.predict(tokenized_val_B).predictions
val_preds_B_uncalibrated = torch.softmax(torch.tensor(val_logits_B), dim=-1).numpy()
logloss_B_uncalibrated = log_loss(y_val, val_preds_B_uncalibrated)
print(f"🎯 [模型 B] 校准前 Validation LogLoss: {logloss_B_uncalibrated:.5f}")

# 8. 保存最佳模型
trainer_B.save_model(os.path.join(OUTPUT_DIR, 'model_B_deberta_lora'))
tokenizer_B.save_pretrained(os.path.join(OUTPUT_DIR, 'model_B_deberta_lora'))
print(f"✅ [模型 B] 已保存到 {os.path.join(OUTPUT_DIR, 'model_B_deberta_lora')}")

del model_B, base_model_B, trainer_B, tokenizer_B
clear_memory()

# === 10: 训练并保存 [模型 D (LoRA RoBERTa-base)] [已修复 num_proc] ===
print("\n--- 正在训练 [模型 D] ---")

# 1. 加载模型
local_model_path = "./roberta-base-local"
if not os.path.exists(local_model_path):
    shutil.copytree(ROBERTA_PATH, local_model_path)
tokenizer_D = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
base_model_D = AutoModelForSequenceClassification.from_pretrained(local_model_path, num_labels=3, local_files_only=True)

# 2. 配置 LoRA
peft_config_D = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.1, bias="none")
model_D = get_peft_model(base_model_D, peft_config_D)

# 3. 数据处理
def preprocess_function_D(examples):
    texts = [f"问题: {p} [SEP] A: {a} [SEP] B: {b}" for p, a, b in zip(examples["prompt"], examples["response_a"], examples["response_b"])]
    return tokenizer_D(texts, truncation=True, padding="max_length", max_length=256)

train_dataset = Dataset.from_pandas(train_df_lora)
val_dataset = Dataset.from_pandas(val_df_lora)

# --- 【最终修复】完全删除 num_proc 参数以禁用多进程 ---
print("⏳ 正在 (单进程) 映射训练集 (RoBERTa)...")
tokenized_train_D = train_dataset.map(preprocess_function_D, batched=True)
print("⏳ 正在 (单进程) 映射验证集 (RoBERTa)...")
tokenized_val_D = val_dataset.map(preprocess_function_D, batched=True)

# 4. 训练配置
training_args_D = TrainingArguments(
    output_dir="./ft_results_roberta_base",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=3e-4,
    save_total_limit=2,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    dataloader_num_workers=0, # 保持为 0
    logging_steps=200,
    evaluation_strategy="epoch",
    fp16=True,
    fp16_full_eval=True,
    report_to=[]
)

trainer_D = Trainer(
    model=model_D,
    args=training_args_D,
    train_dataset=tokenized_train_D,
    eval_dataset=tokenized_val_D,
    tokenizer=tokenizer_D,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# 5. 训练
print("⏳ 开始 LoRA 微调 (RoBERTa-base)...")
trainer_D.train()
print("✅ 微调完成。")

# 6. 评估
val_logits_D = trainer_D.predict(tokenized_val_D).predictions
val_preds_D_uncalibrated = torch.softmax(torch.tensor(val_logits_D), dim=-1).numpy()
logloss_D_uncalibrated = log_loss(y_val, val_preds_D_uncalibrated)
print(f"🎯 [模型 D] 校准前 Validation LogLoss: {logloss_D_uncalibrated:.5f}")

# 7. 保存最佳模型
trainer_D.save_model(os.path.join(OUTPUT_DIR, 'model_D_roberta_lora'))
tokenizer_D.save_pretrained(os.path.join(OUTPUT_DIR, 'model_D_roberta_lora'))
print(f"✅ [模型 D] 已保存到 {os.path.join(OUTPUT_DIR, 'model_D_roberta_lora')}")

del model_D, base_model_D, trainer_D, tokenizer_D
clear_memory()

# === 11: 概率校准与权重优化 ===
print("\n--- 正在执行: 概率校准与权重优化 ---")

def temperature_scale(logits, T):
    logits_T = logits / T
    return torch.softmax(torch.tensor(logits_T), dim=-1).numpy()

def loss_fn_cal(T, logits, labels):
    probs_T = temperature_scale(logits, T)
    return log_loss(labels, probs_T)

# --- 校准模型 B (DeBERTa) ---
res_B = minimize(loss_fn_cal, x0=[1.0], args=(val_logits_B, y_val), 
               bounds=[(0.5, 5.0)], method="L-BFGS-B")
T_opt_B = res_B.x[0]
logloss_B_calibrated = res_B.fun
print(f"📏 [模型 B] DeBERTa T = {T_opt_B:.3f} | 校准后 Loss: {logloss_B_calibrated:.5f}")
val_preds_B = temperature_scale(val_logits_B, T_opt_B)
np.save(os.path.join(OUTPUT_DIR, 'temp_B.npy'), np.array([T_opt_B]))

# --- 校准模型 D (RoBERTa) ---
res_D = minimize(loss_fn_cal, x0=[1.0], args=(val_logits_D, y_val), 
               bounds=[(0.5, 5.0)], method="L-BFGS-B")
T_opt_D = res_D.x[0]
logloss_D_calibrated = res_D.fun
print(f"📏 [模型 D] RoBERTa T = {T_opt_D:.3f} | 校准后 Loss: {logloss_D_calibrated:.5f}")
val_preds_D = temperature_scale(val_logits_D, T_opt_D)
np.save(os.path.join(OUTPUT_DIR, 'temp_D.npy'), np.array([T_opt_D]))

# --- 优化集成权重 (采纳 SLSQP 建议) ---
def loss_fn_ensemble(weights):
    wA, wB, wC = weights
    wD = 1.0 - wA - wB - wC
    if wD < 0 or min(weights) < 0: return 100.0
    ensemble_val_preds = (
        (val_preds_A * wA) + (val_preds_B * wB) +
        (val_preds_C * wC) + (val_preds_D * wD)
    )
    ensemble_val_preds = np.clip(ensemble_val_preds, 1e-7, 1 - 1e-7)
    return log_loss(y_val, ensemble_val_preds)

initial_weights = [0.4, 0.1, 0.4]  # [wA, wB, wC]
bounds = [(0, 1), (0, 1), (0, 1)]
constraints = {'type': 'ineq', 'fun': lambda w: 1.0 - sum(w)}

res = minimize(
    loss_fn_ensemble, initial_weights, method='SLSQP',
    bounds=bounds, constraints=constraints
)

wA_opt, wB_opt, wC_opt = res.x
wD_opt = 1.0 - sum(res.x)
print(f"\n🎯 最佳集成验证 LogLoss: {res.fun:.5f}")
print("--- 最佳权重 ---")
print(f"模型 A (LGBM+MiniLM): {wA_opt:.4f}")
print(f"模型 B (LoRA-DeBERTa): {wB_opt:.4f}")
print(f"模型 C (LGBM+E5):     {wC_opt:.4f}")
print(f"模型 D (LoRA-RoBERTa): {wD_opt:.4f}")

# --- 保存最终权重 ---
final_weights = np.array([wA_opt, wB_opt, wC_opt, wD_opt])
np.save(os.path.join(OUTPUT_DIR, 'ensemble_weights.npy'), final_weights)
print(f"✅ 最终权重已保存到 'ensemble_weights.npy'")

# === 12: 训练完成 ===
print(f"🎉🎉🎉 所有模型训练和优化完毕！🎉🎉🎉")
print(f"所有必需的文件都已保存在你的输出文件夹中: \n{OUTPUT_DIR}")

print("\n你需要打包并上传到 Kaggle Dataset 的文件：")
print("---------------------------------")
print("1. model_baseline.joblib")
print("2. vectorizer_baseline.joblib")
print("3. scaler_baseline.joblib")
print("4. model_A_lgbm.txt")
print("5. model_C_lgbm.txt")
print("6. model_B_deberta_lora/ (整个文件夹)")
print("7. model_D_roberta_lora/ (整个文件夹)")
print("8. temp_B.npy")
print("9. temp_D.npy")
print("10. ensemble_weights.npy")

# === 步骤 14: 高级错误分析 (A/C vs D) ===
print("\n--- 正在执行：高级错误分析 (A/C 失败 vs D 成功) ---")

# 1. 获取所有模型的验证集预测类别
pred_class_A = np.argmax(val_preds_A, axis=1) # (来自单元格 8)
pred_class_C = np.argmax(val_preds_C, axis=1) # (来自单元格 9)
pred_class_D = np.argmax(val_preds_D, axis=1) # (来自单元格 12)
# y_val 是真实标签 (来自单元格 7)

# 2. 找到我们感兴趣的样本索引
#    (A 错了 AND C 错了 AND D 对了)
error_indices = np.where(
    (pred_class_A != y_val) &
    (pred_class_C != y_val) &
    (pred_class_D == y_val)
)[0]

print(f"✅ 找到 {len(error_indices)} 个样本，其中模型 A 和 C 都失败了，但模型 D 成功了。")

# 3. 提取这些样本的原始文本 (使用 'val_indices')
analysis_df = train_df.loc[val_indices[error_indices]].copy()
analysis_df['true_label'] = y_val.iloc[error_indices]
analysis_df['pred_D_label'] = pred_class_D[error_indices]

print("\n--- 正在显示 A/C 的共同盲区 (前 10 个样本) ---")

for idx, row in analysis_df.head(10).iterrows():
    print(f"\n--- 样本 ID: {idx} | 真实标签: {row['true_label']} (模型 D 猜对了) ---")
    
    # 获取 A 和 C 的错误预测
    idx_in_val_set = np.where(val_indices == idx)[0][0]
    print(f"    模型 A 的错误预测: {pred_class_A[idx_in_val_set]} (置信度: {val_preds_A[idx_in_val_set].max():.2%})")
    print(f"    模型 C 的错误预测: {pred_class_C[idx_in_val_set]} (置信度: {val_preds_C[idx_in_val_set].max():.2%})")
    
    print(f"    Prompt: {row['prompt'][:100]}...")
    print(f"    Response A: {row['response_a'][:100]}...")
    print(f"    Response B: {row['response_b'][:100]}...")

# === 步骤 15: 模型 D (RoBERTa-base) 的“定量”错误分析 ===

print("--- 正在为模型 D (RoBERTa-base) 生成混淆矩阵 ---")

# (y_val 是真实标签)
# (val_preds_D 是模型 D 校准后的验证集预测)
y_pred_classes_D = np.argmax(val_preds_D, axis=1)

cm_D = confusion_matrix(y_val, y_pred_classes_D)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_D, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['A Win', 'B Win', 'Tie'], 
            yticklabels=['A Win', 'B Win', 'Tie'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Model D (LoRA RoBERTa-base @ 1.06579)')
plt.savefig("confusion_matrix_model_D.png") # 保存为新文件名
print("✅ 模型 D 的混淆矩阵已保存为 'confusion_matrix_model_D.png'")

# === 步骤 16: 模型 D (RoBERTa-base) 的“定性”错误分析 ===

print("--- 正在为模型 D (RoBERTa-base) 查找最差的 10 个预测 ---")

# 1. (y_val 是真实标签, val_preds_D 是预测概率)
# 2. (val_indices 是验证集在 train_df 中的索引)
val_df_D = train_df.loc[val_indices].copy()
val_df_D['true_label'] = y_val
val_df_D['pred_prob_A'] = val_preds_D[:, 0]
val_df_D['pred_prob_B'] = val_preds_D[:, 1]
val_df_D['pred_prob_Tie'] = val_preds_D[:, 2]

# 3. 找出模型预测的类别
val_df_D['predicted_label'] = y_pred_classes_D

# 4. 找出所有预测错误的样本
error_df_D = val_df_D[val_df_D['true_label'] != val_df_D['predicted_label']].copy()

# 5. 找出错误样本中，模型对“错误答案”的置信度
error_df_D['confidence_in_wrong_answer'] = np.max(val_preds_D[val_df_D.index.isin(error_df_D.index)], axis=1)

# 6. 按“对错误答案的置信度”降序排列，找出最自信的 10 个错误
worst_misses_D = error_df_D.sort_values(by='confidence_in_wrong_answer', ascending=False).head(10)

print("--- 10个模型 D“最自信的错误” (用于报告分析) ---")
# 打印这些样本的关键信息
for idx, row in worst_misses_D.iterrows():
    print(f"\n--- 样本 ID: {idx} | 真实标签: {row['true_label']} | 错误预测: {row['predicted_label']} ---")
    print(f"    模型对(错误的)预测 {row['predicted_label']} 的置信度: {row['confidence_in_wrong_answer']:.2%}")
    print(f"    (A/B/Tie 概率): {row['pred_prob_A']:.2f} / {row['pred_prob_B']:.2f} / {row['pred_prob_Tie']:.2f}")
    print(f"    Prompt: {row['prompt'][:100]}...")
    print(f"    Response A: {row['response_a'][:100]}...")
    print(f"    Response B: {row['response_b'][:100]}...")

worst_misses_D.to_csv("worst_misses_model_D.csv", index=False)
print("\n✅ 模型 D 最差预测的详细信息已保存到 'worst_misses_model_D.csv'")

# === 步骤 15: 优化版模型 A (LGBM+MiniLM) 的混淆矩阵 ===
print("--- 正在为模型 A (LGBM+MiniLM) 生成混淆矩阵 ---")

# (y_val 是真实标签)
# (val_preds_A 是模型 A 的验证集预测)
y_pred_classes_A = np.argmax(val_preds_A, axis=1)

cm_A = confusion_matrix(y_val, y_pred_classes_A)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_A, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['A Win', 'B Win', 'Tie'], 
            yticklabels=['A Win', 'B Win', 'Tie'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Optimized LGBM (MiniLM + 4 Feat @ 1.03534)')
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_model_A.png"))
print(f"✅ 模型 A 的混淆矩阵已保存到: {os.path.join(OUTPUT_DIR, 'confusion_matrix_model_A.png')}")

# === 步骤 16: 优化版模型 C (LGBM+E5) 的混淆矩阵 ===
print("--- 正在为模型 C (LGBM+E5) 生成混淆矩阵 ---")

# (y_val 是真实标签)
# (val_preds_C 是模型 C 的验证集预测)
y_pred_classes_C = np.argmax(val_preds_C, axis=1)

cm_C = confusion_matrix(y_val, y_pred_classes_C)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_C, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['A Win', 'B Win', 'Tie'], 
            yticklabels=['A Win', 'B Win', 'Tie'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Optimized LGBM (E5 + 4 Feat @ 1.03605)')
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_model_C.png"))
print(f"✅ 模型 C 的混淆矩阵已保存到: {os.path.join(OUTPUT_DIR, 'confusion_matrix_model_C.png')}")

# === 步骤 17: (修正版) 训练 [模型 E (LoRA DeBERTa-v3-base)] ===
print("\n--- 正在训练 [模型 E (LoRA DeBERTa-v3-base)] ---")

# 1. 定义新模型路径
DEBERTAv3_BASE_PATH = os.path.join(MODELS_DIR, "deberta-v3-base")

if not os.path.exists(DEBERTAv3_BASE_PATH):
    print(f"❌ 错误: 找不到 DeBERTa-v3-base 模型。")
    print(f"   请确保它存在于: {DEBERTAv3_BASE_PATH}")
else:
    print(f"✅ 找到 DeBERTa-v3-base 模型: {DEBERTAv3_BASE_PATH}")

    # 2. 加载模型 (使用新变量名)
    local_model_path_E = "./deberta-v3-base-local"
    if not os.path.exists(local_model_path_E):
        shutil.copytree(DEBERTAv3_BASE_PATH, local_model_path_E)
        
    tokenizer_E = AutoTokenizer.from_pretrained(local_model_path_E, local_files_only=True)
    base_model_E = AutoModelForSequenceClassification.from_pretrained(local_model_path_E, num_labels=3, local_files_only=True)

    # 3. 配置 LoRA
    # 【修复】: 将 Task_TYPE 修正为 TaskType
    peft_config_E = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.1, bias="none")
    model_E = get_peft_model(base_model_E, peft_config_E)

    # 4. 数据处理 (使用与模型 D 相同的处理函数)
    def preprocess_function_E(examples):
        texts = [f"问题: {p} [SEP] A: {a} [SEP] B: {b}" for p, a, b in zip(examples["prompt"], examples["response_a"], examples["response_b"])]
        return tokenizer_E(texts, truncation=True, padding="max_length", max_length=256)

    # (train_df_lora 和 val_df_lora 来自 步骤 7)
    train_dataset = Dataset.from_pandas(train_df_lora)
    val_dataset = Dataset.from_pandas(val_df_lora)

    print("⏳ 正在 (单进程) 映射训练集 (DeBERTa-v3-base)...")
    tokenized_train_E = train_dataset.map(preprocess_function_E, batched=True)
    print("⏳ 正在 (单进程) 映射验证集 (DeBERTa-v3-base)...")
    tokenized_val_E = val_dataset.map(preprocess_function_E, batched=True)

    # 5. 训练配置 (使用原始 3e-4 学习率)
    training_args_E = TrainingArguments(
        output_dir="./ft_results_deberta_base", # 新的输出目录
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=3e-4, # (使用原始学习率)
        save_total_limit=2,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_num_workers=0, 
        logging_steps=200,
        evaluation_strategy="epoch",
        fp16=True,
        fp16_full_eval=True,
        report_to=[]
    )
    
    # (compute_metrics 来自 步骤 9)
    trainer_E = Trainer(
        model=model_E,
        args=training_args_E,
        train_dataset=tokenized_train_E,
        eval_dataset=tokenized_val_E,
        tokenizer=tokenizer_E,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # 6. 训练
    print("⏳ 开始 LoRA 微调 (DeBERTa-v3-base)...")
    trainer_E.train()
    print("✅ 微调完成。")

    # 7. 评估 (创建新变量)
    val_logits_E = trainer_E.predict(tokenized_val_E).predictions
    val_preds_E_uncalibrated = torch.softmax(torch.tensor(val_logits_E), dim=-1).numpy()
    logloss_E_uncalibrated = log_loss(y_val, val_preds_E_uncalibrated)
    print(f"🎯 [模型 E] 校准前 Validation LogLoss: {logloss_E_uncalibrated:.5f}")

    # 8. 保存最佳模型 (保存到新路径)
    new_model_path_E = os.path.join(OUTPUT_DIR, 'model_E_deberta_base_lora')
    trainer_E.save_model(new_model_path_E)
    tokenizer_E.save_pretrained(new_model_path_E)
    print(f"✅ [模型 E] 已保存到 {new_model_path_E}")

    del model_E, base_model_E, trainer_E, tokenizer_E
    clear_memory()

# === 步骤 18: (新) 5 模型校准与权重优化 (A+B+C+D_orig+E_new) ===
print("\n--- 正在执行: 5 模型校准与权重优化 ---")

# (重用 步骤 11 的函数)
def temperature_scale(logits, T):
    logits_T = logits / T
    return torch.softmax(torch.tensor(logits_T), dim=-1).numpy()

def loss_fn_cal(T, logits, labels):
    probs_T = temperature_scale(logits, T)
    return log_loss(labels, probs_T)

# --- 1. 校准 LoRA 模型 B, D, E ---
# B (来自步骤 9)
res_B = minimize(loss_fn_cal, x0=[1.0], args=(val_logits_B, y_val), bounds=[(0.5, 5.0)], method="L-BFGS-B")
T_opt_B = res_B.x[0]
val_preds_B = temperature_scale(val_logits_B, T_opt_B)
print(f"📏 [模型 B] DeBERTa-small T = {T_opt_B:.3f} | Loss: {res_B.fun:.5f}")
# (temp_B.npy 已在 步骤 11 保存)

# D (原始, 来自步骤 10)
res_D = minimize(loss_fn_cal, x0=[1.0], args=(val_logits_D, y_val), bounds=[(0.5, 5.0)], method="L-BFGS-B")
T_opt_D = res_D.x[0]
val_preds_D = temperature_scale(val_logits_D, T_opt_D)
print(f"📏 [模型 D-Orig] RoBERTa-base T = {T_opt_D:.3f} | Loss: {res_D.fun:.5f}")
# (temp_D.npy 已在 步骤 11 保存)

# E (新, 来自步骤 17)
res_E = minimize(loss_fn_cal, x0=[1.0], args=(val_logits_E, y_val), bounds=[(0.5, 5.0)], method="L-BFGS-B")
T_opt_E = res_E.x[0]
logloss_E_calibrated = res_E.fun
val_preds_E = temperature_scale(val_logits_E, T_opt_E)
print(f"📏 [模型 E-New] DeBERTa-base T = {T_opt_E:.3f} | Loss: {logloss_E_calibrated:.5f}")
# 保存新的温度文件
np.save(os.path.join(OUTPUT_DIR, 'temp_E.npy'), np.array([T_opt_E]))


# --- 2. 优化 5 模型集成权重 ---
# (val_preds_A 来自步骤 8, val_preds_C 来自步骤 9)

def loss_fn_ensemble_5(weights):
    wA, wB, wC, wD = weights
    wE = 1.0 - wA - wB - wC - wD
    if wE < 0 or min(weights) < 0: return 100.0
    ensemble_val_preds = (
        (val_preds_A * wA) + (val_preds_B * wB) +
        (val_preds_C * wC) + (val_preds_D * wD) + 
        (val_preds_E * wE) # 添加模型 E
    )
    ensemble_val_preds = np.clip(ensemble_val_preds, 1e-7, 1 - 1e-7)
    return log_loss(y_val, ensemble_val_preds)

initial_weights_5 = [0.3, 0.1, 0.3, 0.1]  # [wA, wB, wC, wD]
bounds_5 = [(0, 1), (0, 1), (0, 1), (0, 1)]
constraints_5 = {'type': 'ineq', 'fun': lambda w: 1.0 - sum(w)}

res_5 = minimize(
    loss_fn_ensemble_5, initial_weights_5, method='SLSQP',
    bounds=bounds_5, constraints=constraints_5
)

wA_opt_5, wB_opt_5, wC_opt_5, wD_opt_5 = res_5.x
wE_opt_5 = 1.0 - sum(res_5.x)
print(f"\n🎯 [5 模型集成] 最佳集成验证 LogLoss: {res_5.fun:.5f}")
print("--- [5 模型] 最佳权重 ---")
print(f"模型 A (LGBM+MiniLM): {wA_opt_5:.4f}")
print(f"模型 B (LoRA-DeBERTa-small): {wB_opt_5:.4f}")
print(f"模型 C (LGBM+E5):     {wC_opt_5:.4f}")
print(f"模型 D (LoRA-RoBERTa-Orig): {wD_opt_5:.4f}")
print(f"模型 E (LoRA-DeBERTa-base): {wE_opt_5:.4f}")

# --- 3. 保存最终的 5 模型权重 ---
final_weights_5 = np.array([wA_opt_5, wB_opt_5, wC_opt_5, wD_opt_5, wE_opt_5])
np.save(os.path.join(OUTPUT_DIR, 'ensemble_weights_5model.npy'), final_weights_5)
print(f"✅ 最终(5模型)权重已保存到 'ensemble_weights_5model.npy'")

# === 步骤 19: (新) 模型 E (DeBERTa-v3-base) 的“定量”错误分析 ===
print("\n--- 正在为模型 E (DeBERTa-v3-base) 生成混淆矩阵 ---")

# (y_val 是真实标签)
# (val_preds_E 是来自 步骤 18 的 E 校准后预测)
y_pred_classes_E = np.argmax(val_preds_E, axis=1)

cm_E = confusion_matrix(y_val, y_pred_classes_E)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_E, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['A Win', 'B Win', 'Tie'], 
            yticklabels=['A Win', 'B Win', 'Tie'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix for Model E (LoRA DeBERTa-base @ {logloss_E_calibrated:.5f})')
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_model_E.png"))
print(f"✅ 模型 E 的混淆矩阵已保存到: {os.path.join(OUTPUT_DIR, 'confusion_matrix_model_E.png')}")

# === 步骤 21: (新) 为 LGBM 创建 N-gram 差异特征 ===
print("\n--- 正在为 LGBM 创建 N-gram 特征 ---")

# 1. 创建一个包含训练集中所有 response_a 和 response_b 的语料库
print("⏳ 正在构建 N-gram 语料库...")
corpus = pd.concat([
    train_df['response_a'],
    train_df['response_b']
]).astype(str).unique() # 使用 unique 减少 fit 的工作量

# 2. 初始化并训练 CountVectorizer
# 我们使用 2000 个特征, 包含单词(1,1)和双词(2,2)
vectorizer = CountVectorizer(
    max_features=2000,
    ngram_range=(1, 2), # 包含 1-grams 和 2-grams
    stop_words='english',
    dtype=np.float32 # 节省内存
)

print("⏳ 正在训练 CountVectorizer (fit)...")
vectorizer.fit(corpus)

# 3. 转换训练集和测试集
print("⏳ 正在转换 (transform) 训练集 A/B...")
train_ngram_a = vectorizer.transform(train_df['response_a'].astype(str))
train_ngram_b = vectorizer.transform(train_df['response_b'].astype(str))

print("⏳ 正在转换 (transform) 测试集 A/B...")
test_ngram_a = vectorizer.transform(test_df['response_a'].astype(str))
test_ngram_b = vectorizer.transform(test_df['response_b'].astype(str))

# 4. 创建差异特征 (A - B)
# 这会创建一个稀疏矩阵，其中包含 A 中独有/更多的 N-gram（正值）和 B 中独有/更多的 N-gram（负值）
train_ngram_diff = (train_ngram_a - train_ngram_b)
test_ngram_diff = (test_ngram_a - test_ngram_b)

print(f"✅ N-gram 差异特征已创建 (形状: {train_ngram_diff.shape})")

# 5. 保存特征和 vectorizer 到 output
from scipy.sparse import save_npz, load_npz
import joblib

print("⏳ 正在保存 N-gram 特征和 Vectorizer...")
save_npz(os.path.join(OUTPUT_DIR, 'train_ngram_diff.npz'), train_ngram_diff)
save_npz(os.path.join(OUTPUT_DIR, 'test_ngram_diff.npz'), test_ngram_diff)
joblib.dump(vectorizer, os.path.join(OUTPUT_DIR, 'vectorizer_ngram.joblib'))

print("✅ N-gram 模块已保存。")

del corpus, train_ngram_a, train_ngram_b, test_ngram_a, test_ngram_b
clear_memory()

# === 步骤 22: (新) 重新训练 [模型 A-Ngram (LGBM + MiniLM + Ngram)] ===
print("\n--- 正在训练 [模型 A-Ngram] ---")
from scipy.sparse import hstack, load_npz

# 1. 加载所有需要的特征
# (来自 步骤 7)
train_emb_minilm = np.load(os.path.join(OUTPUT_DIR, 'train_emb_minilm.npy'))
all_4_features_train = np.load(os.path.join(OUTPUT_DIR, 'train_features_4.npy'))
# (来自 步骤 21)
train_ngram_diff = load_npz(os.path.join(OUTPUT_DIR, 'train_ngram_diff.npz'))

print(f"  MiniLM 嵌入: {train_emb_minilm.shape}")
print(f"  4 个偏置特征: {all_4_features_train.shape}")
print(f"  N-gram 差异: {train_ngram_diff.shape}")

# 2. 堆叠所有特征
# (hstack 会自动处理稀疏和密集矩阵的堆叠)
X_A_ngram_full = hstack([
    train_emb_minilm,
    all_4_features_train,
    train_ngram_diff
]).tocsr() # 转换为 CSR 格式以便于索引

print(f"✅ 新的 A 特征矩阵已创建 (形状: {X_A_ngram_full.shape})")

# 3. 使用相同的分层抽样索引 (来自 步骤 7)
X_train_A_ngram = X_A_ngram_full[train_indices]
X_val_A_ngram = X_A_ngram_full[val_indices]
# (y_train 和 y_val 来自 步骤 7)

# 4. 训练新的 LGBM 模型
lgbm_model_A_ngram = LGBMClassifier(
    n_estimators=300, 
    learning_rate=0.05, 
    num_leaves=64, 
    random_state=42,
    device='gpu'
)

print("⏳ 正在训练 LGBM (MiniLM + 4 特征 + N-gram)...")
lgbm_model_A_ngram.fit(X_train_A_ngram, y_train)

# 5. 评估 (创建新变量)
val_preds_A_ngram = lgbm_model_A_ngram.predict_proba(X_val_A_ngram)
logloss_A_ngram = log_loss(y_val, val_preds_A_ngram)
print(f"🎯 [模型 A-Ngram] Validation LogLoss: {logloss_A_ngram:.5f}")
print(f"  (原始模型 A LogLoss: {logloss_A:.5f})") # logloss_A 来自 步骤 8

# 6. 保存新模型
lgbm_model_A_ngram.booster_.save_model(os.path.join(OUTPUT_DIR, 'model_A_lgbm_ngram.txt'))
print("✅ [模型 A-Ngram] 已保存为 'model_A_lgbm_ngram.txt'")

del X_A_ngram_full, X_train_A_ngram, X_val_A_ngram, lgbm_model_A_ngram
clear_memory()

# === 步骤 23: (新) 重新训练 [模型 C-Ngram (LGBM + E5 + Ngram)] ===
print("\n--- 正在训练 [模型 C-Ngram] ---")

# 1. 加载所有需要的特征
# (来自 步骤 7)
train_emb_e5 = np.load(os.path.join(OUTPUT_DIR, 'train_emb_e5.npy'))
# (all_4_features_train 和 train_ngram_diff 已在 步骤 22 加载过)

print(f"  E5 嵌入: {train_emb_e5.shape}")
print(f"  4 个偏置特征: {all_4_features_train.shape}")
print(f"  N-gram 差异: {train_ngram_diff.shape}")

# 2. 堆叠所有特征
X_C_ngram_full = hstack([
    train_emb_e5,
    all_4_features_train,
    train_ngram_diff
]).tocsr() 

print(f"✅ 新的 C 特征矩阵已创建 (形状: {X_C_ngram_full.shape})")

# 3. 使用相同的分层抽样索引 (来自 步骤 7)
X_train_C_ngram = X_C_ngram_full[train_indices]
X_val_C_ngram = X_C_ngram_full[val_indices]

# 4. 训练新的 LGBM 模型
lgbm_model_C_ngram = LGBMClassifier(
    n_estimators=300, 
    learning_rate=0.05, 
    num_leaves=64, 
    random_state=42,
    device='gpu'
)

print("⏳ 正在训练 LGBM (E5 + 4 特征 + N-gram)...")
lgbm_model_C_ngram.fit(X_train_C_ngram, y_train)

# 5. 评估 (创建新变量)
val_preds_C_ngram = lgbm_model_C_ngram.predict_proba(X_val_C_ngram)
logloss_C_ngram = log_loss(y_val, val_preds_C_ngram)
print(f"🎯 [模型 C-Ngram] Validation LogLoss: {logloss_C_ngram:.5f}")
print(f"  (原始模型 C LogLoss: {logloss_C:.5f})") # logloss_C 来自 步骤 9

# 6. 保存新模型
lgbm_model_C_ngram.booster_.save_model(os.path.join(OUTPUT_DIR, 'model_C_lgbm_ngram.txt'))
print("✅ [模型 C-Ngram] 已保存为 'model_C_lgbm_ngram.txt'")

del X_C_ngram_full, X_train_C_ngram, X_val_C_ngram, lgbm_model_C_ngram, train_ngram_diff
clear_memory()

# === 步骤 24: (新) 最终 5 模型集成 (使用 A/C N-gram 版) ===
print("\n--- 正在执行: 最终 5 模型集成 (使用 A-Ngram, C-Ngram) ---")

# (val_preds_B, val_preds_D, val_preds_E 来自 步骤 18)
# (val_preds_A_ngram 来自 步骤 22, val_preds_C_ngram 来自 步骤 23)

print(f"  A-Ngram Loss: {logloss_A_ngram:.5f}")
print(f"  C-Ngram Loss: {logloss_C_ngram:.5f}")

# --- 优化 5 模型集成权重 (A-Ngram + B + C-Ngram + D + E) ---

def loss_fn_ensemble_5_ngram(weights):
    wA_ng, wB, wC_ng, wD = weights
    wE = 1.0 - wA_ng - wB - wC_ng - wD
    if wE < 0 or min(weights) < 0: return 100.0
    ensemble_val_preds = (
        (val_preds_A_ngram * wA_ng) +  # 新 A
        (val_preds_B * wB) +           
        (val_preds_C_ngram * wC_ng) +  # 新 C
        (val_preds_D * wD) +           
        (val_preds_E * wE)             
    )
    ensemble_val_preds = np.clip(ensemble_val_preds, 1e-7, 1 - 1e-7)
    return log_loss(y_val, ensemble_val_preds)

initial_weights_5 = [0.3, 0.1, 0.3, 0.1]  # [wA, wB, wC, wD]
bounds_5 = [(0, 1), (0, 1), (0, 1), (0, 1)]
constraints_5 = {'type': 'ineq', 'fun': lambda w: 1.0 - sum(w)}

res_5_ngram = minimize(
    loss_fn_ensemble_5_ngram, initial_weights_5, method='SLSQP',
    bounds=bounds_5, constraints=constraints_5
)

wA_opt_5_ng, wB_opt_5_ng, wC_opt_5_ng, wD_opt_5_ng = res_5_ngram.x
wE_opt_5_ng = 1.0 - sum(res_5_ngram.x)
print(f"\n🎯 [N-gram 5 模型集成] 最佳集成验证 LogLoss: {res_5_ngram.fun:.5f}")
print(f"  (上一次 5 模型 LogLoss: {res_5.fun:.5f})") # res_5 来自 步骤 18

print("--- [N-gram 5 模型] 最佳权重 ---")
print(f"模型 A-Ngram: {wA_opt_5_ng:.4f}")
print(f"模型 B:       {wB_opt_5_ng:.4f}")
print(f"模型 C-Ngram: {wC_opt_5_ng:.4f}")
print(f"模型 D-Orig:  {wD_opt_5_ng:.4f}")
print(f"模型 E-New:   {wE_opt_5_ng:.4f}")

# --- 保存最终的 N-gram 5 模型权重 ---
final_weights_5_ngram = np.array([wA_opt_5_ng, wB_opt_5_ng, wC_opt_5_ng, wD_opt_5_ng, wE_opt_5_ng])
np.save(os.path.join(OUTPUT_DIR, 'ensemble_weights_5model_ngram.npy'), final_weights_5_ngram)
print(f"✅ 最终(N-gram 5模型)权重已保存到 'ensemble_weights_5model_ngram.npy'")

