# Data Processing and Scientific Computing

import pandas as pd
import numpy as np

# Feature Engineering

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

# Models and Evaluation

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# Deep Learning Embedding Models

from sentence_transformers import SentenceTransformer

print("🚀 Team 8 Baseline & Embedding Model Pipeline Started!")

# --- Load Dataset ---

# Use try-except structure to ensure robust data loading

try:
    train = pd.read_csv("/kaggle/input/llm-classification-finetuning/train.csv")
    test = pd.read_csv("/kaggle/input/llm-classification-finetuning/test.csv")
    sample = pd.read_csv("/kaggle/input/llm-classification-finetuning/sample_submission.csv")
    print("✅ Dataset loaded successfully!")
    print(f"  Training set size: {train.shape}")
    print(f"  Test set size: {test.shape}")
except FileNotFoundError:
    print("❌ Data loading failed! Please check if the competition dataset has been correctly added to the Notebook.")

# === Step 2: Feature Engineering (Consistent processing for train and test sets) ===

print("\n--- Executing Step 1: Feature Engineering ---")

# --- 1. Process training set (train) ---
# Convert one-hot encoded labels to single multi-class labels (0, 1, 2)
train['label'] = train[['winner_model_a', 'winner_model_b', 'winner_tie']].values.argmax(axis=1)

# Create complete text containing prompt and response
train['text_a'] = train['prompt'] + " " + train['response_a']
train['text_b'] = train['prompt'] + " " + train['response_b']

# Concatenate two responses with [SEP] separator for model comparison
train['text'] = train['text_a'] + " [SEP] " + train['text_b']

# Calculate length features
train['prompt_len'] = train['prompt'].str.len()
train['resp_a_len'] = train['response_a'].str.len()
train['resp_b_len'] = train['response_b'].str.len()


# --- 2. Process test set (test) with identical method ---
# Note: Test set doesn't have 'label' column, so no need to process
test['text_a'] = test['prompt'] + " " + test['response_a']
test['text_b'] = test['prompt'] + " " + test['response_b']
test['text'] = test['text_a'] + " [SEP] " + test['text_b']
test['prompt_len'] = test['prompt'].str.len()
test['resp_a_len'] = test['response_a'].str.len()
test['resp_b_len'] = test['response_b'].str.len()

print("✅ Feature engineering completed.")


# === Step 3: Baseline Model (Bag of Words + Logistic Regression) ===

print("\n--- Executing Baseline Model ---")

# --- 1. Feature Vectorization ---
# Text features (Bag of Words), considering unigrams and bigrams
vectorizer = CountVectorizer(max_features=5000, ngram_range=(1,2))
X_text_train = vectorizer.fit_transform(train['text'])
X_text_test = vectorizer.transform(test['text'])

# Numerical features (lengths), with standardization
scaler = StandardScaler()
num_features_train = train[['prompt_len','resp_a_len','resp_b_len']]
num_features_test = test[['prompt_len','resp_a_len','resp_b_len']]
X_num_train = scaler.fit_transform(num_features_train)
X_num_test = scaler.transform(num_features_test)

# Combine text and numerical features
X_baseline = np.hstack([X_text_train.toarray(), X_num_train])
X_test_baseline = np.hstack([X_text_test.toarray(), X_num_test])
y = train['label']

# --- 2. Training and Validation ---
# Split baseline features into training and validation sets
X_train_base, X_val_base, y_train_base, y_val_base = train_test_split(
    X_baseline, y, test_size=0.2, random_state=42
)

# Train logistic regression model, increased max_iter to avoid convergence warning
clf_base = LogisticRegression(max_iter=1000)
clf_base.fit(X_train_base, y_train_base)

# Evaluate model performance on validation set
y_pred_val_base = clf_base.predict_proba(X_val_base)
validation_score_base = log_loss(y_val_base, y_pred_val_base)
print(f"📊 Validation LogLoss (Baseline): {validation_score_base:.5f}")

# Note: We won't generate submission.csv from this model, the final submission will be from stronger models.


# === Step 4: Embedding Model (MiniLM + Logistic Regression) ===

print("\n--- Executing Embedding Model ---")

# --- 1. Load pre-trained SentenceTransformer model (offline mode) ---
# Make sure you've added 'sentence-transformers-all-minilm-l6-v2' dataset in Notebook's Input
model_path = '/kaggle/input/sentencetransformersallminilml6v2'

model = None
try:
    model = SentenceTransformer(model_path, device='cuda') # Use GPU acceleration
    print("✅ Embedding model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load embedding model: {e}")
    print("  Please ensure 'sentencetransformersallminilml6v2' dataset is added in the Input panel and path is correct.")


# Only proceed if model loaded successfully
if model is not None:
    # --- 2. Generate sentence embeddings ---
    # Create a new combined field for embedding for both train and test data
    train['combined_for_embedding'] = train['prompt'] + " " + train['response_a'] + " [SEP] " + train['response_b']
    test['combined_for_embedding'] = test['prompt'] + " " + test['response_a'] + " [SEP] " + test['response_b']
    
    print("⏳ Generating sentence embeddings for training set (this may take a few minutes)...")
    train_emb = model.encode(train['combined_for_embedding'].tolist(), show_progress_bar=True, batch_size=128)
    
    print("⏳ Generating sentence embeddings for test set...")
    test_emb = model.encode(test['combined_for_embedding'].tolist(), show_progress_bar=True, batch_size=128)
    print("✅ Sentence embeddings generation completed.")

    # --- 3. Training and Validation ---
    # Split embedding features into training and validation sets
    X_train_emb, X_val_emb, y_train_emb, y_val_emb = train_test_split(
        train_emb, y, test_size=0.2, random_state=42
    )

    # Train logistic regression classifier
    print("⏳ Training classifier for Embedding model...")
    clf_emb = LogisticRegression(max_iter=1000)
    clf_emb.fit(X_train_emb, y_train_emb)
    print("✅ Classifier training completed.")

    # Evaluate model performance on validation set
    y_pred_val_emb = clf_emb.predict_proba(X_val_emb)
    validation_score_emb = log_loss(y_val_emb, y_pred_val_emb)
    print(f"📊 Validation LogLoss (Embedding): {validation_score_emb:.5f}")

    # --- 4. Generate final Kaggle submission file ---
    # Use model trained on partial data for predictions
    # (Better practice would be retraining on full data, but using clf_emb directly for speed and simplicity)
    print("⏳ Generating final predictions for test set...")
    preds_final = clf_emb.predict_proba(test_emb)

    # Create submission DataFrame, ensure filename is "submission.csv"
    submission_final = pd.DataFrame(preds_final, columns=sample.columns[1:])
    submission_final.insert(0, "id", sample["id"])
    submission_final.to_csv("submission.csv", index=False)

    print("\n🎉 Final submission.csv has been generated! Ready to save and submit.")

# === Step 5: Model Extensions (E5 Embedding + LightGBM + Ensemble) ===
print("\n--- Executing Step 3: Model Extensions ---")

from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from sentence_transformers import SentenceTransformer

# --- 1. Load alternative embedding model (E5) ---
try:
    e5_path = "/kaggle/input/e5-small-v2"  # Ensure this is added to Input
    e5_model = SentenceTransformer(e5_path, device='cuda')
    print("✅ E5 model loaded successfully!")
except Exception as e:
    print("❌ Failed to load E5 model:", e)
    e5_model = None

if e5_model is not None:
    print("⏳ Generating E5 sentence embeddings...")
    train_emb_e5 = e5_model.encode(train["combined_for_embedding"].tolist(), batch_size=128, show_progress_bar=True)
    test_emb_e5 = e5_model.encode(test["combined_for_embedding"].tolist(), batch_size=128, show_progress_bar=True)
    print("✅ E5 embeddings generation completed.")

    # --- 2. LightGBM Classifier ---
    print("⏳ Training LightGBM model...")
    lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=64, random_state=42)
    X_train_lgb, X_val_lgb, y_train_lgb, y_val_lgb = train_test_split(train_emb_e5, y, test_size=0.2, random_state=42)
    lgbm.fit(X_train_lgb, y_train_lgb)
    val_pred_lgb = lgbm.predict_proba(X_val_lgb)
    val_logloss_lgb = log_loss(y_val_lgb, val_pred_lgb)
    print(f"📊 Validation LogLoss (E5 + LightGBM): {val_logloss_lgb:.5f}")

    # --- 3. Ensemble with Logistic Regression (MiniLM) ---
    print("⏳ Performing ensemble fusion...")
    ensemble = VotingClassifier(
        estimators=[
            ('minilm', clf_emb),
            ('lgbm', lgbm)
        ],
        voting='soft'
    )
    X_train_ens, X_val_ens, y_train_ens, y_val_ens = train_test_split(
        np.hstack([train_emb, train_emb_e5]), y, test_size=0.2, random_state=42
    )
    ensemble.fit(X_train_ens, y_train_ens)
    y_pred_val_ens = ensemble.predict_proba(X_val_ens)
    val_logloss_ens = log_loss(y_val_ens, y_pred_val_ens)
    print(f"🎯 Validation LogLoss (MiniLM+E5 Ensemble): {val_logloss_ens:.5f}")

    # --- 4. Final Prediction and File Output ---
    preds_final_ens = ensemble.predict_proba(np.hstack([test_emb, test_emb_e5]))
    submission_final_ens = pd.DataFrame(preds_final_ens, columns=sample.columns[1:])
    submission_final_ens.insert(0, "id", sample["id"])
    submission_final_ens.to_csv("submission.csv", index=False)
    print("✅ Ensemble model results saved as submission.csv")


# === Test Set Bias Feature Calculation ===
print("\n--- Executing: Test Set Bias Feature Calculation ---")

for df in [train, test]:
    # Calculate response lengths
    df["resp_a_len"] = df["response_a"].str.len()
    df["resp_b_len"] = df["response_b"].str.len()
    
    # Length difference features
    df["len_diff"] = df["resp_a_len"] - df["resp_b_len"]
    df["len_ratio"] = df["resp_a_len"] / (df["resp_b_len"] + 1e-6)  # Add small epsilon to avoid division by zero
    
    # Lexical diversity features (unique words / total words)
    df["lexical_a"] = df["response_a"].apply(lambda x: len(set(str(x).split())) / (len(str(x).split()) + 1e-6))
    df["lexical_b"] = df["response_b"].apply(lambda x: len(set(str(x).split())) / (len(str(x).split()) + 1e-6))
    df["lexical_diff"] = df["lexical_a"] - df["lexical_b"]

# Display sample features
print(train[["len_diff", "len_ratio", "lexical_diff"]].head())
print("✅ Test set bias feature calculation completed.")

from sklearn.decomposition import PCA
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
import numpy as np

print("\n--- Quick Bias Modeling Experiment ---")

# Combine bias features (MiniLM embedding + 3 bias features)
bias_feats_train = train[["len_diff", "len_ratio", "lexical_diff"]].fillna(0).values
bias_feats_test = test[["len_diff", "len_ratio", "lexical_diff"]].fillna(0).values

# 1️⃣ PCA Dimensionality Reduction
pca = PCA(n_components=128, random_state=42)
train_pca = pca.fit_transform(train_emb)
test_pca = pca.transform(test_emb)

# 2️⃣ Concatenate Bias Features
X_train_bias = np.hstack([train_pca, bias_feats_train])
X_test_bias = np.hstack([test_pca, bias_feats_test])

# 3️⃣ Train Fast LightGBM
lgb_bias = LGBMClassifier(
    n_estimators=200, learning_rate=0.05, num_leaves=64, random_state=42
)
X_train_b, X_val_b, y_train_b, y_val_b = train_test_split(
    X_train_bias, y, test_size=0.2, random_state=42
)
lgb_bias.fit(X_train_b, y_train_b)
val_pred_bias = lgb_bias.predict_proba(X_val_b)
val_logloss_bias = log_loss(y_val_b, val_pred_bias)
print(f"🎯 Validation LogLoss (Bias-aware LGBM + PCA): {val_logloss_bias:.5f}")


# === In-depth Position Bias Analysis (Revised Version) ===
print("\n--- Executing: In-depth Position Bias Analysis ---")

# Randomly sample from training set and reset index
subset = train.sample(1000, random_state=42).reset_index(drop=True)
subset_swapped = subset.copy()

# Swap response_a and response_b
subset_swapped["response_a"], subset_swapped["response_b"] = (
    subset["response_b"], subset["response_a"]
)

# Generate input texts (Prompt + A + B concatenation)
subset_texts = (subset["prompt"] + " " + subset["response_a"] + " " + subset["response_b"]).tolist()
subset_texts_swapped = (subset_swapped["prompt"] + " " + subset_swapped["response_a"] + " " + subset_swapped["response_b"]).tolist()

# Generate embeddings and predictions
subset_emb = model.encode(subset_texts, show_progress_bar=False)
subset_emb_swapped = model.encode(subset_texts_swapped, show_progress_bar=False)

pred_orig = clf_emb.predict_proba(subset_emb)
pred_swap = clf_emb.predict_proba(subset_emb_swapped)

# Calculate "prediction flip rate" (should be high if model truly focuses on content)
flip_rate = np.mean(np.argmax(pred_orig, axis=1) != np.argmax(pred_swap, axis=1))
print(f"🔄 Model prediction flip rate (after A/B swap): {flip_rate:.3f}")

# Compare average prediction probabilities
avg_conf_diff = np.mean(np.abs(pred_orig - pred_swap))
print(f"📊 Average probability change magnitude: {avg_conf_diff:.4f}")


# === Step 5.3: LoRA Fine-tuning (DeBERTa-small, Memory-Optimized) ===
print("\n--- Executing Step 5.3: LoRA Fine-tuning (Memory Optimized) ---")

import os
os.environ["WANDB_MODE"] = "disabled"  # Disable W&B

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import torch
import pandas as pd
import shutil
from sklearn.metrics import accuracy_score

# 1️⃣ Kaggle Input Model Path
input_model_path = "/kaggle/input/deberta-v3-small/deberta-v3-small"
local_model_path = "./deberta-small-local"
if not os.path.exists(local_model_path):
    shutil.copytree(input_model_path, local_model_path)

# 2️⃣ Load Local Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
base_model = AutoModelForSequenceClassification.from_pretrained(
    local_model_path, num_labels=3, local_files_only=True
)

# 3️⃣ Configure LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(base_model, peft_config)

# 4️⃣ Data Processing: Concatenate prompt and options
def preprocess_function(examples):
    texts = [
        f"Question: {p} [SEP] A: {a} [SEP] B: {b}" 
        for p, a, b in zip(examples["prompt"], examples["response_a"], examples["response_b"])
    ]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=256)

train_texts_ft = train[:-2000]
y_train_ft = y[:-2000]
val_texts = train[-2000:]
y_val_ft = y[-2000:]

train_dataset = Dataset.from_dict({
    "prompt": train_texts_ft["prompt"],
    "response_a": train_texts_ft["response_a"],
    "response_b": train_texts_ft["response_b"],
    "label": y_train_ft
})
val_dataset = Dataset.from_dict({
    "prompt": val_texts["prompt"],
    "response_a": val_texts["response_a"],
    "response_b": val_texts["response_b"],
    "label": y_val_ft
})

# Parallel tokenization for speed
tokenized_train = train_dataset.map(preprocess_function, batched=True, num_proc=2)
tokenized_val = val_dataset.map(preprocess_function, batched=True, num_proc=2)

# 5️⃣ Evaluation Function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

# 6️⃣ Training Configuration (Memory Optimized)
training_args = TrainingArguments(
    output_dir="./ft_results",
    per_device_train_batch_size=8,          # Reduced batch size
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,         # 2-step accumulation, equivalent to batch_size=16
    num_train_epochs=3,
    learning_rate=3e-4,
    logging_steps=50,
    fp16=True,                              # Mixed precision training
    report_to=[]                            # Disable W&B
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 7️⃣ Start Fine-tuning
print("⏳ Starting LoRA fine-tuning...")
trainer.train()
print("✅ Fine-tuning completed.")

# 8️⃣ Test Set Prediction
test_texts = [
    f"Question: {p} [SEP] A: {a} [SEP] B: {b}" 
    for p, a, b in zip(test["prompt"], test["response_a"], test["response_b"])
]
test_dataset = Dataset.from_dict({"text": test_texts})
tokenized_test = test_dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=256), batched=True, num_proc=2)

pred_logits = trainer.predict(tokenized_test).predictions
pred_probs = torch.softmax(torch.tensor(pred_logits), dim=-1).numpy()

submission_ft = pd.DataFrame(pred_probs, columns=sample.columns[1:])
submission_ft.insert(0, "id", sample["id"])
submission_ft.to_csv("submission_finetuned.csv", index=False)
print("✅ Fine-tuned model results saved as submission_finetuned.csv")


# === Calibration (Temperature Scaling) ===
print("\n--- Executing: Probability Calibration (Temperature Scaling) ---")

from sklearn.metrics import log_loss
from scipy.optimize import minimize
import numpy as np

# Use validation set logits
val_logits = trainer.predict(tokenized_val).predictions
val_probs = torch.softmax(torch.tensor(val_logits), dim=-1).numpy()

def temperature_scale(logits, T):
    logits_T = logits / T
    exp_T = np.exp(logits_T - np.max(logits_T, axis=1, keepdims=True))
    return exp_T / np.sum(exp_T, axis=1, keepdims=True)

def loss_fn(T):
    probs_T = temperature_scale(val_logits, T)
    return log_loss(y_val_ft, probs_T)

# Optimize temperature parameter T
res = minimize(loss_fn, x0=[1.0], bounds=[(0.5, 5.0)], method="L-BFGS-B")
T_opt = res.x[0]
print(f"📏 Optimal temperature parameter T = {T_opt:.3f}")

# Apply to test set predictions
calibrated_probs = temperature_scale(pred_logits, T_opt)

submission_calibrated = pd.DataFrame(calibrated_probs, columns=sample.columns[1:])
submission_calibrated.insert(0, "id", sample["id"])
submission_calibrated.to_csv("submission_calibrated.csv", index=False)
submission_calibrated.to_csv("submission.csv", index=False)
print("✅ Calibrated results saved as submission_calibrated.csv")




import pandas as pd
import numpy as np
import os
import torch
import gc
import joblib 
import lightgbm as lgb
from scipy.sparse import hstack
import warnings

# Feature Engineering
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Evaluation & Validation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Disable unnecessary warnings
warnings.filterwarnings('ignore')
os.environ["WANDB_MODE"] = "disabled"

# Memory cleanup function
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("🚀 (Clean 5-Fold CV, Model A Only) Pipeline Started!")

# --- 1. Competition Data Paths ---
COMP_DIR = "/kaggle/input/llm-classification-finetuning"
TRAIN_FILE = os.path.join(COMP_DIR, "train.csv")
TEST_FILE = os.path.join(COMP_DIR, "test.csv")
SAMPLE_FILE = os.path.join(COMP_DIR, "sample_submission.csv")

# --- 2. Public Model Paths ---
print("⏳ Defining public model paths...")
BASE_MINILM_PATH = "/kaggle/input/sentencetransformersallminilml6v2"
print(f"  ...MiniLM Path: {BASE_MINILM_PATH}")

print("✅ All paths defined.")
print(f"⏳ Loading train.csv and test.csv...")

# === Cell 2: Feature Enginer ===
try:
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    sample_df = pd.read_csv(SAMPLE_FILE) # sample_df will be used in final cell
    print(f"  Train: {train_df.shape}, Test: {test_df.shape}")
except FileNotFoundError as e:
    print(f"❌ Data loading failed! {e}")
    raise

# --- 2.1 Base Features (len, lexical) ---
print("⏳ Creating base features (len_diff, lexical_diff)...")
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
y_true_full = train_df['label'] # Prepare all labels (57k)

# --- 2.2 SBERT Embeddings (MiniLM) ---
print("⏳ Generating MiniLM embeddings (for A and similarity)...")
model_minilm = SentenceTransformer(BASE_MINILM_PATH, device='cuda') 
train_emb_minilm = model_minilm.encode(train_df['combined_for_embedding'].tolist(), show_progress_bar=True, batch_size=128)
test_emb_minilm = model_minilm.encode(test_df['combined_for_embedding'].tolist(), show_progress_bar=True, batch_size=128)

# --- 2.3 Similarity Features (from MiniLM) ---
print("⏳ Creating similarity features...")
resp_a_emb_train = model_minilm.encode(train_df['response_a'].tolist(), show_progress_bar=True, batch_size=128)
resp_b_emb_train = model_minilm.encode(train_df['response_b'].tolist(), show_progress_bar=True, batch_size=128)
train_df['cosine_similarity'] = np.array([cosine_similarity(resp_a_emb_train[i].reshape(1, -1), resp_b_emb_train[i].reshape(1, -1))[0][0] for i in range(len(resp_a_emb_train))])

resp_a_emb_test = model_minilm.encode(test_df['response_a'].tolist(), show_progress_bar=True, batch_size=128)
resp_b_emb_test = model_minilm.encode(test_df['response_b'].tolist(), show_progress_bar=True, batch_size=128)
test_df['cosine_similarity'] = np.array([cosine_similarity(resp_a_emb_test[i].reshape(1, -1), resp_b_emb_test[i].reshape(1, -1))[0][0] for i in range(len(resp_a_emb_test))])

del model_minilm, resp_a_emb_train, resp_b_emb_train, resp_a_emb_test, resp_b_emb_test
clear_memory()

# --- 2.4 N-gram Features (On-the-fly Training) ---
print("⏳ Training N-gram Vectorizer and creating features...")
corpus = pd.concat([train_df['response_a'], train_df['response_b']]).astype(str).unique()

vectorizer = CountVectorizer(
    max_features=2000,
    ngram_range=(1, 2), # Include 1-grams and 2-grams
    stop_words='english',
    dtype=np.float32 
)
print("  ...Fitting Vectorizer...")
vectorizer.fit(corpus)
del corpus
clear_memory()

print("  ...Transforming train/test sets...")
# [[[ Logic Fix ]]]: Correctly create difference features
train_ngram_a = vectorizer.transform(train_df['response_a'].astype(str))
train_ngram_b = vectorizer.transform(train_df['response_b'].astype(str))
train_ngram_diff = (train_ngram_a - train_ngram_b)

test_ngram_a = vectorizer.transform(test_df['response_a'].astype(str))
test_ngram_b = vectorizer.transform(test_df['response_b'].astype(str))
test_ngram_diff = (test_ngram_a - test_ngram_b)

del vectorizer, train_ngram_a, train_ngram_b, test_ngram_a, test_ngram_b
clear_memory()

# --- 2.5 [[[ NameError Fix ]]] ---
# Extract 4 bias features before deleting train_df/test_df
print("⏳ Extracting 4 bias features...")
all_4_features_train = train_df[["len_diff", "len_ratio", "lexical_diff", "cosine_similarity"]].fillna(0).values
all_4_features_test = test_df[["len_diff", "len_ratio", "lexical_diff", "cosine_similarity"]].fillna(0).values

# --- 2.6 Now safe to delete ---
del train_df, test_df
clear_memory()

print("✅ All feature engineering completed.")



# === Cell 3: Prepare Final Feature Matrices ===
print("\n--- Preparing final train/test feature matrices ---")
# (All variables train_emb_minilm, all_4_features_train, train_ngram_diff etc. already in memory)

# --- Stack Package A (MiniLM + 4 Feat + Ngram) ---
print(f"  ...Stacking Package A (MiniLM + 4 Feat + Ngram, total {384 + 4 + 2000} features)")
X_A_full = hstack([train_emb_minilm, all_4_features_train, train_ngram_diff]).tocsr()
X_test_A_ngram = hstack([test_emb_minilm, all_4_features_test, test_ngram_diff]).tocsr()

print(f"✅ Feature matrices ready. Train: {X_A_full.shape}, Test: {X_test_A_ngram.shape}")

# --- Free Memory ---
del train_emb_minilm, test_emb_minilm, all_4_features_train, all_4_features_test, train_ngram_diff, test_ngram_diff
clear_memory()

# === Cell 4: Define 5-Fold Cross Validation ===
print("\n--- Defining 5-Fold CV ---")

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# (y_true_full from Cell 2)

# Define LGBM Parameters
lgbm_params = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'num_leaves': 64,
    'random_state': 42,
    'device': 'gpu',
    'n_jobs': -1,
    'verbose': -1
}

# --- Initialize Storage ---
oof_preds_A = np.zeros((len(y_true_full), 3))
test_preds_A_list = []

print(f"✅ K-Fold (n_splits={N_SPLITS}) ready.")


