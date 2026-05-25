import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig
)
import numpy as np


from peft import LoraConfig, get_peft_model, TaskType
import os
import sys
import argparse  # --- 新增：用于接收命令行参数 ---
from sklearn.metrics import mean_squared_error  # --- 新增：否则 compute_metrics 会报错 ---
from scipy.stats import pearsonr                # --- 新增：否则 compute_metrics 会报错 ---

# 请确保这里填入的是正确的 40 位 Key
os.environ["WANDB_API_KEY"] = "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"

# ================= 配置区域 =================

MODEL_NAME = "/root/autodl-tmp/pretrain_models/skywork"
DATA_PATH = "/root/autodl-tmp/dialogue-KT0/RLMT/reward_model_train/data/evaluation_results_multidim_200.csv"
BASE_OUTPUT_DIR = "/root/autodl-tmp/dialogue-KT0/RLMT/saves/skywork-8b-reward-model-rag-200" # 基础路径

# 3. 训练超参数
MAX_LENGTH = 4096
BATCH_SIZE = 8          
GRAD_ACCUMULATION = 2   
LEARNING_RATE = 1e-5    
NUM_EPOCHS = 3          

# ================= Prompt 定义 =================
SYSTEM_PROMPT = """You are a professional and rigorous Knowledge Tracing expert. Your task is to analyze a dialogue between a student and a teacher.
We concealed the final round of the dialogue.
Please carefully read the similar cases, student profile, the question, and the dialogue history.
Based on all the information, analyze and predict: At the end of the dialogue, did the student finally master the problem and achieve self-correction?

Your response must include two parts:
1.  **Chain-of-Thought**: Provide a detailed analysis. Compare the current student's behavior with the provided [Similar Cases] to see if they fit a typical pattern.
2.  **Final Prediction**: After the Chain-of-Thought analysis, please start a new line and provide your final conclusion clearly using `[Prediction]: True` or `[Prediction]: False`."""

USER_PROMPT_TEMPLATE = """
[Similar Cases / Context]
Here are some examples of past interactions involving students with the EXACT SAME profile as the current student. Use these to understand the student's typical behavior pattern.
{rag_context}

[Question Content]
{question}

[Student Profile]
{student_profile}

[Student's Initial Incorrect Solution]
{student_incorrect_solution}

[Teacher-Student Dialogue (Masked)]
{conversation}
"""

# ================= 数据处理函数 =================
# 在文件开头添加 sigmoid 函数
def sigmoid(x):
      return 1 / (1 + np.exp(-x))
  
def normalize(scores):
      """Min-max normalization to [0, 1]"""
      min_val = np.min(scores)
      max_val = np.max(scores)
      if max_val - min_val == 0:
          return np.zeros_like(scores)
      return (scores - min_val) / (max_val - min_val)
  
  
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    predictions = predictions.squeeze()
    labels = labels.squeeze()
    
    mse = mean_squared_error(labels, predictions)
    
    try:
        pearson_corr, _ = pearsonr(predictions, labels)
    except:
        pearson_corr = 0.0
        
    return {
        "mse": mse,
        "pearson": pearson_corr
    }

def format_example(row, tokenizer):
    rag_context = str(row.get('rag_context', 'No context provided.'))
    profile = str(row.get('student_profile', 'N/A'))
    question = str(row.get('question', 'N/A'))
    solution = str(row.get('student_incorrect_solution', 'N/A'))
    conversation = str(row.get('conversation', 'N/A'))

    user_content = USER_PROMPT_TEMPLATE.format(
        rag_context=rag_context,
        question=question,
        student_profile=profile,
        student_incorrect_solution=solution,
        conversation=conversation 
    )

    assistant_content = str(row.get('model_output', ''))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]
    
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    return full_text

# ================= 主流程 =================

def main():
    # --- 0. 解析命令行参数 ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_col", type=str, required=True, help="要训练的目标列名")
    args = parser.parse_args()
    
    TARGET_COL = args.target_col
    OUTPUT_DIR = f"{BASE_OUTPUT_DIR}-{TARGET_COL}" # 动态生成输出路径
    
    print(f"=== Starting Training for Target: {TARGET_COL} ===")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Model Path: {MODEL_NAME}")
    
    # --- 1. 加载数据 ---
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH,encoding='utf-8')
    
    
    '''读取原来的奖励列，并进行归一化处理，生成新的 'label' 列
    
    if TARGET_COL not in df.columns:
        print(f"Error: Target column '{TARGET_COL}' not found in CSV!")
        return
    
    print(f"Using label column: {TARGET_COL}")
    
    df = df.dropna(subset=[TARGET_COL, 'model_output'])
    
    print(f"Normalizing labels from 0-5 scale to 0-1 scale...")
    df['label'] = pd.to_numeric(df[TARGET_COL]) / 5.0
    
    '''
    #新奖励计算方法
    # 读取三个维度的分数
    score_accuracy = df['score_accuracy'].values  # Rc
    score_logic = df['score_logic'].values        # Rf
    score_persona = df['score_persona'].values    # Rp

  # 归一化
    norm_rc = normalize(score_accuracy)
    norm_rf = normalize(score_logic)
    norm_rp = normalize(score_persona)

    # 设置 lambda 权重（可调整）
    lambda_weight = 0.5

    # 计算新的总奖励：Rtotal = Norm(Rc) · σ(Norm(Rf)) · (1 + λ·Norm(Rp))
    final_weighted_score = norm_rc * sigmoid(norm_rf) * (1 + lambda_weight * norm_rp)

  # 将计算结果添加回 DataFrame
    df['label'] = final_weighted_score
    
    
    
    
    
    print(f"Dataset Size: {len(df)}")

    # --- 2. 加载 Tokenizer ---
    print("Loading Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    except Exception as e:
        print(f"Error loading local tokenizer: {e}")
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # --- 3. 数据集转换 ---
    raw_dataset = Dataset.from_pandas(df)

    def preprocess_function(examples):
        texts = []
        labels = []
        for i in range(len(examples['label'])):
            row = {k: examples[k][i] for k in examples}
            text = format_example(row, tokenizer)
            texts.append(text)
            labels.append(float(row['label']))
        
        tokenized = tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=MAX_LENGTH
        )
        tokenized["labels"] = labels
        return tokenized

    print("Tokenizing dataset...")
    tokenized_dataset = raw_dataset.map(
        preprocess_function, 
        batched=True, 
        remove_columns=raw_dataset.column_names,
        num_proc=4 
    )
    
    dataset_split = tokenized_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    # --- 4. 加载 Skywork 模型 ---
    print(f"Loading Skywork Reward Model (4-bit)...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1, 
        # quantization_config=bnb_config, # 如果显存够大(80G)，建议注释掉这一行用全量BF16，精度更好
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True 
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id

    # --- 5. LoRA 配置 ---
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        modules_to_save=["score"] 
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --- 6. 训练参数 ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.001,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        logging_steps=5, # 稍微调大一点点
        fp16=False,
        bf16=True,
        remove_unused_columns=False,
        report_to="wandb", 
        
        # --- 动态设置 WandB 的 Run Name ---
        run_name=f"skywork_rm_{TARGET_COL}",
        
        label_names=["labels"],
        gradient_checkpointing=True, 
        optim="paged_adamw_32bit",   
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(f">>> Starting Training for {TARGET_COL}...")
    trainer.train()
    
    print(f"Saving final model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR) 
    print(f"Skywork Reward Model Fine-tuning for {TARGET_COL} Completed!")

if __name__ == "__main__":
    main()