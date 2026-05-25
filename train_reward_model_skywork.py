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
import argparse 
from sklearn.metrics import mean_squared_error  
from scipy.stats import pearsonr               




MODEL_NAME = "/root/autodl-tmp/pretrain_models/skywork"
DATA_PATH = "/root/autodl-tmp/dialogue-KT0/RLMT/reward_model_train/data/evaluation_results_multidim_200.csv"
BASE_OUTPUT_DIR = "/root/autodl-tmp/dialogue-KT0/RLMT/saves/skywork-8b-reward-model-rag-200" 

MAX_LENGTH = 4096
BATCH_SIZE = 8          
GRAD_ACCUMULATION = 2   
LEARNING_RATE = 1e-5    
NUM_EPOCHS = 3          

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_col", type=str, required=True, help="")
    args = parser.parse_args()
    
    TARGET_COL = args.target_col
    OUTPUT_DIR = f"{BASE_OUTPUT_DIR}-{TARGET_COL}" 
    
    print(f"=== Starting Training for Target: {TARGET_COL} ===")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Model Path: {MODEL_NAME}")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH,encoding='utf-8')
    
    
    
    if TARGET_COL not in df.columns:
        print(f"Error: Target column '{TARGET_COL}' not found in CSV!")
        return
    
    print(f"Using label column: {TARGET_COL}")
    
    df = df.dropna(subset=[TARGET_COL, 'model_output'])
    
    print(f"Normalizing labels from 0-5 scale to 0-1 scale...")
    df['label'] = pd.to_numeric(df[TARGET_COL]) / 5.0
    
    score_accuracy = df['score_accuracy'].values  # Rc
    score_logic = df['score_logic'].values        # Rf
    score_persona = df['score_persona'].values    # Rp

    norm_rc = normalize(score_accuracy)
    norm_rf = normalize(score_logic)
    norm_rp = normalize(score_persona)

    lambda_weight = 0.5

    final_weighted_score = norm_rc * sigmoid(norm_rf) * (1 + lambda_weight * norm_rp)

    df['label'] = final_weighted_score
    
    print(f"Dataset Size: {len(df)}")

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

    print(f"Loading Skywork Reward Model (4-bit)...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1, 
        # quantization_config=bnb_config, 
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True 
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id

    print("Configuring LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"], 
        modules_to_save=["score"] 
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.001,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        logging_steps=5,
        fp16=False,
        bf16=True,
        remove_unused_columns=False,
        report_to="wandb", 
        
        run_name=f"skywork_rm_{}",
        
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
        data_collator=data_collator,
    )

    print(f">>> Starting Training for {TARGET_COL}...")
    trainer.train()
    
    print(f"Saving final model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR) 
    print(f"Skywork Reward Model Fine-tuning for {TARGET_COL} Completed!")

if __name__ == "__main__":
    main()
