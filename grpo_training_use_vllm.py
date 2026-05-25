import os
import sys
import gc
import torch
import pandas as pd
import argparse
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

# 🟢 [A100] 显存碎片整理依然保持开启，是个好习惯
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- 导入 RAG 模块 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from rag_util import RAGRetriever 

# 🟢 [兼容性补丁] 防止 ImportError
try:
    from trl.trainer.grpo_trainer import GRPOTrainer
    _original_get_train_sampler = GRPOTrainer._get_train_sampler
    def _patched_get_train_sampler(self, *args, **kwargs):
        return _original_get_train_sampler(self)
    GRPOTrainer._get_train_sampler = _patched_get_train_sampler
except ImportError:
    pass

# --- 固定配置 ---
POLICY_MODEL_PATH = "/root/autodl-tmp/pretrain_models/llama3"
DATA_PATH = "/root/autodl-tmp/dialogue-KT0/RLMT/data_generate/data/annotated/train.csv" 

# GRPO 参数 (A100 高性能版)
LEARNING_RATE = 2e-5          
NUM_GENERATIONS = 8           
# 🟢 [A100优化] 显存够大，可以开回 8192 或 6144
MAX_PROMPT_LENGTH = 8192      
MAX_COMPLETION_LENGTH = 512   
EPOCHS = 2
PROMPTS_PER_BATCH = 1 
GRADIENT_ACCUMULATION_STEPS = 4 

# --- Prompt 模板 ---
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

def mask_conversation(conversation):
    if not isinstance(conversation, str): return ""
    segments = [s for s in conversation.split('|EOM|') if s.strip()]
    if not segments: return ""
    last_turn = segments[-1].strip()
    if last_turn.lower().startswith("teacher"):
        return '|EOM|'.join(segments[:-2]) if len(segments) >= 2 else ""
    else:
        return '|EOM|'.join(segments[:-1]) if len(segments) >= 1 else ""

# --- Reward 函数 ---
class RewardModelScorer:
    def __init__(self, reward_model_path, base_model_path):
        self.__name__ = "reward_model_scorer"
        print(f"Loading Reward Model from {reward_model_path} [GPU Resident]...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 🟢 [A100优化] 使用 bfloat16 计算，精度更高
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16, # A100 原生支持 BF16
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path,
            num_labels=1,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16, # A100
            device_map={"": 0} # 依然把 RM 放在卡 2，物理隔离最稳
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        try:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, reward_model_path)
            print("✅ Loaded LoRA adapter for Reward Model.")
        except Exception as e:
            print(f"⚠️ Warning: Could not load LoRA for RM, using base: {e}")

        self.model.eval()

    def __call__(self, prompts, completions, **kwargs):
        rewards = []
        inputs = [p + c for p, c in zip(prompts, completions)]
        batch_size = 8 # A100 显存大，可以增大 batch
        
        with torch.no_grad():
            for i in range(0, len(inputs), batch_size):
                batch_inputs = inputs[i : i + batch_size]
                tokenized = self.tokenizer(
                    batch_inputs, 
                    padding=True, 
                    truncation=True, 
                    max_length=MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH, 
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**tokenized)
                batch_rewards = outputs.logits.squeeze(-1).float().cpu().tolist()
                rewards.extend(batch_rewards)
        
        return rewards

def build_dataset(data_path, tokenizer):
    print(f"Loading data from {data_path}...")
    try: df = pd.read_csv(data_path, encoding='gbk')
    except: df = pd.read_csv(data_path)
    
    retriever = RAGRetriever(data_path)
    df = df.dropna(subset=['question', 'student_incorrect_solution', 'conversation'])
    
    input_prompts = []
    print("Constructing GRPO Prompts...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        profile = str(row.get('student_profile', ''))
        rag_context = retriever.retrieve_examples(profile, k=3, max_chars=12000)
        masked_conv = mask_conversation(str(row['conversation']))
        
        user_content = USER_PROMPT_TEMPLATE.format(
            rag_context=rag_context,
            question=str(row['question']),
            student_profile=profile,
            student_incorrect_solution=str(row['student_incorrect_solution']),
            conversation=masked_conv
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_prompts.append(formatted_prompt)

    dataset = Dataset.from_dict({"prompt": input_prompts})
    return dataset

def main():
    # --- 0. 解析命令行参数 ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_model_path", type=str, required=True, help="Path to the reward model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model")
    args = parser.parse_args()

    REWARD_MODEL_PATH = args.reward_model_path
    OUTPUT_DIR = args.output_dir
    run_identifier = OUTPUT_DIR.split('-')[-1]

    # --- 1. 设置 WandB ---
    os.environ["WANDB_API_KEY"] = "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"
    WANDB_PROJECT = "RLMT_GRPO_Experiments_200"
    
    print(f"==================================================")
    print(f"🚀 Starting GRPO Training Task (A100 Mode)")
    print(f"   Reward Model: {REWARD_MODEL_PATH}")
    print(f"   Output Dir:   {OUTPUT_DIR}")
    print(f"   WandB Run:    grpo_{run_identifier}")
    print(f"==================================================")

    # --- 4. 配置 GRPO (A100 BF16) ---
    GLOBAL_BATCH_SIZE = PROMPTS_PER_BATCH * NUM_GENERATIONS
    
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=GLOBAL_BATCH_SIZE, 
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        gradient_checkpointing=True, 
        
        # 🟢 [A100 关键] 全面启用 BF16
        fp16=False,      
        bf16=True, 
        
        logging_steps=1,
        save_strategy="epoch",
        report_to="wandb",
        run_name=f"grpo_{run_identifier}", 
        
        beta=0.002, 
        temperature=0.9,
        
        # 🚀 [vLLM 配置]
        use_vllm=False,                
        vllm_device="cuda:0",
        vllm_gpu_memory_utilization=0.4, # A100 显存大，可以多给点
        # 🟢 [A100 关键] vLLM 使用 bfloat16
        vllm_dtype="bfloat16", 
    )

    # --- 5. 加载 Policy Model ---
    print("Loading Policy Model (BF16 + Flash Attention 2)...")
    
    model = AutoModelForCausalLM.from_pretrained(
        POLICY_MODEL_PATH,
        device_map="cpu",
        torch_dtype=torch.bfloat16, # A100
        # 🟢 [A100 关键] 使用 Flash Attention 2 加速
        attn_implementation="sdpa" 
    )
    model.enable_input_require_grads()
    
    peft_config = LoraConfig(
        autocast_adapter_dtype=False,  # 加这行
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", 
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, peft_config)
    model = model.to("cuda:0") # 把 Policy Model 放在卡 0，和 RM 物理隔离
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # --- 6. 准备数据 ---
    dataset = build_dataset(DATA_PATH, tokenizer)

    # --- 7. 初始化 Reward Function ---
    reward_scorer = RewardModelScorer(REWARD_MODEL_PATH, POLICY_MODEL_PATH)

    # --- 8. 训练 ---
    print(f"Starting GRPO Training...")
    torch.cuda.empty_cache()
    gc.collect()

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_scorer, 
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer, 
    )

    trainer.train()

    # --- 9. 稳健保存 ---
    print(f"Saving model to {OUTPUT_DIR}")
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # 递归解包保存
    model_to_save = trainer.model
    while hasattr(model_to_save, "module") or \
          hasattr(model_to_save, "pretrained_model") or \
          (hasattr(model_to_save, "policy") and model_to_save.__class__.__name__ != "PeftModel"):
        if hasattr(model_to_save, "pretrained_model"):
            model_to_save = model_to_save.pretrained_model
        elif hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module
        elif hasattr(model_to_save, "policy"):
            model_to_save = model_to_save.policy
        else:
            break
            
    print(f"✅ Saving underlying model class: {model_to_save.__class__.__name__}")
    model_to_save.save_pretrained(OUTPUT_DIR)
    
    if hasattr(model_to_save, "peft_config"):
        model_to_save.peft_config.save_pretrained(OUTPUT_DIR)

    print("GRPO Training Completed!")

if __name__ == "__main__":
    main()