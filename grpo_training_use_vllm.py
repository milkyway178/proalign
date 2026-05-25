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


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from rag_util import RAGRetriever 


try:
    from trl.trainer.grpo_trainer import GRPOTrainer
    _original_get_train_sampler = GRPOTrainer._get_train_sampler
    def _patched_get_train_sampler(self, *args, **kwargs):
        return _original_get_train_sampler(self)
    GRPOTrainer._get_train_sampler = _patched_get_train_sampler
except ImportError:
    pass


POLICY_MODEL_PATH = 
DATA_PATH = 


LEARNING_RATE = 2e-5          
NUM_GENERATIONS = 8           

MAX_PROMPT_LENGTH = 8192      
MAX_COMPLETION_LENGTH = 512   
EPOCHS = 2
PROMPTS_PER_BATCH = 1 
GRADIENT_ACCUMULATION_STEPS = 4 


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


class RewardModelScorer:
    def __init__(self, reward_model_path, base_model_path):
        self.__name__ = "reward_model_scorer"
        print(f"Loading Reward Model from {reward_model_path} [GPU Resident]...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16, 
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path,
            num_labels=1,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16, 
            device_map={"": 0} 
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
        batch_size = 8 
        
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_model_path", type=str, required=True, help="Path to the reward model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model")
    args = parser.parse_args()

    REWARD_MODEL_PATH = args.reward_model_path
    OUTPUT_DIR = args.output_dir
    run_identifier = OUTPUT_DIR.split('-')[-1]


    GLOBAL_BATCH_SIZE = PROMPTS_PER_BATCH * NUM_GENERATIONS
    
    training_args = GRPOConfig()

    print("Loading Policy Model (BF16 + Flash Attention 2)...")
    
    model = AutoModelForCausalLM.from_pretrained(
        POLICY_MODEL_PATH,
        device_map="cpu",
        torch_dtype=torch.bfloat16, # A100
        attn_implementation="sdpa" 
    )
    model.enable_input_require_grads()
    
    peft_config = LoraConfig(
        autocast_adapter_dtype=False,  
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", 
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, peft_config)
    model = model.to("cuda:0") 
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = build_dataset(DATA_PATH, tokenizer)

    reward_scorer = RewardModelScorer(REWARD_MODEL_PATH, POLICY_MODEL_PATH)

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

    print(f"Saving model to {OUTPUT_DIR}")
    tokenizer.save_pretrained(OUTPUT_DIR)
    
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
