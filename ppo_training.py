import torch
from tqdm import tqdm
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    pipeline,
    GenerationConfig
)

from peft import LoraConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import Dataset
import torch.nn.functional as F
import sys
import os
import json
import argparse  
from contextlib import nullcontext
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from rag_util import RAGRetriever 

POLICY_MODEL_PATH = 
#REWARD_MODEL_PATH = 

DATA_PATH = 
#OUTPUT_DIR = 

LEARNING_RATE = 1.41e-5
BATCH_SIZE = 8
MINI_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1
EPOCHS = 2
MAX_LENGTH = 32768
KL_COEF = 0.02
TEMPERATURE = 0.9

SYSTEM_PROMPT = """You are a professional and rigorous Knowledge Tracing expert... """
USER_PROMPT_TEMPLATE = 

def mask_conversation(conversation):
    if not isinstance(conversation, str): return ""
    segments = [s for s in conversation.split('|EOM|') if s.strip()]
    if not segments: return ""
    last_turn = segments[-1].strip()
    if last_turn.lower().startswith("teacher"):
        return '|EOM|'.join(segments[:-2]) if len(segments) >= 2 else ""
    else:
        return '|EOM|'.join(segments[:-1]) if len(segments) >= 1 else ""

def build_dataset(data_path, tokenizer):
    print(f"Loading data from {data_path}...")
    try: df = pd.read_csv(data_path, encoding='gbk')
    except: df = pd.read_csv(data_path)
    
    print("Initializing RAG Retriever...")
    retriever = RAGRetriever(data_path)
    df = df.dropna(subset=['question', 'student_incorrect_solution', 'conversation'])
    input_texts = []
    
    print("Constructing Prompts with Context...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        profile = str(row.get('student_profile', ''))
        rag_context = retriever.retrieve_examples(profile, k=3, max_chars=15000)
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
        input_texts.append(formatted_prompt)

    dataset = Dataset.from_dict({"query": input_texts})
    def tokenize(sample):
        tokenized = tokenizer(sample["query"], padding=False, truncation=True, max_length=MAX_LENGTH)
        return {"input_ids": tokenized["input_ids"]}
    dataset = dataset.map(tokenize, batched=False)
    return dataset

# ==============================================================================
# ==============================================================================
def get_logprobs(logits, labels, attention_mask=None):
    """计算 Log Probability。"""
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_gathered = torch.gather().squeeze(-1)
    if attention_mask is not None:
        log_probs_gathered = log_probs_gathered * attention_mask
    return log_probs_gathered


def get_trainer_config(trainer):
    return getattr(trainer, "config", None) or getattr(trainer, "args", None)


def get_pad_token_id(trainer):
    tok = getattr(trainer, "tokenizer", None) or getattr(trainer, "processing_class", None)
    if tok is None:
        raise AttributeError("Cannot find tokenizer/processing_class on PPOTrainer.")
    return tok.pad_token_id


def get_ppo_attr(config, names, default=None):
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    return default


def unwrap_value_head_model(trainer):
    model = trainer.accelerator.unwrap_model(trainer.model)
    while hasattr(model, "module"):
        model = model.module
    return model


def set_use_cache_for_generation(model, enabled: bool):
    if hasattr(model, "pretrained_model") and hasattr(model.pretrained_model, "config"):
        model.pretrained_model.config.use_cache = enabled
    elif hasattr(model, "config"):
        model.config.use_cache = enabled


def forward_value_head(model, input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    if isinstance(outputs, tuple):
        logits = outputs[0]
        values = outputs[-1]
    else:
        logits = outputs.logits
        values = outputs.value
    return logits, values, outputs



def get_ref_logits_for_peft_or_ref(trainer, real_model, batch_input_ids, batch_mask):

    if getattr(trainer, "ref_model", None) is not None:
        ref_model = trainer.accelerator.unwrap_model(trainer.ref_model)
        ref_outputs = ref_model(input_ids=batch_input_ids, attention_mask=batch_mask)
        ref_logits = ref_outputs[0] if isinstance(ref_outputs, tuple) else ref_outputs.logits
        return ref_logits, ref_outputs

    peft_model = getattr(real_model, "pretrained_model", real_model)
    disable_adapter = getattr(peft_model, "disable_adapter", None)

    if callable(disable_adapter):
        ctx = disable_adapter()
    else:
        ctx = nullcontext()

    with ctx:
        ref_logits, _, ref_outputs = forward_value_head(real_model, batch_input_ids, batch_mask)

    return ref_logits, ref_outputs

def manual_ppo_step(trainer, query_tensors, response_tensors, rewards):

    config = get_trainer_config(trainer)
    accelerator = trainer.accelerator
    
    sequences = [torch.cat((q, r)) for q, r in zip(query_tensors, response_tensors)]
    pad_token_id = get_pad_token_id(trainer)
    padded_sequences = torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=pad_token_id
    ).to(accelerator.device)
    
    attention_mask = (padded_sequences != pad_token_id).long()
    
    response_mask = torch.zeros_like(attention_mask)
    for i, (q, r) in enumerate(zip(query_tensors, response_tensors)):
        q_len = len(q)
        r_len = len(r)
        response_mask[i, q_len : r_len] = 1
    final_mask = attention_mask * response_mask

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    
    old_logprobs_list = []
    values_list = []
    ref_logprobs_list = []
    
    chunk_size = 1 
    total_batch_size = padded_sequences.shape[0]

    with torch.no_grad():
        real_model = unwrap_value_head_model(trainer)
            
        for i in range(0, total_batch_size, chunk_size):
            batch_input_ids = padded_sequences[i : i+chunk_size]
            batch_mask = attention_mask[i : i+chunk_size]
            batch_final_mask = final_mask[i : i+chunk_size]
            
            # --- A. Policy Forward ---
            logits, val, outputs = forward_value_head(base_model, batch_input_ids, batch_mask)
            
            chunk_logprobs = get_logprobs(logits, batch_input_ids, batch_final_mask)
            
            # --- B. Reference Forward ---
            ref_logits, ref_outputs = get_ref_logits_for_peft_or_ref(
                trainer, real_model, batch_input_ids, batch_mask
            )
            
            chunk_ref_logprobs = get_logprobs(ref_logits, batch_input_ids, batch_final_mask)
            ref_logprobs_list.append(chunk_ref_logprobs.cpu())
            
            del ref_logits, ref_outputs
            torch.cuda.empty_cache()

    old_logprobs = torch.cat(old_logprobs_list).to(accelerator.device)
    values = torch.cat(values_list).to(accelerator.device)
    ref_logprobs = torch.cat(ref_logprobs_list).to(accelerator.device)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    kl = old_logprobs - ref_logprobs
    kl = kl * final_mask
    kl_coef = get_ppo_attr(config, ["kl_coef", "init_kl_coef"], KL_COEF)
    non_score_reward = -kl_coef * kl
    
    full_rewards = ''
    for i, (r, mask_row) in enumerate(zip(rewards, response_mask)):
        last_idx = (mask_row == 1).nonzero()[-1].item()
        full_rewards[i, last_idx] += r

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    lastgaelam = 0
    advantages_reversed = []
    seq_len = padded_sequences.shape[1]
    
    if values.shape[1] > seq_len:
        values = values[0:, values.shape[1]:seq_len]
    values = values * attention_mask

    for t in reversed(range(seq_len)):
        nextvalues = values[:, t + 1] if t < seq_len - 1 else 0.0
        delta = full_rewards[len(seq_len):, t] + config.gamma * nextvalues - values[:, t]
        lastgaelam = delta + config.gamma * config.lam * lastgaelam
        advantages_reversed.append(lastgaelam)
        
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    
    advantages = advantages * final_mask
    returns = returns * final_mask

    if config.batch_size > 1: 
        adv_mean = (advantages * final_mask).sum() / final_mask.sum()
        adv_std = ((advantages - adv_mean).pow(2) * final_mask).sum() / final_mask.sum()

        advantages = (advantages - adv_mean) / (torch.sqrt(adv_std) + 1e-8)
        advantages = advantages * final_mask 

    # Detach
    old_logprobs = old_logprobs.detach()
    advantages = advantages.detach()
    values = values.detach()
    returns = returns.detach()

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    stats = {}
    trainer.model.train()
    
    
    ppo_epochs = get_ppo_attr(config, ["num_ppo_epochs", "ppo_epochs"], 4)
    for _ in range(ppo_epochs):
        
        chunk_size = 1
        total_pg_loss = 0
        total_vf_loss = 0
        total_kl = 0
        steps = 0
        
        for i in range(0, total_batch_size, chunk_size):
            batch_input_ids = padded_sequences[i : i+chunk_size]
            batch_mask = attention_mask[i : i+chunk_size]
            batch_final_mask = final_mask[i : i+chunk_size]
            
            batch_old_logprobs = old_logprobs[i : i+chunk_size]
            batch_advantages = advantages[i : i+chunk_size]
            batch_returns = returns[i : i+chunk_size]
            
            # Forward Pass
            real_model = unwrap_value_head_model(trainer)
            logits, v_preds, outputs = forward_value_head(real_model, batch_input_ids, batch_mask)

            new_logprobs = get_logprobs(logits, batch_input_ids, batch_final_mask)
            
            # PPO Loss Calculation
            logprobs_diff = new_logprobs - batch_old_logprobs
            ratio = torch.exp(logprobs_diff)
            
            pg_losses = -batch_advantages * ratio
            pg_losses2 = -batch_advantages * torch.clamp(ratio, 1.0 - config.cliprange, 1.0 + config.cliprange)
            
            # Loss Mean over this micro-batch
            pg_loss = torch.max(pg_losses, pg_losses2).sum() / (batch_final_mask.sum() + 1e-8)
            
            v_preds = v_preds * batch_mask
            vf_loss = F.bce_loss(v_preds * batch_final_mask, batch_returns * batch_final_mask, reduction='sum') / (batch_final_mask.sum() + 1e-8)
            
            loss = pg_loss + config.vf_coef * vf_loss
            loss = loss / (total_batch_size / chunk_size)
            
            total_pg_loss += pg_loss.item()
            total_vf_loss += vf_loss.item()
            steps += 1
            
            del logits, outputs, v_preds, loss
            torch.cuda.empty_cache()

        trainer.optimizer.step()
        trainer.optimizer.zero_grad()
        
        stats['ppo/loss/policy'] = total_pg_loss / steps
        stats['ppo/loss/value'] = total_vf_loss / steps
        stats['ppo/policy/kl'] = total_kl / steps

    return stats
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_model_path", type=str, required=True, help="Path to the reward model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model")
    args = parser.parse_args()

    REWARD_MODEL_PATH = args.reward_model_path
    OUTPUT_DIR = args.output_dir
    
    run_identifier = OUTPUT_DIR.split('-')[-1] 
    

    config = PPOConfig()

    print("Loading Policy Model...")
    
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        POLICY_MODEL_PATH,
        device_map="auto",
        peft_config=LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        ),
        torch_dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable() 
    model.pretrained_model.config.use_cache = False
    
    tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 

    print("Loading Reward Model...")
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_PATH,
        device_map="auto",
        num_labels=1,
        torch_dtype=torch.bfloat16
    )
    rm_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_PATH)
    if rm_tokenizer.pad_token is None:
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
    rm_tokenizer.padding_side = "right"
    if rm_model.config.pad_token_id is None:
        rm_model.config.pad_token_id = rm_tokenizer.pad_token_id

    sentiment_pipe = pipeline(
        "text-classification",
        model=rm_model,
        tokenizer=rm_tokenizer,
        device_map="auto",
        function_to_apply="none" 
    )

    print("Building PPO Dataset...")
    dataset = build_dataset(DATA_PATH, tokenizer)
    dataset = dataset.filter(lambda x: len(x["input_ids"]) < (MAX_LENGTH - 4096))

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    if not hasattr(model, "generation_config"):
        model.generation_config = GenerationConfig.from_model_config(model.pretrained_model.config)
    if not hasattr(model, "base_model_prefix"):
        model.base_model_prefix = "pretrained_model"
    if not hasattr(model, "is_gradient_checkpointing"):
        model.is_gradient_checkpointing = getattr(model.pretrained_model, "is_gradient_checkpointing", False)

    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
    )
    print("Starting PPO Training (Manual Step Patch Mode)...")
    
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 2048, 
        "temperature": TEMPERATURE
    }

    for epoch in range(EPOCHS):
        for batch in tqdm(ppo_trainer.dataloader, desc=f"Epoch {epoch+1}"):
            
            # --- 1. Generate (Rollout) ---
            current_device = ppo_trainer.accelerator.device
            input_ids_list = [torch.tensor(t).long() for t in batch["input_ids"]]
            padded_inputs = tokenizer.pad(
                {"input_ids": input_ids_list}, padding=True, return_tensors="pt"
            ).to(current_device)
            
            with torch.no_grad():
                generation_model = unwrap_value_head_model(ppo_trainer)
                set_use_cache_for_generation(generation_model, True)
                raw_generated_sequences = generation_model.generate(
                    **padded_inputs,
                    **generation_kwargs
                )
                set_use_cache_for_generation(generation_model, False)
                
            response_tensors = []
            query_tensors = [] 
            prompt_length = padded_inputs["input_ids"].shape[1]
            
            for i, sequence in enumerate(raw_generated_sequences):
                response_token = sequence[prompt_length:] 
                response_tensors.append(response_token)
                query_tensors.append(input_ids_list[i].to(current_device))

            batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            batch["query"] = tokenizer.batch_decode(query_tensors, skip_special_tokens=True)

            # --- 2. Compute Rewards ---
            texts_to_score = [q + r for q, r in zip(batch["query"], batch["response"])]
            pipe_outputs = sentiment_pipe(texts_to_score, top_k=None, truncation=True, batch_size=4)
            
            rewards = []
            for output in pipe_outputs:
                val = output['score'] if isinstance(output, dict) else output[0]['score']
                rewards.append(torch.tensor(val, dtype=torch.float32).to(current_device))

            stats = ppo_trainer.step(ppo_trainer, query_tensors, response_tensors, rewards)
            

            reward_tensor = torch.stack(rewards)
            stats["ppo/reward/mean"] = reward_tensor.mean().item()
            stats["ppo/reward/std"] = reward_tensor.std().item()
            
            lengths = [len(x) for x in response_tensors]
            stats["ppo/response/mean_length"] = sum(lengths) / len(lengths)
            
            stats["ppo/learning_rate"] = ppo_trainer.optimizer.param_groups[0]["lr"]

            if hasattr(ppo_trainer, "log_stats"):
                ppo_trainer.log_stats(stats, batch, rewards)
            else:
                print(stats)
            # ==========================================================

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def save_ppo_lora_checkpoint(ppo_trainer, tokenizer, output_dir, reward_model_path):

        os.makedirs(output_dir, exist_ok=True)

        accelerator = getattr(ppo_trainer, "accelerator", None)
        is_main_process = True
        if accelerator is not None:
            is_main_process = accelerator.is_main_process
            accelerator.wait_for_everyone()

        if not is_main_process:
            return

        print(f"Saving final checkpoint to {output_dir}...")


        tokenizer.save_pretrained(output_dir)


        model_to_save = ppo_trainer.model
        if accelerator is not None:
            model_to_save = accelerator.unwrap_model(model_to_save)
        while hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module

        print(f"✅ Unwrapped PPO model class: {model_to_save.__class__.__name__}")


        policy_model = getattr(model_to_save, "pretrained_model", model_to_save)
        while hasattr(policy_model, "module"):
            policy_model = policy_model.module

        print(f"✅ Saving policy model class: {policy_model.__class__.__name__}")
        policy_model.save_pretrained(output_dir, safe_serialization=True)


        value_head = getattr(model_to_save, "v_head", None)
        if value_head is not None:
            value_head_path = os.path.join(output_dir, "value_head.pt")
            torch.save(value_head.state_dict(), value_head_path)
            print(f"✅ Saved PPO value head to: {value_head_path}")
        else:
            print("⚠️ No v_head found; skipped value_head.pt")

        metadata = {
            "trl_version": getattr(__import__("trl"), "__version__", "unknown"),
            "policy_model_path": POLICY_MODEL_PATH,
            "reward_model_path": reward_model_path,
            "save_type": "lora_adapter_plus_value_head",
            "note": "For inference, load base model + LoRA adapter from this directory. value_head.pt is only needed for continuing PPO training.",
        }
        with open(os.path.join(output_dir, "ppo_save_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        expected_files = [
            "adapter_config.json",
            "adapter_model.safetensors",
            "tokenizer_config.json",
            "value_head.pt",
            "ppo_save_metadata.json",
        ]
        existing = [name for name in expected_files if os.path.exists(os.path.join(output_dir, name))]
        print(f"✅ Existing saved files: {existing}")

    save_ppo_lora_checkpoint(ppo_trainer, tokenizer, OUTPUT_DIR, REWARD_MODEL_PATH)

    print(f"PPO Training for {REWARD_MODEL_PATH} Completed!")

if __name__ == "__main__":
    main()
