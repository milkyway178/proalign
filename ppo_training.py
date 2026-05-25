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
import argparse  # --- 新增：命令行参数解析 ---
from contextlib import nullcontext
# --- 新增：导入 RAG 模块 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from rag_util import RAGRetriever 

# --- 1. 配置路径与参数 ---
POLICY_MODEL_PATH = "/root/autodl-tmp/pretrain_models/llama3"
#REWARD_MODEL_PATH = "dialogue-KT0/RLMT/saves/skywork-8b-reward-model-rag-200"
#dialogue-KT0/RLMT/saves/skywork-8b-reward-model-rag-200
DATA_PATH = "/root/autodl-tmp/dialogue-KT0/RLMT/data_generate/data/annotated/train.csv" 
#OUTPUT_DIR = "dialogue-KT0/RLMT/saves/llama3-8b-skywork-8b-ppo-final-200"

# PPO 超参数
LEARNING_RATE = 1.41e-5
BATCH_SIZE = 8
MINI_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1
EPOCHS = 2
MAX_LENGTH = 32768
KL_COEF = 0.02
TEMPERATURE = 0.9

# --- Prompt 模板 ---
SYSTEM_PROMPT = """You are a professional and rigorous Knowledge Tracing expert... (保持原样)"""
USER_PROMPT_TEMPLATE = """... (保持原样) ..."""

def mask_conversation(conversation):
    # (保持原样)
    if not isinstance(conversation, str): return ""
    segments = [s for s in conversation.split('|EOM|') if s.strip()]
    if not segments: return ""
    last_turn = segments[-1].strip()
    if last_turn.lower().startswith("teacher"):
        return '|EOM|'.join(segments[:-2]) if len(segments) >= 2 else ""
    else:
        return '|EOM|'.join(segments[:-1]) if len(segments) >= 1 else ""

def build_dataset(data_path, tokenizer):
    # (保持原样)
    print(f"Loading data from {data_path}...")
    try: df = pd.read_csv(data_path, encoding='gbk')
    except: df = pd.read_csv(data_path)
    
    print("Initializing RAG Retriever...")
    retriever = RAGRetriever(data_path)
    df = df.dropna(subset=['question', 'student_incorrect_solution', 'conversation'])
    input_texts = []
    
    print("Constructing Prompts with RAG Context...")
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
# 🔴 终极修复补丁：绕过 Wrapper 的 Bug
# ==============================================================================
def get_logprobs(logits, labels, attention_mask=None):
    """计算 Log Probability。"""
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_gathered = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
    if attention_mask is not None:
        log_probs_gathered = log_probs_gathered * attention_mask
    return log_probs_gathered


def get_trainer_config(trainer):
    """兼容 TRL 0.9.x / 0.15.x 的 config 访问。"""
    return getattr(trainer, "config", None) or getattr(trainer, "args", None)


def get_pad_token_id(trainer):
    """TRL 0.9.x 用 tokenizer；新版可能用 processing_class。"""
    tok = getattr(trainer, "tokenizer", None) or getattr(trainer, "processing_class", None)
    if tok is None:
        raise AttributeError("Cannot find tokenizer/processing_class on PPOTrainer.")
    return tok.pad_token_id


def get_ppo_attr(config, names, default=None):
    """兼容不同 TRL 版本中的 PPOConfig 字段名。"""
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    return default


def unwrap_value_head_model(trainer):
    """拿到带 value head 的策略模型。不要剥到纯 CausalLM，否则没有 value。"""
    model = trainer.accelerator.unwrap_model(trainer.model)
    while hasattr(model, "module"):
        model = model.module
    return model


def set_use_cache_for_generation(model, enabled: bool):
    """AutoModelForCausalLMWithValueHead 的 config 通常在 pretrained_model.config。"""
    if hasattr(model, "pretrained_model") and hasattr(model.pretrained_model, "config"):
        model.pretrained_model.config.use_cache = enabled
    elif hasattr(model, "config"):
        model.config.use_cache = enabled


def forward_value_head(model, input_ids, attention_mask):
    """兼容 AutoModelForCausalLMWithValueHead 的输出格式。"""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    if isinstance(outputs, tuple):
        logits = outputs[0]
        values = outputs[-1]
    else:
        logits = outputs.logits
        values = outputs.value
    return logits, values, outputs



def get_ref_logits_for_peft_or_ref(trainer, real_model, batch_input_ids, batch_mask):
    """
    TRL 0.9.6 没有 trainer.null_ref_context()。
    当 ref_model=None 且 policy 是 PEFT/LoRA 模型时，临时关闭 adapter，
    用 base model 作为 reference model 来计算 ref logits。
    如果不是 PEFT 模型，则退化为当前模型 logits，此时 KL≈0。
    """
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
    """
    深度优化版 PPO Step：
    引入 chunking 机制，逐条计算 forward pass，避免 logits 张量撑爆显存。
    """
    config = get_trainer_config(trainer)
    accelerator = trainer.accelerator
    
    # 1. 数据预处理
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
        response_mask[i, q_len : q_len + r_len] = 1
    final_mask = attention_mask * response_mask

    # ------------------------------------------------------------------
    # 2. 获取旧策略的 Logprobs 和 Values (Rollout) - [显存优化核心]
    # ------------------------------------------------------------------
    # 我们不一次性把 batch=8 塞进去，而是切分成 batch=1 的小块处理
    
    old_logprobs_list = []
    values_list = []
    ref_logprobs_list = []
    
    # 每次只处理 1 条数据 (Micro-Batch = 1)
    chunk_size = 1 
    total_batch_size = padded_sequences.shape[0]

    with torch.no_grad():
        # 准备模型：TRL 0.9.6 的 AutoModelForCausalLMWithValueHead 本身就带 value head
        real_model = unwrap_value_head_model(trainer)
            
        for i in range(0, total_batch_size, chunk_size):
            # 切片
            batch_input_ids = padded_sequences[i : i+chunk_size]
            batch_mask = attention_mask[i : i+chunk_size]
            batch_final_mask = final_mask[i : i+chunk_size]
            
            # --- A. Policy Forward ---
            logits, val, outputs = forward_value_head(real_model, batch_input_ids, batch_mask)
            
            # 立即计算 logprobs (结果很小)
            chunk_logprobs = get_logprobs(logits, batch_input_ids, batch_final_mask)
            
            # 存入列表
            old_logprobs_list.append(chunk_logprobs.cpu()) # 暂时放 CPU 省显存
            values_list.append(val.cpu())
            
            # 🔥 立即释放巨大的 logits
            del logits, outputs, val
            torch.cuda.empty_cache() 
            
            # --- B. Reference Forward ---
            # TRL 0.9.6 没有 trainer.null_ref_context()；这里手动兼容 PEFT/LoRA。
            ref_logits, ref_outputs = get_ref_logits_for_peft_or_ref(
                trainer, real_model, batch_input_ids, batch_mask
            )
            
            chunk_ref_logprobs = get_logprobs(ref_logits, batch_input_ids, batch_final_mask)
            ref_logprobs_list.append(chunk_ref_logprobs.cpu())
            
            # 🔥 再次释放
            del ref_logits, ref_outputs
            torch.cuda.empty_cache()

    # 拼接回大张量并放回 GPU
    old_logprobs = torch.cat(old_logprobs_list).to(accelerator.device)
    values = torch.cat(values_list).to(accelerator.device)
    ref_logprobs = torch.cat(ref_logprobs_list).to(accelerator.device)

    # ------------------------------------------------------------------
    # 3. 计算 Rewards
    # ------------------------------------------------------------------
    kl = old_logprobs - ref_logprobs
    kl = kl * final_mask
    kl_coef = get_ppo_attr(config, ["kl_coef", "init_kl_coef"], KL_COEF)
    non_score_reward = -kl_coef * kl
    
    full_rewards = non_score_reward.clone()
    for i, (r, mask_row) in enumerate(zip(rewards, response_mask)):
        last_idx = (mask_row == 1).nonzero()[-1].item()
        full_rewards[i, last_idx] += r

    # ------------------------------------------------------------------
    # 4. 计算 Advantages (GAE)
    # ------------------------------------------------------------------
    lastgaelam = 0
    advantages_reversed = []
    seq_len = padded_sequences.shape[1]
    
    if values.shape[1] > seq_len:
        values = values[:, :seq_len]
    values = values * attention_mask

    for t in reversed(range(seq_len)):
        nextvalues = values[:, t + 1] if t < seq_len - 1 else 0.0
        delta = full_rewards[:, t] + config.gamma * nextvalues - values[:, t]
        lastgaelam = delta + config.gamma * config.lam * lastgaelam
        advantages_reversed.append(lastgaelam)
        
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    
    advantages = advantages * final_mask
    returns = returns * final_mask

    if config.batch_size > 1: # 只有 batch > 1 时归一化才有统计意义
        adv_mean = (advantages * final_mask).sum() / final_mask.sum()
        adv_std = ((advantages - adv_mean).pow(2) * final_mask).sum() / final_mask.sum()
        # 加上 1e-8 防止除以 0
        advantages = (advantages - adv_mean) / (torch.sqrt(adv_std) + 1e-8)
        advantages = advantages * final_mask # 再次 mask 确保 padding 处为 0

    # Detach
    old_logprobs = old_logprobs.detach()
    advantages = advantages.detach()
    values = values.detach()
    returns = returns.detach()

    # ------------------------------------------------------------------
    # 5. 训练循环 (PPO Epochs) - [显存优化核心]
    # ------------------------------------------------------------------
    stats = {}
    trainer.model.train()
    
    # 同样使用 Micro-Batch 进行训练更新
    # 由于我们需要计算梯度，这里使用 gradient accumulation
    
    ppo_epochs = get_ppo_attr(config, ["num_ppo_epochs", "ppo_epochs"], 4)
    for _ in range(ppo_epochs):
        # 重新 shuffle 索引 (可选，简单起见这里按顺序)
        
        chunk_size = 1 # 训练时也一次只算 1 条
        
        total_pg_loss = 0
        total_vf_loss = 0
        total_kl = 0
        steps = 0
        
        for i in range(0, total_batch_size, chunk_size):
            # 切片数据
            batch_input_ids = padded_sequences[i : i+chunk_size]
            batch_mask = attention_mask[i : i+chunk_size]
            batch_final_mask = final_mask[i : i+chunk_size]
            
            # 切片对应的 target 数据
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
            vf_loss = F.mse_loss(v_preds * batch_final_mask, batch_returns * batch_final_mask, reduction='sum') / (batch_final_mask.sum() + 1e-8)
            
            loss = pg_loss + config.vf_coef * vf_loss
            
            # 归一化 Loss (因为我们在累积梯度)
            # 实际上这里是每个 micro-batch 更新一次，或者累积后更新
            # 简单起见，我们每个 micro-batch backward 一次，然后最后 step
            # 但为了模拟大 batch 效果，我们需要除以 accumulation steps
            loss = loss / (total_batch_size / chunk_size)
            
            accelerator.backward(loss)
            
            # 统计
            total_pg_loss += pg_loss.item()
            total_vf_loss += vf_loss.item()
            total_kl += (kl[i:i+chunk_size].sum() / (batch_final_mask.sum() + 1e-8)).item()
            steps += 1
            
            # 释放显存
            del logits, outputs, v_preds, loss
            torch.cuda.empty_cache()

        # 所有 micro-batches 跑完后，更新一次参数
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
    
    # 从输出路径或模型路径提取关键词作为 WandB 的 run name
    run_identifier = OUTPUT_DIR.split('-')[-1] # 例如 score_accuracy
    
    print(f"==================================================")
    print(f"🚀 Starting PPO Training Task")
    print(f"   Reward Model: {REWARD_MODEL_PATH}")
    print(f"   Output Dir:   {OUTPUT_DIR}")
    print(f"   WandB Name:   ppo_{run_identifier}")
    print(f"==================================================")
    # --- 3. 初始化配置（适配 trl==0.9.6） ---
    config = PPOConfig(
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        mini_batch_size=MINI_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        # trl==0.9.6 使用 init_kl_coef
        init_kl_coef=KL_COEF,
    )

    # --- 4. 加载模型 ---
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

    # --- 5. 准备数据 ---
    print("Building PPO Dataset...")
    dataset = build_dataset(DATA_PATH, tokenizer)
    dataset = dataset.filter(lambda x: len(x["input_ids"]) < (MAX_LENGTH - 4096))

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    # 针对部分 wrapper 可能存在的属性缺失问题，手动补全
    if not hasattr(model, "generation_config"):
        model.generation_config = GenerationConfig.from_model_config(model.pretrained_model.config)
    if not hasattr(model, "base_model_prefix"):
        model.base_model_prefix = "pretrained_model"
    if not hasattr(model, "is_gradient_checkpointing"):
        model.is_gradient_checkpointing = getattr(model.pretrained_model, "is_gradient_checkpointing", False)

    # --- 6. 初始化 Trainer：trl==0.9.6 接口 ---
    # 注意：0.9.6 没有 args / processing_class / train_dataset / reward_model / value_model 参数。
    # Reward model 在外部自己打分，然后把 rewards 传给 manual_ppo_step。
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
            # 手动 generate，避免不同 TRL 版本的 generate 包装差异
            current_device = ppo_trainer.accelerator.device
            input_ids_list = [torch.tensor(t).long() for t in batch["input_ids"]]
            padded_inputs = tokenizer.pad(
                {"input_ids": input_ids_list}, padding=True, return_tensors="pt"
            ).to(current_device)
            
            with torch.no_grad():
                # AutoModelForCausalLMWithValueHead 支持 generate，会委托给底层 CausalLM。
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

            # --- 3. PPO Step (调用我们的补丁函数) ---
            # 🔴 这里不再调用 ppo_trainer.step，而是调用 manual_ppo_step
            stats = manual_ppo_step(ppo_trainer, query_tensors, response_tensors, rewards)
            
            # 1. 计算 Reward 的统计信息
            # rewards 是 list of tensors, 先转为 tensor
            reward_tensor = torch.stack(rewards)
            stats["ppo/reward/mean"] = reward_tensor.mean().item()
            stats["ppo/reward/std"] = reward_tensor.std().item()
            
            # 2. 计算回复长度的统计信息
            lengths = [len(x) for x in response_tensors]
            stats["ppo/response/mean_length"] = sum(lengths) / len(lengths)
            
            # 3. 记录 Learning Rate
            # 从 optimizer 获取当前 LR
            stats["ppo/learning_rate"] = ppo_trainer.optimizer.param_groups[0]["lr"]

            # 4. 记录日志：trl==0.9.6 优先使用 log_stats；若不可用则直接 print。
            if hasattr(ppo_trainer, "log_stats"):
                ppo_trainer.log_stats(stats, batch, rewards)
            else:
                print(stats)
            # ==========================================================

    # ------------------------------------------------------------------
    # 7. 稳健保存：兼容 TRL 0.9.6 + AutoModelForCausalLMWithValueHead + LoRA
    # ------------------------------------------------------------------
    def save_ppo_lora_checkpoint(ppo_trainer, tokenizer, output_dir, reward_model_path):
        """
        保存三类内容：
        1) tokenizer：用于后续加载/推理；
        2) LoRA adapter：adapter_model.safetensors + adapter_config.json；
        3) PPO value head：value_head.pt，用于将来继续 PPO 训练时恢复 critic。

        注意：如果只是做最终推理，通常只需要 tokenizer + LoRA adapter。
        value_head 主要用于继续 RL/PPO 训练。
        """
        os.makedirs(output_dir, exist_ok=True)

        accelerator = getattr(ppo_trainer, "accelerator", None)
        is_main_process = True
        if accelerator is not None:
            is_main_process = accelerator.is_main_process
            accelerator.wait_for_everyone()

        if not is_main_process:
            return

        print(f"Saving final checkpoint to {output_dir}...")

        # 1) 保存 tokenizer
        tokenizer.save_pretrained(output_dir)

        # 2) 解包 Accelerate/DDP wrapper，但保留 TRL 的 value-head wrapper
        model_to_save = ppo_trainer.model
        if accelerator is not None:
            model_to_save = accelerator.unwrap_model(model_to_save)
        while hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module

        print(f"✅ Unwrapped PPO model class: {model_to_save.__class__.__name__}")

        # 3) 保存底层 policy/LoRA adapter
        # AutoModelForCausalLMWithValueHead.pretrained_model 通常是 PeftModelForCausalLM。
        policy_model = getattr(model_to_save, "pretrained_model", model_to_save)
        while hasattr(policy_model, "module"):
            policy_model = policy_model.module

        print(f"✅ Saving policy model class: {policy_model.__class__.__name__}")
        policy_model.save_pretrained(output_dir, safe_serialization=True)

        # 4) 额外保存 value head，避免以后继续 PPO 时 critic 丢失
        value_head = getattr(model_to_save, "v_head", None)
        if value_head is not None:
            value_head_path = os.path.join(output_dir, "value_head.pt")
            torch.save(value_head.state_dict(), value_head_path)
            print(f"✅ Saved PPO value head to: {value_head_path}")
        else:
            print("⚠️ No v_head found; skipped value_head.pt")

        # 5) 保存一点元信息，方便之后确认版本和加载方式
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