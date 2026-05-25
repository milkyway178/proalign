import os
import re
import sys
import gc
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))

from rag_util import RAGRetriever


# ===================== Config =====================
BASE_MODEL_PATH = "/root/autodl-tmp/pretrain_models/llama3"
PPO_ADAPTER_PATH = "/root/autodl-tmp/dialogue-KT0/RLMT/saves/llama3-8b-skywork-8b-ppo-200-final_weighted_score"

TRAIN_DATA_PATH = "/root/autodl-tmp/dialogue-KT0/RLMT/data_generate/data/annotated/train.csv"
TEST_DATA_PATH = "/root/autodl-tmp/dialogue-KT0/RLMT/data_generate/data/annotated/test.csv"
OUTPUT_DIR = "/root/autodl-tmp/dialogue-KT0/RLMT/Test/data/experiments_training_model_200_skywork"

BATCH_SIZE = 1
MAX_NEW_TOKENS = 512
MAX_INPUT_LENGTH = 8192
SAMPLE_SIZE = None
RAG_MAX_CHARS = 80000
USE_4BIT = False


# ===================== Prompt =====================
SYSTEM_PROMPT = """You are a professional Knowledge Tracing expert.
The final round of teacher-student dialogue has been concealed.
Predict whether the student explicitly achieves self-correction by the end of the tutoring session.

Your response must end with exactly one of the following labels:
[Prediction]: Yes
[Prediction]: No
"""

USER_PROMPT_TEMPLATE = """
[Similar Cases / Context]
{rag_context}

[Current Task Info]
Question: {question}
Correct Answer: {ground_truth}

[Current Student Profile]
{student_profile}

[Student's Initial Incorrect Solution]
{student_incorrect_solution}

[Dialogue History (Masked)]
{conversation}

Based on the profile and masked dialogue history, predict the final outcome.
"""


# ===================== Utilities =====================
def mask_conversation(conversation: str) -> str:
    if not isinstance(conversation, str):
        return ""

    segments = [s.strip() for s in conversation.split("|EOM|") if s.strip()]
    if not segments:
        return ""

    last_turn = segments[-1].lower()
    if last_turn.startswith("teacher"):
        return "|EOM|".join(segments[:-2]) if len(segments) >= 2 else ""
    return "|EOM|".join(segments[:-1])


def normalize_gt(value) -> str:
    s = str(value).strip().lower()
    if s in {"yes", "true", "1", "1.0", "yes, but i had to reveal the answer"}:
        return "yes"
    if s in {"no", "false", "0", "0.0"}:
        return "no"
    return "unknown"


def extract_prediction(text: str) -> str:
    match = re.search(r"\[Prediction\]:\s*(Yes|No)", str(text), re.IGNORECASE)
    return match.group(1).lower() if match else "unknown"


def build_prompt(row, retriever, strategy: str, k: int) -> str:
    if strategy == "no_rag":
        rag_context = "No examples provided."
    elif strategy == "rag" and retriever is not None:
        profile = str(row.get("student_profile", ""))
        rag_context = retriever.retrieve_examples(profile, k=k, max_chars=RAG_MAX_CHARS)
    else:
        rag_context = "No examples available."

    return USER_PROMPT_TEMPLATE.format(
        rag_context=rag_context,
        question=str(row["question"]),
        ground_truth=str(row.get("ground_truth", "N/A")),
        student_profile=str(row.get("student_profile", "N/A")),
        student_incorrect_solution=str(row["student_incorrect_solution"]),
        conversation=str(row["masked_conversation"]),
    )


# ===================== Local Inference =====================
class LocalInference:
    def __init__(self, base_path: str, adapter_path: str, name: str):
        self.base_path = base_path
        self.adapter_path = adapter_path
        self.name = name
        self.model = None
        self.tokenizer = None

    def load(self):
        print(f"[{self.name}] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        print(f"[{self.name}] Loading model...")
        quantization_config = None
        if USE_4BIT:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )

        if self.adapter_path and os.path.exists(self.adapter_path):
            print(f"[{self.name}] Loading adapter: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            self.model = self.model.merge_and_unload()
        else:
            print(f"[{self.name}] Adapter not found. Using base model.")

        self.model.eval()

    def unload(self):
        print(f"[{self.name}] Unloading model...")
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def infer(self, df: pd.DataFrame, retriever, strategy: str, k: int) -> list[str]:
        prompts = []
        for _, row in df.iterrows():
            user_content = build_prompt(row, retriever, strategy, k)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            prompts.append(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        outputs_all = []
        for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc=f"[{self.name}] {strategy}_k{k}"):
            batch = prompts[i : i + BATCH_SIZE]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_INPUT_LENGTH,
            ).to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            input_len = inputs["input_ids"].shape[1]
            decoded = self.tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
            outputs_all.extend(decoded)

        return outputs_all


# ===================== Evaluation =====================
def evaluate_and_save(df: pd.DataFrame, exp_name: str) -> pd.DataFrame:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = []
    for col in [c for c in df.columns if c.endswith("_output")]:
        model_id = col.replace("_output", "")
        pred_col = f"{model_id}_pred"

        df[pred_col] = df[col].apply(extract_prediction)
        valid = df[df[pred_col] != "unknown"]

        if valid.empty:
            rows.append({
                "Experiment": exp_name,
                "Model": model_id,
                "Valid": 0,
                "Accuracy": 0.0,
                "Precision": 0.0,
                "Recall": 0.0,
                "F1": 0.0,
            })
            continue

        y_true = valid["gt_label"]
        y_pred = valid[pred_col]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            pos_label="yes",
            zero_division=0,
        )

        rows.append({
            "Experiment": exp_name,
            "Model": model_id,
            "Valid": len(valid),
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        })

    df.to_csv(os.path.join(OUTPUT_DIR, f"results_{exp_name}.csv"), index=False)
    return pd.DataFrame(rows)


# ===================== Main =====================
def main():
    retriever = RAGRetriever(TRAIN_DATA_PATH)

    df_test = pd.read_csv(TEST_DATA_PATH, encoding="utf-8")
    if SAMPLE_SIZE:
        df_test = df_test.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

    df_test["masked_conversation"] = df_test["conversation"].apply(mask_conversation)
    df_test["gt_label"] = df_test["self-correctness"].apply(normalize_gt)
    df_test = df_test[df_test["gt_label"] != "unknown"].reset_index(drop=True)

    experiments = [
        {"name": "Exp1_NoRAG", "strategy": "no_rag", "k": 0},
        {"name": "Exp3_RAG_k3", "strategy": "rag", "k": 3},
        {"name": "Exp4_RAG_k5", "strategy": "rag", "k": 5},
    ]

    model_configs = [
        {
            "id": "ppo",
            "base": BASE_MODEL_PATH,
            "adapter": PPO_ADAPTER_PATH,
            "name": "PPO-Model",
        }
    ]

    results_storage = {exp["name"]: df_test.copy() for exp in experiments}

    for cfg in model_configs:
        runner = LocalInference(cfg["base"], cfg["adapter"], cfg["name"])
        runner.load()

        for exp in experiments:
            df_exp = results_storage[exp["name"]]
            df_exp[f"{cfg['id']}_output"] = runner.infer(
                df_exp,
                retriever,
                strategy=exp["strategy"],
                k=exp["k"],
            )

        runner.unload()

    metrics = []
    for exp in experiments:
        metrics.append(evaluate_and_save(results_storage[exp["name"]], exp["name"]))

    metrics_df = pd.concat(metrics, ignore_index=True)
    metrics_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
    metrics_df.to_csv(metrics_path, index=False)

    print(metrics_df)
    print(f"All results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
