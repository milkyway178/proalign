import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from rag_util import RAGRetriever 

BASE_MODEL_PATH = "xxx"
LORA_PATH = "xxx"
DATA_PATH = "xxx"

OUTPUT_PATH = "dialogue-KT0/RLMT/reward_model_train/data/inference_results_200.csv" 
SAMPLE_SIZE = 200 
BATCH_SIZE = 8     
RANDOM_STATE = 42  
RAG_K = 3          
RAG_MAX_CHARS = 40000 


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

def create_llama3_prompt(row, retriever, tokenizer):
    
    question = str(row.get('question', ''))
    solution = str(row.get('student_incorrect_solution', ''))
    profile = str(row.get('student_profile', ''))
    
    raw_conv = str(row.get('conversation', ''))
    masked_conv = mask_conversation(raw_conv)

    rag_context = retriever.retrieve_examples(profile, k=RAG_K, max_chars=RAG_MAX_CHARS)

    user_content = USER_PROMPT_TEMPLATE.format(
        rag_context=rag_context,
        question=question,
        student_profile=profile,
        student_incorrect_solution=solution,
        conversation=masked_conv
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    
    full_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    return full_prompt, rag_context, user_content

def main():
    if not os.path.exists(BASE_MODEL_PATH):
        return
    if not os.path.exists(LORA_PATH):
        return

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True 
    )
    '''
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model = model.merge_and_unload()
    '''
    model.eval()

    retriever = RAGRetriever(DATA_PATH)

    try:
        df = pd.read_csv(DATA_PATH, encoding='gbk') 
    except:
        df = pd.read_csv(DATA_PATH, encoding='utf-8')

    if len(df) > SAMPLE_SIZE:
        df_sample = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    else:
        df_sample = df.copy()

    all_prompts = []
    rag_contexts = []
    user_contents = [] 
    
    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
        full_prompt, rag_ctx, user_text = create_llama3_prompt(row, retriever, tokenizer)
        all_prompts.append(full_prompt)
        rag_contexts.append(rag_ctx)
        user_contents.append(user_text)

    results = []
    
    model_device = next(model.parameters()).device
    
    for i in tqdm(range(0, len(all_prompts), BATCH_SIZE)):
        batch_prompts = all_prompts[i:i + BATCH_SIZE]
        
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=80000
        ).to(model_device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,  
                do_sample=True,      
                temperature=0.7,    
                top_p=0.9,          
                eos_token_id=tokenizer.eos_token_id,
            )
        
        input_lengths = inputs['input_ids'].shape[1]
        batch_responses = [
            tokenizer.decode(out[input_lengths:], skip_special_tokens=True).strip() 
            for out in outputs
        ]
        results.extend(batch_responses)

    output_df = df_sample.copy()

    output_df['model_output'] = results 
    
    output_df['rag_context'] = rag_contexts
    output_df['full_prompt_for_judge'] = user_contents 
    output_df['full_llama3_prompt'] = all_prompts     

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    output_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    main()
