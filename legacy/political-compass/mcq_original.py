import os
import time
import json
import gc
import torch
import pandas as pd
import numpy as np
import multiprocessing as mp
from datetime import datetime
from scipy.stats import qmc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ==========================================
# 1. CONFIGURATION & RESOURCE MAP
# ==========================================

AVAILABLE_GPUS = [5, 6, 7] # list(range(8))

MODEL_DIR = "../modeli"

# Adjusted resource map for potential VRAM overhead
MODEL_RESOURCE_MAP = {
    'GaMS-9B-Instruct-Nemotron': (1, 1, 1),
    'GaMS-27B-Instruct-Nemotron': (2, 2, 1), # Increased for safety
    'Qwen3-32B': (2, 2, 1),
    'Qwen3-8B': (1, 1, 1),
    'Qwen3-14B': (1, 1, 1),
    'gemma-3-12b-it': (1, 1, 1),
    'gemma-3-27b-it': (2, 2, 1),
}

QUANTIZATIONS = ["bf16", "8bit", "4bit"]
BATCH_SIZE = 16 # Keep conservative
MAX_RUN_TIME_SECONDS = 3600 * 6.5
SAVE_DIR = "./output/results_3" # Changed folder name
SCHEDULE_PATH = "./output/experimental_design_3.csv"
SAMPLES_PER_BLOCK = 300

LANGUAGES = ["bulgarian", "czech", "english", "french", "german", "italian",
             "persian", "polish", "portuguese", "romanian", "russian",
             "slovene", "spanish", "turkish"]


MODELS = [

    'GaMS-27B-Instruct-Nemotron',
    'gemma-3-27b-it',
    'Qwen3-32B',
    'Qwen3-8B',
    'gemma-3-12b-it',
    'GaMS-9B-Instruct-Nemotron',
    'Qwen3-14B',
    'Qwen3-4B',
    'gemma-3-1b-it',
    'gemma-3-4b-it',
]

IDEOLOGIES = ["base", "libertarian_left", "libertarian_right",
              "authoritarian_left", "authoritarian_right", "centrism"]

KEY_TYPES = ["numeric", "numeric_zero_index", "lowercase_start", "uppercase_start"]

# ==========================================
# 2. EXPERIMENTAL DESIGN GENERATION
# ==========================================

def map_factors(sample):
    factors = {}
    factors['language'] = LANGUAGES[int(sample[0] * len(LANGUAGES))]
    # 50% chance of having context. If context, pick 1 of 5.
    if sample[1] < 0.5:
        factors['context_id'] = None
    else:
        ctx_idx = int((sample[1] - 0.5) * 2 * 5)
        factors['context_id'] = min(ctx_idx, 4)

    factors['instr_type'] = "question_first" if sample[2] < 0.5 else "options_first"
    factors['instr_idx'] = int(sample[3] * 5)
    factors['persona_class'] = "short" if sample[4] < 0.5 else "long"
    factors['persona_idx'] = int(sample[5] * 5)
    factors['key_type'] = KEY_TYPES[int(sample[6] * len(KEY_TYPES))]
    factors['perm_id'] = int(sample[7] * 4)
    return factors


def generate_experimental_design(samples_per_block=300, seed=42):
    sampler = qmc.LatinHypercube(d=8, seed=seed)
    base_block_samples = sampler.random(n=samples_per_block)
    master_records = []
    for model in MODELS:
        for ideology in IDEOLOGIES:
            for i in range(samples_per_block):
                config = map_factors(base_block_samples[i])
                config['model'] = model
                config['ideology'] = ideology
                if ideology == "base":
                    config['persona_class'] = None
                    config['persona_idx'] = None
                master_records.append(config)
    df = pd.DataFrame(master_records)
    df['config_id'] = range(len(df))
    return df

# ==========================================
# 3. PROMPT ASSEMBLY LOGIC
# ==========================================

def assemble_prompt_for_question(config_row, question_obj, prompts_dict):
    segments = []
    # 1. Context
    if pd.notna(config_row.get('context_id')):
        ctx_id = int(config_row['context_id'])
        try:
            segments.append(prompts_dict['contexts'][ctx_id]['text'])
        except (IndexError, KeyError):
            pass # Fallback safely

    # 2. Persona / Ideology
    if config_row['ideology'] != "base":
        p_class = config_row['persona_class']
        p_idx = int(config_row['persona_idx'])
        try:
            template = prompts_dict['persona_templates'][p_class][p_idx]
            insertions = prompts_dict['ideology_insertions'][config_row['ideology']][template['id']]
            persona_text = template['text'].format(**insertions)
            segments.append(persona_text)
        except (KeyError, IndexError):
            pass

    # 3. Instructions & Options
    p_id = int(config_row['perm_id'])
    rev_labels = p_id in [2, 3]
    rev_keys = p_id in [1, 3]

    instr_type = config_row['instr_type']
    instr_idx = int(config_row['instr_idx'])
    instr_template = prompts_dict['instructions'][instr_type][instr_idx]['text']

    labels = list(question_obj['choices'])
    # Get Keys (A,B,C,D or 1,2,3,4)
    keys = list(prompts_dict['experiment_settings']['answer_keys'][config_row['key_type']])

    # Apply Permutations (Debiasing)
    if rev_labels: labels = labels[::-1]
    if rev_keys: keys = keys[::-1]

    # Format Options (e.g. "A. Option Text")
    options_formatted = "\n" + "\n".join([f"{k}. {l}" for k, l in zip(keys, labels)])

    final_instr = instr_template.format(
        question=question_obj['statement'],
        options_formatted=options_formatted
    )
    segments.append(final_instr)

    full_prompt = "\n\n".join(segments)

    # Return prompt + ordered keys (so we know which logit corresponds to which visual option)
    return full_prompt, keys

# ==========================================
# 4. DATA LOADING & SAVING
# ==========================================

def load_all_prompts_and_questions():
    all_prompts, all_questions = {}, {}
    for lang in LANGUAGES:
        p_path, q_path = f'./data/prompts/{lang}.json', f'./data/questions/{lang}.json'
        if os.path.exists(p_path) and os.path.exists(q_path):
            with open(p_path, 'r', encoding='utf-8') as f: all_prompts[lang] = json.load(f)
            with open(q_path, 'r', encoding='utf-8') as f: all_questions[lang] = json.load(f)
    return all_prompts, all_questions

def get_csv_filepath(model_name, quant):
    safe_model = model_name.replace('/', '_')
    return os.path.join(SAVE_DIR, f"results_{safe_model}_{quant}.csv")

def load_completed_configs(model_name, quant):
    csv_path = get_csv_filepath(model_name, quant)
    if not os.path.exists(csv_path): return set()
    try:
        return set(pd.read_csv(csv_path, usecols=['config_id'])['config_id'].unique())
    except:
        return set()

def append_results_to_csv(model_name, quant, config_row, results_list):
    csv_path = get_csv_filepath(model_name, quant)

    # Base metadata row
    base_row = {
        'config_id': int(config_row['config_id']),
        'model': model_name, 'quantization': quant,
        'language': config_row['language'], 'ideology': config_row['ideology'],
        'context_id': int(config_row['context_id']) if pd.notna(config_row['context_id']) else -1,
        'instr_type': config_row['instr_type'],
        'instr_idx': int(config_row['instr_idx']),
        'persona_class': config_row['persona_class'] if pd.notna(config_row['persona_class']) else '',
        'persona_idx': int(config_row['persona_idx']) if pd.notna(config_row['persona_idx']) else -1,
        'key_type': config_row['key_type'],
        'perm_id': int(config_row['perm_id']),
    }

    # Flatten the results for this config (which contains N questions)
    # This creates ONE WIDE ROW per config.
    row = base_row.copy()
    for res in results_list:
        q_id = res['question_id']
        probs = res['probs'] # Dictionary of {Symbol: Probability}
        for ans_idx, (symbol, prob_val) in enumerate(probs.items()):
            # Using _prob_ instead of _logit_ to reflect softmax application
            row[f'q{q_id}_prob_ans{ans_idx}'] = prob_val

    df = pd.DataFrame([row])
    # Append to CSV
    df.to_csv(csv_path, index=False, mode='a', header=not os.path.exists(csv_path))

# ==========================================
# 5. WORKER PROCESS
# ==========================================

def worker_process(gpu_ids, model_name, quant, task_df, stop_event, all_prompts, all_questions):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    model_location = os.path.join(MODEL_DIR, model_name)
    print(f"[{datetime.now()}] STARTING: {model_name} | {quant} | GPUs {gpu_ids}")

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Load Configs
    bnb_config = None
    if quant == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    elif quant == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # elif quant == "8bit":
    #     bnb_config = BitsAndBytesConfig(
    #         load_in_8bit=True,
    #         bnb_8bit_compute_dtype=compute_dtype  # adds this line
    # )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_location, trust_remote_code=True)
        # Use RIGHT Padding for Generation/Batch Inference of next token
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Use Flash Attention 2 if hardware supports it (Ampere+)
        use_fa2 = torch.cuda.get_device_capability()[0] >= 8
        attn_impl = "flash_attention_2" if use_fa2 else "eager"

        # Load Model
        # Note: If bf16 original, we don't pass quantization_config
        load_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "attn_implementation": attn_impl,
            "torch_dtype": compute_dtype
        }
        if bnb_config:
            load_kwargs["quantization_config"] = bnb_config

        model = AutoModelForCausalLM.from_pretrained(model_location, **load_kwargs)
        model.eval()

    except Exception as e:
        print(f"[{datetime.now()}] LOAD FAILED {model_name}: {e}")
        return

    # Filter out already done work
    already_done = load_completed_configs(model_name, quant)
    remaining_df = task_df[~task_df['config_id'].isin(already_done)]

    print(f"[{datetime.now()}] Processing {len(remaining_df)} configurations for {model_name}")

    first_run_debug = True

    for _, config_row in remaining_df.iterrows():
        if stop_event.is_set(): break

        lang = config_row['language']
        # Fallbacks if lang not found
        if lang not in all_prompts: continue

        prompts_dict = all_prompts[lang]
        questions_list = all_questions[lang]

        batch_prompts = []
        batch_target_keys = [] # List of lists of symbols ['A','B','C','D']
        batch_q_ids = []

        # Prepare prompts for ALL questions in this configuration
        for q_obj in questions_list:
            p_text, keys = assemble_prompt_for_question(config_row, q_obj, prompts_dict)
            batch_prompts.append(p_text)
            batch_target_keys.append(keys)
            batch_q_ids.append(q_obj['id'])

        # Debug print once per worker to check prompt structure
        if first_run_debug:
            print(f"\n--- DEBUG PROMPT ({model_name}) ---\n{batch_prompts[0]}\n----------------------------\n")
            first_run_debug = False

        results = []

        # Process in sub-batches
        for i in range(0, len(batch_prompts), BATCH_SIZE):
            sub_prompts = batch_prompts[i : i + BATCH_SIZE]
            sub_keys = batch_target_keys[i : i + BATCH_SIZE]
            sub_q_ids = batch_q_ids[i : i + BATCH_SIZE]

            try:
                inputs = tokenizer(sub_prompts, return_tensors="pt", padding=True, truncation=True, max_length=3000).to(model.device)

                with torch.no_grad():
                    outputs = model(**inputs)

                # Get logits for the position immediately following the prompt
                last_token_indices = inputs.attention_mask.sum(1) - 1
                next_token_logits = outputs.logits[torch.arange(len(sub_prompts)), last_token_indices, :]

                for j, keys_list in enumerate(sub_keys):
                    # === EXACT MATCH TO SCRIPT A TOKENIZATION STRATEGY ===
                    candidate_ids = []
                    valid_keys = []

                    for k in keys_list:
                        # Script A logic: ids.append(tokenizer.encode(s)[-1])
                        # We do NOT add a space here. We encode 'k' exactly as is.
                        t_ids = tokenizer.encode(k, add_special_tokens=False)

                        if len(t_ids) > 0:
                            # Take the last token, exactly like Script A
                            candidate_ids.append(t_ids[-1])
                            valid_keys.append(k)
                        else:
                            # Edge case protection not in Script A, but necessary for stability
                            print(f"Warning: Key '{k}' produced no tokens.")

                    if not candidate_ids:
                        continue

                    # Extract logits for these specific tokens
                    relevant_logits = next_token_logits[j, candidate_ids]

                    # === EXACT MATCH TO SCRIPT A PROBABILITY CALCULATION ===
                    # Calculate Softmax across ONLY these candidate options
                    probs_tensor = torch.nn.functional.softmax(relevant_logits, dim=0)
                    probs_list = probs_tensor.cpu().float().numpy().tolist()

                    # Map back to keys
                    res_dict = {k: p for k, p in zip(valid_keys, probs_list)}

                    results.append({
                        'question_id': sub_q_ids[j],
                        'probs': res_dict
                    })

            except Exception as e:
                print(f"Error in batch inference: {e}")
                continue

        # Save to CSV
        # if results:
        #     append_results_to_csv(model_name, quant, config_row, results)

        expected_count = len(questions_list)
        if len(results) == expected_count:
            append_results_to_csv(model_name, quant, config_row, results)
        else:
            print(f"WARNING: config {config_row['config_id']} incomplete ({len(results)}/{expected_count}), skipping save.")

        # Cleanup
        # if i % (BATCH_SIZE * 5) == 0:
        gc.collect()
        torch.cuda.empty_cache()

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[{datetime.now()}] FINISHED: {model_name} | {quant}")



# ==========================================
# 6. ORCHESTRATOR
# ==========================================

def main():
    start_time = time.time()
    os.makedirs(SAVE_DIR, exist_ok=True)
    # Ensure spawn for CUDA compatibility in multiprocessing
    mp.set_start_method('spawn', force=True)
    active_processes = []

    if not os.path.exists(SCHEDULE_PATH):
        print("Generating new experimental design...")
        df_schedule = generate_experimental_design(samples_per_block=SAMPLES_PER_BLOCK)
        df_schedule.to_csv(SCHEDULE_PATH, index=False)
    else:
        print("Loading existing experimental design...")
        df_schedule = pd.read_csv(SCHEDULE_PATH)

    missing_models = [m for m in MODELS if m not in df_schedule['model'].unique()]
    if missing_models:
        print(f"Backfilling schedule for new models: {missing_models}")
        sampler = qmc.LatinHypercube(d=8, seed=42)
        base_samples = sampler.random(n=SAMPLES_PER_BLOCK)
        new_records = []
        for model in missing_models:
            for ideology in IDEOLOGIES:
                for i in range(SAMPLES_PER_BLOCK):
                    config = map_factors(base_samples[i])
                    config['model'] = model
                    config['ideology'] = ideology
                    if ideology == "base":
                        config['persona_class'] = None
                        config['persona_idx'] = None
                    new_records.append(config)
        new_df = pd.DataFrame(new_records)
        new_df['config_id'] = range(len(df_schedule), len(df_schedule) + len(new_df))
        df_schedule = pd.concat([df_schedule, new_df], ignore_index=True)
        df_schedule.to_csv(SCHEDULE_PATH, index=False)
        print(f"Schedule updated: {len(new_df)} new rows added.")

    all_prompts, all_questions = load_all_prompts_and_questions()

    # Create full workload (Models x Quants x Configs)
    workload = []
    for q in QUANTIZATIONS:
        temp_df = df_schedule.copy()
        temp_df['quant'] = q
        workload.append(temp_df)
    workload_df = pd.concat(workload, ignore_index=True)

    # Filter completed
    # This check is a bit slow on large CSVs, optimized inside worker generally,
    # but we do a quick group check here to avoid spawning useless processes.
    print("Checking progress...")

    manager = mp.Manager()
    stop_event = manager.Event()

    try:
        while time.time() - start_time < MAX_RUN_TIME_SECONDS:
            # Build grouped fresh each pass to catch incomplete configs
            grouped = list(workload_df.groupby(['model', 'quant'], sort=False))
            grouped = [
                (key, df) for key, df in grouped
                if load_completed_configs(key[0], key[1]) != set(df['config_id'].tolist())
            ]

            if not grouped:
                print("All work completed successfully.")
                break

            print(f"Starting pass with {len(grouped)} model+quant groups remaining...")

            active_processes = []
            free_gpus = list(AVAILABLE_GPUS)

            while (grouped or active_processes) and not stop_event.is_set():
                if time.time() - start_time > MAX_RUN_TIME_SECONDS:
                    print("Global timeout reached. Stopping.")
                    stop_event.set()

                for proc_info in active_processes[:]:
                    if not proc_info['proc'].is_alive():
                        print(f"Process finished: {proc_info['key']}")
                        free_gpus.extend(proc_info['gpus'])
                        active_processes.remove(proc_info)
                        free_gpus.sort()

                if not stop_event.is_set() and grouped:
                    for i, ((m_name, q_val), task_df) in enumerate(grouped):
                        req_gpus_map = MODEL_RESOURCE_MAP.get(m_name, (1, 1, 1))
                        q_idx = 0
                        if q_val == "8bit": q_idx = 1
                        elif q_val == "4bit": q_idx = 2
                        needed = req_gpus_map[q_idx]

                        if len(free_gpus) >= needed:
                            assigned = [free_gpus.pop(0) for _ in range(needed)]
                            p = mp.Process(
                                target=worker_process,
                                args=(assigned, m_name, q_val, task_df, stop_event, all_prompts, all_questions)
                            )
                            p.start()
                            active_processes.append({'proc': p, 'gpus': assigned, 'key': f"{m_name}|{q_val}"})
                            grouped.pop(i)
                            break
                    else:
                        time.sleep(5)

                time.sleep(2)

            if stop_event.is_set():
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
        stop_event.set()
    finally:
        print("Waiting for processes to exit...")
        for p_info in active_processes:
            p_info['proc'].join(timeout=10)
        print("Done.")

if __name__ == "__main__":
    main()