import time
import json
import pandas as pd
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-1.7B"
PROXY_JSON = "time_proxy_model.json"

# Target training batch size used by Aditya Sahu's training script
TARGET_TRAIN_BATCH_SIZE = 4
MAX_SEQ_LEN = 2048

def format_odia_qa_prompt(passage, question, answer, tokenizer):
    """
    Formats an Odia QA example using Qwen3 non-thinking chat template.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions in Odia based strictly on the provided passage."
        },
        {
            "role": "user",
            "content": f"Passage:\n{passage}\n\nQuestion:\n{question}"
        },
        {
            "role": "assistant",
            "content": str(answer)
        }
    ]
    # Explicitly disable thinking mode as required by project spec
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )
    return formatted_text

def predict_sample_time_ms(seq_len, batch_size, coeffs):
    """
    Computes amortized per-sample training time (ms) for an example of length L
    trained at batch_size B.
    """
    L = min(seq_len, MAX_SEQ_LEN)
    B = batch_size
    step_time_ms = (
        coeffs["beta_0"]
        + coeffs["beta_B"] * B
        + coeffs["beta_BL"] * (B * L)
        + coeffs["beta_BL2"] * (B * (L ** 2))
    )
    return step_time_ms / B

def score_dataset(input_file, output_file, passage_col="context", question_col="question", answer_col="answer"):
    with open(PROXY_JSON, "r") as f:
        proxy_meta = json.load(f)
    coeffs = proxy_meta["coefficients"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Load dataset (supports .csv, .json, or .jsonl from Pratyush's split)
    if input_file.endswith(".jsonl"):
        df = pd.read_json(input_file, lines=True)
    elif input_file.endswith(".json"):
        df = pd.read_json(input_file)
    elif input_file.endswith(".parquet"):
        df = pd.read_parquet(input_file)
    else:
        df = pd.read_csv(input_file)

    start_wall_time = time.perf_counter()

    token_lengths = []
    time_costs_ms = []

    for _, row in df.iterrows():
        text = format_odia_qa_prompt(
            row.get(passage_col, ""),
            row.get(question_col, ""),
            row.get(answer_col, ""),
            tokenizer
        )
        # Compute exact token count
        tok_len = len(tokenizer.encode(text, truncation=True, max_length=MAX_SEQ_LEN))
        cost_ms = predict_sample_time_ms(tok_len, TARGET_TRAIN_BATCH_SIZE, coeffs)

        token_lengths.append(tok_len)
        time_costs_ms.append(round(cost_ms, 4))

    proxy_calc_duration_sec = time.perf_counter() - start_wall_time

    df["token_length"] = token_lengths
    df["predicted_time_cost_ms"] = time_costs_ms

    # Save enriched dataset
    if output_file.endswith(".jsonl"):
        df.to_json(output_file, orient="records", lines=True, force_ascii=False)
    elif output_file.endswith(".parquet"):
        df.to_parquet(output_file, index=False) # Add this line
    else:
        df.to_csv(output_file, index=False)

    print(f"Scored {len(df)} examples in {proxy_calc_duration_sec:.3f} seconds.")
    print(f"Average Token Length: {sum(token_lengths)/len(token_lengths):.1f} tokens")
    print(f"Total Predicted Epoch Training Time: {sum(time_costs_ms)/1000:.2f} seconds")
    print(f"Saved cost-annotated dataset to {output_file}")

if __name__ == "__main__":
    # Demo test if Pratyush's file is not yet in the directory
    import os
    sample_file = "train_synthetic_odia.csv"
    if not os.path.exists(sample_file):
        print("Creating a small dummy Odia QA CSV to verify the cost scoring pipeline...")
        dummy_df = pd.DataFrame([
            {"context": "ଓଡ଼ିଶା ଭାରତର ପୂର୍ବ ଉପକୂଳରେ ଅବସ୍ଥିତ ଏକ ରାଜ୍ୟ। ଭୁବନେଶ୍ୱର ଏହାର ରାଜଧାନୀ।", "question": "ଓଡ଼ିଶାର ରାଜଧାନୀ କଣ?", "answer": "ଭୁବନେଶ୍ୱର"},
            {"context": "କୋଣାର୍କ ସୂର୍ଯ୍ୟ ମନ୍ଦିର ୧୩ଶ ଶତାବ୍ଦୀରେ ରାଜା ନରସିଂହ ଦେବଙ୍କ ଦ୍ୱାରା নির্মિત ହୋଇଥିଲା।", "question": "କୋଣାର୍କ ମନ୍ଦିର କିଏ ନିର୍ମାଣ କରିଥିଲେ?", "answer": "ରାଜା ନରସିଂହ ଦେବ"}
        ])
        dummy_df.to_csv(sample_file, index=False)

    score_dataset(
        input_file="../data_and_scripts/odia_synth_cleaned/or_synthetic_filtered.parquet", 
        output_file="../data_and_scripts/odia_synth_cleaned/or_synthetic_filtered_with_time_cost.parquet",
        
        passage_col="context", 
        question_col="question",
        answer_col="answer"
    )