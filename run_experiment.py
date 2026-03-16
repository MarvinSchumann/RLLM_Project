import json
import time
import re
import os
from collections import Counter
from openai import OpenAI

# ----------------------------------
# CONFIG
# ----------------------------------

MODEL = "gpt-4o-mini"
STRATEGY = "self_consistency"   # options: "direct", "cot", "self_consistency"
DATASET_PATH = "data/dataset.json"
RESULT_PATH = "results/results_self_consistency-gpt-4o-mini.json"

TEMPERATURE = 0
SELF_CONSISTENCY_RUNS = 10

# OpenAI Client

client = OpenAI()

#--------------------------------------------------
#Prompt Builder
def build_promt(question, strategy = "cot"):
    if strategy == "direct":
        return f"""
Solve the following task.
Question:
{question}

Answer with only the final answer.str
"""
    
    elif strategy == "cot":
        return f"""
Solve the following reasoning task step by step.

Question:
{question}

Lets think step by step. Make sure the last word is the result. If the result is a number, dont write it as the word, but give just the number! 
"""
    else: raise ValueError("Unknown strategy")
#--------------------------------------------------

# ----------------------------------
# Call Model
# ----------------------------------

def ask_model(prompt):
    response = client.responses.create(
        model = MODEL,
        temperature = TEMPERATURE,
        input = prompt
    )
    return response.output_text


# ----------------------------------
# Parse Prediction
# ----------------------------------

#Parse Prediction
def parse_prediction(text):
    if text is None:
        return None

    text = text.strip()

    if not text:
        return None

    # letzte Zeile nehmen
    last_line = text.split("\n")[-1].strip()

    # letztes Wort extrahieren
    tokens = last_line.split()

    if len(tokens) == 0:
        return None

    return tokens[-1]

#-------------------------------------------------------------


# Self-Consistency
def ask_self_consistency(question, runs=SELF_CONSISTENCY_RUNS):
    raw_outputs = []
    predictions = []

    for i in range(runs):
        prompt = build_promt(question, "cot")

        try:
            raw_output = ask_model(prompt)
        except Exception as e:
            print(f"API error in self-consistency run {i+1}: {e}")
            raw_output = None

        prediction = parse_prediction(raw_output)

        raw_outputs.append(raw_output)
        predictions.append(prediction)

        time.sleep(0.2)

    valid_predictions = [p for p in predictions if p is not None]

    if not valid_predictions:
        final_prediction = None
    else:
        counter = Counter(valid_predictions)
        final_prediction = counter.most_common(1)[0][0]

    return {
        "final_prediction": final_prediction,
        "all_predictions": predictions,
        "all_raw_outputs": raw_outputs
    }

# Main Experiment

def run():
    print("Loading dataset...")

    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)

    results = []

    for task in dataset:
        question = task["question"]
        answer = str(task["answer"])

        print(f"Running task {task['id']}")

        start = time.time()

        if STRATEGY == "self_consistency":
            sc_result = ask_self_consistency(question, runs=SELF_CONSISTENCY_RUNS)

            raw_output = None
            prediction = sc_result["final_prediction"]
            all_predictions = sc_result["all_predictions"]
            all_raw_outputs = sc_result["all_raw_outputs"]

        else:
            prompt = build_promt(question, STRATEGY)

            try:
                raw_output = ask_model(prompt)
            except Exception as e:
                print("API error:", e)
                raw_output = None

            prediction = parse_prediction(raw_output)
            all_predictions = None
            all_raw_outputs = None

            time.sleep(0.2)

        latency = time.time() - start
        correct = prediction == answer

        result = {
            "id": task["id"],
            "task_type": task.get("task_type"),
            "question": question,
            "ground_truth": answer,
            "model": MODEL,
            "strategy": STRATEGY,
            "temperature": TEMPERATURE,
            "raw_output": raw_output,
            "parsed_prediction": prediction,
            "correct": correct,
            "latency": latency
        }

        if STRATEGY == "self_consistency":
            result["self_consistency_runs"] = SELF_CONSISTENCY_RUNS
            result["all_predictions"] = all_predictions
            result["all_raw_outputs"] = all_raw_outputs

        results.append(result)

    print("Saving results...")

    os.makedirs("results", exist_ok=True)

    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Done! Results saved to: {RESULT_PATH}")

# Run

if __name__ == "__main__":
    run()