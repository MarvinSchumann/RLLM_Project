import json
import time
import re
from openai import OpenAI

#CONFIG
MODEL = "gpt-5.1"
STRATEGY = "direct"
RESULT_PATH = "results/results_direct_gpt-5.1.json"
DATASET_PATH = "data/test_dataset.json"

TEMPERATURE = 0

#OpenAI_Client
client = OpenAI()

#--------------------------------------------------
#Prompt Builder
def build_promt(question, strategy = "direct"):
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

Lets think step by step.
"""
    else: raise ValueError("Unknown strategy")
#--------------------------------------------------

#Call Model 
def ask_model(prompt):
    response = client.responses.create(
        model = MODEL,
        temperature = TEMPERATURE,
        input = prompt
    )
    return response.output_text

#Parse Prediction
def parse_prediction(text):
    if text is None:
        return None
    
    text = text.strip()

    #in case the AI answers with more than the final word, we strip all other words:
    numbers = re.findall(r"-?\d+",text)
    if len(numbers) > 0:
        return numbers[-1]
    return text

#-------------------------------------------------------------
#Main Experiment:

def run():
    print("Loading dataset...")
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)

    results = []
    
    for task in dataset:
        question = task["question"]
        answer = str(task["answer"])

        promt = build_promt(question, STRATEGY)
        print(f"Running task{task['id']}")
        start = time.time()

        try:
            raw_output = ask_model(promt)
        except Exception as e:
            print("API error:", e)
            raw_output = None

        latency = time.time() - start
        prediction = parse_prediction(raw_output)
        correct = prediction == answer

        result= {
            "id":task["id"],
            "type": task.get("type"),
            "question": question,
            "ground_truth": answer,
            "model": MODEL,
            "strategy": STRATEGY,
            "raw_output": raw_output,
            "parsed_prediction": prediction,
            "correct": correct,
            "latency": latency
        }

        results.append(result)

        #small time break to prevent rate limits:
        time.sleep(0.2)
    
    print("Saving results...")

    with open(RESULT_PATH, "w") as f:
        json.dump(results,f,indent=2)
    print("done!")


if __name__ == "__main__":
    run()