# RLLM_Project
Project of Jan-Lorenz-Wirth and Marvin Schumann


Logical CAPTCHA Benchmark for Large Language Models

This project evaluates how reliably large language models (LLMs) solve synthetic logical CAPTCHA tasks.
The benchmark focuses on simple reasoning problems that humans can solve easily but may expose weaknesses in LLM reasoning.

# Examples of tasks include:
    - counting vowels
    - symbolic pattern reasoning
    - graph traversal
    - simple logical puzzles
    - rule-based symbol manipulation
    - The goal is to compare different inference strategies and models to analyze when LLMs succeed or fail.


# Dataset: 
    The data consists of synthetic logical CAPTCHA tasks stored in data/dataset.json
    Example Entry:

    |   {
    |       "id": "1",
    |       "task_type": "vowel_count",
    |       "question": "Count the vowels in the string: dgfdsfhuipoai",
    |       "answer": "5"
    |   }

# Inference Strategies

Direct Answering
    The model is asked to output only the final answer.

Chain-of-Thought (CoT)
    The model is encouraged to reason step-by-step before answering.

Self-Consistency
    Self-consistency runs the reasoning process multiple times and selects the majority answer.


# Running the experiment
    You can run the experiment using the simple bash-script:
        ./run_all.sh

    This generates six result files:
        results_direct_gpt-5.1.json
        results_cot_gpt-5.1.json
        results_self_consistency_gpt-5.1.json
        results_direct_gpt-4o-mini.json
        results_cot_gpt-4o-mini.json
        results_self_consistency_gpt-4o-mini.json
    

# Evaluation
    Results are analysed in in analysis.ipynb
    The notebook performs:
        - accuracy comparison across strategies
        - model comparison
        - task type difficulty analysis
        - latency comparison
        - error analysis
        - self-consistency stability analysis
        - ...


# Limitations
    This benchmark uses synthetic tasks and therefore does not represent the full complexity of real-world reasoning.
    However, the controlled setup allows systematic comparison of reasoning strategies.


# Authors
    Project developed as part of a course on **Reasoning with Large Language Models (RLLM)** 
    by Jan-Lorenz Wirth and Marvin Schumann