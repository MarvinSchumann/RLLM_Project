#!/bin/bash

models=("gpt-5.1" "gpt-4o-mini")
strategies=("direct" "cot" "self_consistency")

for model in "${models[@]}"; do
  for strategy in "${strategies[@]}"; do
    echo "Running $model with $strategy"
    
    python run_experiment.py \
      --model $model \
      --strategy $strategy \
      --output results/results_${strategy}_${model}.json

  done
done