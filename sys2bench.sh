#!/bin/bash

scripts=(
    "methods/CoT/AQuA/cot.sh"
    "methods/CoT/binpacking/cot.sh"
    "methods/CoT/blocksworld/cot.sh"
    "methods/CoT/calendarplan/cot.sh"
    "methods/CoT/cube/cot.sh"
    "methods/CoT/game24/cot.sh"
    "methods/CoT/gsm8k/cot.sh"
    "methods/CoT/hotpotQA/cot.sh"
    "methods/CoT/prontoqa/cot.sh"
    "methods/CoT/strategyQA/cot.sh"
    "methods/CoT/tripplan/cot.sh"

    "methods/IO/aqua/o1.sh"
    "methods/IO/binpacking/o1.sh"
    "methods/IO/blocksworld/o1mini.sh"
    "methods/IO/calendarplan/o1.sh"
    "methods/IO/cube/o1.sh"
    "methods/IO/game24/o1.sh"
    "methods/IO/gsm8k/o1.sh"
    "methods/IO/hotpotQA/o1.sh"
    "methods/IO/prontoqa/o1.sh"
    "methods/IO/strategyQA/o1.sh"
    "methods/IO/tripplan/o1.sh"

    "methods/RAP/AQuA/rap.sh"
    "methods/RAP/binpacking/rap.sh"
    "methods/RAP/blocksworld/rap.sh"
    "methods/RAP/game24/rap.sh"
    "methods/RAP/gsm8k/rap.sh"
    "methods/RAP/prontoqa/rap.sh"
    "methods/RAP/strategyQA/rap.sh"

    "methods/ToT/AQuA/tot.sh"
    "methods/ToT/binpacking/tot.sh"
    "methods/ToT/blocksworld/tot.sh"
    "methods/ToT/calendarplan/tot.sh"
    "methods/ToT/cube/tot.sh"
    "methods/ToT/game24/tot.sh"
    "methods/ToT/gsm8k/tot.sh"
    "methods/ToT/prontoqa/tot.sh"
    "methods/ToT/strategyQA/tot.sh"
    "methods/ToT/tripplan/tot.sh"
)

for script in "${scripts[@]}"; do
    if [[ -f "$script" && -x "$script" ]]; then
        echo "Running $script..."
        "$script"
        echo "$script completed."
    fi
done

echo "Sys2Bench complete."