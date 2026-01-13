#!/bin/bash

# VREx Training Queue - Automatically launch remaining configs when GPUs are available
# Monitors for 2 GPUs with 60GB+ free memory each

cd /data/shurui.gui/Projects/gateway/Sys2Bench

echo "VREx Training Queue Started - $(date)"
echo "Monitoring for GPU availability..."

check_gpu_memory() {
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | while read line; do
        gpu_id=$(echo $line | cut -d',' -f1)
        free_mem=$(echo $line | cut -d',' -f2 | tr -d ' ')
        if [ $free_mem -gt 60000 ]; then
            echo $gpu_id
        fi
    done
}

launch_config3() {
    echo "Launching VREx Config 3 (Conservative) at $(date)"
    bash methods/RL/scripts/launch_vrex_config3_1600.sh
    echo "Config 3 launched successfully"
}

launch_config4() {
    echo "Launching VREx Config 4 (High Variance Penalty) at $(date)"
    bash methods/RL/scripts/launch_vrex_config4_1600.sh  
    echo "Config 4 launched successfully"
}

config3_launched=false
config4_launched=false

while [ "$config3_launched" = false ] || [ "$config4_launched" = false ]; do
    available_gpus=($(check_gpu_memory))
    num_available=${#available_gpus[@]}
    
    echo "Available GPUs with 60GB+ free: ${available_gpus[*]} (total: $num_available)"
    
    if [ $num_available -ge 2 ]; then
        if [ "$config3_launched" = false ]; then
            # Modify config3 script to use available GPUs
            sed -i "s/CUDA_VISIBLE_DEVICES=3,4/CUDA_VISIBLE_DEVICES=${available_gpus[0]},${available_gpus[1]}/" methods/RL/scripts/launch_vrex_config3_1600.sh
            launch_config3
            config3_launched=true
            sleep 30  # Give time for GPU allocation
        elif [ "$config4_launched" = false ]; then
            # Modify config4 script to use available GPUs  
            sed -i "s/CUDA_VISIBLE_DEVICES=4,7/CUDA_VISIBLE_DEVICES=${available_gpus[0]},${available_gpus[1]}/" methods/RL/scripts/launch_vrex_config4_1600.sh
            launch_config4
            config4_launched=true
            break
        fi
    else
        echo "Waiting for GPU availability... (need 2 GPUs with 60GB+ free)"
        sleep 60
    fi
done

echo "All VREx configurations launched successfully at $(date)"
echo "Monitor progress with: tail -f methods/RL/logs/vrex_config*_1600.log"