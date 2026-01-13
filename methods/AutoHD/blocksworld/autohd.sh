### Run autohd with heuristic evolution using openai GPT 4o-mini
python methods/AutoHD/blocksworld/heuristic_search.py --temperature 0.8 --base_lm openai --openai_model gpt-4o-mini

### Run autohd with heuristic evolution using openai GPT 4o
python methods/AutoHD/blocksworld/heuristic_search.py --temperature 0.8 --base_lm openai --openai_model gpt-4o

### Run autohd with heuristic evolution using Llama-3.1-70B
python methods/AutoHD/blocksworld/heuristic_search.py --temperature 0.8 --base_lm llamaapi --api_model_id meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo

### Test with the best heuristic function from the log file using openai GPT 4o-mini
python methods/AutoHD/blocksworld/inference.py --heuristic_log_file methods/AutoHD/blocksworld/bw_HeuristicSearch-4o-mini.log --temperature 0.8 --base_lm openai --openai_model gpt-4o-mini
