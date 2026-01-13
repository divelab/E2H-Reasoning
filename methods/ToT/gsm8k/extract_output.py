import os
import pickle
import world_model
import sys 
sys.path.append("/mnt/data/shared/blakeo/LM-Reasoning/Reasoning/reasoners")

folder_path = '/mnt/data/shared/blakeo/LM-Reasoning/Reasoning/logs/gsm8k/ToT/11052024-133819_openai-4omini/algo_output'
data = {}

# Loop through files 1.pkl to 11.pkl
for i in range(1, 12):
    file_path = os.path.join(folder_path, f"{i}.pkl")
    
    # Open each file and load the contents using pickle
    with open(file_path, 'rb') as file:
        try:
            content = pickle.load(file)
            data[i] = content  # Store content in the dictionary with the file number as key
        except Exception as e:
            print(f"Error loading {i}.pkl: {e}")

# Now `data` holds the contents of each .pkl file with file numbers as keys
print(data)