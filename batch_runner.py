import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cluster', type=str, choices=['ut', 'tamu'], required=True)
args = parser.parse_args()

run_dict = {

    'math' : {
        'qwen15' : [
            'balanced',
            'cosine',
            'classic',
            {
                'gaussian' : [
                    {
                        'mu' : 0.25,
                        'sigma' : 0.75
                    },
                    {
                        'mu' : 0.5,
                        'sigma' : 0.5
                    },
                    {
                        'mu' : 0.75,
                        'sigma' : 0.25
                    },
                ]
            },
        ],
    },

    'mathlvl4' : {
        'qwen15' : [
            'balanced',
        ],
    },

    'mathlvl5' : {
        'qwen15' : [
            'balanced',
        ],
    },

    'aqua' : {
        'qwen15' : [
            'balanced',
            'cosine',
            'classic',
            {
                'gaussian' : [
                    {
                        'mu' : 0.25,
                        'sigma' : 0.75
                    },
                    {
                        'mu' : 0.5,
                        'sigma' : 0.5
                    },
                    {
                        'mu' : 0.75,
                        'sigma' : 0.25
                    },
                ]
            },
        ],
        'qwen15_base' : [
            'balanced',
            'cosine',
            'classic',
            {
                'gaussian' : [
                    {
                        'mu' : 0.25,
                        'sigma' : 0.75
                    },
                    {
                        'mu' : 0.5,
                        'sigma' : 0.5
                    },
                    {
                        'mu' : 0.75,
                        'sigma' : 0.25
                    },
                ]
            },
        ]
    },

    'gsm8k' : {
        'qwen15' : [
            'balanced',
            'cosine',
            'classic',
            {
                'gaussian' : [
                    {
                        'mu' : 0.25,
                        'sigma' : 0.75
                    },
                    {
                        'mu' : 0.5,
                        'sigma' : 0.5
                    },
                    {
                        'mu' : 0.75,
                        'sigma' : 0.25
                    },
                ]
            },
        ],
        'qwen15_base' : [
            'balanced',
            'cosine',
            'classic',
            {
                'gaussian' : [
                    {
                        'mu' : 0.25,
                        'sigma' : 0.75
                    },
                    {
                        'mu' : 0.5,
                        'sigma' : 0.5
                    },
                    {
                        'mu' : 0.75,
                        'sigma' : 0.25
                    },
                ]
            },
        ]
    },

}


run_commands = []
for task in run_dict:
    for model in run_dict[task]:
        for schedule in run_dict[task][model]:
            if isinstance(schedule, str):
                schedule_type = schedule
                run_command = f"sbatch run_hprc_{args.cluster}.slurm --model={model} --task={task} --schedule={schedule_type}"
                run_commands.append(run_command)
            elif isinstance(schedule, dict):
                for schedule_type, schedule_param_list in schedule.items():
                    for schedule_params in schedule_param_list:
                        run_command = f"sbatch run_hprc_{args.cluster}.slurm --model={model} --task={task} --schedule={schedule_type}"
                        for k,v in schedule_params.items():
                            run_command += f" --{k}={v}"
                        run_commands.append(run_command)

for run_command in run_commands:
    print("\n\n" + run_command)
    os.system(run_command)
    time.sleep(1)
                    