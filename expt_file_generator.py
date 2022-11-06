
import json

with open("tasks.csv", "r") as fh:
    task_data = fh.readlines()

expt_params = {"name": "Parkour Circuit UQ",
               "N": 1,
               "camera index": 0,
               "outdir": "data",
               "nontaskprefix": "unlabelled",
               "taskprefix": "task",
               "tasks": [[_td.split(',')[1].replace('"', ''),
                          [_str.replace('"', '') for _str in _td.split(',')[2:4]]]
                         for _td in task_data[2:]
                         if len(_td.split(',')[1].replace('"', '')) != 0]}

# Generate task dict.
with open("expt_params.json", "w") as fh:
    json.dump(expt_params, fh, indent=4)