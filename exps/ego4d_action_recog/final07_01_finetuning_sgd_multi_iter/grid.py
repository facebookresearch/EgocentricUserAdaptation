import subprocess
import os

this_file_path = os.path.dirname(os.path.realpath(__file__))
script_to_run = os.path.join(
    this_file_path,
    "ego4d_action_recog.sh"
)

# CONFIGS
static_config = {
    "GRID_RESUME_LATEST": False,
    "NUM_USERS_PER_DEVICE": 1,
}

run_name_list = ["SOLVER.BASE_LR", "TRAIN.INNER_LOOP_ITERS", "GPU_IDS"]
grid_nodes_line = ','.join([x for x in run_name_list if x not in ["GPU_IDS", "NUM_USERS_PER_DEVICE"]])

run_val_lists = (
    # ("1e-1", "2", "5"),
    # ("1e-1", "3", "5"),
    # ("1e-1", "5", "6"),
    ("1e-1", "10", "6"),
)

single_line_static = ' '.join([f"{k} {v}" for k, v in static_config.items()])
for run_val_list in run_val_lists:
    nameval_pairs = [f"{name} {val}" for name, val in zip(run_name_list, run_val_list)]
    single_line = ' '.join(nameval_pairs)

    cmd = f"{script_to_run} '{grid_nodes_line}' '{single_line} {single_line_static}'"  # 2 args
    print(f"Running: {cmd}")

    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)

# exit(0)
