"""

First run 'run_summarize_streams.py' for 'train' and 'test' to get the clip-sampled stream from the annotation dataset.
Each sample is now a 2s clip.

For pretraining this clip-sampling is not required as normal iid training is performed.
To get the path for these annotations, run 'run_usersplit_ego4d_LTA.py' and get the path to the pretrain splits.

We then flatten all the jsons into 1 json that is user-agnostic, with the entire dataset pathlist under the 'clips' key.
This can be used for pretraining  on exactly the data we see during pretrain AND our train/test user streams.
"""
import pickle
import json
import os.path as osp
import numpy as np

# Config
INCLUDE_TEST_USERS = False

if INCLUDE_TEST_USERS:
    NUM_EXPECTED_USERS = 198  # 148 pretrain (incl non-assigned-user), 40 test, 10 train
    title = f"ego4d_ALL_DATA_pretrain_incl_nanusers_and_segmented_train_test_usersplit_{NUM_EXPECTED_USERS}users"
else:
    NUM_EXPECTED_USERS = 158  # 148 pretrain (incl non-assigned-user), 40 test, 10 train
    title = f"ego4d_ALL_DATA_pretrain_incl_nanusers_and_segmented_train_usersplit_{NUM_EXPECTED_USERS}users"

# OUT PATH
json_filepath_out = f'/fb-agios-acai-efs/mattdl/data/ego4d_lta_usersplits/{title}.json'

# INPUT PATHS
pretrain_unsegmented_json = '/fb-agios-acai-efs/mattdl/data/ego4d_lta_usersplits/2022-09-08_17-17-16_ego4d_LTA_usersplit/ego4d_LTA_pretrain_incl_nanusers_usersplit_148users.json'

# Missing clip_uid in summary for pretraining ego4d
# train_segmented_ckpt = "/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/ego4d_action_recog/summarize_streams/logs/2022-10-06_16-26-29_UID212c29c9-1ae7-470d-88c5-6f6653ba4fb0/dataset_entries_train_FEWSHOT=False_ego4d_LTA_train_usersplit_10users.ckpt"
# test_segmented_ckpt = "/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/ego4d_action_recog/summarize_streams/logs/2022-10-06_16-33-57_UID9a2cc977-ab47-4cda-af0f-2924662bbf06/dataset_entries_test_FEWSHOT=False_ego4d_LTA_test_usersplit_40users.ckpt"
train_segmented_ckpt = "/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/ego4d_action_recog/summarize_streams/logs/2022-10-07_04-49-02_UIDa5c4c52b-a8d8-4155-b1f4-bed9cd82374e/dataset_entries_train_FEWSHOT=False_ego4d_LTA_train_usersplit_10users.ckpt"
test_segmented_ckpt = "/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/ego4d_action_recog/summarize_streams/logs/2022-10-07_04-33-34_UIDd679068a-dc6e-40ff-b146-70ffe0671a97/dataset_entries_test_FEWSHOT=False_ego4d_LTA_test_usersplit_40users.ckpt"

# Segmented checkpoints are pickled
with open(train_segmented_ckpt, 'rb') as f:
    train_ds = pickle.load(f)

with open(test_segmented_ckpt, 'rb') as f:
    test_ds = pickle.load(f)

# JSON to dict for PRETRAIN
with open(pretrain_unsegmented_json, 'r') as f:
    pretrain_ds = json.load(f)['users']

# Flatten
final_dict = {}

# First collect in 1 dict make sure no user overlap
datasets = [train_ds, pretrain_ds]
if INCLUDE_TEST_USERS:
    datasets.append(test_ds)

for ds in datasets:
    for user, entries in ds.items():
        assert user not in final_dict, f"user {user} is duplicate!"
        final_dict[user] = entries

assert len(final_dict) == NUM_EXPECTED_USERS, f"Too few users {NUM_EXPECTED_USERS}"

# FLATTEN
final_ds_list = []

for user, entry_list in final_dict.items():
    print(f"Adding {len(entry_list)} entries for user {user}")
    final_ds_list.extend(entry_list)

print(f"TOTAL ENTRIES={len(final_ds_list)}")

final_ds = {'clips': final_ds_list, 'split': title}


# Write to JSON
class NpEncoder(json.JSONEncoder):
    """ Enable serializing numpy objects. """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


with open(json_filepath_out, 'w', encoding='utf-8') as f:
    json.dump(final_ds, f, ensure_ascii=False, indent=4, cls=NpEncoder)
    print(f"Saved JSON: {osp.abspath(json_filepath_out)}")

# Test if we can read them in again
with open(json_filepath_out, 'r') as f:
    json_obj = json.load(f)
print(f"TEST CHECK: Loaded json, with num clips={len(json_obj['clips'])}\n")
