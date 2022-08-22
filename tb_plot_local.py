import os.path as osp
import os
import uuid
import subprocess


def run_tb(loglist, args):
    uid = uuid.uuid4()
    local_parent_out_path = osp.join('/tmp', 'mattdl', f'tensorboard_id_{uid}')
    os.makedirs(local_parent_out_path, exist_ok=False)
    print(f"Created parent path: {local_parent_out_path}")

    """
    alias ec2='ssh -i ~/.ssh/id_rsa matthiasdelange@ec2-3-92-75-17.compute-1.amazonaws.com'
    alias ec22='ssh -i ~/.ssh/id_rsa matthiasdelange@ec2-204-236-251-97.compute-1.amazonaws.com'
    """
    remote_user = "matthiasdelange"
    remote_host = "ec2-3-92-75-17.compute-1.amazonaws.com"
    # privkey_path = "~/.ssh/id_rsa" # Specified in ~/.ssh/config
    # subprocess.run(f"sshfs {remote_user}@{remote_host} -o IdentityFile={privkey_path}", shell=True)

    for entryname, remote_entrypath in loglist:
        for fb in [',', ':', ' ']:
            assert fb not in remote_entrypath
            if entryname is not None:
                entryname = entryname.replace(fb, '_')
                assert fb not in entryname, f"'{fb}' in name: {entryname}"

        # rsync interprets a directory with no trailing slash as copy this directory, and a directory with a trailing slash as copy the contents of this directory.
        remote_entrypath = remote_entrypath + '/'

        # Make local path and copy remote into it
        if entryname is not None:
            local_entrypath = osp.join(local_parent_out_path, entryname)
            os.makedirs(local_entrypath, exist_ok=True)
            print(f"Made local entry path: {local_entrypath}")
        else:
            local_entrypath = local_parent_out_path
            print(f"Assuming remote_entrypath is parentdir for entry-dirs: {remote_entrypath}")

        # Only include tensorboard out files.
        cmd = f'rsync -chavzPm --include="*/" --include="*events.out*" --exclude="*" --stats {remote_user}@{remote_host}:{remote_entrypath} {local_entrypath}'
        subprocess.run(cmd, shell=True)
        print(f"Copied remote to local: {cmd}")

        # tb_holder_path = osp.join(local_parent_out_path, entryname)
        # os.symlink(remote_entrypath, tb_holder_path, target_is_directory=True)

    subprocess.run(f"tensorboard --logdir={local_parent_out_path} --port {args.port}", shell=True)


############## CONFIGS ########################

""" Pretraining loss/acc curves for our user-pretrainsplit on ego4d. Validation=our 10-user trainsplit"""
validation_pretrain_nan_vs_nonan = [
    (
        "no-NAN-user_30_epochs",
        "/fb-agios-acai-efs/mattdl/ego4d_models/continual_ego4d_pretrained_models_usersplit/pretrain_147usersplit_excl_nan/2022-07-28_17-06-20_UIDe499d926-a3ff-4a28-9632-7d01054644fe/lightning_logs/version_0",
    ),

    ("NAN-user - 30 epochs",
     "/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/ego4d_action_recog/pretrain_slowfast/logs/2022-08-13_09-41-26_UID8196dadf-1ce7-4ed5-85c1-9bd3d1e6ffe6/lightning_logs/version_0"
     )
]

""" Testing run check for pretrain w.r.t. no-Nan user pretrain. """
test_pretrain_orig_ego4d_vs_nonan = [
    (
        "Orig_ego4d",
        # "/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/ego4d_action_recog/eval_pretrain_orig_ego4d_slowfast/logs/2022-08-09_17-48-34_UIDc3512861-ab9b-476f-8d25-fa71aff9db11/lightning_logs/version_0"
        "/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/ego4d_action_recog/eval_pretrain_orig_ego4d_slowfast/logs/2022-08-15_18-39-02_UIDf65471cf-bd1c-4632-b3d7-1c29c6e38b69/lightning_logs/version_0"
        # Incl Action-paths
    ),

    (
        "nonan_pretrain",
        # "/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/ego4d_action_recog/eval_pretrain_usersplit_train_excl_nan_ego4d_slowfast/logs/2022-08-09_18-20-47_UID9289c372-b6c0-43dc-a074-ee7560601511/lightning_logs/version_0"
        "/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/ego4d_action_recog/eval_pretrain_usersplit_train_excl_nan_ego4d_slowfast/logs/2022-08-15_18-41-21_UIDff77109f-6cab-43ab-9f43-feca9dac3560/lightning_logs/version_0"
        # Incl Actions testing
    )

]

exp01_01_finetuning = [
    # 100-sampling of future/past + fix Forgetting + speed-up small batch size multiple workers
    (None,  # Parent dir, users are entry-names
     "/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/ego4d_action_recog/exp01_01_finetuning/logs/2022-08-20_17-30-48_UID61c906b6-2f71-4d24-9dfa-60efa9b001bb/tb",
     # "/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/ego4d_action_recog/exp01_01_finetuning/logs/2022-08-20_12-18-08_UID505e310d-5db1-43c5-9af8-0d82395b8b0e/tb",
     # "/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/ego4d_action_recog/exp01_01_finetuning/logs/2022-08-20_11-44-11_UIDa889b3fc-e160-4ad6-b487-b3132c008911/tb",
     # "/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/ego4d_action_recog/exp01_01_finetuning/logs/2022-08-18_21-12-20_UIDe57eb203-32ac-4534-86db-20e166af80e4/tb"
     ),

    # New run, updated metrics (start orig ego4d, full stream eval)
    # (None,  # Parent dir, users are entry-names
    #  "/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/ego4d_action_recog/exp01_01_finetuning/logs/2022-08-18_21-12-20_UIDe57eb203-32ac-4534-86db-20e166af80e4/tb"
    #  ),

    #     # Old run that was interupted randomly
    # (None,
    #     "/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/ego4d_action_recog/exp01_01_finetuning/logs/2022-08-18_10-33-52_UID03c7c17a-b016-46b8-af7f-7f2db7c9618e/tb",
    #     # "/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/ego4d_action_recog/exp01_01_finetuning/logs/2022-08-17_15-08-28_UID9cdf57fa-deb8-4423-89e8-cfc007e020d0/tb"
    # ),
]

exp02_replay_fullmem = [
    (None,
     "/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/ego4d_action_recog/exp02_01_replay_unlimited/logs/2022-08-20_17-31-43_UID21f66e51-99a0-4ea9-90b5-0234b7adae67_GRID_METHOD-REPLAY-MEMORY_SIZE_SAMPLES=1000000_METHOD-REPLAY-STORAGE_POLICY=reservoir_action/tb",
     # "/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/ego4d_action_recog/exp02_01_replay_unlimited/logs/2022-08-20_17-31-43_UID21f66e51-99a0-4ea9-90b5-0234b7adae67_GRID_METHOD-REPLAY-MEMORY_SIZE_SAMPLES=1000000_METHOD-REPLAY-STORAGE_POLICY=reservoir_action/tb"
     # "/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/ego4d_action_recog/exp02_01_replay_unlimited/logs/2022-08-20_10-58-58_UIDb8ea4d6d-fb12-446a-9941-0db654eed34d_GRID_METHOD-REPLAY-MEMORY_SIZE_SAMPLES=1000000_METHOD-REPLAY-STORAGE_POLICY=reservoir_action/tb"
     ),
]


if __name__ == "__main__":
    """
    Run locally and define pairs of <NAME,REMOTE_DIRECT_TB_DIR> entries.
    The remote paths are copied locally and then plotted locally so no port-forwarding is required.
    """
    import argparse

    p = argparse.ArgumentParser(description="Tensorboard local viewer of remote results")
    p.add_argument('--port', type=int, help='Local port nb to host on',
                   default=6006)
    args = p.parse_args()

    run_tb(exp02_replay_fullmem, args)
