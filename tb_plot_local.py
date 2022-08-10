import os.path as osp
import os
import uuid
import subprocess


def run_tb(loglist):
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
        for fb in [',', ':']:
            assert fb not in entryname
            assert fb not in remote_entrypath

        # rsync interprets a directory with no trailing slash as copy this directory, and a directory with a trailing slash as copy the contents of this directory.
        remote_entrypath = remote_entrypath + '/'

        # Make local path and copy remote into it
        local_entrypath = osp.join(local_parent_out_path, entryname)
        os.makedirs(local_entrypath, exist_ok=True)

        # Only include tensorboard out files.
        cmd = f'rsync -chavzPm --include="*/" --include="*events.out*" --exclude="*" --stats {remote_user}@{remote_host}:{remote_entrypath} {local_entrypath}'
        subprocess.run(cmd, shell=True)
        print(f"Copied remote to local: {cmd}")

        # tb_holder_path = osp.join(local_parent_out_path, entryname)
        # os.symlink(remote_entrypath, tb_holder_path, target_is_directory=True)

    subprocess.run(f"tensorboard --logdir={local_parent_out_path}", shell=True)


############## CONFIGS ########################

""" Pretraining loss/acc curves for our user-pretrainsplit on ego4d. Validation=our 10-user trainsplit"""
pretrain_nan_vs_nonan = [
    (
        "no-NAN-user",
        "/fb-agios-acai-efs/mattdl/ego4d_models/continual_ego4d_pretrained_models_usersplit/pretrain_147usersplit_excl_nan/2022-07-28_17-06-20_UIDe499d926-a3ff-4a28-9632-7d01054644fe/lightning_logs/version_0",
    ),

    # NAN is BUGGED (Is exact same as no-nan)
    # (
    #     "NAN-user",
    #     # "/fb-agios-acai-efs/mattdl/ego4d_models/continual_ego4d_pretrained_models_usersplit/pretrain_148usersplit_incl_nan/2022-08-07_10-58-41_UIDb107f026-abad-42bc-a66e-77442d07ef0a/lightning_logs/version_0",
    #     "/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/ego4d_action_recog/pretrain_slowfast/logs/2022-08-07_10-58-41_UIDb107f026-abad-42bc-a66e-77442d07ef0a/lightning_logs/version_0"
    # ),
]

""" Testing run check for pretrain w.r.t. no-Nan user pretrain. """
pretrain_orig_ego4d_vs_nonan = [
    (
        "Orig_ego4d",
        "/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/ego4d_action_recog/eval_pretrain_orig_ego4d_slowfast/logs/2022-08-09_17-48-34_UIDc3512861-ab9b-476f-8d25-fa71aff9db11/lightning_logs/version_0"
    ),

    (
        "nonan_pretrain",
        "/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/ego4d_action_recog/eval_pretrain_usersplit_train_excl_nan_ego4d_slowfast/logs/2022-08-09_18-20-47_UID9289c372-b6c0-43dc-a074-ee7560601511/lightning_logs/version_0"
    )
]

if __name__ == "__main__":
    """
    Run locally and define pairs of <NAME,REMOTE_DIRECT_TB_DIR> entries.
    The remote paths are copied locally and then plotted locally so no port-forwarding is required.
    """

    run_tb(pretrain_nan_vs_nonan)
