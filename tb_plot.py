if __name__ == "__main__":
    """
    EXAMPLE OUTPUT: tensorboard --logdir=LOGNAME:PATH,LOGNAME:PATH
    """
    loglist = [
        (
            "no-NAN-user",
            "/fb-agios-acai-efs/mattdl/ego4d_models/continual_ego4d_pretrained_models_usersplit/pretrain_147usersplit_excl_nan/2022-07-28_17-06-20_UIDe499d926-a3ff-4a28-9632-7d01054644fe/lightning_logs/version_0",
        ),
        (
            "NAN-user",
            "/fb-agios-acai-efs/mattdl/ego4d_models/continual_ego4d_pretrained_models_usersplit/pretrain_148usersplit_incl_nan/2022-08-07_10-58-41_UIDb107f026-abad-42bc-a66e-77442d07ef0a/lightning_logs/version_0",
        ),
    ]

    import os.path as osp
    import os
    import uuid
    import subprocess

    uid = uuid.uuid4()
    parent_out_path = osp.join('/tmp', 'mattdl', f'tensorboard_id_{uid}')
    os.makedirs(parent_out_path, exist_ok=False)
    print(f"Created parent path: {parent_out_path}")

    cmd = "tensorboard --logdir_spec "

    logdirs = []
    for entryname, entrypath in loglist:
        for fb in [',', ':']:
            assert fb not in entryname
            assert fb not in entrypath

        tb_holder_path = osp.join(parent_out_path, entryname)
        os.symlink(entrypath, tb_holder_path, target_is_directory=True)

    subprocess.run(f"tensorboard --logdir={parent_out_path}", shell=True)
