from continual_ego4d.datasets.continual_action_recog_dataset import get_user_to_dataset_dict, extract_json

if __name__ == "__main__":
    # USER-SPLIT DATASETS
    train_json = "/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/forecasting/continual_ego4d/usersplit_data/2022-08-05_18-22-53_ego4d_LTA_usersplit/ego4d_LTA_train_usersplit_10users.json"
    test_json = "/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/forecasting/continual_ego4d/usersplit_data/2022-08-05_18-22-53_ego4d_LTA_usersplit/ego4d_LTA_test_usersplit_40users.json"

    # pretrain_json_excl_nanuser = "/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/forecasting/continual_ego4d/usersplit_data/2022-07-27_21-05-14_ego4d_LTA_usersplit/ego4d_LTA_pretrain_usersplit_147users.json"
    pretrain_json_incl_nanuser = "/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/forecasting/continual_ego4d/usersplit_data/2022-08-05_18-22-53_ego4d_LTA_usersplit/ego4d_LTA_pretrain_incl_nanusers_usersplit_148users.json"
    pretrain_json = pretrain_json_incl_nanuser

    train_ds = extract_json(train_json)['clips']
    test_ds = extract_json(test_json)['clips']
    pretrain_ds = extract_json(pretrain_json)['clips']

    print(f"USERSPLITS: train_len={len(train_ds)}, test_len={len(test_ds)},pretrain_ds={len(pretrain_ds)}")

    # Now load the ego4d pretrain
    ego4d_pretrain_json = "/fb-agios-acai-efs/Ego4D/ego4d_data/v1/annotations/fho_lta_train.json"

    ego4d_pretrain_ds = extract_json(ego4d_pretrain_json)['clips']
    print(f"EGO4d pretrain single epoch iterations: {len(ego4d_pretrain_ds)}")
    print(f"pretrainsplit / EGO4d ratio: {len(pretrain_ds)/len(ego4d_pretrain_ds)}")
