from continual_ego4d.datasets.continual_action_recog_dataset import extract_json
import pickle


def get_pretrain_sets_iterations_vs_epoch_ratio():
    """ego4d=30 epochs pretrain but has a lot more data than our pretrain set.
    See what the ratio is that we can use to increase our nb of epochs.
    """
    ORIG_EPOCHS = 30

    # SEGMENTED (ALREADY CLIP-SAMPLED) DATA IN THE USER STREAMS:
    train_segmented_ckpt = "/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/ego4d_action_recog/summarize_streams/logs/2022-10-07_04-49-02_UIDa5c4c52b-a8d8-4155-b1f4-bed9cd82374e/dataset_entries_train_FEWSHOT=False_ego4d_LTA_train_usersplit_10users.ckpt"
    test_segmented_ckpt = "/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/ego4d_action_recog/summarize_streams/logs/2022-10-07_04-33-34_UIDd679068a-dc6e-40ff-b146-70ffe0671a97/dataset_entries_test_FEWSHOT=False_ego4d_LTA_test_usersplit_40users.ckpt"

    # Segmented checkpoints are pickled
    with open(train_segmented_ckpt, 'rb') as f:
        train_segmented_user_ds = pickle.load(f)

    with open(test_segmented_ckpt, 'rb') as f:
        test_segmented_user_ds = pickle.load(f)

    train_segmented_ds = []
    for user, user_ds in train_segmented_user_ds.items():
        train_segmented_ds.extend(user_ds)

    test_segmented_ds = []
    for user, user_ds in test_segmented_user_ds.items():
        test_segmented_ds.extend(user_ds)

    print(
        f"SEGMENTED (2S CLIPS IN STREAMS): Train users total = {len(train_segmented_ds)}, test = {len(test_segmented_ds)}")

    # USER-SPLIT DATASETS
    train_json = '/fb-agios-acai-efs/mattdl/data/ego4d_lta_usersplits/2022-09-08_17-17-16_ego4d_LTA_usersplit/ego4d_LTA_train_usersplit_10users.json'
    test_json = '/fb-agios-acai-efs/mattdl/data/ego4d_lta_usersplits/2022-09-08_17-17-16_ego4d_LTA_usersplit/ego4d_LTA_test_usersplit_40users.json'

    # pretrain_json_excl_nanuser = "/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/forecasting/continual_ego4d/usersplit_data/2022-07-27_21-05-14_ego4d_LTA_usersplit/ego4d_LTA_pretrain_usersplit_147users.json"
    pretrain_json_incl_nanuser = '/fb-agios-acai-efs/mattdl/data/ego4d_lta_usersplits/2022-09-08_17-17-16_ego4d_LTA_usersplit/ego4d_LTA_pretrain_incl_nanusers_usersplit_148users.json'
    pretrain_json = pretrain_json_incl_nanuser

    # For pretrain baseline
    merged_json = "/fb-agios-acai-efs/mattdl/data/ego4d_lta_usersplits/ego4d_ALL_DATA_pretrain_incl_nanusers_and_segmented_train_test_usersplit_198users.json"

    train_ds = extract_json(train_json)['clips']
    test_ds = extract_json(test_json)['clips']
    pretrain_ds = extract_json(pretrain_json)['clips']
    merged_ds = extract_json(merged_json)['clips']
    print(
        f"USERSPLITS (full annotation timeranges, split per user): train_len={len(train_ds)}, test_len={len(test_ds)},pretrain_ds={len(pretrain_ds)}, merged_ds={len(merged_ds)}")

    # Now load the original ego4d pretrain which we use as a reference for nb of iterations training
    ego4d_pretrain_json = "/fb-agios-acai-efs/Ego4D/ego4d_data/v1/annotations/fho_lta_train.json"

    ego4d_pretrain_ds = extract_json(ego4d_pretrain_json)['clips']
    print(f"EGO4d pretrain single epoch iterations: {len(ego4d_pretrain_ds)}")

    ratio = len(pretrain_ds) / len(ego4d_pretrain_ds)
    print(f"pretrainsplit / EGO4d ratio: {ratio}. EPOCHS: {ORIG_EPOCHS} -> {ORIG_EPOCHS / ratio}")

    ratio = len(merged_ds) / len(ego4d_pretrain_ds)
    print(f"merged_ds / EGO4d ratio: {ratio}. EPOCHS: {ORIG_EPOCHS} -> {ORIG_EPOCHS / ratio}")


if __name__ == "__main__":
    get_pretrain_sets_iterations_vs_epoch_ratio()
