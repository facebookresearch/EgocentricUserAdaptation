# Tutorial: https://medium.com/analytics-vidhya/jupyterlab-on-aws-ec2-d6b2cb945e54
# Run this on the remote AWS EC2 instance.
# CONDA ENV_NAME=matt_ego4d
"""


# Level1 keys:['date', 'version', 'description', 'videos', 'clips', 'concurrent_video_sets', 'physical_settings', 'moments_labels']

# meta_data_obj['videos'][0].keys()
# dict_keys(['video_uid', 'duration_sec', 'scenarios', 'video_metadata', 'split_em', 'split_av', 'split_fho', 's3_path', 'manifold_path', 'origin_video_id', 'video_source', 'device', 'ph
# ysical_setting_name', 'fb_participant_id', 'is_stereo', 'has_imu', 'has_gaze', 'imu_metadata', 'gaze_metadata', 'video_components', 'concurrent_sets', 'has_redacted_regions', 'redacted
# _intervals', 'gaps'])
# -> fb_participant_id, duration_sec
"""

def get_meta_data_obj():
    import json
    meta_data_file_path = "/fb-agios-acai-efs/Ego4D/ego4d_data/ego4d.json"

    with open(meta_data_file_path, 'r') as meta_data_file:
        meta_data_obj = json.load(meta_data_file)
    return meta_data_obj

def get_key_info(meta_data_obj):
    l1_keys = [k for k in meta_data_obj.keys()]
    print(f"L1:{l1_keys}")

    for l1_key in l1_keys:
        subdict = meta_data_obj[l1_key]
        if isinstance(subdict, dict):
            print(f"{l1_key}:{subdict.keys()}")

    import pdb;
    pdb.set_trace()




def plot_nb_videos_per_user(out_dir='./out'):
    from collections import defaultdict
    data = get_meta_data_obj()

    videos = data['videos']

    user_video_count = defaultdict(int)
    for video in videos:
        user_id = int(video['fb_participant_id'])
        user_video_count[user_id] += 1


    user_video_count_s = sorted(list(user_video_count.values()))
    x_axis = [idx for idx in range(len(user_video_count_s))]

    # Plot

    # Save






def main():
    plot_nb_videos_per_user()


if __name__ == "__main__":
    main()
