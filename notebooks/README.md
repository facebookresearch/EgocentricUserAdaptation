# Notebooks description

The Ego4D Long-term Action Anticipation (LTA) data that is labeled with actions.
The action annotations are linked to the meta-data through video_uid.
- ego4d_LTA_SINGLE_SPLITS_analysis.ipynb: Analyses single splits of train/val
- ego4d_LTA_MERGED_TRAINVAL_analysis.ipynb: Analyses merged data from train+val. 
This is eventually all labeled data that we can use for our LTA benchmark. 
The analysis is most extensive for this subset, both for actions and scenarios.
Note that functions are imported from the *ego4d_LTA_SINGLE_SPLITS_analysis.ipynb*.


The full Ego4D data (unlabeled + labeled), based on the meta-data:
- ego4d_METADATA_video_analysis.ipynb: Full ego4d dataset analysis of total and per-user video stats (nb videos, video minutes,...).
- ego4d_METADATA_scenario_analysis.ipynb: Analysis of distribution over scenarios of all videos.

# Paths

We use the Ego4D dataset on the server, accessible with the EC2 instances.
- Ego4D root: */fb-agios-acai-efs/Ego4D/*
- Meta data: */fb-agios-acai-efs/Ego4D/ego4d_data/ego4d.json*
- Annotations: */fb-agios-acai-efs/Ego4D/ego4d_data/v1/annotations*
- For the official Ego4D LTA benchmark description, see p71: https://arxiv.org/pdf/2110.07058.pdf 