# Header variables and parameters.
import sys
import os
import numpy as np
import pandas as pd

from default_config.masif_opts import masif_opts

"""
masif_bsc_cache_training_data.py: TODO
"""

params = masif_opts['ppi_search']
parent_in_dir = params['masif_precomputation_dir']

# read training and testing list
train_df = pd.read_csv(params['training_list'])  # should contains：p1, p2, label (0/1)
test_df = pd.read_csv(params['testing_list'])

binder_rho_wrt_center, binder_theta_wrt_center, binder_input_feat, binder_mask = [], [], [], []
pos_rho_wrt_center, pos_theta_wrt_center, pos_input_feat, pos_mask = [], [], [], []
neg_rho_wrt_center, neg_theta_wrt_center, neg_input_feat, neg_mask = [], [], [], []

training_idx, test_idx = [], []
pos_names, neg_names = [], []

idx_count = 0

def load_sample(p1_id, p2_id, label, split):
    global idx_count

    try:
        in_dir1 = os.path.join(parent_in_dir, p1_id)
        in_dir2 = os.path.join(parent_in_dir, p2_id)

        # p1 is the binder in ppi_search, p2 is the target
        rho1 = np.load(os.path.join(in_dir1, 'p1_rho_wrt_center.npy'))
        theta1 = np.load(os.path.join(in_dir1, 'p1_theta_wrt_center.npy'))
        input_feat1 = np.load(os.path.join(in_dir1, 'p1_input_feat.npy'))
        mask1 = np.load(os.path.join(in_dir1, 'p1_mask.npy'))

        rho2 = np.load(os.path.join(in_dir2, 'p1_rho_wrt_center.npy'))
        theta2 = np.load(os.path.join(in_dir2, 'p1_theta_wrt_center.npy'))
        input_feat2 = np.load(os.path.join(in_dir2, 'p1_input_feat.npy'))
        mask2 = np.load(os.path.join(in_dir2, 'p1_mask.npy'))

        # add binder
        binder_rho_wrt_center.append(rho1)
        binder_theta_wrt_center.append(theta1)
        binder_input_feat.append(input_feat1)
        binder_mask.append(mask1)

        pair_name = f"{p1_id}_{p2_id}"

        if label == 1:
            pos_rho_wrt_center.append(rho2)
            pos_theta_wrt_center.append(theta2)
            pos_input_feat.append(input_feat2)
            pos_mask.append(mask2)
            pos_names.append(pair_name)
        else:
            neg_rho_wrt_center.append(rho2)
            neg_theta_wrt_center.append(theta2)
            neg_input_feat.append(input_feat2)
            neg_mask.append(mask2)
            neg_names.append(pair_name)

        # 保存 index
        if split == 'train':
            training_idx.append(idx_count)
        elif split == 'test':
            test_idx.append(idx_count)

        idx_count += 1

    except Exception as e:
        print(f"Error loading {p1_id}, {p2_id}: {e}")


# deal with trainig and testing set
for _, row in train_df.iterrows():
    load_sample(row['p1'], row['p2'], row['label'], 'train')

for _, row in test_df.iterrows():
    load_sample(row['p1'], row['p2'], row['label'], 'test')


# save in cache
if not os.path.exists(params['cache_dir']):
    os.makedirs(params['cache_dir'])

def save_npy(name, array):
    np.save(os.path.join(params['cache_dir'], name + '.npy'), array)

save_npy('binder_rho_wrt_center', np.array(binder_rho_wrt_center))
save_npy('binder_theta_wrt_center', np.array(binder_theta_wrt_center))
save_npy('binder_input_feat', np.array(binder_input_feat))
save_npy('binder_mask', np.array(binder_mask))

save_npy('pos_rho_wrt_center', np.array(pos_rho_wrt_center))
save_npy('pos_theta_wrt_center', np.array(pos_theta_wrt_center))
save_npy('pos_input_feat', np.array(pos_input_feat))
save_npy('pos_mask', np.array(pos_mask))
save_npy('pos_names', pos_names)

save_npy('neg_rho_wrt_center', np.array(neg_rho_wrt_center))
save_npy('neg_theta_wrt_center', np.array(neg_theta_wrt_center))
save_npy('neg_input_feat', np.array(neg_input_feat))
save_npy('neg_mask', np.array(neg_mask))
save_npy('neg_names', neg_names)

save_npy('pos_training_idx', training_idx)
save_npy('pos_test_idx', test_idx)
save_npy('neg_training_idx', training_idx)
save_npy('neg_test_idx', test_idx)

print(f"number of positives: {len(pos_names)} | negatives: {len(neg_names)}")
