# masif_ppi_search_cache_triplet_data.py
import os
import numpy as np

from default_config.masif_opts import masif_opts

"""
This script caches training data for MaSIF-search using triplet format lists:
Each line: anchor positive negative
Each protein is stored in a subfolder named after its ID, with p1_*.npy feature files.
"""

# Load parameters
params = masif_opts['ppi_search']
parent_in_dir = params['masif_precomputation_dir']

with open(params['training_list']) as f:
    train_lines = f.readlines()

with open(params['testing_list']) as f:
    test_lines = f.readlines()

np.random.seed(0)
training_idx, test_idx = [], []

def load_features(protein_id, base_dir):
    folder = os.path.join(base_dir, protein_id)
    rho = np.load(os.path.join(folder, 'p1_rho_wrt_center.npy'))
    theta = np.load(os.path.join(folder, 'p1_theta_wrt_center.npy'))
    feat = np.load(os.path.join(folder, 'p1_input_feat.npy'))
    mask = np.load(os.path.join(folder, 'p1_mask.npy'))
    return rho, theta, feat, mask

def process_triplets(triplet_list, idx_offset):
    binder_rho, binder_theta, binder_feat, binder_mask = [], [], [], []
    pos_rho, pos_theta, pos_feat, pos_mask = [], [], [], []
    neg_rho, neg_theta, neg_feat, neg_mask = [], [], [], []
    idx_list = []

    for line in triplet_list:
        anchor, positive, negative = line.strip().split()

        try:
            a_rho, a_theta, a_feat, a_mask = load_features(anchor, parent_in_dir)
            p_rho, p_theta, p_feat, p_mask = load_features(positive, parent_in_dir)
            n_rho, n_theta, n_feat, n_mask = load_features(negative, parent_in_dir)
        except Exception as e:
            print(f"Skipping triplet due to error: {anchor}, {positive}, {negative} -> {e}")
            continue

        binder_rho.append(a_rho)
        binder_theta.append(a_theta)
        binder_feat.append(a_feat)
        binder_mask.append(a_mask)

        pos_rho.append(p_rho)
        pos_theta.append(p_theta)
        pos_feat.append(p_feat)
        pos_mask.append(p_mask)

        neg_rho.append(n_rho)
        neg_theta.append(n_theta)
        neg_feat.append(n_feat)
        neg_mask.append(n_mask)

        idx_list.append(idx_offset)
        idx_offset += 1

    return (
        binder_rho, binder_theta, binder_feat, binder_mask,
        pos_rho, pos_theta, pos_feat, pos_mask,
        neg_rho, neg_theta, neg_feat, neg_mask,
        idx_list, idx_offset
    )


idx_count = 0
(tr_b_rho, tr_b_theta, tr_b_feat, tr_b_mask,
 tr_p_rho, tr_p_theta, tr_p_feat, tr_p_mask,
 tr_n_rho, tr_n_theta, tr_n_feat, tr_n_mask,
 tr_idx, idx_count) = process_triplets(train_lines[0:2], idx_count)

(ts_b_rho, ts_b_theta, ts_b_feat, ts_b_mask,
 ts_p_rho, ts_p_theta, ts_p_feat, ts_p_mask,
 ts_n_rho, ts_n_theta, ts_n_feat, ts_n_mask,
 ts_idx, idx_count) = process_triplets(test_lines[0:2], idx_count)

# Combine all
binder_rho_wrt_center = np.concatenate(tr_b_rho + ts_b_rho, axis=0)
binder_theta_wrt_center = np.concatenate(tr_b_theta + ts_b_theta, axis=0)
binder_input_feat = np.concatenate(tr_b_feat + ts_b_feat, axis=0)
binder_mask = np.concatenate(tr_b_mask + ts_b_mask, axis=0)

pos_rho_wrt_center = np.concatenate(tr_p_rho + ts_p_rho, axis=0)
pos_theta_wrt_center = np.concatenate(tr_p_theta + ts_p_theta, axis=0)
pos_input_feat = np.concatenate(tr_p_feat + ts_p_feat, axis=0)
pos_mask = np.concatenate(tr_p_mask + ts_p_mask, axis=0)

neg_rho_wrt_center = np.concatenate(tr_n_rho + ts_n_rho, axis=0)
neg_theta_wrt_center = np.concatenate(tr_n_theta + ts_n_theta, axis=0)
neg_input_feat = np.concatenate(tr_n_feat + ts_n_feat, axis=0)
neg_mask = np.concatenate(tr_n_mask + ts_n_mask, axis=0)

training_idx = np.array(tr_idx)
test_idx = np.array(ts_idx)

# Save all
if not os.path.exists(params['cache_dir']):
    os.makedirs(params['cache_dir'])

np.save(os.path.join(params['cache_dir'], 'binder_rho_wrt_center.npy'), binder_rho_wrt_center)
np.save(os.path.join(params['cache_dir'], 'binder_theta_wrt_center.npy'), binder_theta_wrt_center)
np.save(os.path.join(params['cache_dir'], 'binder_input_feat.npy'), binder_input_feat)
np.save(os.path.join(params['cache_dir'], 'binder_mask.npy'), binder_mask)

np.save(os.path.join(params['cache_dir'], 'pos_rho_wrt_center.npy'), pos_rho_wrt_center)
np.save(os.path.join(params['cache_dir'], 'pos_theta_wrt_center.npy'), pos_theta_wrt_center)
np.save(os.path.join(params['cache_dir'], 'pos_input_feat.npy'), pos_input_feat)
np.save(os.path.join(params['cache_dir'], 'pos_mask.npy'), pos_mask)
np.save(os.path.join(params['cache_dir'], 'pos_training_idx.npy'), training_idx)
np.save(os.path.join(params['cache_dir'], 'pos_test_idx.npy'), test_idx)
# TODO: add validation set

np.save(os.path.join(params['cache_dir'], 'neg_rho_wrt_center.npy'), neg_rho_wrt_center)
np.save(os.path.join(params['cache_dir'], 'neg_theta_wrt_center.npy'), neg_theta_wrt_center)
np.save(os.path.join(params['cache_dir'], 'neg_input_feat.npy'), neg_input_feat)
np.save(os.path.join(params['cache_dir'], 'neg_mask.npy'), neg_mask)
np.save(os.path.join(params['cache_dir'], 'neg_training_idx.npy'), training_idx)
np.save(os.path.join(params['cache_dir'], 'neg_test_idx.npy'), test_idx)

print("Triplet data cached successfully.")