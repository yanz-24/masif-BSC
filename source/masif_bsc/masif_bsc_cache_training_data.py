import sys
import os
import numpy as np
import importlib
from IPython.core.debugger import set_trace

from default_config.masif_opts import masif_opts

"""
masif_ppi_search_cache_triplet_data.py:
    Function to cache all training/validation/testing data for MaSIF-search based on triplet lists.
    Each triplet: (anchor, positive, negative), separated by space.
Pablo Gainza - LPDI STI EPFL 2019 (modified by user)
Released under an Apache License 2.0
"""

# Load configuration
params = masif_opts['ppi_search']
parent_in_dir = params['masif_precomputation_dir']

# Output containers
anchor_rho_wrt_center = []
anchor_theta_wrt_center = []
anchor_input_feat = []
anchor_mask = []

pos_rho_wrt_center = []
pos_theta_wrt_center = []
pos_input_feat = []
pos_mask = []

neg_rho_wrt_center = []
neg_theta_wrt_center = []
neg_input_feat = []
neg_mask = []

anchor_names = []
pos_names = []
neg_names = []

train_idx, val_idx, test_idx = [], [], []

# 读取 triplet 列表
def read_triplet_list(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    triplets = [line.strip().split() for line in lines]
    return triplets

print("Reading triplet files...")
training_triplets = read_triplet_list(params['training_triplets'])
validation_triplets = read_triplet_list(params['validation_triplets'])
testing_triplets = read_triplet_list(params['testing_triplets'])

print(f"Loaded {len(training_triplets)} training, {len(validation_triplets)} validation, {len(testing_triplets)} testing triplets.")

idx_count = 0

# Helper function to read the feature of a single point
def load_single_feature(in_dir, pid):
    rho = np.load(os.path.join(in_dir, f"{pid}_rho_wrt_center.npy"))
    theta = np.load(os.path.join(in_dir, f"{pid}_theta_wrt_center.npy"))
    feat = np.load(os.path.join(in_dir, f"{pid}_input_feat.npy"))
    mask = np.load(os.path.join(in_dir, f"{pid}_mask.npy"))
    return rho, theta, feat, mask

print("Processing triplets...")
def load_one_set_of_triplets(triplets, source):
    global idx_count
    for i, (anchor_name, positive_name, negative_name) in enumerate(triplets):
        # 处理文件名
        anchor_name = f'{anchor_name[:4]}_{anchor_name[4:]}'
        positive_name = f'{positive_name[:4]}_{positive_name[4:]}'
        negative_name = f'{negative_name[:4]}_{negative_name[4:]}'

        pid = 'p1'

        try:
            # anchor
            in_dir = os.path.join(parent_in_dir, anchor_name)
            rho, theta, feat, mask = load_single_feature(in_dir, pid)
            anchor_rho_wrt_center.append(rho)
            anchor_theta_wrt_center.append(theta)
            anchor_input_feat.append(feat)
            anchor_mask.append(mask)
            anchor_names.append(anchor_name)

            # positive
            in_dir = os.path.join(parent_in_dir, positive_name)
            rho, theta, feat, mask = load_single_feature(in_dir, pid)
            pos_rho_wrt_center.append(rho)
            pos_theta_wrt_center.append(theta)
            pos_input_feat.append(feat)
            pos_mask.append(mask)
            pos_names.append(positive_name)

            # negative
            in_dir = os.path.join(parent_in_dir, negative_name)
            rho, theta, feat, mask = load_single_feature(in_dir, pid)
            neg_rho_wrt_center.append(rho)
            neg_theta_wrt_center.append(theta)
            neg_input_feat.append(feat)
            neg_mask.append(mask)
            neg_names.append(negative_name)

            # 保存 idx
            if source == 'train':
                train_idx.append(idx_count)
            elif source == 'val':
                val_idx.append(idx_count)
            elif source == 'test':
                test_idx.append(idx_count)

            idx_count += 1
        except Exception as e:
            print(f"Error processing triplet {anchor_name} {positive_name} {negative_name}: {str(e)}")
            # set_trace()

load_one_set_of_triplets(training_triplets, 'train')
load_one_set_of_triplets(validation_triplets, 'val')
load_one_set_of_triplets(testing_triplets, 'test')
print("Finished processing all triplets.")

# 保存 npy 文件
if not os.path.exists(params['cache_dir']):
    os.makedirs(params['cache_dir'])

print("Saving to cache...")

# Convert to numpy arrays
anchor_rho_wrt_center = np.concatenate(anchor_rho_wrt_center, axis=0)
anchor_theta_wrt_center = np.concatenate(anchor_theta_wrt_center, axis=0)
anchor_input_feat = np.concatenate(anchor_input_feat, axis=0)
anchor_mask = np.concatenate(anchor_mask, axis=0)

pos_rho_wrt_center = np.concatenate(pos_rho_wrt_center, axis=0)
pos_theta_wrt_center = np.concatenate(pos_theta_wrt_center, axis=0)
pos_input_feat = np.concatenate(pos_input_feat, axis=0)
pos_mask = np.concatenate(pos_mask, axis=0)

neg_rho_wrt_center = np.concatenate(neg_rho_wrt_center, axis=0)
neg_theta_wrt_center = np.concatenate(neg_theta_wrt_center, axis=0)
neg_input_feat = np.concatenate(neg_input_feat, axis=0)
neg_mask = np.concatenate(neg_mask, axis=0)

train_idx = np.array(train_idx)
val_idx = np.array(val_idx)
test_idx = np.array(test_idx)

print(f"Read {len(neg_rho_wrt_center)} negative shapes")
print(f"Read {len(anchor_rho_wrt_center)} positive shapes")

# Save anchor (save as "binder_" for compatibility)
np.save(os.path.join(params['cache_dir'], 'binder_rho_wrt_center.npy'), anchor_rho_wrt_center)
np.save(os.path.join(params['cache_dir'], 'binder_theta_wrt_center.npy'), anchor_theta_wrt_center)
np.save(os.path.join(params['cache_dir'], 'binder_input_feat.npy'), anchor_input_feat)
np.save(os.path.join(params['cache_dir'], 'binder_mask.npy'), anchor_mask)
np.save(os.path.join(params['cache_dir'], 'binder_names.npy'), anchor_names)

# Save positive
np.save(os.path.join(params['cache_dir'], 'pos_rho_wrt_center.npy'), pos_rho_wrt_center)
np.save(os.path.join(params['cache_dir'], 'pos_theta_wrt_center.npy'), pos_theta_wrt_center)
np.save(os.path.join(params['cache_dir'], 'pos_input_feat.npy'), pos_input_feat)
np.save(os.path.join(params['cache_dir'], 'pos_mask.npy'), pos_mask)
np.save(os.path.join(params['cache_dir'], 'pos_names.npy'), pos_names)
np.save(os.path.join(params['cache_dir'], 'pos_training_idx.npy'), train_idx)
np.save(os.path.join(params['cache_dir'], 'pos_val_idx.npy'), val_idx)
np.save(os.path.join(params['cache_dir'], 'pos_test_idx.npy'), test_idx)

# Save negative
np.save(os.path.join(params['cache_dir'], 'neg_rho_wrt_center.npy'), neg_rho_wrt_center)
np.save(os.path.join(params['cache_dir'], 'neg_theta_wrt_center.npy'), neg_theta_wrt_center)
np.save(os.path.join(params['cache_dir'], 'neg_input_feat.npy'), neg_input_feat)
np.save(os.path.join(params['cache_dir'], 'neg_mask.npy'), neg_mask)
np.save(os.path.join(params['cache_dir'], 'neg_names.npy'), neg_names)
np.save(os.path.join(params['cache_dir'], 'neg_training_idx.npy'), train_idx)
np.save(os.path.join(params['cache_dir'], 'neg_val_idx.npy'), val_idx)
np.save(os.path.join(params['cache_dir'], 'neg_test_idx.npy'), test_idx)

print("Done. All cached.")
