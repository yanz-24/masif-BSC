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

# 标记来源
all_triplets = []
all_sources = []
for triplet in training_triplets:
    all_triplets.append(triplet)
    all_sources.append('train')
for triplet in validation_triplets:
    all_triplets.append(triplet)
    all_sources.append('val')
for triplet in testing_triplets:
    all_triplets.append(triplet)
    all_sources.append('test')

print(f"Loaded {len(training_triplets)} training, {len(validation_triplets)} validation, {len(testing_triplets)} testing triplets.")

idx_count = 0

# Helper函数，读取单个点的feature
def load_single_feature(in_dir, pid, index):
    rho = np.load(os.path.join(in_dir, f"{pid}_rho_wrt_center.npy"))[int(index)]
    theta = np.load(os.path.join(in_dir, f"{pid}_theta_wrt_center.npy"))[int(index)]
    feat = np.load(os.path.join(in_dir, f"{pid}_input_feat.npy"))[int(index)]
    mask = np.load(os.path.join(in_dir, f"{pid}_mask.npy"))[int(index)]
    return rho, theta, feat, mask

print("Processing triplets...")
for i, (anchor_name, positive_name, negative_name) in enumerate(all_triplets):
    try:
        # anchor
        fields = anchor_name.split('_') #TODO: 这里的分割方式需要调整
        ppi_id = fields[0]
        pid = fields[1]
        idx = fields[2]
        in_dir = os.path.join(parent_in_dir, ppi_id)
        rho, theta, feat, mask = load_single_feature(in_dir, pid, idx)
        anchor_rho_wrt_center.append(rho)
        anchor_theta_wrt_center.append(theta)
        anchor_input_feat.append(feat)
        anchor_mask.append(mask)
        anchor_names.append(anchor_name)

        # positive
        fields = positive_name.split('_')
        ppi_id = fields[0]
        pid = fields[1]
        idx = fields[2]
        in_dir = os.path.join(parent_in_dir, ppi_id)
        rho, theta, feat, mask = load_single_feature(in_dir, pid, idx)
        pos_rho_wrt_center.append(rho)
        pos_theta_wrt_center.append(theta)
        pos_input_feat.append(feat)
        pos_mask.append(mask)
        pos_names.append(positive_name)

        # negative
        fields = negative_name.split('_')
        ppi_id = fields[0]
        pid = fields[1]
        idx = fields[2]
        in_dir = os.path.join(parent_in_dir, ppi_id)
        rho, theta, feat, mask = load_single_feature(in_dir, pid, idx)
        neg_rho_wrt_center.append(rho)
        neg_theta_wrt_center.append(theta)
        neg_input_feat.append(feat)
        neg_mask.append(mask)
        neg_names.append(negative_name)

        # 保存 idx
        if all_sources[i] == 'train':
            train_idx.append(idx_count)
        elif all_sources[i] == 'val':
            val_idx.append(idx_count)
        else:
            test_idx.append(idx_count)

        idx_count += 1
    except Exception as e:
        print(f"Error processing triplet {anchor_name} {positive_name} {negative_name}: {str(e)}")
        set_trace()

print("Finished processing all triplets.")

# 保存 npy 文件
if not os.path.exists(params['cache_dir']):
    os.makedirs(params['cache_dir'])

print("Saving to cache...")

# Convert to numpy arrays
anchor_rho_wrt_center = np.array(anchor_rho_wrt_center)
anchor_theta_wrt_center = np.array(anchor_theta_wrt_center)
anchor_input_feat = np.array(anchor_input_feat)
anchor_mask = np.array(anchor_mask)

pos_rho_wrt_center = np.array(pos_rho_wrt_center)
pos_theta_wrt_center = np.array(pos_theta_wrt_center)
pos_input_feat = np.array(pos_input_feat)
pos_mask = np.array(pos_mask)

neg_rho_wrt_center = np.array(neg_rho_wrt_center)
neg_theta_wrt_center = np.array(neg_theta_wrt_center)
neg_input_feat = np.array(neg_input_feat)
neg_mask = np.array(neg_mask)

train_idx = np.array(train_idx)
val_idx = np.array(val_idx)
test_idx = np.array(test_idx)

# Save anchor
np.save(os.path.join(params['cache_dir'], 'anchor_rho_wrt_center.npy'), anchor_rho_wrt_center)
np.save(os.path.join(params['cache_dir'], 'anchor_theta_wrt_center.npy'), anchor_theta_wrt_center)
np.save(os.path.join(params['cache_dir'], 'anchor_input_feat.npy'), anchor_input_feat)
np.save(os.path.join(params['cache_dir'], 'anchor_mask.npy'), anchor_mask)
np.save(os.path.join(params['cache_dir'], 'anchor_names.npy'), anchor_names)

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
