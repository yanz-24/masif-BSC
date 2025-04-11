# Header variables and parameters.
import sys
import os
import glob
import numpy as np
import pandas as pd

from default_config.masif_opts import masif_opts

"""
masif_bsc_cache_training_data.py: TODO
"""

# get parameters
params = masif_opts['ppi_search']
parent_in_dir = params['masif_precomputation_dir']
cache_dir = params['cache_dir']
val_range = 1 - params['range_val_samples']

# output directory
os.makedirs(cache_dir, exist_ok=True) # create cache directory if it doesn't exist
tmp_dir = os.path.join(cache_dir, 'tmp_chunks') # create tmp directory for output
os.makedirs(tmp_dir, exist_ok=True)

# read training and testing list
train_df = pd.read_csv(params['training_list'])  # should contains：p1, p2, label (0/1)
test_df = pd.read_csv(params['testing_list'])

# Split training and validation
np.random.seed(0)
train_df_shuffled = train_df.sample(frac=1).reset_index(drop=True)
val_cutoff = int(val_range * len(train_df_shuffled))

val_rows = train_df_shuffled.iloc[:val_cutoff]
train_rows = train_df_shuffled.iloc[val_cutoff:]

# Index tracking
training_idx, val_idx, test_idx = [], [], []
pos_names, neg_names = [], []

idx_count = 0

def save_chunk(name, array):
    filename = os.path.join(tmp_dir, f"{name}_{idx_count}.npy")
    np.save(filename, array)

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
        save_chunk('binder_rho_wrt_center', rho1)
        save_chunk('binder_theta_wrt_center', theta1)
        save_chunk('binder_input_feat', input_feat1)
        save_chunk('binder_mask', mask1)

        pair_name = f"{p1_id}_{p2_id}"

        if label == 1:
            save_chunk('pos_rho_wrt_center', rho2)
            save_chunk('pos_theta_wrt_center', theta2)
            save_chunk('pos_input_feat', input_feat2)
            save_chunk('pos_mask', mask2)
            pos_names.append(pair_name)
        elif label == 0:
            save_chunk('neg_rho_wrt_center', rho2)
            save_chunk('neg_theta_wrt_center', theta2)
            save_chunk('neg_input_feat', input_feat2)
            save_chunk('neg_mask', mask2)
            neg_names.append(pair_name)

        # 保存 index
        if split == 'train':
            training_idx.append(idx_count)
        elif split == 'test':
            test_idx.append(idx_count)
        elif split == 'val':
            val_idx.append(idx_count)

        idx_count += 1

    except Exception as e:
        print(f"Error loading {p1_id}, {p2_id}: {e}")


# deal with trainig, validation and testing set
for _, row in train_rows[0:2].iterrows():
    load_sample(row['p1'], row['p2'], row['label'], 'train')

for _, row in val_rows[0:2].iterrows():
    load_sample(row['p1'], row['p2'], row['label'], 'val')

for _, row in test_df[0:2].iterrows():
    load_sample(row['p1'], row['p2'], row['label'], 'test')


def concatenate_chunks(name):
    files = sorted(glob.glob(os.path.join(tmp_dir, f"{name}_*.npy")))
    arrays = [np.load(f) for f in files]
    if arrays and isinstance(arrays[0], np.ndarray):
        out = np.concatenate(arrays, axis=0)
    else:
        out = arrays
    np.save(os.path.join(cache_dir, f"{name}.npy"), out)


# Final saving
for name in [
    'binder_rho_wrt_center', 'binder_theta_wrt_center', 'binder_input_feat', 'binder_mask',
    'pos_rho_wrt_center', 'pos_theta_wrt_center', 'pos_input_feat', 'pos_mask',
    'neg_rho_wrt_center', 'neg_theta_wrt_center', 'neg_input_feat', 'neg_mask'
]:
    concatenate_chunks(name)

# Save metadata
np.save(os.path.join(cache_dir, 'pos_names.npy'), np.array(pos_names))
np.save(os.path.join(cache_dir, 'neg_names.npy'), np.array(neg_names))

np.save(os.path.join(cache_dir, 'pos_training_idx.npy'), np.array(training_idx))
np.save(os.path.join(cache_dir, 'pos_test_idx.npy'), np.array(test_idx))
np.save(os.path.join(cache_dir, 'pos_val_idx.npy'), np.array(val_idx))

np.save(os.path.join(cache_dir, 'neg_training_idx.npy'), np.array(training_idx))
np.save(os.path.join(cache_dir, 'neg_test_idx.npy'), np.array(test_idx))
np.save(os.path.join(cache_dir, 'neg_val_idx.npy'), np.array(val_idx))

print(f"Number of positives: {len(pos_names)} | negatives: {len(neg_names)}")
