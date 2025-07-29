# Header variables and parameters.
import sys
import os
import numpy as np

from default_config.masif_opts import masif_opts

# print pwd
print("Current working directory:", os.getcwd())
# cd to the data_preparation directory 
if not os.path.exists('/Users/yanyz/Ratar/masif-BSC/data/masif_bsc'):
    print("data_preparation directory does not exist. Please run the script from the correct directory.")
    sys.exit(1)
os.chdir('/Users/yanyz/Ratar/masif-BSC/data/masif_bsc')

def cache_data(
        triplets,
        patches_per_protein=32,):
    """
    This script caches data for the MaSIF binding site classification task.
    It reads triplets of binder, positive, and negative samples, processes them,
    and saves the relevant features and indices for training, validation, and testing.
    """

    params = masif_opts['ppi_search']

    # Read input file containing triplets: binder, positive, negative
    # triplets = [line.strip().split() for line in open(triplet_file)]

    # Output directory
    if not os.path.exists(params['cache_dir']):
        os.makedirs(params['cache_dir'])

    binder_rho_wrt_center = []
    binder_theta_wrt_center = []
    binder_input_feat = []
    binder_mask = []

    pos_rho_wrt_center = []
    pos_theta_wrt_center = []
    pos_input_feat = []
    pos_mask = []

    neg_rho_wrt_center = []
    neg_theta_wrt_center = []
    neg_input_feat = []
    neg_mask = []

    training_idx = []
    val_idx = []
    test_idx = []

    pos_names = []
    neg_names = []

    idx_count = 0
    np.random.seed(0)

    for binder_id, pos_id, neg_id in triplets:
        # print(f"Caching: Binder: {binder_id}, Positive: {pos_id}, Negative: {neg_id}")
        
        for pid, rho_list, theta_list, feat_list, mask_list, name_list in [
            (binder_id, binder_rho_wrt_center, binder_theta_wrt_center, binder_input_feat, binder_mask, pos_names),
            (pos_id, pos_rho_wrt_center, pos_theta_wrt_center, pos_input_feat, pos_mask, pos_names),
            (neg_id, neg_rho_wrt_center, neg_theta_wrt_center, neg_input_feat, neg_mask, neg_names)
        ]:
            basename = f"{pid[:4]}_{pid[-1]}" 
            folder = os.path.join(params['masif_precomputation_dir'], basename)
            # Load the numpy arrays
            rho = np.load(os.path.join(folder, f"p1_rho_wrt_center.npy"))
            theta = np.load(os.path.join(folder, f"p1_theta_wrt_center.npy"))
            feat = np.load(os.path.join(folder, f"p1_input_feat.npy"))
            mask = np.load(os.path.join(folder, f"p1_mask.npy"))

            # randomly select 32 patches from the loaded data
            if rho.shape[0] > patches_per_protein:
                indices = np.random.choice(rho.shape[0], patches_per_protein, replace=False)
                rho = rho[indices]
                theta = theta[indices]
                feat = feat[indices]
                mask = mask[indices]
            elif rho.shape[0] < patches_per_protein:
                print(f"Warning: {pid} has less than {patches_per_protein} patches. Using all available patches.")
                # No need to modify rho, theta, feat, mask as they are already smaller than 32
            # elif: No need to modify when rho.shape[0] == patches_per_protein

            rho_list.append(rho)
            theta_list.append(theta)
            feat_list.append(feat)
            mask_list.append(mask)
            for i in range(rho.shape[0]):
                name_list.append(f"{pid}_{i}")

        # Assign to train/val/test
        split_rand = np.random.random()
         # 80% for training, 10% for validation, 10% for testing
        params['range_val_samples'] = 0.8 # tmp modification to make it work with the current data
        if split_rand < params['range_val_samples']:
            training_idx.extend(range(idx_count, idx_count + rho.shape[0]))
        elif split_rand < params['range_val_samples'] + 0.1:
            val_idx.extend(range(idx_count, idx_count + rho.shape[0]))
        else:
            test_idx.extend(range(idx_count, idx_count + rho.shape[0]))
        idx_count += rho.shape[0]
    
    output = {
        'binder_rho_wrt_center': binder_rho_wrt_center,
        'binder_theta_wrt_center': binder_theta_wrt_center,
        'binder_input_feat': binder_input_feat,
        'binder_mask': binder_mask,
        'pos_rho_wrt_center': pos_rho_wrt_center,
        'pos_theta_wrt_center': pos_theta_wrt_center,
        'pos_input_feat': pos_input_feat,
        'pos_mask': pos_mask,
        'neg_rho_wrt_center': neg_rho_wrt_center,
        'neg_theta_wrt_center': neg_theta_wrt_center,
        'neg_input_feat': neg_input_feat, 
        'neg_mask': neg_mask,
        'pos_names': pos_names,
        'neg_names': neg_names,
        'pos_training_idx': training_idx,
        'pos_val_idx': val_idx,
        'pos_test_idx': test_idx,
        'neg_training_idx': training_idx,  # Reuse the same indices for negatives
        'neg_val_idx': val_idx,
        'neg_test_idx': test_idx,  # Reuse the same indices for negatives 
    }

    return output

    """
    output = {
        'binder_rho_wrt_center': np.concatenate(binder_rho_wrt_center),
        'binder_theta_wrt_center': np.concatenate(binder_theta_wrt_center),
        'binder_input_feat': np.concatenate(binder_input_feat),
        'binder_mask': np.concatenate(binder_mask),
        'pos_rho_wrt_center': np.concatenate(pos_rho_wrt_center),
        'pos_theta_wrt_center': np.concatenate(pos_theta_wrt_center),
        'pos_input_feat': np.concatenate(pos_input_feat),
        'pos_mask': np.concatenate(pos_mask),
        'neg_rho_wrt_center': np.concatenate(neg_rho_wrt_center),
        'neg_theta_wrt_center': np.concatenate(neg_theta_wrt_center),
        'neg_input_feat': np.concatenate(neg_input_feat),
        'neg_mask': np.concatenate(neg_mask),
        'pos_names': np.array(pos_names),
        'neg_names': np.array(neg_names),
        'pos_training_idx': np.array(training_idx),
        'pos_val_idx': np.array(val_idx),
        'pos_test_idx': np.array(test_idx),
        'neg_training_idx': np.array(training_idx),
        'neg_val_idx': np.array(val_idx),
        'neg_test_idx': np.array(test_idx), 
    }

    return output
    
    # Concatenate and save
    def save(name, array):
        np.save(os.path.join(params['cache_dir'], name + ".npy"), array)

    save('binder_rho_wrt_center', np.concatenate(binder_rho_wrt_center))
    save('binder_theta_wrt_center', np.concatenate(binder_theta_wrt_center))
    save('binder_input_feat', np.concatenate(binder_input_feat))
    save('binder_mask', np.concatenate(binder_mask))

    save('pos_rho_wrt_center', np.concatenate(pos_rho_wrt_center))
    save('pos_theta_wrt_center', np.concatenate(pos_theta_wrt_center))
    save('pos_input_feat', np.concatenate(pos_input_feat))
    save('pos_mask', np.concatenate(pos_mask))
    save('pos_names', np.array(pos_names))

    save('neg_rho_wrt_center', np.concatenate(neg_rho_wrt_center))
    save('neg_theta_wrt_center', np.concatenate(neg_theta_wrt_center))
    save('neg_input_feat', np.concatenate(neg_input_feat))
    save('neg_mask', np.concatenate(neg_mask))
    save('neg_names', np.array(neg_names))

    save('pos_training_idx', np.array(training_idx))
    save('pos_val_idx', np.array(val_idx))
    save('pos_test_idx', np.array(test_idx))

    # Reuse the same indices for negatives
    save('neg_training_idx', np.array(training_idx))
    save('neg_val_idx', np.array(val_idx))
    save('neg_test_idx', np.array(test_idx))

    print(f"Finished saving cached data to {params['cache_dir']}")
    """