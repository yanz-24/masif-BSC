import tensorflow as tf
import numpy as np
from masif_bsc import MaSIF_bsc  # Assuming your class is in this file
from masif_cache import cache_data  # Import your caching function
from default_config.masif_opts import masif_opts

# Model parameters
MODEL_PARAMS = {
    'max_rho': 12.0,            # Maximum radial distance
    'n_thetas': 16,             # Number of angular bins
    'n_rhos': 5,                # Number of radial bins
    'n_gamma': 1.0,             # Gamma parameter
    'n_rotations': 16,          # Number of rotations for data augmentation
    'idx_gpu': "/device:GPU:0", # GPU device (use "/device:CPU:0" for CPU)
    'feat_mask': [1.0, 1.0, 1.0, 1.0, 1.0]  # Feature mask (which features to use)
}

# Training parameters
TRAINING_PARAMS = {
    'learning_rate': 1e-3,
    'keep_prob': 0.8,           # Dropout keep probability
    'n_epochs': 3,              # Number of epochs to train
    'batch_size': 1,            # Number of quadruplets [binder, pos, neg, neg2] in one batch. 
    'print_every': 10,          # Print loss every N iterations
    'num_iterations' : 1000,    # Total number of training iterations
    'test_every': 10,           # Number of iterations to test the model, i.e. evaluate the model on the test set every N iterations
    'cache_every' : 10,         # Cache the data every N iterations
}

# Data parameters
DATA_PARAMS = {
    'data_path': '/Users/yanyz/Ratar/masif-BSC/data/masif_bsc/nn_models/sc05/cache',  # TODO
    'patches_per_protein': 32,  # Number of patches per protein
    'triplets': '/Users/yanyz/Ratar/masif-BSC/source/masif_bsc/training_list100.txt',  # TODO 
}

class MaSIFTrainer:
    def __init__(self, model_params, training_params):
        """
        Initialize the MaSIF trainer
        
        Args:
            model_params: Dictionary with model parameters
            training_params: Dictionary with training parameters
        """
        self.model_params = model_params
        self.training_params = training_params
        
        # Initialize the MaSIF model
        self.model = MaSIF_bsc(
            max_rho=model_params['max_rho'],
            n_thetas=model_params['n_thetas'],
            n_rhos=model_params['n_rhos'],
            n_gamma=model_params['n_gamma'],
            learning_rate=training_params['learning_rate'],
            n_rotations=model_params['n_rotations'],
            idx_gpu=model_params['idx_gpu'],
            feat_mask=model_params['feat_mask']
        )
        
    def prepare_batch_data(self, batch_data):
        """
        Prepare batch data for training
        
        Args:
            batch_data: Dictionary containing batch data
            
        Returns:
            feed_dict: Dictionary for TensorFlow session
        """
        feed_dict = {
            self.model.rho_coords: batch_data['rho_coords'],
            self.model.theta_coords: batch_data['theta_coords'],
            self.model.input_feat: batch_data['input_feat'],
            self.model.mask: batch_data['mask'],
            self.model.keep_prob: self.training_params['keep_prob']
        }
        return feed_dict
    
    def train_step(self, batch_data):
        """
        Perform one training step (forward + backward + parameter update)

        Args:
            batch_data (dict): A dictionary containing one training batch with the following keys:
                - 'rho_coords':  [batch_size, n_vertices, 1]   radial coordinates of the patch
                - 'theta_coords':[batch_size, n_vertices, 1]   angular coordinates of the patch
                - 'input_feat':  [batch_size, n_vertices, n_feat] feature vectors for each vertex
                - 'mask':        [batch_size, n_vertices, 1]   mask for valid vertices (1=valid,0=padding)

        Returns:
            loss (float):      The computed data loss for this batch (before parameter update)
            grad_norm (float): L2 norm of the gradients (used to monitor training stability)

        Training flow:
        1. Prepare the feed_dict (maps TensorFlow placeholders to actual NumPy data)
        2. Run a single optimization step:
            - Forward pass → compute loss
            - Backward pass → compute gradients
            - Optimizer applies gradients → update model parameters
        3. Also return current loss & gradient norm for logging/debugging.
        """
        # convert numpy data to tensorflow feed_dict (maps TensorFlow placeholders to actual NumPy data)
        feed_dict = self.prepare_batch_data(batch_data)
        
        # Run optimization step
        _, loss, grad_norm = self.model.session.run(
            [self.model.optimizer, # gradient descent optimizer
             self.model.data_loss, # computed current batch loss
             self.model.norm_grad], # compute dradient norm
            feed_dict=feed_dict
        )
        
        return loss, grad_norm
    
    def evaluate(self, batch_data):
        """
        Evaluate model on batch data
        
        Args:
            batch_data: Dictionary containing batch data
            
        Returns:
            loss: Evaluation loss
            scores: Computed scores
        """
        feed_dict = self.prepare_batch_data(batch_data)
        feed_dict[self.model.keep_prob] = 1.0  # No dropout during evaluation
        
        loss, scores = self.model.session.run(
            [self.model.data_loss, self.model.score],
            feed_dict=feed_dict
        )
        
        return loss, scores
    
    def save_model(self, save_path):
        """Save the trained model"""
        self.model.saver.save(self.model.session, save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path):
        """Load a pre-trained model"""
        self.model.saver.restore(self.model.session, load_path)
        print(f"Model loaded from {load_path}")

def construct_batch(
    binder_rho_wrt_center,
    binder_theta_wrt_center,
    binder_input_feat,
    binder_mask,
    pos_rho_wrt_center,
    pos_theta_wrt_center,
    pos_input_feat,
    pos_mask,
    neg_rho_wrt_center,
    neg_theta_wrt_center,
    neg_input_feat,
    neg_mask,
):
    """
    Load your actual data here
    
    This function should return data in the format expected by the model:
    - rho_coords: [batch_size, n_vertices, 1] - radial coordinates
    - theta_coords: [batch_size, n_vertices, 1] - angular coordinates  
    - input_feat: [batch_size, n_vertices, n_feat] - feature vectors
    - mask: [batch_size, n_vertices, 1] - vertex mask
    
    The data should contain 4 groups:
    - Group 1: Positive patches (binding sites)
    - Group 2: Corresponding binder patches  
    - Group 3: Negative patches (non-binding)
    - Group 4: Corresponding non-binder patches
    """

    c_pos_training_idx = np.arange(len(pos_rho_wrt_center))
    c_neg_training_idx = np.arange(len(neg_rho_wrt_center))

    batch_rho_coords_binder = np.expand_dims(
        binder_rho_wrt_center[c_pos_training_idx], 2
    )
    batch_theta_coords_binder = np.expand_dims(
        binder_theta_wrt_center[c_pos_training_idx], 2
    )
    batch_input_feat_binder = binder_input_feat[c_pos_training_idx]
    batch_mask_binder = binder_mask[c_pos_training_idx]

    batch_rho_coords_pos = np.expand_dims(pos_rho_wrt_center[c_pos_training_idx], 2)
    batch_theta_coords_pos = np.expand_dims(pos_theta_wrt_center[c_pos_training_idx], 2)
    batch_input_feat_pos = pos_input_feat[c_pos_training_idx]
    batch_mask_pos = pos_mask[c_pos_training_idx]

    # Negate the input_features of the binder, except the last column.
    batch_input_feat_binder = -batch_input_feat_binder
    # TODO: This should not be like this ... it is a hack.
    if batch_input_feat_binder.shape[2] == 5 or batch_input_feat_binder.shape[2] == 3:
        batch_input_feat_binder[:, :, -1] = -batch_input_feat_binder[
            :, :, -1
        ]  # Do not negate hydrophobicity.
    # Also negate the theta coords for the binder.
    batch_theta_coords_binder = 2 * np.pi - batch_theta_coords_binder

    batch_rho_coords_neg = np.expand_dims(neg_rho_wrt_center[c_neg_training_idx], 2)
    batch_theta_coords_neg = np.expand_dims(neg_theta_wrt_center[c_neg_training_idx], 2)
    batch_input_feat_neg = neg_input_feat[c_neg_training_idx]
    batch_mask_neg = neg_mask[c_neg_training_idx]

    batch_rho_coords_neg_2 = batch_rho_coords_binder.copy()
    batch_theta_coords_neg_2 = batch_theta_coords_binder.copy()
    batch_input_feat_neg_2 = batch_input_feat_binder.copy()
    batch_mask_neg_2 = batch_mask_binder.copy()

    batch_rho_coords = np.concatenate(
        [
            batch_rho_coords_pos,
            batch_rho_coords_binder,
            batch_rho_coords_neg,
            batch_rho_coords_neg_2,
        ],
        axis=0,
    )
    batch_theta_coords = np.concatenate(
        [
            batch_theta_coords_pos,
            batch_theta_coords_binder,
            batch_theta_coords_neg,
            batch_theta_coords_neg_2,
        ],
        axis=0,
    )
    batch_input_feat = np.concatenate(
        [
            batch_input_feat_pos,
            batch_input_feat_binder,
            batch_input_feat_neg,
            batch_input_feat_neg_2,
        ],
        axis=0,
    )
    batch_mask = np.concatenate(
        [batch_mask_pos, batch_mask_binder, batch_mask_neg, batch_mask_neg_2], axis=0
    )
    # expand the last dimension of the mask (batch_size, max_points_patch, 1)
    batch_mask = np.expand_dims(batch_mask, 2)

    return batch_rho_coords, batch_theta_coords, batch_input_feat, batch_mask


def create_dummy_data(n_samples, n_vertices, n_feat):
    """
    Create dummy data for testing
    
    Args:
        n_samples: Number of samples
        n_vertices: Number of vertices per sample
        n_feat: Number of features
        
    Returns:
        batch_data: Dictionary with dummy data
    """
    batch_data = {
        'rho_coords': np.random.uniform(0, 1, (n_samples, n_vertices, 1)).astype(np.float32),
        'theta_coords': np.random.uniform(0, 2*np.pi, (n_samples, n_vertices, 1)).astype(np.float32),
        'input_feat': np.random.normal(0, 1, (n_samples, n_vertices, n_feat)).astype(np.float32),
        'mask': np.ones((n_samples, n_vertices, 1)).astype(np.float32)
    }
    return batch_data

def train_masif_bsc(
        model_params=MODEL_PARAMS,
        training_params=TRAINING_PARAMS, 
        data_params=DATA_PARAMS,
        ):    
    # Initialize trainer
    trainer = MaSIFTrainer(model_params, training_params)

    # load data
    print("Loading data...")
    triplets = [line.strip().split() for line in open(DATA_PARAMS['triplets'])]
    triplets_trainig = triplets[:int(len(triplets) * 0.8)]  # 80% for training
    triplets_validation = triplets[int(len(triplets) * 0.8):int(len(triplets) * 0.9)]  # 10% for validation
    triplets_testing = triplets[int(len(triplets) * 0.9):]  # 10% for testing
    
    iteration = 0
    # Training loop
    print("Starting training...")
    for epoch in range(training_params['n_epochs']):
        np.random.shuffle(triplets_trainig)
        while iteration <= training_params['num_iterations']:
            # cache data every 'cache_every' iterations
            if iteration % training_params['cache_every'] == 0:
                print(f"Iteration {iteration}, caching data...")
                triplets_cache = triplets_trainig[iteration:iteration + training_params['cache_every']]
                cached_data = cache_data(
                    triplets_cache,
                    data_params['patches_per_protein'],
                )
                print(f"Epoch {epoch} Iteration {iteration}: Caching done.")

            for i in range(min(training_params['cache_every'], len(cached_data['binder_rho_wrt_center']))):
                # Construct batch data from cached data
                # Load batch data. A batch data is a group of 4 protein [binder, pos, neg1, neg2].
                # each protein has 32 patches, and the total batch size is 32*4 = 128.

                batch_data = construct_batch(
                    binder_rho_wrt_center=cached_data['binder_rho_wrt_center'][i],
                    binder_theta_wrt_center=cached_data['binder_theta_wrt_center'][i],
                    binder_input_feat=cached_data['binder_input_feat'][i],
                    binder_mask=cached_data['binder_mask'][i],
                    pos_rho_wrt_center=cached_data['pos_rho_wrt_center'][i],
                    pos_theta_wrt_center=cached_data['pos_theta_wrt_center'][i],
                    pos_input_feat=cached_data['pos_input_feat'][i],
                    pos_mask=cached_data['pos_mask'][i],
                    neg_rho_wrt_center=cached_data['neg_rho_wrt_center'][i],
                    neg_theta_wrt_center=cached_data['neg_theta_wrt_center'][i],
                    neg_input_feat=cached_data['neg_input_feat'][i],
                    neg_mask=cached_data['neg_mask'][i],
                )

                batch_data = {
                    'rho_coords': batch_data[0],
                    'theta_coords': batch_data[1],
                    'input_feat': batch_data[2],
                    'mask': batch_data[3]
                }

                # Training step
                loss, grad_norm = trainer.train_step(batch_data)
                print("Done")

            print(f"Iteration {iteration + i}, Loss: {loss:.4f}, Grad Norm: {grad_norm:.4f}")
            iteration += training_params['cache_every']

            if iteration % training_params['test_every'] == 0:
                
            
            # Save model periodically
            if epoch % training_params['test_every_iter'] == 0 and epoch > 0:
                save_path = f"models/masif_model_epoch_{epoch}.ckpt"
                trainer.save_model(save_path)
        
    # Final model save
    final_save_path = "models/masif_model_final.ckpt"
    trainer.save_model(final_save_path)
    
    # Example evaluation
    test_data_cached = cache_data(
                    triplets_testing, 
                    data_params['patches_per_protein'],
                )
    
    print("\nEvaluating model...")
    test_data = construct_batch(test_data_cached)  # Load test data
    test_loss, test_scores = trainer.evaluate(test_data)
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test Scores shape: {test_scores.shape}")

