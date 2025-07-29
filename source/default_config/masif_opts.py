import tempfile

masif_opts = {}
# Default directories
masif_opts["raw_pdb_dir"] = "data_preparation/00-raw_pdbs/"
masif_opts["pdb_chain_dir"] = "data_preparation/01-benchmark_pdbs/"
masif_opts["ply_chain_dir"] = "data_preparation/01-benchmark_surfaces/"
masif_opts["tmp_dir"] = tempfile.gettempdir()
masif_opts["ply_file_template"] = masif_opts["ply_chain_dir"] + "/{}_{}.ply"

# Surface features
masif_opts["use_hbond"] = True
masif_opts["use_hphob"] = True
masif_opts["use_apbs"] = True
masif_opts["compute_iface"] = True
# Mesh resolution. Everything gets very slow if it is lower than 1.0
masif_opts["mesh_res"] = 1.0
masif_opts["feature_interpolation"] = True


# Coords params
masif_opts["radius"] = 12.0

# Neural network patch application specific parameters.
masif_opts["ppi_search"] = {}
masif_opts["ppi_search"]["training_list"] = "lists/training.txt"
masif_opts["ppi_search"]["testing_list"] = "lists/testing.txt"
masif_opts["ppi_search"]["training_triplets"] = "lists/training_list100.txt"
masif_opts["ppi_search"]["validation_triplets"] = "lists/validation_list100.txt"
masif_opts["ppi_search"]["testing_triplets"] = "lists/testing_list100.txt"
masif_opts["ppi_search"]["max_shape_size"] = 200
masif_opts["ppi_search"]["max_distance"] = 12.0  # Radius for the neural network.
masif_opts["ppi_search"][
    "masif_precomputation_dir"
] ="data_preparation/04b-precomputation_12A/precomputation/" # old: "data_preparation/04b-precomputation_12A/precomputation/" 
masif_opts["ppi_search"]["feat_mask"] = [1.0] * 5
masif_opts["ppi_search"]["max_sc_filt"] = 1.0
masif_opts["ppi_search"]["min_sc_filt"] = 0.5
masif_opts["ppi_search"]["pos_surf_accept_probability"] = 1.0
masif_opts["ppi_search"]["pos_interface_cutoff"] = 1.0
masif_opts["ppi_search"]["range_val_samples"] = 0.9  # 0.9 to 1.0
masif_opts["ppi_search"]["cache_dir"] = "nn_models/sc05/cache/"
masif_opts["ppi_search"]["model_dir"] = "nn_models/sc05/all_feat/model_data/"
masif_opts["ppi_search"]["desc_dir"] = "descriptors/sc05/all_feat/"
masif_opts["ppi_search"]["gif_descriptors_out"] = "gif_descriptors/"
# Parameters for shape complementarity calculations.
masif_opts["ppi_search"]["sc_radius"] = 12.0
masif_opts["ppi_search"]["sc_interaction_cutoff"] = 1.5
masif_opts["ppi_search"]["sc_w"] = 0.25

# Neural network patch application specific parameters.
masif_opts["site"] = {}
masif_opts["site"]["training_list"] = "lists/training.txt"
masif_opts["site"]["testing_list"] = "lists/testing.txt"
masif_opts["site"]["max_shape_size"] = 100
masif_opts["site"]["n_conv_layers"] = 3
masif_opts["site"]["max_distance"] = 9.0  # Radius for the neural network.
masif_opts["site"][
    "masif_precomputation_dir"
] = "data_preparation/04a-precomputation_9A/precomputation/"
masif_opts["site"]["range_val_samples"] = 0.9  # 0.9 to 1.0
masif_opts["site"]["model_dir"] = "nn_models/all_feat_3l/model_data/"
masif_opts["site"]["out_pred_dir"] = "output/all_feat_3l/pred_data/"
masif_opts["site"]["out_surf_dir"] = "output/all_feat_3l/pred_surfaces/"
masif_opts["site"]["feat_mask"] = [1.0] * 5

# Neural network ligand application specific parameters.
masif_opts["ligand"] = {}
masif_opts["ligand"]["assembly_dir"] = "data_preparation/00b-pdbs_assembly"
masif_opts["ligand"]["ligand_coords_dir"] = "data_preparation/00c-ligand_coords"
masif_opts["ligand"][
    "masif_precomputation_dir"
] = "data_preparation/04a-precomputation_12A/precomputation/"
masif_opts["ligand"]["max_shape_size"] = 200
masif_opts["ligand"]["feat_mask"] = [1.0] * 5
masif_opts["ligand"]["train_fract"] = 0.9 * 0.8
masif_opts["ligand"]["val_fract"] = 0.1 * 0.8
masif_opts["ligand"]["test_fract"] = 0.2
masif_opts["ligand"]["tfrecords_dir"] = "data_preparation/tfrecords"
masif_opts["ligand"]["max_distance"] = 12.0
masif_opts["ligand"]["n_classes"] = 7
masif_opts["ligand"]["feat_mask"] = [1.0, 1.0, 1.0, 1.0, 1.0]
masif_opts["ligand"]["costfun"] = "dprime"
masif_opts["ligand"]["model_dir"] = "nn_models/all_feat/"
masif_opts["ligand"]["test_set_out_dir"] = "test_set_predictions/"

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
    'keep_prob': 0.8,                   # Dropout keep probability
    'n_epochs': 3,                      # Number of epochs to train
    'batch_size': 1,                    # Number of quadruplets [binder, pos, neg, neg2] in one batch. 
    'num_iterations' : 1000,            # Total number of training iterations
    'print_every': 10,                  # Print loss every N iterations
    'val_every': 10,                    # Number of iterations to validate the model, i.e. evaluate the model on the test set every N iterations
    'cache_every' : 10,                 # Cache the data every N iterations
    'early_stopping': True,             # Whether to use early stopping
    'early_stopping_patience': None,      # Number of iterations to wait for improvement before stopping training
    'early_stopping_delta': 0.01,       # Minimum change to qualify as an improvement
    'early_stopping_metric': 'loss',    # Metric to monitor for early stopping, 'roc_auc' or 'loss'
}

# Data parameters
DATA_PARAMS = {
    'data_path': '/Users/yanyz/Ratar/masif-BSC/data/masif_bsc/nn_models/sc05/cache',  # TODO
    'patches_per_protein': 32,  # Number of patches per protein
    'triplets': '/Users/yanyz/Ratar/masif-BSC/source/masif_bsc/training_list100.txt',
    'models': '/Users/yanyz/Ratar/masif-BSC/data/masif_bsc/models' 
}