from masif_bsc.MaSIF_bsc import MaSIF_bsc, load_checkpoint, make_feed_dict, compute_descriptor
import numpy as np

# some application functions for MaSIF_bsc
# TODO: this is just a draft, need to be improved

def predict_similarity(input_a, input_b, model_path, config):
    """
    Predicts similarity score between two patches using a trained model.

    Args:
        input_a: (rho, theta, feat, mask) for patch A
        input_b: same for patch B
        model_path: path to a saved checkpoint
        config: dict with model init params (e.g. max_rho, etc.)

    Returns:
        float: similarity score (e.g., Euclidean distance)
    """
    model = MaSIF_bsc(**config)
    load_checkpoint(model, model_path)

    # Combine into one batch: A and B
    rho = np.concatenate([input_a[0], input_b[0]], axis=0)
    theta = np.concatenate([input_a[1], input_b[1]], axis=0)
    feat = np.concatenate([input_a[2], input_b[2]], axis=0)
    mask = np.concatenate([input_a[3], input_b[3]], axis=0)

    feed_dict = make_feed_dict(model, rho, theta, feat, mask)
    descriptors = compute_descriptor(model, feed_dict)

    return np.linalg.norm(descriptors[0] - descriptors[1])

def extract_descriptor(input_data, model_path, config):
    """
    Returns learned descriptor from a patch.

    Args:
        input_data: (rho, theta, feat, mask)
        model_path: path to checkpoint
        config: model hyperparameters

    Returns:
        np.array: descriptor vector
    """
    model = MaSIF_bsc(**config)
    load_checkpoint(model, model_path)
    feed_dict = make_feed_dict(model, *input_data)
    return compute_descriptor(model, feed_dict)[0]

def main():
    """
    Main function to demonstrate training and prediction with MaSIF_bsc.
    """
    # Define model config
    config = {
        "max_rho": 12.0,
        "n_thetas": 16,
        "n_rhos": 5,
        "n_rotations": 16,
        "learning_rate": 0.001,
        "idx_gpu": "/device:GPU:0"
    }

    # Load your training data
    train_dataset = load_my_dataset("my_dataset_path")

    # Train model
    model = train_model_on_data(train_dataset, config, output_dir="checkpoints")

    # Use the model
    score = predict_similarity(input_a, input_b, "checkpoints/model.ckpt", config)
    print("Similarity Score:", score)
