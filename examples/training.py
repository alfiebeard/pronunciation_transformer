from data.data_loader import load_data
from model_training.training import train


# Load datasets
ds = load_data()

# Set hyperparameters
hyperparams = {"epochs": 5, "batch_size": 1000, "num_layers": 2, "d_model": 256, "num_heads": 8, "dff": 1024, 
    "pe_input": 500, "pe_target": 2000, "position_encoding_denominator": 1000, "dropout_rate": 0.05, "beta_1": 0.9,
    "beta_2": 0.98, "epsilon": 1e-9}

# Train model and save to saved_model_outputs/saved_models/pronouncer
train(ds=ds, hyperparameters=hyperparams)
