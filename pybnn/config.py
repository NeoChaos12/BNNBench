defaultMlpParams = {
    "hidden_layer_sizes": [50, 50, 50],
    "input_dims": 1,
    "output_dims": 1,
}


defaultModelParams = {
    "num_epochs": 500,
    "batch_size": 10,
    "learning_rate": 0.01,
    "normalize_input": True,
    "normalize_output": True,
}


expParams = {
    "rng": None,
    "debug": True,
    "tb_logging": True,
    "tb_log_dir": f"runs/default/",
    "tb_exp_name": f"experiment",
}