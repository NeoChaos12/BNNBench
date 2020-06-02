from sklearn.datasets import load_boston

dataloader_args = {
    "boston": {"return_X_y": True}
}

dataloaders = {
    "boston": load_boston
}