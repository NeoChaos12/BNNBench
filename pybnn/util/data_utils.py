from sklearn.datasets import load_boston, fetch_openml

dataloader_args = {
    "boston": {"return_X_y": True},
    "concrete": {"name": "Concrete_Data", "return_X_y": True},
    "energy": {"name": "energy-efficiency", "return_X_y": True},
    "kin8nm": {"name": "kin8nm", "return_X_y": True},
    "yacht": {"name": "yacht_hydrodynamics", "return_X_y": True},
}

dataloaders = {
    "boston": load_boston,
    "concrete": fetch_openml,
    "energy": fetch_openml,
    "kin8nm": fetch_openml,
    "yacht": fetch_openml,
}