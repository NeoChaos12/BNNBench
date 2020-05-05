from pybnn.config import ExpConfig as conf

exp_params = {
    "debug": False,
    "tb_logging": True,
    "tb_log_dir": f"runs/tensorboard_test/",
    # "tb_exp_name": "lr 0.1 epochs 1000 minba 64 hu 50 trainsize 100" + str(datetime.datetime.today()),
    "tb_exp_name": f"test_5",
    # 'model_logger': model_logger
}

conf.read_exp_params(exp_params)

writer = conf.tb_writer()

for i in range(100):
    writer.add_scalar('test_scalar', scalar_value=i**2, global_step=i+1)

writer.close()