import time
import config_file as cf
import c_advmc_cnet_fixed_sign_net as advmc
import torch

if __name__ == "__main__":
    config = cf.Config()
    torch.cuda.synchronize()
    advmc_start = time.time()
    advmc.setup_seed(config.advmc_seed)
    advmc.progress_advmc(config)
    torch.cuda.synchronize()
    advmc_end = time.time()
    print("\nadvmc_total:", advmc_end - advmc_start)
