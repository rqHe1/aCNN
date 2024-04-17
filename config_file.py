import attr
import torch
import os
from os.path import dirname, abspath
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

net_structure = "aCNN"

@attr.s(auto_attribs=True)
class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = dirname(abspath(__file__))
    cpu: str = 'cpu'
    dtype = torch.float64

    # network
    channel: int = 3
    kernel: int = 5
    deep_step_up_block: int = 14 #  the number of residual block

    # model super params
    TFIsingModel: bool = False
    HeisenbergModel: bool = True
    Hamiltonian: str = "nH"  # ["H", "nH", "sH"] H: no sign structure; nH: checkerboard sign structure; sH: stripy sign structure;
    pd_md: str = 'circular'
    ly: int = 10             # [8, 10, 16, 24]
    lx: int = 10             # [8, 10, 16, 24]
    size = lx * ly
    n_occ: int = size // 2
    J: torch.float64 = 1.0
    Jp: torch.float64 = 0.55   # [0., 0.4, 0.55]
    lambda_f: torch.float64 = 3.05

    # advmc
    run_advmc: bool = True
    advmc_warmup: int = 50
    advmc_n_mc: int = 256    # note: if out of memory, please reduce this param.
    advmc_sweep_iter: int = 2
    advmc_nsweep = size * advmc_sweep_iter
    advmc_lr_max: float = 0.002
    advmc_lr_step_n: int = 1000
    advmc_lr_step = advmc_lr_max / advmc_lr_step_n
    advmc_milestones: int = [10000, 40000, 100000, 160000, 220000, 280000, 290000]
    advmc_milestones_gamma: float = 0.5
    advmc_regularize_param: float = 1.e-7
    advmc_seed: int = 1
    advmc_max_iter: int = 50000
    advmc_save_interval: int = 10000

    # test the model or not
    advmc_relay: bool = True
    if TFIsingModel:
        relay_path = path + "/read_mdlparams/TFIM/TFIM_L{}_c4v.pth".format(ly)
    elif HeisenbergModel:
        if Jp == 0.:
            relay_path = path + "/read_mdlparams/HM/HM_L{}_c4v.pth".format(ly)
        else:
            relay_path = path + "/read_mdlparams/HJ1J2M/L{}_Jp{}_c4v.pth".format(ly, Jp)

    n_bin: int = advmc_max_iter // 100   ## calc std in advmc amp optimization process, n_bin is the number of bin(set or segment) in MCMC chain we cut.
    debug: bool = True

    if HeisenbergModel:
        file_name_H = './results_advmc/real-nH_HM_{}_nmc{}_L{}{}_iter{}_lr{}_seed{}_Jp{}.txt'.format(
                    device, advmc_n_mc, ly, lx, advmc_max_iter, advmc_lr_max, advmc_seed, Jp)
        file_name_ssf_real = './results_advmc/real-ssf_real-nH_HM_{}_nmc{}_L{}{}_iter{}_lr{}_seed{}_Jp{}.txt'.format(
                    device, advmc_n_mc, ly, lx, advmc_max_iter, advmc_lr_max, advmc_seed, Jp)
        file_name_ssf_imag = './results_advmc/imag-ssf_real-nH_HM_{}_nmc{}_L{}{}_iter{}_lr{}_seed{}_Jp{}.txt'.format(
                    device, advmc_n_mc, ly, lx, advmc_max_iter, advmc_lr_max, advmc_seed, Jp)
        file_name_ssf_real_mean = './results_advmc/real-ssf-mean_real-nH_HM_{}_nmc{}_L{}{}_iter{}_lr{}_seed{}_Jp{}.txt'.format(
                    device, advmc_n_mc, ly, lx, advmc_max_iter, advmc_lr_max, advmc_seed, Jp)
    elif TFIsingModel:
        file_name_H = './results_advmc/real-nH_TFIM_{}_nmc{}_L{}{}_iter{}_lr{}_seed{}_lambda{}.txt'.format(
                    device, advmc_n_mc, ly, lx, advmc_max_iter, advmc_lr_max, advmc_seed, lambda_f)
        file_name_ssf_real = './results_advmc/real-ssf_real-nH_TFIM_{}_nmc{}_L{}{}_iter{}_lr{}_seed{}_lambda{}.txt'.format(
                    device, advmc_n_mc, ly, lx, advmc_max_iter, advmc_lr_max, advmc_seed, lambda_f)
        file_name_ssf_imag = './results_advmc/imag-ssf_real-nH_TFIM_{}_nmc{}_L{}{}_iter{}_lr{}_seed{}_lambda{}.txt'.format(
                    device, advmc_n_mc, ly, lx, advmc_max_iter, advmc_lr_max, advmc_seed, lambda_f)
        file_name_ssf_real_mean = './results_advmc/real-ssf-mean_real-nH_TFIM_{}_nmc{}_L{}{}_iter{}_lr{}_seed{}_lambda{}.txt'.format(
                    device, advmc_n_mc, ly, lx, advmc_max_iter, advmc_lr_max, advmc_seed, lambda_f)
    else:
        print("\n\nPlease check the file name! \n\n")
        quit()
