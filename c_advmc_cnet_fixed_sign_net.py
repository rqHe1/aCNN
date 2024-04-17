import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import random
import c_my_nets as mn
import time


class Idn(torch.autograd.Function):
    """
    The realize of P / detach(P) 's forward and backward progress.
    """
    @staticmethod
    def forward(self, input_):
        self.save_for_backward(input_)   # indispensable for back propagation
        output = torch.ones_like(input_)    # output = input_ / input_
        return output

    @staticmethod
    def backward(self, grad_output):
        input_, = self.saved_tensors
        return grad_output                   # 1 * input_grad

    def __call__(self, input_):
        res = self.forward(self, input_)
        return res


class ADVMC(nn.Module):
    """
    Class to realize the forward process of the network and advmc detach progress.
    including MCMC progress and calculation of h_loc.
    """

    def __init__(self, s0, config):
        super(ADVMC, self).__init__()
        self.cpu = torch.device('cpu')
        self.config = config
        self.n_mc = config.advmc_n_mc
        self.dtype = config.dtype
        self.device = config.device
        self.lx = config.lx
        self.ly = config.ly
        self.size = config.lx * config.ly
        self.J = config.J
        self.Jp = config.Jp

        self.s0 = s0.to(self.config.device)
        self.amplitude_0 = torch.rand(1, dtype=config.dtype).to(self.config.device)
        self.net_phi_0 = torch.rand(1, dtype=config.dtype).to(self.config.device)
        self.amplitude_1 = torch.rand(1, dtype=config.dtype).to(self.config.device)
        self.net_phi_1 = torch.rand(1, dtype=config.dtype).to(self.config.device)
        self.s1 = torch.ones_like(s0)
        self.accept = torch.zeros(1, dtype=config.dtype)
        self.prcsn = 2.98023223876953125e-9 if self.config.dtype == torch.float32 else \
            5.5511151231257827021181583404541e-17
        self.regularize_param = config.advmc_regularize_param

        self.amplitude_net = mn.AmpNetCnnCnV(self.config.ly, self.config.lx, self.config.device, self.config.dtype,
                                             self.config.channel, self.config.kernel,
                                             self.config.deep_step_up_block, self.config.pd_md).to(self.config.device)
        self.amp_net = self.amplitude_net.to(self.config.device)
        self.vhloc = torch.zeros(1)


        if config.TFIsingModel:
            self.h_loc = self.h_loc_matrix_periodic_TFIsing
        if config.HeisenbergModel:
            self.h_loc = self.h_loc_matrix_periodic_Heisenberg

        print("Using periodic bundary condition!\n")  # cyclic
            
        self.index = torch.arange(self.n_mc * 4 * self.ly * self.lx, dtype=torch.int64, device=self.device) + 1
        self.sign_t = torch.zeros([self.n_mc * 4 * self.ly * self.lx], dtype=torch.int64, device=self.device)
        self.amp_t = torch.zeros([self.n_mc * 4 * self.ly * self.lx], dtype=self.dtype, device=self.device)
       
        # default using nH
        if self.config.Hamiltonian == "H":
            self.sign_y = 1.
            self.sign_x = 1.
            self.sign_yx = 1.
            self.sign_y_x = 1.
            print("Using Hamiltonian: H !\n")
        elif self.config.Hamiltonian == "sH":
            self.sign_y = -1.
            self.sign_x = 1.
            self.sign_yx = -1.
            self.sign_y_x = -1.
            print("Using Hamiltonian: sH !\n")
        elif self.config.Hamiltonian == "nH":
            self.sign_y = -1.
            self.sign_x = -1.
            self.sign_yx = 1.
            self.sign_y_x = 1.
            print("Using Hamiltonian: nH !\n")
        else:
            print("\nWaring: Please chose Hamiltonian from: H, nH, sH !\n")
            quit()


    def forward(self, input_):
        """
        connection of the network and advmc.
        :param input_: configuration
        :return: mean value of energy of model
        """

        amplitude = self.amp_net(input_)
        loss_regularize = torch.sum(self.regularize_param * torch.pow(amplitude, 4), dim=0, keepdim=True) / self.n_mc
        output2 = Idn.apply(2 * amplitude)  # input log(|phi|^2)
        with torch.no_grad():
            self.vhloc = self.h_loc()
        energy = (torch.sum(output2 * self.vhloc) / torch.sum(output2)) / self.size  # self.vhloc.detach()
        energy_std = torch.std(self.vhloc) / torch.sqrt(torch.tensor(self.n_mc)) / self.size

        return energy, energy_std, loss_regularize

    def update_net_phi0(self):
        """
        update log_phi0 when the network is updated.
        """
        with torch.no_grad():
            self.amplitude_0 = self.amp_net(self.s0.to(self.device))
            self.net_phi_0 = torch.exp(self.amplitude_0)
        return None

    def update_conf(self):
        """
        running with n_t MCMC step.
        :return: new configuration and accept rate of last step
        """
        for j in range(self.config.advmc_nsweep):
            self.s1, self.amplitude_1, self.net_phi_1, self.accept = self.update()
            self.amplitude_0 = self.amplitude_1.clone()
            self.net_phi_0 = self.net_phi_1.clone()
            self.s0 = self.s1.clone()
        return self.s1, self.accept

    def h_loc_matrix_periodic_Heisenberg(self):
        """
        matrix form could be runing on Gpu, and fast.
        """
        amplitude_0 = self.amp_net(self.s0)
        sign_0 = torch.ones_like(amplitude_0, dtype=torch.int64, device=self.device)

        onehot_matrix_site_wised = (
            torch.eye(self.ly * self.lx).view(self.ly, self.lx, self.ly, self.lx)).to(self.device)
        # direction: y
        onehot_y_shift = torch.cat((onehot_matrix_site_wised[:, :, [-1], :], onehot_matrix_site_wised[:, :, :-1, :]), 2)
        # direction: x
        onehot_x_shift = torch.cat((onehot_matrix_site_wised[:, :, :, [-1]], onehot_matrix_site_wised[:, :, :, :-1]), 3)
        # direction: y, x
        onehot_yx_shift = torch.cat((onehot_y_shift[:, :, :, [-1]], onehot_y_shift[:, :, :, :-1]), 3)
        # direction: y, -x
        onehot_y_x_shift = torch.cat((onehot_y_shift[:, :, :, 1:], onehot_y_shift[:, :, :, [0]]), 3)
        hop = 1 - 2 * torch.cat((onehot_matrix_site_wised + onehot_y_shift.unsqueeze(0),
                                onehot_matrix_site_wised + onehot_x_shift.unsqueeze(0),
                                onehot_matrix_site_wised + onehot_yx_shift.unsqueeze(0),
                                onehot_matrix_site_wised + onehot_y_x_shift.unsqueeze(0)), 0)   # [4, ly, lx, ly, lx]

        n = self.s0.shape[0]
        spin0 = self.s0.view(n, self.ly, self.lx) - 1 / 2

        # diagonalization term of <i|H|j>
        # num_chain; (flip_site_x, flip_site_y)
        positive_shift_y = torch.cat((spin0[:, 1:, :], spin0[:, [0], :]), 1)
        # band_* : 1/4 or -1/4
        band_y = spin0 * positive_shift_y
        band_x = spin0 * torch.cat((spin0[:, :, 1:], spin0[:, :, [0]]), 2)
        band_yx = spin0 * torch.cat((positive_shift_y[:, :, 1:], positive_shift_y[:, :, [0]]), 2)
        band_y_x = spin0 * torch.cat((positive_shift_y[:, :, [-1]], positive_shift_y[:, :, :-1]), 2)

        # [n, 4, ly, lx]
        # nearest-neighbor
        band_1 = torch.cat((band_y.view(n, 1, self.ly, self.lx), band_x.view(n, 1, self.ly, self.lx)), 1)
        # next nearest-neighbor
        band_2 = torch.cat((band_yx.view(n, 1, self.ly, self.lx), band_y_x.view(n, 1, self.ly, self.lx)), 1)
        band = torch.cat((band_1, band_2), 1)

        # [n, 4, ly, lx]
        flag = 1 / 2 - 2 * band   # flag the nearest neighbor and sub neighbor which spin are opposite (flag as 1 or 0)
        flag_bool = flag.to(torch.bool)
        n_flag = torch.sum(flag).to(torch.int64)
        index = flag.view(n * 4 * self.ly * self.lx) * self.index
        cfg_flip = hop * spin0.view(n, 1, 1, 1, self.ly, self.lx)  # [n, 4, ly, lx, ly, lx]
        cfg_flip_flag = cfg_flip[flag_bool]

        with torch.no_grad():
            amp_1_flag = self.amp_net((cfg_flip_flag + 1 / 2).view(n_flag, self.ly * self.lx))
            sign_1_flag = torch.ones_like(amp_1_flag, dtype=torch.int64, device=self.device)
        sign = self.sign_t.scatter(0, index[index>0].to(torch.int64) - 1, sign_1_flag).view(n, 4, self.ly, self.lx)
        amp = self.amp_t.scatter(0, index[index>0].to(torch.int64) - 1, amp_1_flag).view(n, 4, self.ly, self.lx)
        
        d_phi = calc_the_wave_function_advmc(sign.view(n, 4, self.ly, self.lx) / sign_0.view(n, 1, 1, 1),
                                             (amp.view(n, 4, self.ly, self.lx) - amplitude_0.view(n, 1, 1, 1)))
        d_phi_y, d_phi_x, d_phi_yx, d_phi_y_x = torch.split(d_phi, 1, dim=1)

        h_loc = self.J * (band_1.sum([1, 2, 3]) + self.sign_y * 0.5 * d_phi_y.sum([1, 2, 3]) + self.sign_x * 0.5 * d_phi_x.sum([1, 2, 3])) +\
            self.Jp * (band_2.sum([1, 2, 3]) + self.sign_yx * 0.5 * d_phi_yx.sum([1, 2, 3]) + self.sign_y_x * 0.5 * d_phi_y_x.sum([1, 2, 3]))

        # torch.cuda.synchronize()
        # print("after h_loc:", torch.cuda.memory_allocated()/8, torch.cuda.max_memory_allocated()/8, flush=True)
        # torch.cuda.reset_peak_memory_stats()

        return h_loc.detach()


    def h_loc_matrix_periodic_TFIsing(self):
        """
        matrix form could be runing on Gpu, and fast.
        """

        amplitude_0 = self.amp_net(self.s0)
        # sign_0 = torch.ones_like(amplitude_0, dtype=torch.int64, device=self.device)

        onehot_matrix_site_wised = (
            torch.eye(self.lx * self.ly).view(self.ly, self.lx, self.ly, self.lx)).to(self.device)
        hop = 1 - 2 * onehot_matrix_site_wised  # [ly, lx, ly, lx]

        n = self.s0.shape[0]
        spin0 = self.s0.view(n, self.ly, self.lx) - 1 / 2

        # diagonalization term of <i|H|j>
        # num_chain; (flip_site_x, flip_site_y)
        # band_* : 1/4 or -1/4
        band_y = spin0 * torch.cat((spin0[:, 1:, :], spin0[:, [0], :]), 1)
        band_x = spin0 * torch.cat((spin0[:, :, 1:], spin0[:, :, [0]]), 2)

        # [n, 2, ly, lx]
        # nearest-neighbor
        band = torch.cat((band_y.view(n, 1, self.ly, self.lx), band_x.view(n, 1, self.ly, self.lx)), 1)

        cfg_flip = hop * spin0.view(n, 1, 1, self.ly, self.lx)  # [n, ly, lx, ly, lx]

        with torch.no_grad():
            amp_1 = self.amp_net((cfg_flip + 1 / 2).view(-1, self.ly * self.lx))
            # sign_1 = torch.ones_like(amp_1, dtype=torch.int64, device=self.device)
        
        # d_phi = calc_the_wave_function_advmc(sign_1.view(n, self.ly, self.lx) / sign_0.view(n, 1, 1),
        d_phi = calc_the_wave_function_advmc(1, (amp_1.view(n, self.ly, self.lx) - amplitude_0.view(n, 1, 1)))

        h_loc = - 4 * band.sum([1, 2, 3]) - self.config.lambda_f * d_phi.sum([1, 2])

        return h_loc.detach()

    def spin_structure_factor(self):
        s0_z = (self.s0.clone() - 0.5).view(self.n_mc, self.ly, self.lx)
        s_div_N = s0_z / self.size
        s_q = torch.fft.fft2(s_div_N).view(self.n_mc, self.size)
        s_q_star = s_q.conj()
        SSFactor = (s_q * s_q_star)
        SSFactor_expectation = torch.sum(SSFactor, dim=0) / self.n_mc
        return SSFactor_expectation

    def update(self):
        # only shows spin flip update preserving total spin sz,
        # the most general case is not implemented here
        s0 = self.s0

        batch_size = s0.shape[0]
        if self.config.TFIsingModel:
            o = torch.randint(0, self.size, [batch_size, 1]).to(self.config.device)
            eo = torch.reshape(torch.nn.functional.one_hot(o, self.size), [batch_size, self.size])
        if self.config.HeisenbergModel:
            o = (torch.multinomial(torch.cat((s0, 1 - s0)), 1)[:, 0])
            eo = torch.sum(torch.reshape(torch.nn.functional.one_hot(o, self.size), [2, batch_size, self.size]), dim=0)
        ds = eo - 2 * eo * s0  # choose where -1 to 1 and where 1 to -1.
        s1 = s0 + ds  # distinguish the rolling-over way of specified location
        choose_ratio = 1.0

        amplitude_0 = self.amp_net(s1.to(self.config.device))
        net_phi = torch.exp(amplitude_0)

        flag = (torch.le(
            (torch.rand([self.n_mc], dtype=torch.float64)).to(self.config.device) + 5.5511151231257827021181583404541e-17,
            (torch.pow((net_phi / self.net_phi_0), 2) * choose_ratio)
        )).float()

        amplitude_0 = amplitude_0 * flag + self.amplitude_0 * (1.0 - flag)
        net_phi = net_phi * flag + self.net_phi_0 * (1.0 - flag)
        accept = torch.mean(flag)
        # flag = torch.reshape(flag, [1, self.n_mc])
        # flag = torch.t(flag.repeat(self.size, 1))
        flag = torch.reshape(flag, [self.n_mc, 1]).repeat(1, self.size)
        s1 = s0 + flag * ds

        return s1, amplitude_0, net_phi, accept

    def set_optimizer(self, opt, **kw):
        raise Exception("no implementation in base class ADVMC")

    def get_trainable_pars(self):
        lt = []
        for group in self.parameters():
            if group.requires_grad:
                lt.append(group.view(-1))
        pars = torch.cat(lt, dim=0)
        return pars.to(self.cpu)

    def get_hloc(self):
        return self.vhloc.detach()

    def save_net(self, di):
        if not os.path.isdir("./mdlparams_advmc"):
            os.mkdir("./mdlparams_advmc")
        # save_sign_net = {"sign_net_state_dict": self.sign_net.state_dict()}
        save_amplitude_net = {"amplitude_net_state_dict": self.amplitude_net.state_dict()}
        # torch.save(save_sign_net, './mdlparams_advmc/advmc_sign_net_{}.pth'.format(di))
        torch.save(save_amplitude_net, './mdlparams_advmc/advmc_amplitude_net_{}.pth'.format(di))


def init_cfg(size, batch_size, n_occ, dtype):
    """
    :return: init config and an identity tensor(used in h_loc)
    """
    s0 = torch.reshape(
        torch.t(
            torch.stack(  # torch.stack()
                [
                    torch.ones([n_occ], dtype=dtype),  # [lx * ly / 2 ]   torch.ones()
                    torch.zeros([size - n_occ], dtype=dtype),  # torch.zeros()
                ]
            )
        ),
        [1, size],
    ).repeat(batch_size, 1)  # [n_mc, self.size]
    return s0


def random_init_cfg(size, config, batch_size, n_occ, dtype, device):
    cfg = torch.reshape(torch.t(torch.cat((torch.ones([1, n_occ], dtype=dtype),
                        torch.zeros([1, size - n_occ], dtype=dtype)), dim=1)),
                        [1, size]).repeat(batch_size, 1).to(device)  # [n_mc, self.size]
    s0 = cfg.clone()

    if config.TFIsingModel:
        for i in range(size):
            o = torch.randint(0, size, [batch_size, 1]).to(device)
            eo = torch.reshape(torch.nn.functional.one_hot(o, size), [batch_size, size])
            ds = eo - 2 * eo * s0  # choose where -1 to 1 and where 1 to -1.
            s0 = s0 + ds  # distinguish the rolling-over way of specified location

    if config.HeisenbergModel:
        for i in range(size):
            o = (torch.multinomial(torch.cat(
                (torch.reshape(s0, [batch_size, size]),
                torch.reshape((1 - s0), [batch_size, size])), 0).to(dtype), 1)[:, 0]).detach()  # [n_mc]
            onehot_o1, onehot_o2 = torch.chunk(F.one_hot(o, size), 2, dim=0)
            eo = onehot_o1 + onehot_o2
            ds = eo - 2 * eo * s0  # choose where -1 to 1 and where 1 to -1.
            s0 = s0 + ds  # (distinguish the rolling-over way of specified location)

    print("init cfg0 over!", s0.dtype, s0.shape, "\n")

    return s0


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def progress_advmc(config):
    print("advmc progress start!!!\n", flush=True)

    if config.TFIsingModel:
        print("TFIM: Ly={}, lx={}\n".format(config.ly, config.lx), flush=True)
    if config.HeisenbergModel:
        if config.Jp == 0.:
            print("HM: Ly={}, lx={}\n".format(config.ly, config.lx), flush=True)
        else:
            print("HJ1J2M: Ly={}, lx={}, J2/J1={}\n".format(config.ly, config.lx, config.Jp), flush=True)
    print("run program on:", config.device, "\n", flush=True)
    if not os.path.isdir("./results_advmc"):
        os.mkdir("./results_advmc")

    advmc_lr_list = []

    # init s0
    s0 = random_init_cfg(config.size, config, config.advmc_n_mc, config.n_occ, dtype=config.dtype, device=config.device)

    model = ADVMC(s0, config)
    optimizer = torch.optim.Adam(model.amplitude_net.parameters(), lr=0)
    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.advmc_milestones,
                                                       gamma=config.advmc_milestones_gamma)
    
    if config.advmc_relay:
        model_path = config.relay_path
        if os.path.exists(model_path):
            print('init model parameters from FILE!', flush=True)
            checkpoint = torch.load(model_path)
            model.amplitude_net.load_state_dict(checkpoint['amplitude_net_state_dict'])
        else:
            print('error! no pth model in the path:./read_mdlparams', flush=True)
            quit()

    model.update_net_phi0()

    if config.advmc_warmup > 0:
        with torch.no_grad():
            print("advmc_warmup:", config.advmc_warmup, flush=True)
            for i in range(config.advmc_warmup):
                _, _ = model.update_conf()
                if config.debug:
                    print("advmc_warmup Epoch:{:<6}".format(i), flush=True)

    if config.advmc_relay:
        # Test the model
        print("\nStart test the model!!!\n")
        with torch.no_grad():
            for i in range(config.advmc_max_iter):
                t_step_start = time.time()
                s0, accept_rate = model.update_conf()
                energy, energy_std, _ = model(model.s0)
                ssf = model.spin_structure_factor()
                advmc_lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
                t_step_end = time.time()
                t_step = t_step_end - t_step_start

                if config.debug:
                    if i == 0:
                        print("{:<6}".format("step"),
                            "{:<21}".format("energy"),
                            "{:<16}".format("accept_rate"),
                            "{:<10}".format("time"),
                            flush=True,
                        )
                    print(
                        "{:<6}".format(i),
                        "{:<21.16f}".format(energy),
                        "{:<16.6%}".format(accept_rate),
                        "{:<10.6f}".format(t_step),
                        flush=True,
                    )

                with open(config.file_name_H, 'ab') as file_:
                    file_.write(bytes('{:<6}\t'.format(i), 'UTF-8'))
                    np_energy_and_std = np.hstack((energy.detach().to("cpu").numpy(), energy_std.detach().to("cpu"))).reshape(1, 2)
                    np.savetxt(file_, np_energy_and_std, fmt='%.16f', delimiter='\t')
                with open(config.file_name_ssf_real, 'ab') as file_:
                    # file_.write(bytes('{:<6} '.format(i), 'UTF-8'))
                    np_ssf = ssf.real.to('cpu').numpy().reshape(1, config.size)
                    np.savetxt(file_, np_ssf, fmt='%.6f', delimiter=' ')
                with open(config.file_name_ssf_imag, 'ab') as file_:
                    # file_.write(bytes('{:<6} '.format(i), 'UTF-8'))
                    np_ssf = ssf.imag.to('cpu').numpy().reshape(1, config.size)
                    np.savetxt(file_, np_ssf, fmt='%.6f', delimiter=' ')

            calc_std(config)
            calc_ssf_mean(config)

    else:
        # Train the model
        print("\nStart train the model!!!\n")
        for i in range(config.advmc_max_iter):
            t_step_start = time.time()
            with torch.no_grad():
                s0, accept_rate = model.update_conf()

            energy, energy_std, loss_reg = model(model.s0)
            loss = energy + loss_reg
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if i < config.advmc_lr_step_n:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = param_group["lr"] + config.advmc_lr_step

            lr_schedule.step()
            model.update_net_phi0()
            advmc_lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
            t_step_end = time.time()
            t_step = t_step_end - t_step_start

            if config.debug:
                if i == 0:
                    print("{:<6}".format("step"),
                        "{:<21}".format("energy"),
                        "{:<21}".format("loss"),
                        "{:<21}".format("reg"),
                        "{:<16}".format("accept_rate"),
                        "{:<10}".format("time"),
                        "{:<12}".format("lr"),
                        flush=True,
                    )
                print(
                    "{:<6}".format(i),
                    "{:<21.16f}".format(energy),
                    "{:<21.16f}".format(loss[0]),
                    "{:<21.16f}".format(loss_reg[0]),
                    "{:<16.6%}".format(accept_rate),
                    "{:<10.6f}".format(t_step),
                    "{:<12.8f}".format(advmc_lr_list[i]),
                    flush=True,
                )

            with torch.no_grad():
                with open(config.file_name_H, 'ab') as file_:
                    file_.write(bytes('{:<6}\t'.format(i), 'UTF-8'))
                    np_energy_and_std = np.hstack((energy.detach().to("cpu").numpy(),
                                                   energy_std.detach().to("cpu"), loss.to("cpu"), loss_reg.to("cpu"))).reshape(1, 4)
                    np.savetxt(file_, np_energy_and_std, fmt='%.16f', delimiter='\t')

            # if (i + 1) % config.advmc_save_interval == 0:
            #     model.save_net(i)
        model.save_net("finally")


def calc_std(config):
    n_bin = config.n_bin
    file_name = config.file_name_H
    print("\n Calc std_row and std_column !!! \n", flush=True)
    datas = np.loadtxt(file_name)
    print("shape of datas:", datas.shape, flush=True)
    bin_size = datas.shape[0] // n_bin
    energy = datas[:, 1].reshape((-1, bin_size))
    print("energy array column:{} row:{}".format(energy.shape[0], energy.shape[1]), flush=True)
    std_mean_row = np.std(np.mean(energy, axis=1), ddof=1) / np.sqrt(n_bin)
    var_mean_std_column = np.power(np.mean(np.std(energy, axis=0, ddof=1)) * np.sqrt(config.advmc_n_mc), 2)
    print("var_mean_std_column \t std_mean_row\n",
          "{:<24.16f}\t".format(var_mean_std_column),
          "{:<24.16f}\n".format(std_mean_row),
          flush=True)


def calc_ssf_mean(config):
    file_name = config.file_name_ssf_real
    datas = np.loadtxt(file_name)
    mean_ssf = datas.mean(axis=0).reshape([config.ly, config.lx])
    ssf_0_2pi_dim1 = np.append(mean_ssf, mean_ssf[0, :].reshape(1, config.lx), axis = 0)
    ssf_0_2pi = np.append(ssf_0_2pi_dim1, ssf_0_2pi_dim1[:, 0].reshape(config.ly + 1, 1), axis = 1)

    print(ssf_0_2pi, ssf_0_2pi.shape, flush=True)

    with open(config.file_name_ssf_real_mean, 'ab') as file_:
        np_ssf = ssf_0_2pi
        np.savetxt(file_, np_ssf, fmt='%.6f', delimiter=' ')


def calc_the_wave_function_advmc(sign, amplitude):
    phi = sign * torch.exp(amplitude)
    return phi

