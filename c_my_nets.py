import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class InputBlockACnn(nn.Module):
    """
    in_l, out_l and inner_l: input size, output size and inner size.
    """
    def __init__(self, in_channel, out_channel, ly, lx, kernel, pd_md, device, dtype):
        super(InputBlockACnn, self).__init__()
        self.c_0 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel,
                             padding=kernel // 2, padding_mode=pd_md, device=device, dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.c_0(input))
        return out


class StepUpBlockAcnn(nn.Module):
    """
    in_l, out_l and inner_l: input size, output size and inner size.
    """
    def __init__(self, in_channel, inner_channel, out_channel, ly, lx, kernel, pd_md, device, dtype):
        super(StepUpBlockAcnn, self).__init__()
        self.c_0 = nn.Conv2d(in_channel, inner_channel, kernel_size=kernel,
                             padding=kernel // 2, padding_mode=pd_md, device=device, dtype=dtype)
        self.c_1 = nn.Conv2d(inner_channel, out_channel, kernel_size=kernel,
                             padding=kernel // 2, padding_mode=pd_md, device=device, dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.c_0(input))
        out = F.relu(self.c_1(x) + input)
        return out


class OutputBlockACnn(nn.Module):
    """
    in_l, out_l and inner_l: input size, output size and inner size.
    """
    def __init__(self, in_channel, out_channel, ly, lx, kernel, pd_md, device, dtype):
        super(OutputBlockACnn, self).__init__()
        self.c_out = nn.Conv2d(in_channel, out_channel, kernel_size=kernel,
                             padding=kernel // 2, padding_mode=pd_md, device=device, dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.c_out(input)
        return out




class AmpNetCnnCnV(nn.Module):
    """
    split a and b in the end.
    """
    def __init__(self, ly, lx, device, dtype, channel, kernel, num_step_up, pd_md):
        super(AmpNetCnnCnV, self).__init__()
        self.lx = lx
        self.ly = ly
        self.input_size = lx * ly
        self.device = device
        self.dtype = dtype
        self.channel = channel
        self.num_step_up = num_step_up

        self.in_block = InputBlockACnn(1, channel, self.ly, self.lx, kernel, pd_md,
                                       self.device, self.dtype)
        self.step_up_layer = nn.ModuleList([StepUpBlockAcnn(channel, channel, channel, self.ly, self.lx,
                                                            kernel, pd_md, self.device, self.dtype)
                                            for _ in range(self.num_step_up)])
        self.out_block = OutputBlockACnn(channel, 1, self.ly, self.lx, kernel, pd_md,
                                         self.device, self.dtype)

        # if self.ly == self.lx:
        #     self.forward = self.forwardC4V
        # else:
        #     self.forward = self.forwardC2V

    # # forward_no-cv
    # def forward(self, x) -> [torch.tensor, torch.tensor]:
    #     x = torch.reshape(x, [-1, 1, self.ly, self.lx])
    #     out = self.in_block(x)
    #     for step_layer in self.step_up_layer:
    #         out = step_layer(out)
    #     out = self.out_block(out)

    #     out = torch.mean(out.view([-1, self.ly, self.lx]), [1, 2], keepdim=False).reshape(-1, 1) # * 16
    #     return out.reshape(-1)

    # # forwardC2V
    # def forward(self, x) -> [torch.tensor, torch.tensor]:
    #     x = torch.reshape(x, [-1, 1, self.ly, self.lx])
    #     xx = torch.cat([x, torch.rot90(x, -2, [2, 3])])
    #     xxx = torch.cat([xx.view([-1, 1, self.ly, self.lx]),
    #                      torch.reshape(torch.fliplr(xx.view([-1, self.ly, self.lx])), [-1, 1, self.ly, self.lx])])
        
    #     out = self.in_block(xxx)
    #     for step_layer in self.step_up_layer:
    #         out = step_layer(out)
    #     out = self.out_block(out)

    #     # out = torch.sum(out.view([2, 2, -1, self.ly, self.lx]), [0, 1, 3, 4], keepdim=False)
    #     out = torch.mean(out.view([2, 2, -1, self.ly, self.lx]), [0, 1, 3, 4], keepdim=False).reshape(-1, 1) # * 16
    #     return out.reshape(-1)
    
    # forwardC4V
    def forward(self, x) -> [torch.tensor, torch.tensor]:
        x = torch.reshape(x, [-1, 1, self.ly, self.lx])
        # print(x.shape, flush=True)
        xx = torch.cat([x, torch.rot90(x, -1, [2, 3]), torch.rot90(x, -2, [2, 3]), torch.rot90(x, -3, [2, 3])])
        # print(xx.shape, flush=True)
        xxx = torch.cat([xx.view([-1, 1, self.ly, self.lx]),
                         torch.reshape(torch.fliplr(xx.view([-1, self.ly, self.lx])), [-1, 1, self.ly, self.lx])])
        # print(xxx.shape, flush=True)

        out = self.in_block(xxx)
        for step_layer in self.step_up_layer:
            out = step_layer(out)
        out = self.out_block(out)

        # out = torch.sum(out.view([2, 4, -1, self.ly, self.lx]), [0, 1, 3, 4], keepdim=False)
        out = torch.mean(out.view([2, 4, -1, self.ly, self.lx]), [0, 1, 3, 4], keepdim=False).reshape(-1, 1) # * 16
        return out.reshape(-1)

    def __call__(self, xx):
        """
        let the instance of this class is callable.
        """
        # if self.ly == self.lx:
        #     return self.forwardC4V(xx)
        # else:
        #     return self.forwardC2V(xx)
        return self.forward(xx)


class AmpNetCnnC6VTriangular(nn.Module):
    """
    split a and b in the end.
    """
    def __init__(self, ly, lx, device, dtype, channel, kernel, num_step_up, pd_md):
        super(AmpNetCnnC6VTriangular, self).__init__()
        self.lx = lx
        self.ly = ly
        input_size = lx * ly
        in_channel = 1
        lattice = "Square"
        sym = SymmetryHexagon(ly, lx, device)
        # self.sym_index = sym.sym_index.reshape(-1, ly * lx).to("cpu")
        self.sym_index = Parameter(sym.sym_index.reshape(-1, ly * lx).to(device), requires_grad=False)

        self.in_block = InputBlockACnn(1, channel, self.ly, self.lx, kernel, pd_md,
                                       self.device, self.dtype)
        self.step_up_layer = nn.ModuleList([StepUpBlockAcnn(channel, channel, channel, self.ly, self.lx,
                                                            kernel, pd_md, self.device, self.dtype)
                                            for _ in range(num_step_up)])
        self.out_block = OutputBlockACnn(channel, 1, self.ly, self.lx, kernel, pd_md,
                                         self.device, self.dtype)

    def forward(self, x) -> [torch.tensor, torch.tensor]:
        # x = torch.reshape(x, [-1, 1, self.ly, self.lx])
        # print(x.shape, flush=True)
        xx = torch.cat([x, x * (-1) + 1], dim=0).reshape(-1, self.ly * self.lx)
        # print(xx.device, self.sym_index.device)
        xxx = torch.cat([xx,
                         xx[:, self.sym_index[1]],  # .reshape(-1, self.ly*self.lx)
                         xx[:, self.sym_index[2]],
                         xx[:, self.sym_index[3]],
                         xx[:, self.sym_index[4]],
                         xx[:, self.sym_index[5]],
                         xx[:, self.sym_index[6]],
                         xx[:, self.sym_index[7]],
                         xx[:, self.sym_index[8]],
                         xx[:, self.sym_index[9]],
                         xx[:, self.sym_index[10]],
                         xx[:, self.sym_index[11]]], dim=0).reshape(-1, 1, self.ly, self.lx)

        out = self.in_block(xxx)
        for step_layer in self.step_up_layer:
            out = step_layer(out)
        out = self.out_block(out)
        
        out = torch.mean(out.view([12, 2, -1, self.ly, self.lx]), [0, 1, 3, 4], keepdim=False).reshape(-1, 1) * 16
        return out.reshape(-1)

    def __call__(self, xx):
        """
        let the instance of this class is callable.
        """
        return self.forward(xx)


class SymmetryHexagon():
    def __init__(self, ly, lx, device):
        self.ly = ly
        self.lx = lx
        self.device = device
        dtype = torch.int64
        self.lattice = torch.arange(0, ly*lx, 1, requires_grad=False, device=device, dtype=dtype).reshape(ly, lx)
        self.sym_index = torch.zeros([6*2, self.ly, self.lx], requires_grad=False, device=device, dtype=dtype)
        # print("before:\n", self.sym_index)
        self.sym_index[0] += self.lattice
        self.rotC6(self.lattice, self.sym_index[1])
        self.rotC6(self.sym_index[1], self.sym_index[2])
        self.rotC6(self.sym_index[2], self.sym_index[3])
        self.rotC6(self.sym_index[3], self.sym_index[4])
        self.rotC6(self.sym_index[4], self.sym_index[5])
        self.refl0(self.sym_index[0], self.sym_index[6])
        self.refl0(self.sym_index[1], self.sym_index[7])
        self.refl0(self.sym_index[2], self.sym_index[8])
        self.refl0(self.sym_index[3], self.sym_index[9])
        self.refl0(self.sym_index[4], self.sym_index[10])
        self.refl0(self.sym_index[5], self.sym_index[11])
        # print("after:\n", self.sym_index)

        # r2p = ((lattice.reshape(1, -1)).gather(dim=1, index = r2.reshape(1, -1))).reshape(self.ly, self.lx)
        # print(r2p)

    def rotC6(self, index_in, index_out):
        # C6 rotation 60° 
        for i in range(self.ly*self.lx):
            r_y = i//self.ly
            r_x = i%self.lx
            rp_x = r_y%self.ly%self.lx
            rp_y = ((self.lx-r_x%self.lx) + r_y%self.ly)%self.ly
            index_out[rp_y, rp_x] += index_in[r_y, r_x]

    def refl0(self, index_in, index_out):
        # reflection align lx
        for i in range(self.ly*self.lx):
            r_y = i//self.ly
            r_x = i%self.lx
            rp_x = (r_x%self.lx+self.lx-r_y%self.ly)%self.lx
            rp_y = (self.ly-r_y%self.ly)%self.ly
            index_out[rp_y, rp_x] += index_in[r_y, r_x]
