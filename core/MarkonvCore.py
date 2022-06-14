import pdb

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Our module!
from torch.utils.cpp_extension import load

file_dir = os.path.split(os.path.realpath(__file__))[0]

## Needs Ninja to load (pip install Ninja)
os.makedirs(file_dir+"/build", exist_ok=True)
os.makedirs(file_dir+"/build_stride", exist_ok=True)
markonv_stride_cpp = load(name='markonv_stride_cpp', sources=[file_dir+'/Markonv_cuda_stride.cpp', file_dir+'/Markonv_cuda_kernel_stride.cu'], 
                   extra_cflags=["-Wno-deprecated-declarations"],
                   build_directory=file_dir+"/build_stride", verbose=True)

markonv_cpp = load(name='markonv_cpp', sources=[file_dir+'/Markonv_cuda.cpp', file_dir+'/Markonv_cuda_kernel.cu'],
                   extra_cflags=["-Wno-deprecated-declarations"],
                   ## extra_cflags=["-g"], extra_cuda_cflags=["-g", "-G"],
                   build_directory=file_dir+"/build", verbose=True)

class MarkonvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, Kernel_Full_4DTensor):
        input = input.float()
        Kernel_Full_4DTensor = Kernel_Full_4DTensor.float()
        output_old = torch.zeros([input.shape[0], Kernel_Full_4DTensor.shape[3], input.shape[2] - Kernel_Full_4DTensor.shape[0]]).to(torch.device("cuda"))
        output_new = markonv_cpp.forward(input, Kernel_Full_4DTensor, output_old)
        ctx.save_for_backward(input, Kernel_Full_4DTensor)
        return output_new[0]

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.float()
        input, Kernel_Full_4DTensor = ctx.saved_tensors
        grad_input, grad_Kernel_Full_4DTensor = markonv_cpp.backward(grad_output.contiguous(), input, Kernel_Full_4DTensor)
        return grad_input, grad_Kernel_Full_4DTensor

class MarkonvFunction_stride(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, Kernel_Full_4DTensor, stride):
        input = input.float()
        Kernel_Full_4DTensor = Kernel_Full_4DTensor.float()
        output_old = torch.zeros([input.shape[0], Kernel_Full_4DTensor.shape[3], math.ceil((input.shape[2] - Kernel_Full_4DTensor.shape[0])/stride)]).to(torch.device("cuda")) 
        output_new = markonv_stride_cpp.forward(input, Kernel_Full_4DTensor, output_old, stride)
        ctx.save_for_backward(input, Kernel_Full_4DTensor, torch.from_numpy(np.asarray(stride)))
        return output_new[0]

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.float()
        input, Kernel_Full_4DTensor, stride = ctx.saved_tensors
        grad_input, grad_Kernel_Full_4DTensor = markonv_stride_cpp.backward(grad_output.contiguous(), input, Kernel_Full_4DTensor, stride)
        return grad_input, grad_Kernel_Full_4DTensor, None


class Markonv_stride(torch.nn.Module):
    def __init__(self, kernel_length, kernel_number, channel_size, stride):
        ## Initialize the instance
        super(Markonv_stride, self).__init__()
        ## Save configs
        self.kernel_length = kernel_length
        self.kernel_number = kernel_number
        self.channel_size = channel_size
        self.stride = stride
        ## Build weights
        self.Kernel_Full_4DTensor = torch.nn.Parameter(torch.empty(
            self.kernel_length,
            self.channel_size,
            self.channel_size,
            self.kernel_number
        ))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.Kernel_Full_4DTensor, a=0.0, b=1.0)

    def forward(self, input):
        return MarkonvFunction_stride.apply(input, self.Kernel_Full_4DTensor, self.stride)


class Markonv(torch.nn.Module):
    def __init__(self, kernel_length, kernel_number, channel_size, channel_last=1):
        ## Initialize the instance
        super(Markonv, self).__init__()
        ## Save configs
        self.kernel_length = kernel_length
        self.kernel_number = kernel_number
        self.channel_size = channel_size
        self.channel_last = channel_last
        ## Build weights
        self.Kernel_Full_4DTensor = torch.nn.Parameter(torch.empty(
            self.kernel_length,
            self.channel_size,
            self.channel_size,
            self.kernel_number
        ))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.Kernel_Full_4DTensor, a=-0.05, b=0.05)
        #torch.nn.init.uniform_(self.Kernel_Full_4DTensor, a=-0.3, b=0.3)
        # torch.nn.init.xavier_uniform_(self.Kernel_Full_4DTensor)
        # torch.nn.init.kaiming_uniform_(self.Kernel_Full_4DTensor)

    def forward(self, input):
        if self.channel_last:
            input = input.permute((0,2,1)).contiguous().float()
        return MarkonvFunction.apply(input, self.Kernel_Full_4DTensor)

class MarkonvR(torch.nn.Module):
    def __init__(self, kernel_length, kernel_number, channel_size, channel_last=1, bias=False):
        ## Initialize the instance
        super(MarkonvR, self).__init__()
        ## Save configs
        self.channel_last = channel_last
        self.kernel_length = kernel_length
        self.kernel_number = kernel_number
        self.channel_size = channel_size
        ## Build weights
        self.Kernel_Full_4DTensor = torch.nn.Parameter(torch.empty(
            self.kernel_length,
            self.channel_size,
            self.channel_size,
            int(self.kernel_number/2)
        ))
        self.Kernel_Full_4DTensorR = torch.nn.Parameter(torch.empty(
            self.kernel_length,
            self.channel_size,
            self.channel_size,
            int(self.kernel_number/2)
        ))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(kernel_number))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.uniform_(self.Kernel_Full_4DTensor, a=-0.3, b=0.3)
        torch.nn.init.xavier_uniform_(self.Kernel_Full_4DTensor)
        torch.nn.init.xavier_uniform_(self.Kernel_Full_4DTensorR)
        # torch.nn.init.kaiming_uniform_(self.Kernel_Full_4DTensor)
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.Kernel_Full_4DTensor)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.channel_last:
            input = input.permute((0,2,1)).contiguous().float()
        #### Calculate forward prob
        kernel_forward = self.Kernel_Full_4DTensor
        probforward = MarkonvFunction.apply(input, kernel_forward)
        #### Calculate forward reverse prob
        inputR = torch.flip(input, dims=[2])
        kernel_reverse = self.Kernel_Full_4DTensorR
        probreverse = MarkonvFunction.apply(inputR, kernel_reverse)

        output = torch.cat((probforward, probreverse), dim=1)

        if self.bias is not None:
            for i in range(len(self.bias)):
                output[:,i,:] += self.bias[i]
        return output


class MarkonvV(torch.nn.Module):
    def __init__(self, kernel_length, kernel_number, channel_size, initKernelLen=None, channel_last=1, bias=False):
        ## Initialize the instance
        super(MarkonvV, self).__init__()
        ## Save configs
        self.kernel_length = kernel_length
        self.kernel_number = kernel_number
        self.channel_size = channel_size
        self.channel_last = channel_last
        if initKernelLen:
            self.initKernelLen = initKernelLen
        else:
            self.initKernelLen = int(self.kernel_length/3*2)
        self.padlen = int((self.kernel_length -self.initKernelLen)/2)+1

        ## Build weights
        self.Kernel_Full_4DTensor = torch.nn.Parameter(torch.empty(
            self.kernel_length,
            self.channel_size,
            self.channel_size,
            int(self.kernel_number/2)
        ))

        self.Kernel_Full_4DTensorR = torch.nn.Parameter(torch.empty(
            self.kernel_length,
            self.channel_size,
            self.channel_size,
            int(self.kernel_number/2)
        ))

        self.k_weights = torch.nn.Parameter(torch.from_numpy(self.initMask()).float())
        self.k_weightsR = torch.nn.Parameter(torch.from_numpy(self.initMask()).float())
        
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(kernel_number))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.Kernel_Full_4DTensor)
        torch.nn.init.xavier_uniform_(self.Kernel_Full_4DTensorR)
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.Kernel_Full_4DTensor)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def initMask(self):
        k_weights_shape = (2,) + (1, int(self.kernel_number/2))
        k_weights = np.zeros(k_weights_shape)
        init_num = int(self.kernel_number/2)
        init_part_left = np.zeros([1, k_weights_shape[1], init_num]) + (self.kernel_length - self.initKernelLen) / 2
        init_part_right = np.zeros((1, k_weights_shape[1], init_num)) + (self.kernel_length + self.initKernelLen) / 2
        k_weights[0, :, :] = init_part_left
        k_weights[1, :, :] = init_part_right
        return k_weights


    def init_left(self,k_weights):
        """
        Used to generate a leftmask
        :return:
        """
        #[self.kernel_length]
        k_weights_tem_2d_left = torch.arange(self.kernel_length).to(device=k_weights.device)
        #[self.kernel_length,1]
        k_weights_tem_2d_left = torch.unsqueeze(k_weights_tem_2d_left, 1)
        #[self.kernel_length,self.kernel_number/2]
        k_weights_tem_2d_left= k_weights_tem_2d_left
        k_weights_tem_3d_left = torch.repeat_interleave(k_weights_tem_2d_left,k_weights.shape[2], dim=1) - k_weights[0, :, :]
        #[self.kernel_length,1,self.kernel_number/2]
        return torch.unsqueeze(k_weights_tem_3d_left, 1)

    def init_right(self,k_weights):
        """
        Used to generate a rightmask
        :return:
        """

        k_weights_tem_2d_right = torch.arange(self.kernel_length).to(device=k_weights.device)
        k_weights_tem_2d_right = torch.unsqueeze(k_weights_tem_2d_right, 1)
        k_weights_tem_2d_right = k_weights_tem_2d_right
        k_weights_tem_3d_right = - (torch.repeat_interleave(k_weights_tem_2d_right, k_weights.shape[2],
                                                          dim=1)- k_weights[1, :, :])
        return torch.unsqueeze(k_weights_tem_3d_right, 1)

    def GeneRateMask(self,k_weights):
        """

        Returns:

        """
        k_weights_3d_left = self.init_left(k_weights)
        k_weights_3d_right = self.init_right(k_weights)
        k_weights_left = torch.sigmoid(k_weights_3d_left)
        k_weights_right = torch.sigmoid(k_weights_3d_right)
        #[self.kernel_length,1,self.kernel_number/2]
        MaskFinal = k_weights_left + k_weights_right - 1
        #[kernel length, self.channels, kernel number]
        mask = torch.repeat_interleave(MaskFinal, self.channel_size, dim=1)
        #[kernel length,1, self.channels, kernel number]
        mask = torch.unsqueeze(mask, 1)
        #[kernel length,self.channels, self.channels, kernel number]
        mask = torch.repeat_interleave(mask, self.channel_size, dim=1)
        return mask


    def forward(self, input):
        if self.channel_last:
            input = input.permute((0,2,1)).contiguous().float()
        p1d = (self.padlen, self.padlen)
        #### Calculate forward prob
        padinput = F.pad(input, p1d, "constant", 0)
        mask_forward = self.GeneRateMask(self.k_weights)
        kernel_forward = self.Kernel_Full_4DTensor * mask_forward
        probforward = MarkonvFunction.apply(padinput, kernel_forward)


        #### Calculate forward reverse prob
        inputR = torch.flip(input, dims=[2])
        padinputR = F.pad(inputR, p1d, "constant", 0)
        mask_reverse = self.GeneRateMask(self.k_weightsR)
        kernel_reverse = self.Kernel_Full_4DTensorR * mask_reverse
        probreverse = MarkonvFunction.apply(padinputR, kernel_reverse)
        output = torch.cat((probforward, probreverse), dim=1)
        if self.bias is not None:
            for i in range(len(self.bias)):
                output[:,i,:] += self.bias[i]

        return output


class MarkonvVS(torch.nn.Module):
    def __init__(self, kernel_length, kernel_number, channel_size, initKernelLen=None, channel_last=1, bias=False, stride=1):
        ## Initialize the instance
        super(MarkonvVS, self).__init__()
        ## Save configs
        self.kernel_length = kernel_length
        self.kernel_number = kernel_number
        self.channel_size = channel_size
        self.channel_last = channel_last
        if initKernelLen:
            self.initKernelLen = initKernelLen
        else:
            self.initKernelLen = int(self.kernel_length/3*2)
        self.padlen = int((self.kernel_length -self.initKernelLen)/2)+1

        ## Build weights
        self.Kernel_Full_4DTensor = torch.nn.Parameter(torch.empty(
            self.kernel_length,
            self.channel_size,
            self.channel_size,
            int(self.kernel_number/2)
        ))

        self.Kernel_Full_4DTensorR = torch.nn.Parameter(torch.empty(
            self.kernel_length,
            self.channel_size,
            self.channel_size,
            int(self.kernel_number/2)
        ))

        self.k_weights = torch.nn.Parameter(torch.from_numpy(self.initMask()).float())
        self.k_weightsR = torch.nn.Parameter(torch.from_numpy(self.initMask()).float())
        
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(kernel_number))
        else:
            self.register_parameter('bias', None)
        self.stride = stride
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.Kernel_Full_4DTensor)
        torch.nn.init.xavier_uniform_(self.Kernel_Full_4DTensorR)
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.Kernel_Full_4DTensor)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def initMask(self):
        k_weights_shape = (2,) + (1, int(self.kernel_number/2))
        k_weights = np.zeros(k_weights_shape)
        init_num = int(self.kernel_number/2)
        init_part_left = np.zeros([1, k_weights_shape[1], init_num]) + (self.kernel_length - self.initKernelLen) / 2
        init_part_right = np.zeros((1, k_weights_shape[1], init_num)) + (self.kernel_length + self.initKernelLen) / 2
        k_weights[0, :, :] = init_part_left
        k_weights[1, :, :] = init_part_right
        return k_weights


    def init_left(self,k_weights):
        """
        Used to generate a leftmask
        :return:
        """
        #[self.kernel_length]
        k_weights_tem_2d_left = torch.arange(self.kernel_length).to(device=k_weights.device)
        #[self.kernel_length,1]
        k_weights_tem_2d_left = torch.unsqueeze(k_weights_tem_2d_left, 1)
        #[self.kernel_length,self.kernel_number/2]
        k_weights_tem_2d_left= k_weights_tem_2d_left
        k_weights_tem_3d_left = torch.repeat_interleave(k_weights_tem_2d_left,k_weights.shape[2], dim=1) - k_weights[0, :, :]
        #[self.kernel_length,1,self.kernel_number/2]
        return torch.unsqueeze(k_weights_tem_3d_left, 1)

    def init_right(self,k_weights):
        """
        Used to generate a rightmask
        :return:
        """

        k_weights_tem_2d_right = torch.arange(self.kernel_length).to(device=k_weights.device)
        k_weights_tem_2d_right = torch.unsqueeze(k_weights_tem_2d_right, 1)
        k_weights_tem_2d_right = k_weights_tem_2d_right
        k_weights_tem_3d_right = - (torch.repeat_interleave(k_weights_tem_2d_right, k_weights.shape[2],
                                                          dim=1)- k_weights[1, :, :])
        return torch.unsqueeze(k_weights_tem_3d_right, 1)

    def GeneRateMask(self,k_weights):
        """

        Returns:

        """
        k_weights_3d_left = self.init_left(k_weights)
        k_weights_3d_right = self.init_right(k_weights)
        k_weights_left = torch.sigmoid(k_weights_3d_left)
        k_weights_right = torch.sigmoid(k_weights_3d_right)
        #[self.kernel_length,1,self.kernel_number/2]
        MaskFinal = k_weights_left + k_weights_right - 1
        #[kernel length, self.channels, kernel number]
        mask = torch.repeat_interleave(MaskFinal, self.channel_size, dim=1)
        #[kernel length,1, self.channels, kernel number]
        mask = torch.unsqueeze(mask, 1)
        #[kernel length,self.channels, self.channels, kernel number]
        mask = torch.repeat_interleave(mask, self.channel_size, dim=1)
        return mask


    def forward(self, input):
        if self.channel_last:
            input = input.permute((0,2,1)).contiguous().float()
        p1d = (self.padlen, self.padlen)
        #### Calculate forward prob
        padinput = F.pad(input, p1d, "constant", 0)
        mask_forward = self.GeneRateMask(self.k_weights)
        kernel_forward = self.Kernel_Full_4DTensor * mask_forward
        probforward = MarkonvFunction_stride.apply(padinput, kernel_forward, self.stride)


        #### Calculate forward reverse prob
        inputR = torch.flip(input, dims=[2])
        padinputR = F.pad(inputR, p1d, "constant", 0)
        mask_reverse = self.GeneRateMask(self.k_weightsR)
        kernel_reverse = self.Kernel_Full_4DTensorR * mask_reverse
        probreverse = MarkonvFunction_stride.apply(padinputR, kernel_reverse, self.stride)
        output = torch.cat((probforward, probreverse), dim=1)
        if self.bias is not None:
            for i in range(len(self.bias)):
                output[:,i,:] += self.bias[i]

        return output
