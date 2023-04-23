import numpy as np
import pandas as pd
import cvxpy as cp
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
from torch.autograd import Variable
import math

def to_tensor(x):
    if type(x) == np.ndarray:
        return torch.from_numpy(x).float()
    elif type(x) == torch.Tensor:
        return x
    else:
        print("Type error. Input should be either numpy array or torch tensor")

def adjust_lr_zt(optimizer, lr0, epoch):
    lr = lr0 * (1.0 / np.sqrt(epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr      

def SDRO_eval(theta, Lambda, Reg, X, Y):
    n,d = X.shape

    ratio_1 = Lambda / (Lambda - 2 * np.linalg.norm(theta)**2)
    residual = np.mean((X@theta - Y)**2,0)
    obj_1 = ratio_1 * residual
    obj_2 = Lambda*Reg/2 * np.linalg.slogdet(np.eye(d) - theta@theta.T*2/Lambda)[1]
    return obj_1[0] - obj_2, np.sqrt(residual[0])

def SDRO_oracle(dataloader, theta, Lambda, Reg, X_all ,y_all,
                        silence=False, test_iterations=10, maxiter = 50, step_size0 = 1e-2):
    #   Input:
    # Feature: N samples of R^d [dim: N*d]
    #  Target: labels of N samples [dim: N*1]
    #   theta: initial guess for optimization
    #  Output:
    #   theta: optimized decision
    #  objval: optimal value
    _,d = X_all.shape
    iter = 0
    theta = Variable(to_tensor(theta), requires_grad=True).type(torch.float64)


    theta_avg = torch.zeros_like(theta)



    for epoch in range(maxiter):
        step_size = step_size0 / np.sqrt(1 + epoch)
        for _, (data, target, idx) in enumerate(dataloader):
            # obtain gradient oracle
            
            ratio_1 = Lambda / (Lambda - 2 * torch.linalg.norm(theta)**2)
            residual = torch.mean((data@theta - target)**2,0)
            obj_1 = ratio_1 * residual
            obj_2 = Lambda*Reg/2 * torch.logdet(torch.eye(d) - theta@theta.T*2/Lambda)

            loss_theta = obj_1 - obj_2
            v = torch.autograd.grad(loss_theta, theta)[0]

            theta = theta - step_size * v

            if torch.linalg.norm(theta) > 0.95*np.sqrt(Lambda/2):
                theta = theta / torch.linalg.norm(theta) * 0.95*np.sqrt(Lambda/2)

            iter += 1
            theta_avg = theta_avg * (1 - 1/(iter+1)) + 1/(iter+1) * theta.detach().clone()

            if (silence == False) and (iter % test_iterations == 0):
                Loss_eval,_ = SDRO_eval(theta_avg.detach().numpy(), Lambda, Reg, X_all, y_all)
                #print(Loss_eval)
                print("Iter: {}, Loss: {:.2f}".format(iter, Loss_eval))
    return theta_avg.detach().numpy()

def SDRO_SG_solver(dataloader, theta, Reg, Lambda, 
                        maxiter = 1, learning_rate = 5e-2, silence=False, K_sample_max=5, test_iterations = 10):
    """
    2-SDRO Approach with SG estimator for Regression Problem
    #   Input:
    # Feature: N samples of R^d [dim: N*d]
    #  Target: labels of N samples [dim: N*1]
    #   theta: initial guess for optimization
    #  Lambda: Lagrangian multiplier
    #     Reg: bandwidth
    #  Output:
    #   theta: optimized decision
    """
    iter       = 0
    Lambda_Reg = Lambda * Reg

    theta_torch     = torch.tensor(theta, dtype=torch.float, requires_grad=True)
    optimizer_theta = torch.optim.SGD([theta_torch], lr=learning_rate)

    theta_hist = []
    cost_hist  = []
    theta_hist.append(theta_torch.clone().detach().numpy())
    cost_hist.append(0)

    for epoch in range(maxiter):
        for _, (data, target, idx) in enumerate(dataloader):
            iter = iter + 1

            # generate stochastic samples
            N, d         = data.shape
            data, target = Variable(data), Variable(target)
            m            = int(2**K_sample_max)

            optimizer_theta.zero_grad()
            data_noise     = torch.randn([m, N, d]) * np.sqrt(Reg) + data.reshape([1,N,d])
            data_noise_vec = data_noise.reshape([-1,d])
            target_noise   = target.repeat(m,1)

            haty     = data_noise_vec @ theta_torch.type(torch.float64)
            obj_vec  = (haty - target_noise) ** 2
            obj_mat  = obj_vec.reshape([m, N])
            Residual = obj_mat / Lambda_Reg

            Loss_SDRO     = torch.log(torch.mean(torch.exp(Residual), dim=0)) * Lambda_Reg
            Loss_SDRO_avg = torch.mean(Loss_SDRO)

            Loss_SDRO_avg.backward()
            optimizer_theta.step()

            theta_hist.append(theta_torch.clone().detach().numpy())
            cost_hist.append(m*N + cost_hist[-1])

            
            if torch.linalg.norm(theta_torch) > 0.95*np.sqrt(Lambda/2):
                theta_torch = theta_torch / torch.linalg.norm(theta_torch) * 0.95*np.sqrt(Lambda/2)
    
            if (silence == False) and (iter % test_iterations == 0):
                print("Iter: {}, Ksample: {}, m: {}, Loss: {:.2f}".format(iter, K_sample_max, m, Loss_SDRO_avg.item()))

        adjust_lr_zt(optimizer_theta,learning_rate,epoch+1)
    return theta_torch.detach().numpy(), theta_hist, cost_hist

def SDRO_MLMC_solver(dataloader, theta, Reg, Lambda, 
                        maxiter = 1, learning_rate = 5e-2, silence=False, K_sample_max=5, test_iterations = 10):
    """
    2-SDRO Approach with MLMC estimator for Regression Problem
    #   Input:
    # Feature: N samples of R^d [dim: N*d]
    #  Target: labels of N samples [dim: N*1]
    #   theta: initial guess for optimization
    #  Lambda: Lagrangian multiplier
    #     Reg: bandwidth
    #  Output:
    #   theta: optimized decision
    """
    iter       = 0
    Lambda_Reg = Lambda * Reg
    
    theta_torch     = torch.tensor(theta, dtype=torch.float, requires_grad=True)
    optimizer_theta = torch.optim.SGD([theta_torch], lr=learning_rate)
    N_ell_hist      = np.int_(2**(np.arange(K_sample_max) + 1))

    theta_hist = []
    cost_hist  = []
    theta_hist.append(theta_torch.clone().detach().numpy())
    cost_hist.append(0)


    
    for epoch in range(maxiter):
        for _, (data, target, idx) in enumerate(dataloader):
            iter = iter + 1
            # generate stochastic samples
            N, d         = data.shape
            data, target = Variable(data), Variable(target)
            optimizer_theta.zero_grad()

            m_total = 0
            for K_sample in np.arange(K_sample_max):
                m          = int(2**K_sample)
                N_ell      = N_ell_hist[-K_sample-1]
                data_ell   = data[:N_ell, :]
                N_ell, d   = data_ell.shape
                target_ell = target[:N_ell]

                data_noise     = torch.randn([m, N_ell, d]) * np.sqrt(Reg) + data_ell.reshape([1,N_ell,d])
                m_total       += m * N_ell
                data_noise_vec = data_noise.reshape([-1,d])
                target_noise   = target_ell.repeat(m,1)

                haty     = data_noise_vec @ theta_torch.type(torch.float64)
                obj_vec  = (haty - target_noise) ** 2
                obj_mat  = obj_vec.reshape([m, N_ell])
                Residual = obj_mat / Lambda_Reg

                if K_sample == 0:
                    Loss_SDRO = torch.log(torch.mean(torch.exp(Residual), dim=0)) * Lambda_Reg
                    Loss_SDRO_avg_K_sample = torch.mean(Loss_SDRO)
                    Loss_SDRO_avg_sum = Loss_SDRO_avg_K_sample
                else:
                    m1 = int(m/2)
                    Residual_half   = Residual[:m1,:]
                    Residual_remain = Residual[m1:,:]
                    Loss_SDRO_1     = torch.log(torch.mean(torch.exp(Residual), dim=0)) * Lambda_Reg
                    Loss_SDRO_avg_1 = torch.mean(Loss_SDRO_1)
                    Loss_SDRO_2     = (torch.log(torch.mean(torch.exp(Residual_half), dim=0)) + torch.log(torch.mean(torch.exp(Residual_remain), dim=0))) * Lambda_Reg
                    Loss_SDRO_avg_2 = torch.mean(Loss_SDRO_2)
                    Loss_SDRO_avg_K_sample   = Loss_SDRO_avg_1 - 0.5 * Loss_SDRO_avg_2                
                    Loss_SDRO_avg_sum = Loss_SDRO_avg_sum + Loss_SDRO_avg_K_sample

            Loss_SDRO_avg_sum.backward()
            optimizer_theta.step()
            theta_hist.append(theta_torch.clone().detach().numpy())
            cost_hist.append(m_total + cost_hist[-1])

            if torch.linalg.norm(theta_torch) > 0.95*np.sqrt(Lambda/2):
                theta_torch = theta_torch / torch.linalg.norm(theta_torch) * 0.95*np.sqrt(Lambda/2)
    
            if (silence == False) and (iter % test_iterations == 0):
                print("Iter: {}, Ksample: {}, m: {}, Loss: {:.2f}".format(iter, K_sample, m, Loss_SDRO_avg_sum.item()))
        adjust_lr_zt(optimizer_theta,learning_rate,epoch+1)

    return theta_torch.detach().numpy(), theta_hist, cost_hist

def SDRO_RTMLMC_solver(dataloader, theta, Reg, Lambda, 
                        maxiter = 1, learning_rate = 5e-2, silence=False, K_sample_max=5, test_iterations = 10):
    """
    2-SDRO Approach with RTMLMC estimator for Regression Problem
    #   Input:
    # Feature: N samples of R^d [dim: N*d]
    #  Target: labels of N samples [dim: N*1]
    #   theta: initial guess for optimization
    #  Lambda: Lagrangian multiplier
    #     Reg: bandwidth
    #  Output:
    #   theta: optimized decision
    """

    iter            = 0
    Lambda_Reg      = Lambda * Reg
    theta_torch     = torch.tensor(theta, dtype=torch.float, requires_grad=True)
    optimizer_theta = torch.optim.SGD([theta_torch], lr=learning_rate)

    # sampling from truncated gemoetric distribution
    elements      = np.arange(K_sample_max)
    probabilities = (0.5) ** (elements)
    probabilities = probabilities / np.sum(probabilities)

    theta_hist = []
    cost_hist  = []
    theta_hist.append(theta_torch.clone().detach().numpy())
    cost_hist.append(0)

    for epoch in range(maxiter):
        for _, (data, target, idx) in enumerate(dataloader):
            iter = iter + 1
            # generate stochastic samples
            N, d         = data.shape
            data, target = Variable(data), Variable(target)

            K_sample       = int(np.random.choice(list(elements), 1, list(probabilities)))
            m              = int(2**K_sample)
            optimizer_theta.zero_grad()
            data_noise     = torch.randn([m, N, d]) * np.sqrt(Reg) + data.reshape([1,N,d])
            data_noise_vec = data_noise.reshape([-1,d])
            target_noise   = target.repeat(m,1)

            haty       = data_noise_vec @ theta_torch.type(torch.float64)
            obj_vec    = (haty - target_noise) ** 2
            obj_mat    = obj_vec.reshape([m, N])
            Residual   = obj_mat / Lambda_Reg

            if m == 1:
                Loss_SDRO = torch.log(torch.mean(torch.exp(Residual), dim=0)) * Lambda_Reg
                Loss_SDRO_avg = torch.mean(Loss_SDRO)
            else:
                m1 = int(2**(K_sample-1))
                Residual_half   = Residual[:m1,:]
                Residual_remain = Residual[m1:,:]
                Loss_SDRO_1     = torch.log(torch.mean(torch.exp(Residual), dim=0)) * Lambda_Reg
                Loss_SDRO_avg_1 = torch.mean(Loss_SDRO_1)
                Loss_SDRO_2     = (torch.log(torch.mean(torch.exp(Residual_half), dim=0)) + torch.log(torch.mean(torch.exp(Residual_remain), dim=0))) * Lambda_Reg
                Loss_SDRO_avg_2 = torch.mean(Loss_SDRO_2)
                Loss_SDRO_avg   = Loss_SDRO_avg_1 - 0.5 * Loss_SDRO_avg_2
            
            Loss_SDRO_avg = 1/probabilities[K_sample]* Loss_SDRO_avg
            Loss_SDRO_avg.backward()
            optimizer_theta.step()

            theta_hist.append(theta_torch.clone().detach().numpy())
            cost_hist.append(m*N + cost_hist[-1])

            
            if torch.linalg.norm(theta_torch) > 0.95*np.sqrt(Lambda/2):
                theta_torch = theta_torch / torch.linalg.norm(theta_torch) * 0.95*np.sqrt(Lambda/2)
    
            if (silence == False) and (iter % test_iterations == 0):
                print("Iter: {}, Ksample: {}, m: {}, Loss: {:.2f}".format(iter, K_sample, m, Loss_SDRO_avg.item()))
        adjust_lr_zt(optimizer_theta,learning_rate,epoch+1)

    return theta_torch.detach().numpy(), theta_hist, cost_hist

def SDRO_RUMLMC_solver(dataloader, theta, Reg, Lambda, 
                        maxiter = 1, learning_rate = 5e-2, silence=False, test_iterations = 10):
    """
    2-SDRO Approach with RRMLMC estimator for Regression Problem
    #   Input:
    # Feature: N samples of R^d [dim: N*d]
    #  Target: labels of N samples [dim: N*1]
    #   theta: initial guess for optimization
    #  Lambda: Lagrangian multiplier
    #     Reg: bandwidth
    #  Output:
    #   theta: optimized decision
    """

    iter            = 0
    Lambda_Reg      = Lambda * Reg
    theta_torch     = torch.tensor(theta, dtype=torch.float, requires_grad=True)
    optimizer_theta = torch.optim.SGD([theta_torch], lr=learning_rate)

    theta_hist = []
    cost_hist  = []
    theta_hist.append(theta_torch.clone().detach().numpy())
    cost_hist.append(0)


    for epoch in range(maxiter):
        for _, (data, target, idx) in enumerate(dataloader):
            iter = iter + 1
            # generate stochastic samples
            N, d         = data.shape
            data, target = Variable(data), Variable(target)

            K_sample             = int(np.random.geometric(p=0.5)) - 1
            probability_K_sample = (0.5)**(K_sample+1)
            m                    = int(2**K_sample)
            optimizer_theta.zero_grad()
            data_noise       = torch.randn([m, N, d]) * np.sqrt(Reg) + data.reshape([1,N,d])
            data_noise_vec   = data_noise.reshape([-1,d])
            target_noise     = target.repeat(m,1)

            haty     = data_noise_vec @ theta_torch.type(torch.float64)
            obj_vec  = (haty - target_noise) ** 2
            obj_mat  = obj_vec.reshape([m, N])
            Residual = obj_mat / Lambda_Reg

            if m == 1:
                Loss_SDRO = torch.log(torch.mean(torch.exp(Residual), dim=0)) * Lambda_Reg
                Loss_SDRO_avg = torch.mean(Loss_SDRO)
            else:
                m1 = int(2**(K_sample-1))
                Residual_half   = Residual[:m1,:]
                Residual_remain = Residual[m1:,:]
                Loss_SDRO_1     = torch.log(torch.mean(torch.exp(Residual), dim=0)) * Lambda_Reg
                Loss_SDRO_avg_1 = torch.mean(Loss_SDRO_1)
                Loss_SDRO_2     = (torch.log(torch.mean(torch.exp(Residual_half), dim=0)) + torch.log(torch.mean(torch.exp(Residual_remain), dim=0))) * Lambda_Reg
                Loss_SDRO_avg_2 = torch.mean(Loss_SDRO_2)
                Loss_SDRO_avg   = Loss_SDRO_avg_1 - 0.5 * Loss_SDRO_avg_2
            
            Loss_SDRO_avg = 1/probability_K_sample* Loss_SDRO_avg
            Loss_SDRO_avg.backward()
            optimizer_theta.step()

            theta_hist.append(theta_torch.clone().detach().numpy())
            cost_hist.append(m*N + cost_hist[-1])
            
            if torch.linalg.norm(theta_torch) > 0.95*np.sqrt(Lambda/2):
                theta_torch = theta_torch / torch.linalg.norm(theta_torch) * 0.95*np.sqrt(Lambda/2)
    
            if (silence == False) and (iter % test_iterations == 0):
                print("Iter: {}, Ksample: {}, m: {}, Loss: {:.2f}".format(iter, K_sample, m, Loss_SDRO_avg.item()))
        adjust_lr_zt(optimizer_theta,learning_rate,epoch+1)
    return theta_torch.detach().numpy(), theta_hist, cost_hist

def SDRO_RRMLMC_solver(dataloader, theta, Reg, Lambda, 
                        maxiter = 1, learning_rate = 5e-2, silence=False, test_iterations = 10):
    """
    2-SDRO Approach with RRMLMC estimator for Regression Problem
    #   Input:
    # Feature: N samples of R^d [dim: N*d]
    #  Target: labels of N samples [dim: N*1]
    #   theta: initial guess for optimization
    #  Lambda: Lagrangian multiplier
    #     Reg: bandwidth
    #  Output:
    #   theta: optimized decision
    """
    iter            = 0
    Lambda_Reg      = Lambda * Reg
    theta_torch     = torch.tensor(theta, dtype=torch.float, requires_grad=True)
    optimizer_theta = torch.optim.SGD([theta_torch], lr=learning_rate)

    theta_hist = []
    cost_hist  = []
    theta_hist.append(theta_torch.detach().numpy())
    cost_hist.append(0)

    for epoch in range(maxiter):
        for _, (data, target, idx) in enumerate(dataloader):
            iter = iter + 1
            # generate stochastic samples
            N, d         = data.shape
            data, target = Variable(data), Variable(target)
            K_sample_max = int(np.random.geometric(p=0.5)) - 1
            optimizer_theta.zero_grad()
            m_total = 0

            for K_sample in np.arange(K_sample_max + 1):
                m         = int(2**K_sample)
                m_total  += m*N

                data_noise     = torch.randn([m, N, d]) * np.sqrt(Reg) + data.reshape([1,N,d])
                data_noise_vec = data_noise.reshape([-1,d])
                target_noise   = target.repeat(m,1)

                haty     = data_noise_vec @ theta_torch.type(torch.float64)
                obj_vec  = (haty - target_noise) ** 2
                obj_mat  = obj_vec.reshape([m, N])
                Residual = obj_mat / Lambda_Reg

                if m == 1:
                    Loss_SDRO = torch.log(torch.mean(torch.exp(Residual), dim=0)) * Lambda_Reg
                    Loss_SDRO_avg_K_sample = torch.mean(Loss_SDRO)
                    Loss_SDRO_avg_sum = Loss_SDRO_avg_K_sample
                else:
                    m1 = int(2**(K_sample-1))
                    Residual_half   = Residual[:m1,:]
                    Residual_remain = Residual[m1:,:]
                    Loss_SDRO_1     = torch.log(torch.mean(torch.exp(Residual), dim=0)) * Lambda_Reg
                    Loss_SDRO_avg_1 = torch.mean(Loss_SDRO_1)
                    Loss_SDRO_2     = (torch.log(torch.mean(torch.exp(Residual_half), dim=0)) + torch.log(torch.mean(torch.exp(Residual_remain), dim=0))) * Lambda_Reg
                    Loss_SDRO_avg_2 = torch.mean(Loss_SDRO_2)
                    Loss_SDRO_avg_K_sample   = Loss_SDRO_avg_1 - 0.5 * Loss_SDRO_avg_2
                    cdf_q_ell = 1 - ((0.5)**(K_sample+1))
                    p_ell = 1 / (1 - cdf_q_ell)
                    Loss_SDRO_avg_sum = Loss_SDRO_avg_sum + Loss_SDRO_avg_K_sample * p_ell

            Loss_SDRO_avg_sum.backward()
            optimizer_theta.step()
            theta_hist.append(theta_torch.clone().detach().numpy())
            cost_hist.append(m_total + cost_hist[-1])

            if torch.linalg.norm(theta_torch) > 0.95*np.sqrt(Lambda/2):
                theta_torch = theta_torch / torch.linalg.norm(theta_torch) * 0.95*np.sqrt(Lambda/2)
            if (silence == False) and (iter % test_iterations == 0):
                print("Iter: {}, Ksample: {}, m: {}, Loss: {:.2f}".format(iter, K_sample, m, Loss_SDRO_avg_sum.item()))
        adjust_lr_zt(optimizer_theta,learning_rate,epoch+1)
    return theta_torch.detach().numpy(), theta_hist, cost_hist