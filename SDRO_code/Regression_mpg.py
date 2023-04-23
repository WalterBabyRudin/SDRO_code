import os
from os.path import exists
import numpy as np
import random
import cvxpy as cp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import sklearn.datasets
import cvxpy as cp
import SDRO_optimizer
from torch.utils.data import TensorDataset, DataLoader, Dataset


random_state_num = 42 + 7*1
np.random.seed(random_state_num)
torch.manual_seed(random_state_num)
torch.cuda.manual_seed(random_state_num)

class MyDataset(Dataset):
    def __init__(self, Feature_Tr_tensor, Target_Tr_mat_tensor):
        self.Data = TensorDataset(Feature_Tr_tensor, Target_Tr_mat_tensor)
    
    def __getitem__(self, index):
        data, target = self.Data[index]

        return data, target, index

    def __len__(self):
        return len(self.Data)


#######################################################
################# Data Loading ########################
data = sklearn.datasets.load_svmlight_file("data/mpg_scale.txt")
X_all, y_all = np.float32(np.array(data[0].todense())), np.float32(np.array(data[1]).reshape([-1,1]))
N = len(y_all)
X_all = np.concatenate((np.ones([N,1]), X_all), axis=1)
_, d = np.shape(X_all)
# initial guess
theta0 = np.float32(np.zeros([d,1]))


# # pytorch format
Feature_tensor = torch.tensor(X_all)
Target_tensor  = torch.tensor(y_all)

Dataset_tensor = MyDataset(Feature_tensor, Target_tensor)
#######################################################
################# Solving Oracle Problem ##############
SDRO_Lambda = 1000
SDRO_Reg    = 0.1
N,d = np.shape(X_all)

Dataloader_oracle = DataLoader(Dataset_tensor, batch_size=700, shuffle=True)
# theta_SDRO_opt = SDRO_optimizer.SDRO_oracle(Dataloader_oracle, theta0, SDRO_Lambda, SDRO_Reg, X_all ,y_all,
#                 maxiter = 10000, step_size0=1e-1, test_iterations=500)

# # Report Statistics
# SDRO_optval, residual = SDRO_optimizer.SDRO_eval(theta_SDRO_opt, SDRO_Lambda, SDRO_Reg, X_all, y_all)
SDRO_optval = 113.62537979145378
residual = 5.370712145509769
print("Lambda: ", SDRO_Lambda, "Reg: ", SDRO_Reg, "SDRO optval: ", SDRO_optval, "Residual: ", residual)

SDRO_0, residual_0 = SDRO_optimizer.SDRO_eval(theta0, SDRO_Lambda, SDRO_Reg, X_all, y_all)
print([SDRO_0, residual_0])



############################################################
################# Testing Performance of SGD Method ########
input_method = input("Input Method: ")
Dataloader_all = DataLoader(Dataset_tensor, batch_size=100, shuffle=True)



if input_method == "SG":
    theta_SG, theta_SG_hist, cost_SG_hist = SDRO_optimizer.SDRO_SG_solver(Dataloader_all, theta0, SDRO_Reg, SDRO_Lambda, 
                        maxiter = 1000, learning_rate = 5e-3, silence=False, K_sample_max=4, test_iterations = 10)
    
    SDRO_SG_optval, SDRO_SG_residual = SDRO_optimizer.SDRO_eval(theta_SG, SDRO_Lambda, SDRO_Reg, X_all, y_all)
    print("SG Method    Optval: {:.4e}, Residual: {:.4e}".format(SDRO_SG_optval, SDRO_SG_residual))

    obj_hist      = []
    residual_hist = []
    for i in range(len(theta_SG_hist)):
        SDRO_SG_optval_i, SDRO_SG_residual_i = SDRO_optimizer.SDRO_eval(theta_SG_hist[i], SDRO_Lambda, SDRO_Reg, X_all, y_all)
        obj_hist.append(SDRO_SG_optval_i)
        residual_hist.append(SDRO_SG_residual_i)

    obj_SG_hist = np.array(obj_hist)
    residual_SG_hist = np.array(residual_hist)
    cost_SG_hist = np.array(cost_SG_hist)
    np.save("results/obj_SG_hist_mpg.npy", obj_SG_hist)
    np.save("results/residual_SG_hist_mpg.npy", residual_SG_hist)
    np.save("results/cost_SG_hist_mpg.npy", cost_SG_hist)


if input_method == "RTMLMC":
    theta_RTMLMC, theta_RTMLMC_hist, cost_RTMLMC_hist = SDRO_optimizer.SDRO_RTMLMC_solver(Dataloader_all, theta0, SDRO_Reg, SDRO_Lambda, 
                        maxiter = 1000, learning_rate = 1e-2, silence=True, K_sample_max=4, test_iterations = 10)
    SDRO_RTMLMC_optval, SDRO_RTMLMC_residual = SDRO_optimizer.SDRO_eval(theta_RTMLMC, SDRO_Lambda, SDRO_Reg, X_all, y_all)
    print("RTMLMC Method    Optval: {:.4e}, Residual: {:.4e}".format(SDRO_RTMLMC_optval, SDRO_RTMLMC_residual))
    
    obj_hist      = []
    residual_hist = []
    for i in range(len(theta_RTMLMC_hist)):
        SDRO_RTMLMC_optval_i, SDRO_RTMLMC_residual_i = SDRO_optimizer.SDRO_eval(theta_RTMLMC_hist[i], SDRO_Lambda, SDRO_Reg, X_all, y_all)
        obj_hist.append(SDRO_RTMLMC_optval_i)
        residual_hist.append(SDRO_RTMLMC_residual_i)

    obj_RTMLMC_hist = np.array(obj_hist)
    residual_RTMLMC_hist = np.array(residual_hist)
    cost_RTMLMC_hist = np.array(cost_RTMLMC_hist)
    print(cost_RTMLMC_hist.shape)
    print(residual_RTMLMC_hist.shape)    
    np.save("results/obj_RTMLMC_hist_mpg.npy", obj_RTMLMC_hist)
    np.save("results/residual_RTMLMC_hist_mpg.npy", residual_RTMLMC_hist)
    np.save("results/cost_RTMLMC_hist_mpg.npy", cost_RTMLMC_hist)

    # print(obj_RTMLMC_hist)
    # print(residual_RTMLMC_hist)
    # print(cost_RTMLMC_hist)

if input_method == "MLMC":
    Dataloader_all = DataLoader(Dataset_tensor, batch_size=int(2**8), shuffle=True)
    theta_MLMC, theta_MLMC_hist, cost_MLMC_hist = SDRO_optimizer.SDRO_MLMC_solver(Dataloader_all, theta0, SDRO_Reg, SDRO_Lambda, 
                        maxiter = 1000, learning_rate = 1e-2, silence=False, K_sample_max=4, test_iterations = 10)
    SDRO_MLMC_optval, SDRO_MLMC_residual = SDRO_optimizer.SDRO_eval(theta_MLMC, SDRO_Lambda, SDRO_Reg, X_all, y_all)
    print("MLMC Method    Optval: {:.4e}, Residual: {:.4e}".format(SDRO_MLMC_optval, SDRO_MLMC_residual))
    print(cost_MLMC_hist)

    obj_hist      = []
    residual_hist = []
    for i in range(len(theta_MLMC_hist)):
        SDRO_MLMC_optval_i, SDRO_MLMC_residual_i = SDRO_optimizer.SDRO_eval(theta_MLMC_hist[i], SDRO_Lambda, SDRO_Reg, X_all, y_all)
        obj_hist.append(SDRO_MLMC_optval_i)
        residual_hist.append(SDRO_MLMC_residual_i)

    obj_MLMC_hist = np.array(obj_hist)
    residual_MLMC_hist = np.array(residual_hist)
    cost_MLMC_hist = np.array(cost_MLMC_hist)
    np.save("results/obj_MLMC_hist_mpg.npy", obj_MLMC_hist)
    np.save("results/residual_MLMC_hist_mpg.npy", residual_MLMC_hist)
    np.save("results/cost_MLMC_hist_mpg.npy", cost_MLMC_hist)


if input_method == "RRMLMC":
    theta_RRMLMC, theta_RRMLMC_hist, cost_RRMLMC_hist = SDRO_optimizer.SDRO_RRMLMC_solver(Dataloader_all, theta0, SDRO_Reg, SDRO_Lambda, 
                        maxiter = 1000, learning_rate = 4e-3, silence=False, test_iterations = 10)
    SDRO_RRMLMC_optval, SDRO_RRMLMC_residual = SDRO_optimizer.SDRO_eval(theta_RRMLMC, SDRO_Lambda, SDRO_Reg, X_all, y_all)
    print("RRMLMC Method    Optval: {:.4e}, Residual: {:.4e}".format(SDRO_RRMLMC_optval, SDRO_RRMLMC_residual))
    print(cost_RRMLMC_hist)

    obj_hist      = []
    residual_hist = []
    for i in range(len(theta_RRMLMC_hist)):
        SDRO_RRMLMC_optval_i, SDRO_RRMLMC_residual_i = SDRO_optimizer.SDRO_eval(theta_RRMLMC_hist[i], SDRO_Lambda, SDRO_Reg, X_all, y_all)
        obj_hist.append(SDRO_RRMLMC_optval_i)
        residual_hist.append(SDRO_RRMLMC_residual_i)

    obj_RRMLMC_hist = np.array(obj_hist)
    residual_RRMLMC_hist = np.array(residual_hist)
    cost_RRMLMC_hist = np.array(cost_RRMLMC_hist)
    np.save("results/obj_RRMLMC_hist_mpg.npy", obj_RRMLMC_hist)
    np.save("results/residual_RRMLMC_hist_mpg.npy", residual_RRMLMC_hist)
    np.save("results/cost_RRMLMC_hist_mpg.npy", cost_RRMLMC_hist)


    

if input_method == "RUMLMC":
    theta_RUMLMC, theta_RUMLMC_hist, cost_RUMLMC_hist = SDRO_optimizer.SDRO_RUMLMC_solver(Dataloader_all, theta0, SDRO_Reg, SDRO_Lambda, 
                        maxiter = 1000, learning_rate = 4e-3, silence=False, test_iterations = 10)
    SDRO_RUMLMC_optval, SDRO_RUMLMC_residual = SDRO_optimizer.SDRO_eval(theta_RUMLMC, SDRO_Lambda, SDRO_Reg, X_all, y_all)
    print("RUMLMC Method    Optval: {:.4e}, Residual: {:.4e}".format(SDRO_RUMLMC_optval, SDRO_RUMLMC_residual))
    print(cost_RUMLMC_hist)

    obj_hist      = []
    residual_hist = []
    for i in range(len(theta_RUMLMC_hist)):
        SDRO_RUMLMC_optval_i, SDRO_RUMLMC_residual_i = SDRO_optimizer.SDRO_eval(theta_RUMLMC_hist[i], SDRO_Lambda, SDRO_Reg, X_all, y_all)
        obj_hist.append(SDRO_RUMLMC_optval_i)
        residual_hist.append(SDRO_RUMLMC_residual_i)

    obj_RUMLMC_hist = np.array(obj_hist)
    residual_RUMLMC_hist = np.array(residual_hist)
    cost_RUMLMC_hist = np.array(cost_RUMLMC_hist)
    np.save("results/obj_RUMLMC_hist_mpg.npy", obj_RUMLMC_hist)
    np.save("results/residual_RUMLMC_hist_mpg.npy", residual_RUMLMC_hist)
    np.save("results/cost_RUMLMC_hist_mpg.npy", cost_RUMLMC_hist)











