import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from Learn2Rank.XGboost import test_input_xgboost


def make_test(select):
    datasets = ["data_ogbn-products", "data_Reddit", "data_Flickr"]
    models = ["/EnGCN", "/SIGN", "/GraphSAGE", "/ClusterGCN"]
         
    # select the dataset and GNN model
    dataset_name = datasets[select[0]]
    model_name = models[select[1]]
    
    # The probability distribution of output from target GNN 
    out = torch.load("my_data/" + dataset_name + model_name +'/out/out')     
    
    
    ###### There are graph node attributes
    
    # The probability distribution of output from MLP
    x_out = torch.load("my_data/" + dataset_name + '/data_x/x_out')
    x_out = F.softmax(x_out, dim=1)
    
    # The information entropy of MLP output
    x_HE = torch.load("my_data/" + dataset_name + '/data_x/x_HE')
    
    ######
    
    # The degree list for all nodes,  shape=(N)
    deg = torch.load("my_data/" + dataset_name + '/deg/deg')
    
    # The failure mask of all nodes. (e.g. [1,0,1,0,0,1]: The 0, 2, 5 nodes are failure.)
    y_error = torch.load("my_data/" + dataset_name + model_name+'/y_error')     

    split_masks = torch.load("my_data/" + dataset_name + '/split/my_split')
 
    # The neighbors of a nodes, used to Atributes Enhancement.
    T = torch.load("my_data/" + dataset_name +'/edge/T')

    # The probability distribution of output from GNN model.
    out = F.softmax(out, dim=1)
 
    # The information entropy of output from GNNs.
    HE = torch.load("my_data/" + dataset_name + model_name +'/out/HE')
    
    # The uncertainty of GNNs.
    HE_uncer = torch.load("my_data/" + dataset_name + model_name +'/dropout/HE_uncer')

    DeepGini = 1 - torch.sum(torch.pow(out, 2), dim=1)
    DeepGini = DeepGini.unsqueeze(dim=1)
    
    ###### There are all attributes, concatenate them.
    out_last = torch.zeros(size=(out.shape[0],0), dtype=torch.float32)

    out_last = torch.cat((out_last, out), dim=1)
    out_last = torch.cat((out_last, x_out), dim=1)
    out_last = torch.cat((out_last, HE), dim=1)
    out_last = torch.cat((out_last, x_HE), dim=1)
    out_last = torch.cat((out_last, deg), dim=1)
    out_last = torch.cat((out_last, HE_uncer), dim=1) 
    out_last = torch.cat((out_last, DeepGini), dim=1)
    #####
    
    ##### There is the Attributes Enhancement.
    out_agg = torch.zeros_like(out_last, dtype=torch.float32)
    for i in tqdm(range(len(T))):
        out_nei = out_last[T[i]]
        out_nei_mean = torch.mean(out_nei, dim=0)
        out_agg[i] = out_nei_mean
        
    out_last = torch.cat((out_last, out_agg), dim=1)
    #####
        
    n = y_error[split_masks['test']].sum().item()

    ##### test input prioritization
    test_input_xgboost(out_last, y_error, split_masks, n, "valid", "test")
    
    return
    
   
   
   
if __name__ == "__main__":
    
    # for select in [[0,0],[0,1],[0,2],[0,3], [1,0],[1,1],[1,2],[1,3], [2,0],[2,1],[2,2],[2,3]]:
    for select in [[2,0],[2,1],[2,2],[2,3]]:
        make_test(select)