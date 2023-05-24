import numpy as np
import torch

import torch.nn.functional as F

from tqdm import tqdm


from Learn2Rank.XGboost import test_input_xgboost


from data_process import process_HE


def make_y_error(out, y_class):
    y_error = torch.zeros(size=[y_class.shape[0]], dtype=torch.int64)
    for i in tqdm(range(y_class.shape[0])):
        if out[i].max(-1)[1].item() != y_class[i]:
            y_error[i] = 1
        else :
            y_error[i] = 0
    return y_error
  

def make_test(select):

    datasets = ["data_ogbn-products", "data_Reddit", "data_Flickr"]
    models = ["/EnGCN", "/SIGN", "/GraphSAGE", "/ClusterGCN"]
         
    dataset_name = datasets[select[0]]
    model_name = models[select[1]]
        
    print("dataset : {}, model : {}".format(dataset_name, model_name))
    
    out = torch.load("../" + dataset_name + model_name +'/out/out')   
    
    out_mean = torch.torch.zeros_like(torch.load("../" + dataset_name + model_name +'/drop_out/out_9'))
    for i in tqdm(range(10)):      # get probabilistic output attributes
        out_mean += torch.load("../" + dataset_name + model_name +'/drop_out/out_{}'.format(i))
    out_mean /= 10
    
    logpx = F.log_softmax(out_mean, dim=1).tolist()
    logpx = np.array(logpx)
    uncertainty = -np.sum(np.multiply(logpx, np.exp(logpx)), axis = 1)
    uncertainty = torch.tensor(uncertainty).unsqueeze(dim=1)

    x_out = torch.load("../" + dataset_name + '/data_x/x_out')     # get graph node attributes
    
    split_masks = torch.load("../" + dataset_name + '/split/split_masks')    

    deg = torch.load("../" + dataset_name + '/deg/deg')    # get graph structure attributes
    
    y_class = torch.load("../" + dataset_name + '/y/y_class')
    y_error = make_y_error(out, y_class)          # 分类错误的样本置1，正确的样本置0，二分类      

    T = torch.load("../" + dataset_name +'/edge/T')    # The neighobors of node  {node_id_1:[nei_id_1, nei_id_2, ……],   node_id_2:[nei_id_1, nei_id_2, ……]}

    HE = process_HE(out)
    x_HE = process_HE(x_out)
   

    out = F.softmax(out, dim=1)
    x_out = F.softmax(x_out, dim=1)
    DeepGini = 1 - torch.sum(torch.pow(out, 2), dim=1)
    DeepGini = DeepGini.unsqueeze(dim=1)

    out_last = torch.zeros(size=(out.shape[0],0), dtype=torch.float32)

    out_last = torch.cat((out_last, out), dim=1)
    out_last = torch.cat((out_last, x_out), dim=1)
    out_last = torch.cat((out_last, HE), dim=1)
    out_last = torch.cat((out_last, x_HE), dim=1)
    out_last = torch.cat((out_last, deg), dim=1)
    out_last = torch.cat((out_last, uncertainty), dim=1) 
    out_last = torch.cat((out_last, DeepGini), dim=1)

    out_agg = torch.zeros_like(out_last, dtype=torch.float32)   # Attributes Enhancement
    for i in tqdm(range(len(T))):
        out_nei = out_last[T[i]]
        out_nei_mean = torch.mean(out_nei, dim=0)
        out_agg[i] = out_nei_mean
    out_last = torch.cat((out_last, out_agg), dim=1)


    n = y_error[split_masks['test']].sum().item()
    test_input_xgboost(out_last, y_error, split_masks, n, "valid", "test")

    return
    
   
   
   
if __name__ == "__main__":

    for select in [[2,0],[2,1],[2,2],[2,3]]:
        make_test(select)
        print()

   

    