
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from scipy.optimize import linear_sum_assignment
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
import torch.optim as optim
from tqdm.auto import tqdm, trange
import math
import numpy as np
import argparse
import random
def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", default=5, type=int, help="Number of datasets.")
    parser.add_argument("--lambda0", default=1, type=float, help="Penalty factor for alignment.")
    parser.add_argument("--lambda1", default=0.5, type=float, help="Penalty factor for group lasso")
    parser.add_argument("--lambda2", default=0.2, type=float, help="Penalty factor for L2 loss")
    parser.add_argument("--num_epochs", default=16, type=int, help="Number of epochs in training.")
    parser.add_argument("--num_epochs_test", default=32, type=int, help="Number of epochs in testing.")
    parser.add_argument("--dvc", default='cuda:0', type=str, help="Current device")
    parser.add_argument("--tor", default=20, type=int, help="max tolerate")
    parser.add_argument("--epochs_all", default=450, type=int, help="All of epochs")
    parser.add_argument("--num_b", default=15, type=int, help="Number of main effects.")
    parser.add_argument("--l_r", default=0.001, type=float, help="Initial learning Rate")
    parser.add_argument("--delta", default=0.2, type=float, help="Relation Threshold")
    parser.add_argument("--alpha_all", default=[0.25,0.5,1,1.5,2,4,8,16], type=list, help="Alpha set")
    parser.add_argument("--scheme", default=0, type=int, help="Scheme of learning rate dacay")
    parser.add_argument("--align_first_layer", default=1, type=int, help="Align the input layer or not.")
    parser.add_argument("--print_log", default=True, type=bool, help="Print the result during training or not.")
    parser.add_argument("--save_result", default=False, type=bool, help="Save the result or not.")
    return parser

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dp=torch.nn.Dropout(0.02)
        self.fc1 = nn.Linear(605, 800) 
        self.fc2 = nn.Linear(800, 400)
        self.fc3 = nn.Linear(400, 100)
        self.fc4 = nn.Linear(100, 1)
        self.tan = torch.nn.Tanh()
        self.tanh = torch.nn.Hardtanh(-3,3)
        self.double()
    def forward(self, x):
        x = self.tan(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = self.fc4(x)
        return x
        

def align(args,net1,net2):
    nt1 = copy.deepcopy(net1)
    w_key = {}
    b_key = {}
    next_key = {}
    shape_key = {}
    last_key = None
    for key in nt1.state_dict().keys():
        if 'fc' in key and 'wei' in key:
            key0 = key.replace('.weight','')
            w_key[key0] = key
            b_key[key0] = key.replace('weight','bias')
            next_key[key0] = None
            shape_key[key0] = nt1.state_dict()[key].shape
            if last_key!=None:
                next_key[last_key] = key0
            last_key = key0
    trans_matrix_left = {}
    trans_matrix_right = {}
    for key in next_key.keys():
        n = shape_key[key][0]
        trans_matrix_left[key] = torch.zeros([n,n]).double().to(args.dvc)
    for key in trans_matrix_left.keys():
        key0 = w_key[key]
        key1 = b_key[key]
        if next_key[key]!=None:
            all_vec1 = torch.cat((net1.state_dict()[key0],net1.state_dict()[key1].unsqueeze(1)),1)
            if trans_matrix_right.get(key)!=None:
                all_vec2 = torch.cat((torch.mm(net2.state_dict()[key0],trans_matrix_right[key]),net2.state_dict()[key1].unsqueeze(1)),1)
            else:
                all_vec2 = torch.cat((net2.state_dict()[key0],net2.state_dict()[key1].unsqueeze(1)),1)
            all_vecn2=-all_vec2
            rind, cind, pnc = get_match(all_vec1,all_vec2,all_vecn2) #get matching results 
            trans_matrix_left[key][rind,cind] = pnc.double().to(args.dvc)
            trans_matrix_right[next_key[key]] = trans_matrix_left[key].t()
        else:
            trans_matrix_left[key] = torch.eye(trans_matrix_left[key].shape[0])
    for key in trans_matrix_left.keys():
        trans_matrix_left[key] = trans_matrix_left[key].double().to(args.dvc)
    for key in trans_matrix_right.keys():
        trans_matrix_right[key] = trans_matrix_right[key].double().to(args.dvc)
    return trans_matrix_left,trans_matrix_right  # the corresponding transformation matrix

def get_match(all_vec1,all_vec2,all_vecn2):
    n = all_vec1.shape[0]
    dismatrix = torch.zeros([n,n]) #distance matrix
    pnmatrix = torch.zeros([n,n]) # matrix that indicate positive or negative
    for i in range(n):
        pos = torch.norm(all_vec2-all_vec1[i],p=2,dim=1) #positive distance
        neg = torch.norm(all_vecn2-all_vec1[i],p=2,dim=1) #negative distance
        judge = torch.where(pos>neg)[0]
        pos[judge] = neg[judge]
        pnindex = torch.ones(n)
        pnindex[judge] = -1 
        dismatrix[i] = pos
        pnmatrix[i] = pnindex
    row_ind, col_ind = linear_sum_assignment(dismatrix) # Hungarian algorithm
    pncondition = pnmatrix[row_ind,col_ind] #situation of postive and negative
    return row_ind, col_ind, pncondition

def get_loss(args,net,index,weight,trans_left_dic=None,trans_right_dic=None):
    # loss0 alignment penalty
    # loss1 group-lasso penalty
    # loss2 l2 penalty
    net_n1 = net[index]
    loss0 = 0
    is_prepare = 1
    if trans_left_dic == None and weight.sum() !=0:
        trans_left_dic = {}
        trans_right_dic = {}
        is_prepare = 0
    for i in range(len(weight)):
        if weight[i] != 0:
            net_n2 = net[i]
            if is_prepare == 0:
                trans_left,trans_right = align(args,net_n1,net_n2)
                trans_left_dic[str(i)] = trans_left
                trans_right_dic[str(i)] = trans_right
            else:
                trans_left = trans_left_dic[str(i)]
                trans_right = trans_right_dic[str(i)]
            loss_tan = 0
            for key, parms in net_n1.named_parameters():
                if 'fc1' not in key:
                    if 'weight' in key:
                        key0=key.replace('.weight','')               
                        if trans_right.get(key0)!=None:
                            loss_tan = loss_tan + torch.norm(torch.mm(torch.mm(trans_left[key0],net_n2.state_dict()[key]),trans_right[key0])-parms,p=2,dim=(0,1))**2
                        else:
                            loss_tan = loss_tan + torch.norm(torch.mm(trans_left[key0],net_n2.state_dict()[key])-parms,p=2,dim=(0,1))**2
                else:
                    if 'weight' in key:  
                        if args.align_first_layer==1:
                            key0=key.replace('.weight','')  
                            loss_tan = loss_tan + torch.norm(torch.mm(trans_left[key0],net_n2.state_dict()[key])-parms,p=2,dim=(0,1))**2
                        else:
                            loss_tan = loss_tan + torch.norm(net_n2.state_dict()[key]-parms,p=2,dim=(0,1))**2
            loss0 = loss0 + weight[i]*loss_tan
    loss1 = 0
    loss2 = 0
    for key, parms in net_n1.named_parameters():
        if 'fc1' in key:
           if 'weight' in key:
                loss1 = loss1 + torch.sum(torch.norm(parms,p=2,dim=0))
        else:
            if 'weight' in key:
                loss2 = loss2 + torch.norm(parms,p=2,dim=(0,1))**2
            #if 'bias' in key:
                #loss2 = loss2 + torch.norm(parms,p=2)
    if is_prepare == 1:
        return args.lambda0*loss0**0.5 + args.lambda1*loss1 + args.lambda2*loss2
    else:
        return args.lambda0*loss0**0.5 + args.lambda1*loss1 + args.lambda2*loss2, trans_left_dic, trans_right_dic


def test_result(args,model,loader): #return mean MSE loss
    criterion = nn.MSELoss(reduction='sum')
    model.eval()
    loss_all = 0
    sum_all = 0
    for batch in loader:
        batch[0] = batch[0].to(args.dvc)
        batch[1] = batch[1].to(args.dvc)
        output = model(batch[0])
        loss_all = loss_all + criterion(output, batch[1].unsqueeze(-1))
        sum_all = sum_all + batch[0].shape[0]
    model.train()
    return(loss_all/sum_all)


def test_y(args,model,loader): #return y
    model.eval()
    y_test = []
    for batch in loader:
        batch[0]=batch[0].to(args.dvc)
        batch[1]=batch[1].to(args.dvc)
        output=model(batch[0])
        y_test = y_test + list(output[:,0].detach().numpy())
    model.train()
    return y_test

def train(args,net,optimizer,loader_train,loader_test,loader_val,rel):
    n = args.n
    in_train = list(range(n))
    er = [1000]*n
    er_new = [0]*n
    pace = [0]*n
    res_val = [0]*n
    res_test = [0]*n
    left = [None]*n
    right = [None]*n
    criterion = nn.MSELoss()
    for epoch in range(args.epochs_all):
        for i in range(n):
            if i in in_train:
                for batch in loader_train[i]:
                    batch[0] = batch[0].to(args.dvc)
                    batch[1] = batch[1].to(args.dvc)
                    output = net[i](batch[0])
                    loss = criterion(output, batch[1].unsqueeze(-1))
                    if ((epoch % 10 == 0 and epoch<=20) or (epoch % 20 == 0 and (epoch>20 and epoch<=80)) or (epoch % 40 == 0 and epoch>80)) and rel[i,:].sum() != 0 :
                        loss_par, left[i], right[i] = get_loss(args,net,i,rel[i,:])
                    else:
                        loss_par = get_loss(args,net,i,rel[i,:],left[i],right[i])
                    loss = loss + loss_par #add penalty
                    optimizer[i].zero_grad()
                    loss.backward()
                    optimizer[i].step()
        for i in range(n):
            if i in in_train:
                er_new[i] = test_result(args,net[i],loader_val[i])
                if er_new[i]*1.01 < er[i]:
                    er[i] = er_new[i]
                    res_val[i] = er_new[i]
                    res_test[i] = test_result(args,net[i],loader_test[i])
                    pace[i] = 0
                    for j in range(n):
                        if j not in in_train and rel[i,j] > 0:
                            in_train.append(j)
                            optimizer[j].param_groups[0]["lr"] = 1e-3
                            #optimizer[j].param_groups[0]["lr"] = optimizer[i].param_groups[0]["lr"]*2
                            net[j].load_state_dict(torch.load('net{}.pth'.format(j)))
                            pace[j] = 0
                    torch.save(net[i].state_dict(), 'net{}.pth'.format(i) )
                else:
                    pace[i] = pace[i] + 1
                    if pace[i] > args.tor:
                        optimizer[i].param_groups[0]["lr"] = optimizer[i].param_groups[0]["lr"]*0.75
                        pace[i] = 0
                        if args.scheme == 0:
                            if optimizer[i].param_groups[0]["lr"]<1e-5:
                                in_train.remove(i)
                        else:
                            if optimizer[i].param_groups[0]["lr"]<1e-4:
                                optimizer[i].param_groups[0]["lr"] = 1e-3*np.random.rand(1)[0]
                        net[i].load_state_dict(torch.load('net{}.pth'.format(i)))
            if args.print_log==True:
                if epoch % 100 == 0:
                    print("{} epoch {}-th data's test result:{}".format(epoch+1,i+1,res_test[i]))
        if in_train == None:
            break
    for i in range(n):
        net[i].load_state_dict(torch.load('net{}.pth'.format(i)))
    return res_val, res_test

def e(x,alpha):
    return math.exp(-alpha*x)
def get_cov(args,net,loader_val,alpha_all, delta):
    n = args.n
    ex = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
                ex[i,j] = test_result(args,net[i],loader_val[j])
    aa = {}
    for alpha in alpha_all:
        aa[str(alpha)] = np.zeros([n,n])
        for i in range(n):
            for j in range(n):
                if i != j:
                    aa[str(alpha)][i,j] = e((ex[j,i]-ex[i,i])/ex[i,i]+(ex[i,j]-ex[j,j])/ex[j,j],alpha)
        aa[str(alpha)]=aa[str(alpha)]/aa[str(alpha)].max()
        for i in range(n):
            for j in range(n):
                if aa[str(alpha)][i,j] < delta:
                    aa[str(alpha)][i,j] = 0
    return aa

def get_TP(args,net,indA,m):
    n = args.n
    M = np.array([0]*n)
    I = np.array([0]*n)
    for i in range(n):
        aa = torch.norm(net[i].state_dict()['fc1.weight'],p=2,dim=0)
        _ , indcs = torch.sort(aa)
        TP = 0
        ind0 = indA[i][:args.num_b]
        for j in range(len(ind0)):
            if ind0[j] in indcs[-m:]:
                TP = TP +1
        M[i] = TP
        ind0 = indA[i][args.num_b:]
        TP = 0
        for j in range(len(ind0)):
            if ind0[j] in indcs[-m:]:
                TP = TP +1
        I[i] = TP
    return M, I

args = get_argparse().parse_args()
na = globals()
### Parameters###
#args.n = 5
#args.num_epochs = 16
#args.num_epochs_test = 32
#args.dvc = 'cuda:0'
#args.epochs_all = 400
args.l_r = [args.l_r]*args.n
#args.alpha_all = [0.5,1,2,4,8,16,32]
#args.delta = 0.2
#args.num_b = 15
#args.tor = 20
#args.scheme = 1
#args.print_log = True
### Data collection###
dataset_name=[0]*args.n
dataset_name_test=[0]*args.n
dataset_name_val=[0]*args.n
net_name=[0]*args.n
loader_name = [0]*args.n
loader_name_test = [0]*args.n
loader_name_val = [0]*args.n
optimizer_name = [0]*args.n
x_name = [0]*args.n
y_name = [0]*args.n
x_val = [0]*args.n
y_val = [0]*args.n
x_test = [0]*args.n
y_test = [0]*args.n
ind = [0]*args.n
for i in range(args.n):
    dataset_name[i] = 'data{}'.format(i+1)
    dataset_name_test[i] = 'datat{}'.format(i+1)
    dataset_name_val[i] = 'datav{}'.format(i+1)
    net_name[i] = 'net{}'.format(i+1)
    loader_name[i] = 'data_loader{}'.format(i+1)
    loader_name_test[i] = 'data_loader_test{}'.format(i+1)
    loader_name_val[i] = 'data_loader_val{}'.format(i+1)
    optimizer_name[i] = 'optimizer_loader{}'.format(i+1)
    x_name[i] = 'x{}'.format(i+1)
    y_name[i] = 'y{}'.format(i+1)
    x_val[i] = 'xv{}'.format(i+1)
    y_val[i] = 'yv{}'.format(i+1)
    x_test[i] = 'xt{}'.format(i+1)
    y_test[i] = 'yt{}'.format(i+1)
    ind[i] = 'ind{}'.format(i+1)
for i in range(args.n):
    na[x_val[i]] = np.load('data/{}.npy'.format(x_val[i]))
    na[x_test[i]] = np.load('data/{}.npy'.format(x_test[i]))
    na[x_name[i]] = np.load('data/{}.npy'.format(x_name[i]))
    na[y_val[i]] = np.load('data/{}.npy'.format(y_val[i]))
    na[y_test[i]] = np.load('data/{}.npy'.format(y_test[i]))
    na[y_name[i]] = np.load('data/{}.npy'.format(y_name[i]))
    na[ind[i]] = np.load('data/{}.npy'.format(ind[i]))
for i in range(args.n):
    na[dataset_name[i]] = [[na[x_name[i]][j],na[y_name[i]][j]] for j in range(na[x_name[i]].shape[0])]
    na[dataset_name_test[i]] = [[na[x_test[i]][j],na[y_test[i]][j]] for j in range(na[x_test[i]].shape[0])]
    na[dataset_name_val[i]] = [[na[x_val[i]][j],na[y_val[i]][j]] for j in range(na[x_val[i]].shape[0])]

for i in range(args.n):
    na[loader_name[i]] = DataLoader(
        dataset=na[dataset_name[i]],
        batch_size=args.num_epochs,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    na[loader_name_test[i]] = DataLoader(
        dataset=na[dataset_name_test[i]],
        batch_size=args.num_epochs_test,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    na[loader_name_val[i]] = DataLoader(
        dataset=na[dataset_name_val[i]],
        batch_size=args.num_epochs_test,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
### Network,Optimizer Initialization### 

ind_all = [na[ind[i]] for i in range(args.n)]
loader_train_all = [na[loader_name[i]] for i in range(args.n)]
loader_test_all = [na[loader_name_test[i]] for i in range(args.n)]
loader_val_all = [na[loader_name_val[i]] for i in range(args.n)]

TP_20M = np.zeros([1,3,5])
TP_40M = np.zeros([1,3,5])
TP_20I = np.zeros([1,3,5])
TP_40I = np.zeros([1,3,5])
noob = np.array([[0.]*3])
def initial():
    np.save('data/NTP20M',np.array([]))
    np.save('data/NTP20I',np.array([]))
    np.save('data/NTP40M',np.array([]))
    np.save('data/NTP40I',np.array([]))
    np.save('data/Nres',np.array([]))



### Tuning and select (1) DNNs
tuning_lambda1 = [1/16,0.125,0.25]
tuning_lambda2 = [1/16,0.125]
relationship = torch.zeros([args.n,args.n])
best0 = [1000]*args.n #best results in tuning set
best2 = [0]*args.n    #corresponding results in testing set
for par2 in tuning_lambda2:
    for par1 in tuning_lambda1:
        args.lambda1 = par1
        args.lambda2 = par2
        for key in net_name:
            na[key]=Net().to(args.dvc)
            for m in na[key].modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)
        for i in range(args.n):
            na[optimizer_name[i]]=optim.AdamW(na[net_name[i]].parameters(),lr=args.l_r[i])
        net_all = [na[net_name[i]] for i in range(args.n)]
        optimizer_all = [na[optimizer_name[i]] for i in range(args.n)]
        result_val, result_test = train(args,net_all,optimizer_all,loader_train_all,loader_test_all,loader_val_all,relationship)
        print("New result({},{}):{}".format(args.lambda1,args.lambda2,sum(result_val)),flush=True)
        for i in range(args.n):
            if result_val[i] < best0[i]:
                best0[i] = result_val[i] #weighted
                best2[i] = result_test[i]
                torch.save(net_all[i].state_dict(), 'net_normal{}.pth'.format(i))
                print('The current best2 is:{}'.format(sum(best2)),flush=True)

for i in range(args.n):
    net_all[i].load_state_dict(torch.load('net_normal{}.pth'.format(i)))
TP_20M[0,0,:], TP_20I[0,0,:] = get_TP(args,net_all,ind_all,20)
TP_40M[0,0,:], TP_40I[0,0,:] = get_TP(args,net_all,ind_all,40)
noob[0,0] = sum(best2)
print(noob[0,0],flush=True)
print(TP_40M[0,0,:],flush=True)
print(TP_40I[0,0,:],flush=True)

### Tuning and select (2) ANNI
rel = get_cov(args,net_all,loader_val_all,args.alpha_all, args.delta)
tuning_lambda0 = [8]
tuning_lambda1 = [0.125,0.25]
tuning_lambda2 = [0.125,0.25]
tuning_alpha = [2,4]
best0 = [1000]*args.n
best2 = [0]*args.n
for par3 in tuning_alpha:
    for par0 in tuning_lambda0:
        for par2 in tuning_lambda2:
            for par1 in tuning_lambda1:
                args.alpha0 = par3
                args.lambda0 = par0
                args.lambda1 = par1
                args.lambda2 = par2
                for key in net_name:
                    na[key]=Net().to(args.dvc)
                    for m in na[key].modules():
                        if isinstance(m, (nn.Conv2d, nn.Linear)):
                            nn.init.xavier_uniform_(m.weight)
                for i in range(args.n):
                    na[optimizer_name[i]] = optim.AdamW(na[net_name[i]].parameters(),lr=args.l_r[i])
                net_all = [na[net_name[i]] for i in range(args.n)]
                optimizer_all = [na[optimizer_name[i]] for i in range(args.n)]
                result_val, result_test = train(args,net_all,optimizer_all,loader_train_all,loader_test_all,loader_val_all,rel[str(args.alpha0)])
                print("New result({},{},{},{}):{}".format(args.alpha0,args.lambda0,args.lambda1,args.lambda2,sum(result_test)),flush=True)
                for i in range(args.n):
                    if result_val[i] < best0[i]:
                        best0[i] = result_val[i]
                        best2[i] = result_test[i]
                        torch.save(net_all[i].state_dict(), 'net_rel{}.pth'.format(i))
                        print('The current best2 is:{}'.format(sum(best2)),flush=True)

for i in range(args.n):
    net_all[i].load_state_dict(torch.load('net_rel{}.pth'.format(i)))
TP_20M[0,1,:], TP_20I[0,1,:] = get_TP(args,net_all,ind_all,20)
TP_40M[0,1,:], TP_40I[0,1,:] = get_TP(args,net_all,ind_all,40)
noob[0,1] = sum(best2)
print(noob[0,1],flush=True)
print(TP_40M[0,1,:],flush=True)
print(TP_40I[0,1,:],flush=True)



### Tuning and select (3) ANNI-e
def con_rel(relationship_original,sigma):
    relationship = copy.deepcopy(relationship_original)
    for i in range(relationship.shape[0]):
        for j in range(relationship.shape[0]):
            if j<=i:
                error = random.gauss(1,sigma)
                if error < 0:
                    error = 0
                relationship[i,j] = relationship[i,j]*error
                relationship[j,i] = relationship[i,j]
    if relationship.max()>0:
        relationship = relationship/relationship.max()
    return relationship

relationship = con_rel(rel[str(2)],1)
tuning_lambda0 = [8]
tuning_lambda1 = [0.125,0.25]
tuning_lambda2 = [0.125,0.25]
best0 = [1000]*args.n
best2 = [0]*args.n
for par0 in tuning_lambda0:
    for par2 in tuning_lambda2:
        for par1 in tuning_lambda1:
            args.lambda0 = par0
            args.lambda1 = par1
            args.lambda2 = par2
            for key in net_name:
                na[key]=Net().to(args.dvc)
                for m in na[key].modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        nn.init.xavier_uniform_(m.weight)
            for i in range(args.n):
                na[optimizer_name[i]]=optim.AdamW(na[net_name[i]].parameters(),lr=args.l_r[i])
            net_all = [na[net_name[i]] for i in range(args.n)]
            optimizer_all = [na[optimizer_name[i]] for i in range(args.n)]
            result_val, result_test = train(args,net_all,optimizer_all,loader_train_all,loader_test_all,loader_val_all,relationship)
            #print("New result({},{},{},{}):{}".format(args.alpha0,args.lambda0,args.lambda1,args.lambda2,sum(result_test)),flush=True)
            for i in range(args.n):
                if result_val[i] < best0[i]:
                    best0[i] = result_val[i] 
                    best2[i] = result_test[i]
                    torch.save(net_all[i].state_dict(), 'net_rel{}.pth'.format(i))
                    #print('The current best2 is:{}'.format(sum(best2)),flush=True)
for i in range(args.n):
    net_all[i].load_state_dict(torch.load('net_rel{}.pth'.format(i)))
TP_20M[0,2,:], TP_20I[0,2,:] = get_TP(args,net_all,ind_all,20)
TP_40M[0,2,:], TP_40I[0,2,:] = get_TP(args,net_all,ind_all,40)
noob[0,2] = sum(best2)
print(noob[0,2],flush=True)
print(TP_40M[0,2,:],flush=True)
print(TP_40I[0,2,:],flush=True)


if args.save_result == True:
    noob0 = np.load('data/Nres.npy')
    if len(noob0) == 0:
        noob0 = noob
        np.save('data/Nres',noob0)
    else:
        noob0 = np.concatenate((noob0,noob),axis=0)
        np.save('data/Nres',noob0)


    TP_20M0 = np.load('data/NTP20M.npy')
    if len(TP_20M0) == 0:
        TP_20M0 = TP_20M
        np.save('data/NTP20M',TP_20M0)
    else:
        TP_20M0 = np.concatenate((TP_20M0,TP_20M),axis=0)
        np.save('data/NTP20M',TP_20M0)

    TP_20I0 = np.load('data/NTP20I.npy')
    if len(TP_20I0) == 0:
        TP_20I0 = TP_20I
        np.save('data/NTP20I',TP_20I0)
    else:
        TP_20I0 = np.concatenate((TP_20I0,TP_20I),axis=0)
        np.save('data/NTP20I',TP_20I0)

    TP_40M0 = np.load('data/NTP40M.npy')
    if len(TP_40M0) == 0:
        TP_40M0 = TP_40M
        np.save('data/NTP40M',TP_40M0)

    else:
        TP_40M0 = np.concatenate((TP_40M0,TP_40M),axis=0)
        np.save('data/NTP40M',TP_40M0)


    TP_40I0 = np.load('data/NTP40I.npy')
    if len(TP_40I0) == 0:
        TP_40I0 = TP_40I
        np.save('data/NTP40I',TP_40I0)
    else:
        TP_40I0 = np.concatenate((TP_40I0,TP_40I),axis=0)
        np.save('data/NTP40I',TP_40I0)






