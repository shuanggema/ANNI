'''
This is the data generation for nonlinear regression, 
which corresponds to the simulation in Table S20-S22.
'''
import numpy as np
import random
import copy

E = 5 #E dimension
p = 100 #G dimension
n = 150 #number of train and tuning set
indp = 10 #G effct
indi = 15 #Interaction effect
signal = 0 # 0:weel signal，1:strong signal
sigma2 = 1 
relation = 'AR'

def get_AR_cov(dim):
    cov = np.zeros([dim,dim])
    for i in range(dim):
        for j in range(dim):
                cov[i,j] = 0.3**abs(i-j)
    return cov

def get_E_cov(dim):
    cov = np.zeros([dim,dim])
    for i in range(dim):
        for j in range(dim):
                cov[i,j] = 0.2**abs(i-j)
    return cov
    
def get_Band_cov(dim):
    cov = np.zeros([dim,dim])
    for i in range(dim):
        for j in range(dim):
            if abs(i-j) == 0:
                cov[i,j] = 1
            if abs(i-j) == 1:
                cov[i,j] = 0.33
    return cov  

def get_mean(dim):
    return np.zeros(dim)


def get_alldata(n,E,p,relation,index1,index2):
    Evar = np.random.multivariate_normal(get_mean(E), get_E_cov(E), n, 'raise') 
    if relation == 'AR':
        Pvar = np.random.multivariate_normal(get_mean(p), get_AR_cov(p), n, 'raise')
    elif relation == 'Band':
        Pvar = np.random.multivariate_normal(get_mean(p), get_Band_cov(p), n, 'raise')
    else:
        Pvar = np.random.multivariate_normal(get_mean(p), get_E_cov(p), n, 'raise')
    Ivar = np.zeros([n,E*p])
    bias = np.ones([n,1])
    for i in range(n):
        vec_tan = []
        for j in range(E):
            vec_tan = vec_tan + list(Evar[i,j]*Pvar[i,:])
        Ivar[i,:] = np.array(vec_tan)
    Evar = np.sin(Evar)
    Pvar = np.cos(Pvar)
    Ivar = np.tanh(Ivar)
    X = np.concatenate((bias,Evar,Pvar,Ivar),axis=1)
    index = np.zeros(E+p+E*p+1)
    index[:(E+1)] = 1
    index[index1] = 1
    index[index2] = 1
    beta = np.zeros(E+p+E*p+1)
    for i in range(E+p+E*p+1):
        if index[i]==1:
            if signal == 0:
                beta[i]=random.uniform(0.2,0.8)
            else:
                beta[i]=random.uniform(0.6,1.2)
    error = np.array([random.gauss(0,sigma2) for i in range(n)])
    Y_T = np.matmul(X,beta)
    Y = Y_T + error
    return X, Y, Y_T, beta, index
    
def get_alldata_beta(n,E,p,relation,beta):
    Evar = np.random.multivariate_normal(get_mean(E), get_E_cov(E), n, 'raise') 
    if relation == 'AR':
        Pvar = np.random.multivariate_normal(get_mean(p), get_AR_cov(p), n, 'raise')
    elif relation == 'Band':
        Pvar = np.random.multivariate_normal(get_mean(p), get_Band_cov(p), n, 'raise')
    else:
        Pvar = np.random.multivariate_normal(get_mean(p), get_E_cov(p), n, 'raise')
    Ivar = np.zeros([n,E*p])
    bias = np.ones([n,1])
    for i in range(n):
        vec_tan = []
        for j in range(E):
            vec_tan = vec_tan + list(Evar[i,j]*Pvar[i,:])
        Ivar[i,:] = np.array(vec_tan)
    Evar = np.sin(Evar)
    Pvar = np.cos(Pvar)
    Ivar = np.tanh(Ivar)
    X = np.concatenate((bias,Evar,Pvar,Ivar),axis=1)
    error = np.array([random.gauss(0,sigma2) for i in range(n)])
    Y_T = np.matmul(X,beta)
    Y = Y_T + error
    return X, Y, Y_T, beta

def get_index_different(E,p,nump,numi,index1,index2):
    dif = 0
    while dif==0:
        dif=1
        pindex0 = random.sample(range(E+1,E+p+1),nump)
        iindex0 = random.sample(range(E+p+1,E+p+E*p+1),numi)
        for i in range(len(index1)):
            if index1[i] in pindex0:
                dif=0
                break
        for i in range(len(index2)):
            if index2[i] in iindex0:
                dif=0
                break
    return pindex0, iindex0
    
def change_beta(beta,index1=None,index2=None,index3=None):
    if index1!=None:
        for key in index1:
            if signal == 0:
                beta[key] = random.uniform(0.2,0.8)
            else:
                beta[key] = random.uniform(0.6,1.2)
    if index2!=None:
        for key in index2:
            beta[key] = 0
    if index3!=None:
        for key in index3:
            if signal == 0:
                beta[key] = random.uniform(0.2,0.8)
            else:
                beta[key] = random.uniform(0.6,1.2)
    return beta
    
    
 
pindex = random.sample(range(E+1,E+p+1),indp)
iindex = random.sample(range(E+p+1,E+p+E*p+1),indi)
x1, y1, y1_T, beta1, index= get_alldata(n+n+1000,E,p,relation,pindex,iindex)
index1 = [1,2,3,4,5] + pindex[:5]
beta2 = copy.deepcopy(beta1)
beta2 = change_beta(beta2,index1)

index2 = iindex[:4]
index3 = get_index_different(E,p,0,4,pindex,iindex)
index3 = index3[0] + index3[1]
beta3 = copy.deepcopy(beta1)
beta3 = change_beta(beta3,index2=index2,index3=index3)

x2, y2, y2_T, beta2 = get_alldata_beta(n+n+1000,E,p,relation,beta2)
x3, y3, y3_T, beta3 = get_alldata_beta(n+n+1000,E,p,relation,beta3)


pindex4, iindex4 = get_index_different(E,p,indp,indi,pindex,iindex)
x4, y4, y4_T, beta4, index4= get_alldata(n+n+1000,E,p,relation,pindex4,iindex4)
index2 = pindex4[:3] + iindex4[:4]
index3 = get_index_different(E,p,3,4,pindex4,iindex4)
index3 = index3[0] + index3[1]
beta5 = copy.deepcopy(beta4)
beta5 = change_beta(beta5,index2=index2,index3=index3)
x5, y5, y5_T, beta5 = get_alldata_beta(n+n+1000,E,p,relation,beta5)



x_name=['x1','x2','x3','x4','x5']
y_name=['y1','y2','y3','y4','y5']
x_val=['xv1','xv2','xv3','xv4','xv5']
y_val=['yv1','yv2','yv3','yv4','yv5']
x_test=['xt1','xt2','xt3','xt4','xt5']
y_test=['yt1','yt2','yt3','yt4','yt5']
ind =['ind1','ind2','ind3','ind4','ind5']
ind1 = np.where(beta1!=0)[0][1:]-1
ind2 = np.where(beta2!=0)[0][1:]-1
ind3 = np.where(beta3!=0)[0][1:]-1
ind4 = np.where(beta4!=0)[0][1:]-1
ind5 = np.where(beta5!=0)[0][1:]-1


na =globals()
for i in range(5):
    na[x_name[i]] = na[x_name[i]][:,1:]
    na[x_val[i]] = na[x_name[i]][n:(n+n),:]
    np.save('data/{}'.format(x_val[i]),na[x_val[i]])
    na[x_test[i]] = na[x_name[i]][(n+n):,:]
    np.save('data/{}'.format(x_test[i]),na[x_test[i]])
    na[x_name[i]] = na[x_name[i]][:n,:]
    np.save('data/{}'.format(x_name[i]),na[x_name[i]])
    na[y_val[i]] = na[y_name[i]][n:(n+n)]
    np.save('data/{}'.format(y_val[i]),na[y_val[i]])
    na[y_test[i]] = na[y_name[i]][(n+n):]
    np.save('data/{}'.format(y_test[i]),na[y_test[i]])
    na[y_name[i]] = na[y_name[i]][:n]
    np.save('data/{}'.format(y_name[i]),na[y_name[i]])
    np.save('data/{}'.format(ind[i]),na[ind[i]])




