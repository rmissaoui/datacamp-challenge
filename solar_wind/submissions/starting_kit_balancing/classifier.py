from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBClassifier
import numpy as np
###
# Import Library
import scipy
import operator
import scipy.stats
import pandas as pd
from scipy.special import gammaln, psi
from numpy import transpose as t
from numpy.linalg import eigvals, eig, det, inv
from numpy import transpose, matrix, multiply, sqrt, diag, log, pi


class Classifier(BaseEstimator):
    def __init__(self):
        self.name = "logreg"
        if self.name == "logreg":
            self.model = make_pipeline(StandardScaler(), LogisticRegression())
            
        elif self.name == "xgboost":
            self.algo = XGBClassifier(learning_rate=0.05, n_estimators=140, max_depth=1,
                        min_child_weight=3, gamma=0.2, subsample=0.8, colsample_bytree=1.0,
                        objective='binary:logistic', nthread=12, scale_pos_weight=1, seed=0)
            
            self.model = make_pipeline(StandardScaler(), self.algo)

    def fit(self, X, y):
        
        if self.name == "xgboost":
            #cross validate
            self.model.fit(compute_pca(X, y), y)

        elif self.name == "logreg":
            self.model.fit(X, y)


    def predict_proba(self, X):
        return self.model.predict_proba(X)


##########################

def compute_pca(data, labels, n_comp=6):
    
    labels2 = labels.copy()
    
    
    ## Compute Pearson Correlation
    data2 = data.copy()
    #data2.iloc[:,:] = Y
    ## Compute Pearson Correlation
    correlation = {}
    for i in range(0,data2.shape[1]):
        x = data2.iloc[:,i]
        correlation.update({data.columns[i]: 
                            np.abs(scipy.stats.pearsonr(x, labels2)[0])})
    top_dict = sorted(correlation.items(),
                         key=operator.itemgetter(1),reverse=True)[:n_comp]


    #selected top 50 series
    multi_data = data2[[i[0] for i in top_dict]]
    multi_data.head()
    Y = np.mat(multi_data)
    T,q = Y.shape
    Y = transpose(Y)
    
    
    ## forward filtering Multivariate DLM
    model = multi_filtering(Y = Y , m0 = matrix(np.ones([1,q])),
                            beta = 0.99, delta = 0.99, 
                                backwardSmooth = False,Print = False)

    # construct variance proportion explained
    sSt = model.sSt
    sdCt = model.sdCt
    sMt = model.sMt
    sloglik = model.sloglik
    eigs = model.eigs
    eigs_vec = model.eig_vec
    eigs_frac = 100*eigs/np.sum(eigs,axis = 0)
    
    component = project_component(Y,eigs_vec,n_comp)
    component = component - np.mean(component,0)
    component = component / np.std(component,0)

    df_component = pd.DataFrame(component)
    
    return df_component

# Function setup

# Log Likelihood
def ltpdf(x,m,w,n,D):
    q = x.shape[0]
    C = transpose(np.linalg.cholesky(D));
    e = inv(C)@(x-m);
    d = n+q-1;
    ltpdf = q*np.log(2)/2 - q*np.log(2*pi*w)/2 - sum(np.log(diag(C))
                                                    ) - (d+1)*np.log(
        1+(transpose(e)@e)/w)/2;
    ltpdf += sum(gammaln(1 + d - np.array(list(range(0,q))))/2
                ) - sum(gammaln(d - np.array(list(range(0,q))))/2)
    return(ltpdf)

## Multivariate DLM
## only for local level , thus p = 1
class multi_filtering(object):
    
    def __init__(self, Y  ,m0 = matrix(np.ones([1,6])),
                 c0 = 0.01 , n0 = 3 ,beta = 0.9, delta = 0.9, 
                 backwardSmooth = False, Print = True):
        
        ## tune beta
        q = 6

        
        #change discounts to assess, compare marginal likelihood
        p=1;   
        F=np.matrix(np.ones([1,T]));

        #delta = 0.9; # discount level
        #beta = 0.9;  # discount volatility

        n0 = n0 ; 
        h0=n0+q-1; 
        D0=np.matrix(h0*np.identity(q));

        z = np.ones([p,q],dtype=complex);  
        zq=np.zeros([q,1],dtype=complex); 

        M0 = m0; 
        r=0.99; # priors
        Mt = M0;

        C0= np.matrix( c0 *np.identity(p))
        Ct=C0;        # initial Theta prior 

        n = n0; 
        h=h0;
        D = D0; 
        St=D/h;         # initial Sigma prior


        sMt = np.ones([p,q,T],dtype=complex); 
        sCt=np.zeros([p,p,T],dtype=complex);
        sdCt=np.zeros([p,q,T],dtype=complex);
        sSt=np.zeros([q,q,T],dtype=complex);  
        snt=np.zeros(T,dtype=complex);
        sloglik=np.zeros(T,dtype=complex); 

        sEt = matrix(np.zeros([q,T],dtype=complex))

        eigs = np.zeros([q,T],dtype=complex)
        eig_vec = []
        eig_sort_index = []
        
        self.M0 = M0
        self.C0 = C0
        self.S0 = St
        self.delta = delta
        self.beta = beta
        
        if Print:
            print('Start forward filtering...')
        
        # # forward filtering: 
        for t in range(T):
            ft = np.matrix(transpose(Mt) @ F[:,t])
            et = Y[:,t] - ft;
            Rt = Ct/delta; 
            h  = beta*h; 
            n=h-q+1; 
            D = beta*D;  
            snt[t]=n;  
            qvt = 1 + transpose(F[:,t])@Rt@F[:,t]; 
            sEt[:,t] = transpose(np.squeeze(et)/np.sqrt(qvt*np.diag(St)))

            At = Rt@F[:,t]/qvt;
            h += 1; 
            n += 1; 
            D += et @ transpose(et)/qvt
            St=D/h; 
            St=(St+transpose(St))/2; 
            Mt = Mt + At@transpose(et);
            Ct = Rt - At@np.transpose(At)*qvt;

            #
            sloglik[t] = ltpdf(et,zq,qvt,n,D);    

            # PCA
            if backwardSmooth == False:
                           
                eig_vals, eig_vecs = eig(St)
                eig_vals_sorted = np.sort(eig_vals)[::-1]
                eig_vecs_sorted = eig_vecs[:, eig_vals.argsort()[::-1]]
                eigs[:,t] = eig_vals_sorted
                eig_sort_index.append(eig_vals.argsort()[::-1])

                if (t>0):
                    a = eig_vecs_sorted - eig_vec[t-1]; 
                    f = -eig_vecs_sorted - eig_vec[t-1];
                    eig_a = np.sum(np.multiply(a,a),axis = 0)
                    eig_f = np.sum(np.multiply(f,f),axis = 0)

                    compare = np.argmin([eig_a,eig_f],0)
                    temp_compare = np.array(compare[0])
                    index = list(np.where(temp_compare == 1)[0])
                    eig_vecs_sorted[:,index] = -1 * eig_vecs_sorted[:,index]

                eig_vec.append(eig_vecs_sorted)
            
            # save
            sCt[:,:,t]=Ct;
            sSt[:,:,t]=St; 
            sMt[:,:,t] = Mt; 
            sdCt[:,:,t] = sqrt(transpose(diag(Ct)*diag(St)));
        
        
        if Print:
            print('Start Backward Smoothing...')
                
        if backwardSmooth:
            # reverse smoothing 
            K=inv(sSt[:,:,-1]); 
            n=snt[:-1]; 
            Mt = sMt[:,:,-1]; 
            Ct = sCt[:,:,-1]; 

            for t in list(range(1,T-1))[::-1]:
                K=(1-beta)*inv(sSt[:,:,t])+beta*K;         
                St = inv(K); 
                sSt[:,:,t]=St;  
                Mt = (1-delta)*sMt[:,:,t] + delta*Mt;  
                sMt[:,:,t] = Mt; 
                Ct = (1-delta)*sCt[:,:,t] + np.power(delta,2)*Ct;    
                sCt[:,:,t] = Ct; 
                sdCt[:,:,t] = sqrt(diag(Ct)* transpose(diag(St)));
            
        #print(eig_a)
        #print(eig_f)
        #print(np.argmin([eig_a,eig_f],0))

        self.sCt = sCt
        self.sSt = sSt
        self.sMt = sMt
        self.sdCt = sdCt
        self.eigs = eigs
        self.sloglik = sloglik
        self.eig_vec = eig_vec
        self.eig_sort_index = eig_sort_index
        
        if Print:
            print('Finished...')
    

## PC projection
def project_component(Y,eigs_vec,n):
    component = np.zeros([len(eigs_vec),n])
    for i in range(len(eigs_vec)):
        component[i,:] = np.abs(transpose(Y[:,i]) @ eigs_vec[i][:,0:n])
    return(component)


