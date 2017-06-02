from functools import partial
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import pandas as pd
from matplotlib import colors
#import proximal as prox
import cvxpy as cvx
from joblib import Parallel, delayed

from scipy import stats

import spams

def colorGen():
    i = 0
    _color = [x for x in colors.ColorConverter.colors.keys() if x != 'w']
    while True:
        i = i + 1
        yield _color[i % len(_color)]


if False:

    



    dat160 = pd.read_excel('data/Plastic Raw.xlsx', sheetname=0, index_col=0, header=0)
    dat160.to_pickle('dat160.pkl')

    dat65 = pd.read_excel('data/Plastic Raw.xlsx', sheetname=1, index_col=0, header=0)
    dat65.to_pickle('dat65.pkl')

    F = pd.read_excel('data/F(64x64).xlsx')
    F.to_pickle('F.pkl')    



if True:
    # Checking the inherent variance of things
    from scipy.io import loadmat
    Dstar = loadmat('chris_D.mat')['D']
    newF = loadmat('newF.mat')['newF']

    _dat160 = pd.read_pickle('dat160.pkl')
    _dat65 = pd.read_pickle('dat65.pkl')
    _F = pd.read_pickle('F.pkl')

    maxWavelength = np.min([np.max(np.array(_dat65.index)), np.max(np.array(_F.columns))])
    minWavelength = np.max([np.min(np.array(_dat65.index)), np.min(np.array(_F.columns))])
    
    

    F = _F.loc[:, (_F.columns >= minWavelength) & (_F.columns <= maxWavelength)]
    dat160 = _dat160.loc[(_dat160.index >= minWavelength) & (_dat160.index <= maxWavelength), :]
    dat65 = _dat65.loc[(_dat65.index >= minWavelength) & (_dat65.index <= maxWavelength), :]/100.

    bigF = np.vstack([ np.interp(np.array(dat65.index), np.array(F.columns), np.array(F)[i,:] ).reshape((1,-1)) for i in range(len(F))])


    X160 = np.array(dat160)
    X65 = np.array(dat65)
    # Phi = bigF.copy() + np.random.randn(bigF.shape[0],bigF.shape[1])*1e-3

    m = bigF.shape[0]
    n = bigF.shape[1]


    def lineFit(x1, x2):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x1,x2)
        _x1 = x1*slope+intercept
        return _x1

    def calError(X, trueX):
        _X = X
        # if trueX.shape[1] == 1:
        #     _X = np.vstack([lineFit(x, trueX.ravel()) for x in X.T]).T
        # else:
        #     _X = np.vstack([lineFit(x, truex) for (x,truex) in zip(X.T,trueX.T) ]).T

        # return np.mean( np.linalg.norm((_X-trueX), axis=0) )
        return np.median( np.linalg.norm((_X-trueX), axis=0) / np.linalg.norm(trueX,axis=0) )
        # return np.mean( np.linalg.norm((X-X.mean(0))-(trueX-trueX.mean(0)), axis=0) )

    run_ids = np.array([int(run_id.split('_')[0]) for run_id in _dat160.columns]).astype(int)
    color = colorGen()
    plt.figure(5);plt.clf()
    Err = []
    for c in np.unique(run_ids):
        dat_c = X160[ :, np.where(run_ids==c)[0][:10] ]
        # dat_c = dat_c - dat_c.mean(0).reshape(1,-1)
        dat_c_mean = np.median(dat_c, axis=1)
        err = calError(dat_c, dat_c_mean.reshape(-1,1))

        print err
        Err.append(err)
        plt.plot(dat_c, color.next())
    print 'Average Error',np.median(Err)




if True:

    # M = [64]
    m = 64
    #k = 1e-1
    k = 5
    atoms = 20    
    _lambda = 1e4
    _lambda2 = 1e-3        

    # Param = range(5,40,5)
    # Param = range(16,65,4)
    Param = np.linspace(2,8,10)

    # Error = {'TV':[], 'SC':[], 'step1':[], 'step2':[], 'step3':[], 'M':[]}
    Error = {'Ideal':[], 'TV':[], 'SC':[], 'step1':[], 'step2':[], 'step3':[], 'param':[]}
    #for param in Param:
    if True: 
        param = 4
        # m = param

        _lambda = 10**param
        # k = 10**param        
        Phi = np.random.randn(m,n)
        # Phi = newF.copy()




        Y160 = np.dot(Phi, X160)
        Y65 = np.dot(Phi, X65)

        n = X160.shape[0]
        m = Phi.shape[0]

        y = Y160[:,0]


        def _encodeTV(y, Phi, _lambda):
            x = cvx.Variable(n, 1)
            obj = cvx.Minimize( cvx.norm( y - Phi*x ) + _lambda*cvx.sum_squares(cvx.diff(x))  )
            constraints = []
            prob = cvx.Problem(obj, constraints)
            if _lambda > 1:
                prob.solve(verbose=False, solver=cvx.SCS)
            else:
                prob.solve(verbose=False, solver=cvx.ECOS)
            return np.array(x.value)

        def encodeTV(Y, Phi, _lambda):    

            encoder = partial(_encodeTV, Phi=Phi, _lambda=_lambda)
            tmp = Parallel(n_jobs=-1, verbose=10, backend="multiprocessing")(map(delayed(encoder), Y.T))  #Chris modified
            #tmp = map(encoder, Y.T)
            # alternative backend = "threading"
            return np.hstack(tmp)



        stageA_160 = encodeTV(Y160, Phi, _lambda)
        stageAresidual_160 = X160 - stageA_160

        stageA_65 = encodeTV(Y65, Phi, _lambda)
        stageAresidual_65 = X65 - stageA_65
        stageAresidualY_65 = Y65-np.dot(Phi, stageA_65)


        def learnD(X, atoms, k):
             _D =  spams.trainDL(np.asfortranarray(X),return_model= False,model= None,D = None,
                   numThreads = -1,batchsize = -1,
                   K= atoms,lambda1= k,lambda2= 10e-10,iter=-1,t0=1e-5,mode=3,
                   posAlpha=False,posD=False,expand=False,modeD=0,whiten=False,
                   clean=True,verbose=True,gamma1=0.,gamma2=0.,rho=1.0,iter_updateD=None,
                   stochastic_deprecated=False,modeParam=0,batch=False,log_deprecated=False,
                   logName='')
             return np.array(_D)
        #def learnD(X, atoms, k):
        #   _D =  spams.trainDL(np.asfortranarray(X),return_model= False,model= None,D = None,
        #          numThreads = -1,batchsize = -1,
        #          K= atoms,lambda1= k,lambda2= 0,iter=-1,t0=1e-5,mode=2,
        #          posAlpha=False,posD=False,expand=False,modeD=0,whiten=False,
        #          clean=True,verbose=True,gamma1=0.1,gamma2=0.,rho=1.0,iter_updateD=None,
        #          stochastic_deprecated=False,modeParam=0,batch=False,log_deprecated=False,
        #          logName='')
        #    return np.array(_D)

        
        # atoms = 30
        D = learnD(X160,atoms, k=k)
        Dresidual = learnD(stageAresidual_160, atoms, k=k)
        # Dresidual = Dstar.copy()


        def encodeSC(Y, Psi, k):  
             atom_norm = np.linalg.norm(Psi, axis=0) 
             # _Psi = Psi
             _Psi = Psi/atom_norm
             Z = spams.omp(np.asfortranarray(Y), np.asfortranarray(_Psi), L=k, 
                 eps= None,lambda1 = None,return_reg_path = False,numThreads = -1)  
             Z = np.array(Z.todense())/atom_norm.reshape(-1,1)
             return Z
        #def encodeSC(Y, Psi, k):  
        #    Z = spams.lasso(np.asfortranarray(Y),D= np.asfortranarray(Psi),
        #            Q = None, q = None,return_reg_path = False,L= -1,lambda1= k,
        #            lambda2= 0.,mode= 2,pos= False,ols= True,numThreads= -1,
        #            max_length_path= -1,verbose=False,cholesky= False)
        #    Z = np.array(Z.todense())
        #    return Z

        stageB_160z = encodeSC(Y160, np.dot(Phi,D) , k=k)
        stageB_160 = np.dot(D, stageB_160z)

        stageB_65z = encodeSC(Y65, np.dot(Phi,D) , k=k)
        stageB_65 = np.dot(D, stageB_65z)




        # Second step of the 3-step method
        Psi = np.dot(Phi,Dresidual)
        stageAB_65z = encodeSC(stageAresidualY_65, Psi , k=k)
        stageAB_65 = np.dot(Dresidual, stageAB_65z)
        stageABresidual_65 = Y65-np.dot(Psi, stageAB_65z)

        print np.max(stageAB_65z)
        print np.max(stageB_65z)


        # Third step of the 3-step method
        stageC_65 = encodeTV(Y65, Phi, _lambda2)
        stageABC_65 = encodeTV(stageABresidual_65, Phi, _lambda=_lambda2)

        Error['param'].append( param )
        Error['Ideal'].append( np.median(Err) )
        Error['TV'].append( calError(stageC_65, X65) )
        Error['SC'].append( calError(stageB_65, X65) )
        Error['step1'].append( calError(stageA_65, X65) )
        Error['step2'].append( calError(stageAB_65 + stageA_65, X65) )
        Error['step3'].append( calError(stageAB_65 + stageABC_65, X65) )

    

        print "TV", calError(stageC_65, X65)
        print "SC", calError(stageB_65, X65)
        print "step 1", calError(stageA_65, X65)
        print "step 2", calError(stageAB_65 + stageA_65, X65)
        print "step 3", calError(stageAB_65 + stageABC_65, X65)
        print "Average inherent error", np.mean(Err)



        if False:
            tid = np.random.randint(0,65)
            # tid = 8
            # plt.figure(4);
            # plt.clf()
            # plt.title(tid)
            # plt.subplot(3,1,1)
            # plt.plot(X65[:,tid],'b')
            # plt.plot(stageA_65[:,tid],'m')
            # # plt.plot(stageB_65[:,tid],'r')
            # plt.plot(stageAB_65[:,tid] + stageA_65[:, tid],'g')
            # plt.plot(stageAB_65[:,tid] + stageABC_65[:, tid],'c')
            # plt.subplot(3,1,2)
            # plt.plot(X65[:,tid]-stageA_65[:,tid],'b')
            # plt.plot(stageAB_65[:,tid],'g')
            # plt.subplot(3,1,3)
            # plt.plot(X65[:,tid]-stageAB_65[:,tid],'b')
            # plt.plot(stageABC_65[:,tid],'g')
            # plt.plot(stageA_65[:,tid],'g', alpha=0.5)

            plt.figure(3); plt.clf()
            plt.plot(X65[:,tid],'b', label='x')
            plt.plot(stageC_65[:,tid],'r', label='TV')
            # plt.plot(stageB_65[:,tid],'m', label='SC')
            # plt.plot(stageA_65[:,tid],'r', label='1step')
            # plt.plot(stageA_65[:,tid] + stageAB_65[:, tid],'m', label='2step')
            plt.plot(stageAB_65[:,tid] + stageABC_65[:, tid],'g', label='3step')
            plt.legend(loc='best'); plt.grid()
            print "TV", calError(X65[:,tid].reshape(-1,1), stageC_65[:,tid].reshape(-1,1))
            print "SC", calError(X65[:,tid].reshape(-1,1), stageB_65[:,tid].reshape(-1,1))
            print "1step", calError(X65[:,tid].reshape(-1,1), (stageA_65[:,tid]).reshape(-1,1))
            print "2step", calError(X65[:,tid].reshape(-1,1), (stageA_65[:,tid] + stageAB_65[:, tid]).reshape(-1,1))
            print "3step", calError(X65[:,tid].reshape(-1,1), (stageAB_65[:,tid] + stageABC_65[:, tid]).reshape(-1,1))



if False:
    import matplotlib.pyplot as plt
    plt.ion()    
    import scipy.io as io
    import numpy as np
    # Error = io.loadmat('over_m.mat',Error)

    plt.figure(2)
    plt.clf()
    plt.plot(Error['param'], Error['SC'], 'm',label='SC')
    plt.plot(Error['param'], Error['TV'], 'r',label='TV')
    plt.plot(Error['param'], Error['step1'], 'g-.',label='step1', alpha=0.3)
    plt.plot(Error['param'], Error['step2'], 'g-.',label='step2', alpha=0.6)
    plt.plot(Error['param'], Error['step3'], 'g',label='step3')
    plt.plot(Error['param'], [np.median(Err)]*len(Error['param']), 'k-.',label='Target')
    plt.legend(loc='upper right'); plt.grid()
    plt.xlabel('parameter')
    plt.ylabel('error')

if False:

        _lambda = 1e-2
        tid = 0
        # y = stageABresidual_65[:,tid]
        PPhi = Phi.copy()
        # PPhi = np.random.randn(PPhi.shape[0],PPhi.shape[1])

        y = Y65[:,tid].reshape(-1,1)-np.dot(Psi, stageAB_65z[:,tid].reshape(-1,1))
        y2 = np.dot(PPhi, X65[:,tid]-stageAB_65[:,tid]).reshape(-1,1)
        y3 = stageABresidual_65[:,tid].reshape(-1,1)

        x = cvx.Variable(n, 1)
        obj = cvx.Minimize( cvx.norm( y - PPhi*x ) + _lambda*cvx.sum_squares(cvx.diff(x))  )
        constraints = []
        prob = cvx.Problem(obj, constraints)
        # prob.solve(verbose=True, solver=cvx.CVXOPT)
        prob.solve(verbose=True, solver=cvx.ECOS)

        plt.clf()
        plt.plot(X65[:,tid]-stageAB_65[:,tid] ,'g')
        plt.plot(np.array(x.value),'b')
        t2 = np.array(x.value)


        # x = cvx.Variable(n, 1)
        # obj = cvx.Minimize( cvx.norm( y - PPhi*x ) + _lambda*cvx.sum_squares(x[1:]-x[:-1])  )
        # constraints = []
        # prob = cvx.Problem(obj, constraints)
        # prob.solve(verbose=True, solver=cvx.SCS)


        # plt.plot(np.array(x.value),'r')



