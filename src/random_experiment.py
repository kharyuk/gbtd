from gtcd import initializeCBT, tcd, recover, refactorizeDicts, Bkl
from group import group_constraint, projector_group_constraint
import multiprocessing

import numpy as np
import copy
import time

import os
import re
def stringSplitByNumbers(x):
    '''
    from comment here
    http://code.activestate.com/recipes/135435-sort-a-string-using-numeric-order/
    '''
    r = re.compile('(\d+)')
    l = r.split(x)
    return [int(y) if y.isdigit() else y for y in l]

def sub2ind(subs, shape):
    return np.ravel_multi_index(subs, shape, order='F')

def ind2sub(index, shape):
    return np.unravel_index(index, shape, order='F')

def printStats(fnm):
    df = np.load(fnm)
    fv = df['resultFV']
    tms = df['resultTime']
    names = df['methods']
    minv = fv[:,:,-1].min(axis=0)
    maxv = fv[:,:,-1].max(axis=0)
    medv = np.median(fv[:,:,-1], axis=0)
    meanv = np.mean(fv[:,:,-1], axis=0)
    mint = tms[:,:].min(axis=0)
    maxt = tms[:,:].max(axis=0)
    medt = np.median(tms[:,:], axis=0)
    meant = np.mean(tms[:,:], axis=0)    
    print "\t%s" % (fnm)
    print "\t%s" % (reduce(lambda x, y: str(x)+'\t'+str(y), names)) #\t ALS\t dogleg\t scg_q\t scg_fn"
    print "Min.fv"+" %.3e"*minv.size % tuple(minv)
    print "Med.fv"+" %.3e"*medv.size % tuple(medv)
    print "Mean.fv"+" %.3e"*meanv.size % tuple(meanv)
    print "Max.fv"+" %.3e"*maxv.size % tuple(maxv)
    print
    print "Min.tm"+" %.3e"*mint.size % tuple(mint)
    print "Med.tm"+" %.3e"*medt.size % tuple(medt)
    print "Mean.tm"+" %.3e"*meant.size % tuple(meant)
    print "Max.tm"+" %.3e"*maxt.size % tuple(maxt)

def projector(canonical_dict=None, lro_dict=None, tucker_dict=None):
    rv = projector_group_constraint(canonical_dict, lro_dict, tucker_dict, dimnums=constraints['dimnum'], vect=False)
    return rv

def ex1_Convergence(
    n, methods, canonical_dict=None, lro_dict=None, tucker_dict=None,
    fnm='ex1-1_conv', Nrun=10, overestimated=None, maxitnum=100,
    mynum=None, constraints=None, projector=None
):
    '''
    Test perfomance of algorithms without any constraints except term parameters (such as rank, number of terms, ..)
    Actual realisation will iterate over all lower ranks, choosing between maximal(exact)+overestimate (if derived)
    and iterated.
    '''
    if isinstance(mynum, int):
        np.random.seed(mynum)
    cFlag = canonical_dict is not None
    lFlag = lro_dict is not None
    tFlag = tucker_dict is not None
    assert cFlag or lFlag or tFlag
    result_rank_modes = []
    d = len(n)
    if cFlag:
        Rc = canonical_dict['Rc']
        result_rank_modes.append(Rc)
        if isinstance(overestimated, int) and (overestimated > 0):
            result_rank_modes[-1] += overestimated
    if lFlag:
        L = lro_dict['L']
        P = lro_dict['P']
        if (constraints is not None) and (not tFlag):
            result_rank_modes.append(max(L[:-1]))
        else:
            result_rank_modes.append(max(L))
        if isinstance(overestimated, int) and (overestimated > 0):
            result_rank_modes[-1] += overestimated
        fmc = None
        if 'fullModesConfig' in lro_dict.keys():
            fmc = lro_dict['fullModesConfig']
            tmp = filter(lambda x: x is not None, fmc)
            tmp = [x[0] for x in tmp]
            result_rank_modes.append(max(tmp))
            if isinstance(overestimated, int) and (overestimated > 0):
                result_rank_modes[-1] += overestimated
    if tFlag:
        r = tucker_dict['r']
        if constraints is not None: # must be modified for any other
            result_rank_modes.append(r[:,:-1].max())
        else:
            result_rank_modes.append(r.max())
        if isinstance(overestimated, int) and (overestimated > 0):
            result_rank_modes[-1] += overestimated
    
    resultFV = np.zeros([Nrun, len(methods)] + result_rank_modes + [maxitnum+1])
    resultGV = np.zeros([Nrun, len(methods)] + result_rank_modes + [maxitnum])
    resultTimeClock = np.zeros([Nrun, len(methods)] + result_rank_modes)
    resultTimeTime = np.zeros([Nrun, len(methods)] + result_rank_modes)

    result_rank_modes = np.array(result_rank_modes)

    Ngrid_param = sub2ind(result_rank_modes-1, result_rank_modes) + 1
    for itRun in xrange(Nrun):
        C, B, A, G = initializeCBT(
            n, canonical_param=canonical_dict, lro_param=lro_dict, tucker_param=tucker_dict, rtype='normal', normalize=True
        )
        if cFlag:
            canonical_dict['C'] = C
        if lFlag:
            lro_dict['B'] = B
            if 'E' in lro_dict.keys():
                del lro_dict['E']
        if tFlag:
            tucker_dict['A'] = A
            tucker_dict['G'] = G
        if projector is not None:
            canonical_dict, lro_dict, tucker_dict = projector(canonical_dict, lro_dict, tucker_dict)
        a = recover(n, canonical_dict=canonical_dict, lro_dict=lro_dict, tucker_dict=tucker_dict)
        a /= np.linalg.norm(a)
        for gridParam in xrange(Ngrid_param):
            indices = ind2sub(gridParam, result_rank_modes)
            ind = 0
            cdN = None
            if cFlag:
                cdN = copy.deepcopy(canonical_dict)
                cdN['Rc'] = indices[ind]+1
                ind += 1
            ldN = None
            if lFlag:
                ldN = copy.deepcopy(lro_dict)
                if (constraints is not None) and (not tFlag): # must be modified for any other
                    ldN['L'][:-1] = np.minimum(np.array(ldN['L'][:-1]), indices[ind]+1)
                else:
                    ldN['L'] = np.minimum(np.array(ldN['L']), indices[ind]+1)
                if 'E' in ldN.keys():
                    del ldN['E']
                ind += 1
                if fmc is not None:
                    copy_fmc = copy.deepcopy(fmc)
                    for k in xrange(d):
                        if copy_fmc[k] is None:
                            continue
                        copy_fmc[k][0] = min(fmc[k][0], indices[ind]+1)
                    ldN['fullModesConfig'] = copy_fmc
                    ind += 1
            tdN = None
            if tFlag:
                tdN = copy.deepcopy(tucker_dict)
                if constraints is not None: # must be modified for any other
                    tdN['r'][:, :-1] = np.minimum(tdN['r'][:, :-1], indices[ind]+1)                    
                else:
                    tdN['r'] = np.minimum(tdN['r'], indices[ind]+1)                    
            Cinit, Binit, Ainit, Ginit = initializeCBT(
                n, canonical_param=cdN, lro_param=ldN, tucker_param=tdN, rtype='normal', normalize=True
            )
            if projector is not None:
                cdN, ldN, tdN = refactorizeDicts(cdN, ldN, tdN, Cinit, Binit, Ainit, Ginit)
                cdN, ldN, tdN = projector(cdN, ldN, tdN)
                if cFlag:
                    Cinit = copy.deepcopy(cdN['C'])
                if lFlag:
                    Binit = copy.deepcopy(ldN['B'])
                if tFlag:
                    Ainit = copy.deepcopy(tdN['A'])
                    Ginit = copy.deepcopy(tdN['G'])
                    
            if (cdN is not None) and ('C' in cdN.keys()):
                del cdN['C']
            if (ldN is not None) and ('B' in ldN.keys()):
                del ldN['B']
            if (tdN is not None) and ('A' in tdN.keys()) and ('G' in tdN.keys()):
                del tdN['A'], tdN['G']
            
            times = []
            for itAlg in xrange(len(methods)):
                x0 = {
                    'C': copy.deepcopy(Cinit),
                    'B': copy.deepcopy(Binit),
                    'A': copy.deepcopy(Ainit),
                    'G': copy.deepcopy(Ginit)
                }
                method_name, alg = methods[itAlg]
                t01 = time.time(); t0 = time.clock()
                cdN, ldN, tdN, info = alg(a, x0, cdN, ldN, tdN, maxitnum, constraints)
                t1 = time.clock(); t11 = time.time()
                if len(info['gnorm']) == 0:
                    info['gnorm'] = [np.nan]
                if isinstance(mynum,int):
                    print "%d: Run:%d\t Alg: %s\t Clock: %.2f\t Time: %.2f\t Fv: %.8f\t Gv: %.8f; Ranks %s" % (mynum, itRun+1,
                    method_name, t1-t0, t11-t01, info['funval'][-1], info['gnorm'][-1], str(np.array(indices)+1))
                else:
                    print "Run:%d\t Alg: %s\t Clock: %.2f\t Time: %.2f\t Fv: %.8f\t Gv: %.8f; Ranks %s" % (itRun+1,
                    method_name, t1-t0, t11-t01, info['funval'][-1], info['gnorm'][-1], str(np.array(indices)+1))
                ###################3
                relres = np.linalg.norm(recover(n, cdN, ldN, tdN) - a) / np.linalg.norm(a)
                conder = 0.
                if constraints is not None:
                    for dimnum in constraints['dimnum']:
                        if lFlag and tFlag:
                            if (dimnum < P) and (fmc is not None) and (fmc[dimnum] is not None):
                                Bk = Bkl(ldN, dimnum,full=0)
                                conder += np.linalg.norm(np.dot(tdN['A'][dimnum].T, Bk[0]))**2.
                                if (Bk[2] is not None):
                                    conder += np.linalg.norm(np.dot(tdN['A'][dimnum].T, Bk[2]))**2.
                            else:
                                conder += np.linalg.norm(np.dot(tdN['A'][dimnum].T, ldN['B'][dimnum]))**2.
                        elif lFlag:
                            if (dimnum < P) and (fmc is not None) and (fmc[dimnum] is not None):
                                Bk = Bkl(ldN, dimnum,full=0)
                                Bkg = Bk[2][:, -ldN['L'][-1]:]
                                conder += np.linalg.norm(np.dot(Bkg.T, Bk[0]))**2.
                                if (Bk[2].shape > ldN['L'][-1]):
                                    conder += np.linalg.norm(np.dot(Bkg.T, Bk[2][:, :-ldN['L'][-1]]))**2.
                            else:
                                conder += np.linalg.norm(np.dot(ldN['B'][dimnum][:, -ldN['L'][-1]:].T, ldN['B'][dimnum][:, :-ldN['L'][-1]]))**2.
                    print fnm, relres, conder*0.5
                indexToAssign = [[itRun], [itAlg]] + [[x] for x in list(indices)]
                #indexToAssign = 
                resultTimeClock[indexToAssign] = t1 - t0
                resultTimeTime[indexToAssign] = t11 - t01
                flist = info['funval']
                glist = info['gnorm']
                flist += max(0, 1+maxitnum-len(flist))*[flist[-1]]
                glist += max(0, maxitnum-len(glist))*[glist[-1]]
                #indexToAssign = np.reshape(np.array(indexToAssign), [1, -1])
                #print fnm, indexToAssign, resultFV[indexToAssign].shape
                resultFV[indexToAssign] = np.array(flist)
                resultGV[indexToAssign] = np.array(glist)
                np.savez_compressed(
                    fnm, resultFV=resultFV, resultGV=resultGV, resultTimeClock=resultTimeClock,
                    resultTimeTime=resultTimeTime, itRun=itRun, itAlg=itAlg, canonical_dict=canonical_dict,
                    lro_dict=lro_dict, tucker_dict=tucker_dict, n=n, methods=methods,
                    result_rank_modes=result_rank_modes, indices=indices
                )


def launcher(N_jobs, dirname, n, methods, canonical_dict, lro_dict, tucker_dict, fnm, Nrun,
             maxitnum=100, constraints=None, projector=None):
    jobs = []
    Nrun_per_process = Nrun / N_jobs
    Number_of_hard_workers = Nrun % N_jobs
    for i in xrange(N_jobs):
        Nrun_local = Nrun_per_process
        if i < Number_of_hard_workers:
            Nrun_local += 1
        p = multiprocessing.Process(
            target=ex1_Convergence, args=(
                n, methods, canonical_dict, lro_dict, tucker_dict, dirname+fnm+'_part'+str(i), Nrun_local,
                None, maxitnum, i+1, constraints, projector
            )
        )
        jobs.append(p)
        p.start()
        print "process %d with %d starts" % (i+1, Nrun_local)
    for p in jobs:
        p.join()
    
    print "========================================================"
    print "Launcher has successfully finished his work"
    filename_base = fnm + '_part'
    #def gatherResults(dirname, filename_base):
    fnms = os.listdir(dirname)
    fnms = filter(lambda x: x.startswith(filename_base), fnms)
    fnms = sorted(fnms, key=stringSplitByNumbers)
    resultFV = None
    resultGV = None
    resultTimeTime = None
    resultTimeClock = None
    n = None
    methods_names = None
    for k in xrange(len(fnms)):
        filename = filename_base + str(k) + '.npz'
        df = np.load(dirname+filename)
        if k > 0:
            resultFV = np.concatenate([resultFV, df['resultFV']], axis=0)
            resultGV = np.concatenate([resultGV, df['resultGV']], axis=0)
            resultTimeTime = np.concatenate([resultTimeTime, df['resultTimeTime']], axis=0)
            resultTimeClock = np.concatenate([resultTimeClock, df['resultTimeClock']], axis=0)
        else:
            resultFV = df['resultFV'].copy()
            resultGV = df['resultGV'].copy()
            resultTimeTime = df['resultTimeTime'].copy()
            resultTimeClock = df['resultTimeClock'].copy()
            n = df['n']
            methods = df['methods']
    np.savez_compressed(
        dirname+fnm, resultFV=resultFV, resultGV=resultGV, resultTimeTime=resultTimeTime, resultTimeClock=resultTimeClock,
        n=n, methods=methods
    )

if __name__ == '__main__':
    Nrun = 100
    maxitnum = 20
    maxInnerIt = 15
    tolRes = 1e-8
    tolGrad = 1e-8
    tolSwamp = 1e-8
    verbose = 0
    N_jobs = 8
    dirname = './experimental_1/'
    # set up different algorithms
    def alsM(a, x0, cdN, ldN, tdN, constraints):
        return tcd(
            a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
            tolRes=tolRes, tolGrad=tolGrad, tolSwamp=tolSwamp, method='als', verbose=verbose, 
            regTGD=None, regPGD=None, doSA=0, constraints=constraints
        )
    def gdM(a, x0, cdN, ldN, tdN, constraints):
        return tcd(
            a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
            tolRes=tolRes, tolGrad=tolGrad,tolSwamp=tolSwamp, method='gd', backtrack=True, 
            verbose=verbose, regTGD=None, regPGD=None, doSA=0, constraints=constraints
        )
    def gdrtM(a, x0, cdN, ldN, tdN, constraints):
        return tcd(
            a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
            tolRes=tolRes, tolGrad=tolGrad, tolSwamp=tolSwamp, method='gd', backtrack=True,
            verbose=verbose, regTGD=1e-3, regPGD=None, doSA=0, constraints=constraints
        )
    def gdrpM(a, x0, cdN, ldN, tdN, constraints):
        return tcd(
            a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
            tolRes=tolRes, tolGrad=tolGrad, tolSwamp=tolSwamp, method='gd', backtrack=True, 
            verbose=verbose, regTGD=None, regPGD=1e-3, doSA=0, constraints=constraints
        )
    def cgfrM(a, x0, cdN, ldN, tdN, constraints):
        return tcd(
            a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum,
            tolRes=tolRes, tolGrad=tolGrad, tolSwamp=tolSwamp, method='cg', betaCG='fr', 
            verbose=verbose, regTGD=None, regPGD=None, doSA=0, constraints=constraints
        )
    def cgprM(a, x0, cdN, ldN, tdN, constraints):
        return tcd(
            a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
            tolRes=tolRes, tolGrad=tolGrad, tolSwamp=tolSwamp, method='cg', betaCG='pr', 
            verbose=verbose, regTGD=None, regPGD=None, doSA=0, constraints=constraints
        )
    def cghsM(a, x0, cdN, ldN, tdN, constraints):
        return tcd(
            a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum,
            tolRes=tolRes, tolGrad=tolGrad, tolSwamp=tolSwamp, method='cg', betaCG='hs',
            verbose=verbose, regTGD=None, regPGD=None, doSA=0, constraints=constraints
        )
                                                
    def cgdyM(a, x0, cdN, ldN, tdN, constraints):
        return tcd(
            a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
            tolRes=tolRes, tolGrad=tolGrad, tolSwamp=tolSwamp, method='cg', betaCG='dy', 
            verbose=verbose, regTGD=None, regPGD=None, doSA=0, constraints=constraints
        )
    def gnM(a, x0, cdN, ldN, tdN, constraints):
        return tcd(
            a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
            tolRes=tolRes,tolGrad=tolGrad,tolSwamp=tolSwamp, method='gn', backtrack=True, 
            verbose=verbose, regTGD=None, regPGD=None, doSA=0, constraints=constraints
        )
    def lmqM(a, x0, cdN, ldN, tdN, constraints):
        return tcd(
            a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
            tolRes=tolRes, tolGrad=tolGrad, tolSwamp=tolSwamp, method='lm', epsilonLM=1e-8,
            lmSetup='Quadratic', muInit=1., verbose=verbose, regTGD=None, regPGD=None,
            doSA=0, constraints=constraints
        )
    def lmnM(a, x0, cdN, ldN, tdN, constraints):
        return tcd(
            a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
            tolRes=tolRes, tolGrad=tolGrad, tolSwamp=tolSwamp, method='lm', epsilonLM=1e-8,
            lmSetup='Nielsen', muInit=1., verbose=verbose, regTGD=None, regPGD=None,
            doSA=0, constraints=constraints
        )
    def doglegM(a, x0, cdN, ldN, tdN, constraints):
        return tcd(
            a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
            maxInnerIt=maxInnerIt, tolRes=tolRes, tolGrad=tolGrad, tolSwamp=tolSwamp, method='tr', 
            verbose=verbose, doSA=0, constraints=constraints, trStep='dogleg',
            trDelta0=1.2,trEta=0.23
        )
    def scg_qnM(a, x0, cdN, ldN, tdN, constraints):
        return tcd(
            a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
            maxInnerIt=maxInnerIt, tolRes=tolRes, tolGrad=tolGrad, tolSwamp=tolSwamp, method='tr', 
            verbose=verbose, doSA=0, constraints=constraints, curvature=0, trStep='scg',
            trDelta0=1.2, trEta=0.23
        )
    def scg_fnM(a, x0, cdN, ldN, tdN, constraints):
        return tcd(
            a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
            maxInnerIt=maxInnerIt, tolRes=tolRes, tolGrad=tolGrad, tolSwamp=tolSwamp, method='tr', 
            verbose=verbose, doSA=0, constraints=constraints, curvature=1, trStep='scg',
            trDelta0=1.2, trEta=0.23
        )

    ##############
    methods_names = [
        "ALS", "GD", "CG-FR", "CG-PR", "CG-HS", "CG-DY", "GN", "LM-Q", "LM-N", "TR-DL", "SCG-QN", "SCG-FN"
    ]
    algs = [alsM, gdM, cgfrM, cgprM, cghsM, cgdyM, gnM, lmqM, lmnM, doglegM, scg_qnM, scg_fnM]
    #methods_names = ["ALS", "GN", "LM-Q", "LM-N", "TR-DL", "SCG-QN", "SCG-FN"]
    #algs = [alsM, gnM, lmqM, lmnM, doglegM, scg_qnM, scg_fnM]
    ## parameters ###################################
    d = 3
    n = [20]*d
    constraints = None
    ## experiment 1: convergence with exact and inexact ranks without constraints
    
    NTcores = 1
    r = 3*np.ones([NTcores, d], dtype=np.int)
    Rl = 5
    fset = 3
    #L = range(fset, Rl+fset)
    L = [fset]*Rl
    P = d-1
    lro_dict = {
        'L': L,
        'P': P
    }
    tucker_dict = {
        'r': r
    }
    canonical_dict = None
    fnm = 'experiment1_Tucker+LR1-act.npz'
    #ex1_Convergence(n, algs, canonical_dict, lro_dict, tucker_dict, fnm, Nrun, methods_names=methods_names, overestimated=None)
    #launcher_experiments1(N_jobs, dirname, n, algs, canonical_dict, lro_dict, tucker_dict, fnm, Nrun, methods_names)
    #stop
    print "Tucker+(Lr,1): convergence without constraints"
    #printStats(fnm)
    fnm = 'experiment2_only_LR1-act.npz'
    tucker_dict = None
    #ex1_Convergence(n, algs, canonical_dict, lro_dict, tucker_dict, fnm, Nrun, methods_names=methods_names, overestimated=None)
    #launcher_experiments1(N_jobs, dirname, n, algs, canonical_dict, lro_dict, tucker_dict, fnm, Nrun, methods_names)
    print "Full (Lr,1): convergence without constraints"
    #printStats(fnm)
    fmc = [None]*d
    fmc[0] = [3, 3]
    fmc[1] = [2, 2]
    lro_dict['fullModesConfig'] = fmc
    tucker_dict = {
        'r': r
    }
    fnm = 'experiment1_Tucker+LR1_with_fmc-act.npz'
    #ex1_Convergence(n, algs, canonical_dict, lro_dict, tucker_dict, fnm, Nrun, methods_names=methods_names, overestimated=None)
    #launcher_experiments1(N_jobs, dirname, n, algs, canonical_dict, lro_dict, tucker_dict, fnm, Nrun, methods_names)
    print "Tucker+(Lr,1): convergence without constraints with fmc"
    #printStats(fnm)
    fnm = 'experiment2_only_LR1_with_fmc-act.npz'
    tucker_dict = None
    #ex1_Convergence(n, algs, canonical_dict, lro_dict, tucker_dict, fnm, Nrun, methods_names=methods_names, overestimated=None)
    #launcher_experiments1(N_jobs, dirname, n, algs, canonical_dict, lro_dict, tucker_dict, fnm, Nrun, methods_names)
    print "Full (Lr,1): convergence without constraints with fmc"
    #printStats(fnm)
    #stop
    #############################3
    
    
    ## experiment 1.2: convergence with group constraint
    Nsubj = 5
    NTcores = 1
    d = 3
    n = [20]*(d-1)+[Nsubj]
    r = 3*np.ones([NTcores, d], dtype=np.int)
    r[:, -1] = Nsubj
    Rl = Nsubj
    L = [3]*Rl
    P = d-1

    lro_dict = {
        'L': L,
        'P': P
    }
    tucker_dict = {
        'r': r
    }
    constraints = group_constraint(n, [0], otype='projected')
    #algs = [doglegM, scg_qnM, scg_fnM]
    #algs = [alsM, gdM, gdrtM, gdrpM, cgfrM, cgprM, cghsM, cgdyM, gnM, lmqM, lmnM, doglegM, scg_qnM, scg_fnM]
    fnm = 'ex2_Tucker-LR1_conv_group_projected.npz'
    '''
    launcher_experiments1(
        N_jobs, dirname, n, algs, canonical_dict, lro_dict, tucker_dict, fnm, Nrun,
        methods_names=methods_names, projector=projector, constraints=constraints
    )'''
    print "Tucker+(Lr,1): convergence with proj group constraint"
    #printStats(fnm)
    constraints = group_constraint(n, [0], otype='Lagrange')
    methods_names = [
        "GN", "LM-Q", "LM-N", "TR-DL", "SCG-QN", "SCG-FN"
    ]
    algs = [gnM, lmqM, lmnM, doglegM, scg_qnM, scg_fnM]
    fnm = 'ex2_Tucker-LR1_conv_group_lmm.npz'
    '''
    launcher_experiments1(
        N_jobs, dirname, n, algs, canonical_dict, lro_dict, tucker_dict, fnm, Nrun,
        methods_names=methods_names, projector=projector, constraints=constraints
    )'''
    print "Tucker+(Lr,1): convergence with Lag.mult. group constraint"
    tucker_dict = None
    lro_dict = {
        'L': L + [len(L)],
        'P': P
    }
    constraints = group_constraint(n, [0], otype='projected')
    #algs = [doglegM, scg_qnM, scg_fnM]
    #algs = [alsM, gdM, gdrtM, gdrpM, cgfrM, cgprM, cghsM, cgdyM, gnM, lmqM, lmnM, doglegM, scg_qnM, scg_fnM]
    fnm = 'ex2_full-LR1_conv_group_projected.npz'
    #fnm = 'ex3_A_uconv.npz'
    methods_names = [
        "ALS", "GD", "CG-FR", "CG-PR", "CG-HS", "CG-DY", "GN", "LM-Q", "LM-N", "TR-DL", "SCG-QN", "SCG-FN"
    ]
    algs = [alsM, gdM, cgfrM, cgprM, cghsM, cgdyM, gnM, lmqM, lmnM, doglegM, scg_qnM, scg_fnM]
    '''launcher_experiments1(
        N_jobs, dirname, n, algs, canonical_dict, lro_dict, tucker_dict, fnm, Nrun,
        methods_names=methods_names, projector=projector, constraints=constraints
    )'''
    print "Full (Lr,1): convergence with proj group constraint"
    #printStats(fnm)
    constraints = group_constraint(n, [0], otype='Lagrange')
    methods_names = [
        "GN", "LM-Q", "LM-N", "TR-DL", "SCG-QN", "SCG-FN"
    ]
    algs = [gnM, lmqM, lmnM, doglegM, scg_qnM, scg_fnM]
    #algs = [doglegM, scg_qnM, scg_fnM]
    #algs = [gnM, lmqM, lmnM, doglegM, scg_qnM, scg_fnM]
    fnm = 'ex2_full-LR1_conv_group_lmm.npz'
    '''launcher_experiments1(
        N_jobs, dirname, n, algs, canonical_dict, lro_dict, tucker_dict, fnm, Nrun,
        methods_names=methods_names, projector=projector, constraints=constraints
    )'''
    print "Full (Lr,1): convergence with Lag.mult. group constraint"
    #printStats(fnm)
    
    ################################3 fmc
    
    fmc = [None]*d
    fmc[0] = [3, 3]
    fmc[1] = [2, 2]
    
    tucker_dict = {
        'r': r
    }
    lro_dict = {
        'L': L,
        'P': P
    }
    lro_dict['fullModesConfig'] = fmc
    constraints = group_constraint(n, [0], otype='projected')
    #algs = [doglegM, scg_qnM, scg_fnM]
    #algs = [alsM, gdM, gdrtM, gdrpM, cgfrM, cgprM, cghsM, cgdyM, gnM, lmqM, lmnM, doglegM, scg_qnM, scg_fnM]
    fnm = 'ex2_Tucker-LR1_conv_group_projected_FMC.npz'
    '''
    methods_names = [
        "ALS", "GD", "CG-FR", "CG-PR", "CG-HS", "CG-DY", "GN", "LM-Q", "LM-N", "TR-DL", "SCG-QN", "SCG-FN"
    ]'''
    #algs = [alsM, gdM, cgfrM, cgprM, cghsM, cgdyM, gnM, lmqM, lmnM, doglegM, scg_qnM, scg_fnM]
    methods_names = [
        "ALS", "CG-FR", "CG-DY", "LM-Q", "LM-N", "TR-DL", "SCG-QN", "SCG-FN"
    ]
    algs = [alsM, cgfrM, cgdyM, lmqM, lmnM, doglegM, scg_qnM, scg_fnM]
    launcher_experiments1(
        N_jobs, dirname, n, algs, canonical_dict, lro_dict, tucker_dict, fnm, Nrun,
        methods_names=methods_names, projector=projector, constraints=constraints
    )
    print "Tucker+(Lr,1): convergence with proj group constraint"
    #printStats(fnm)
    constraints = group_constraint(n, [0], otype='Lagrange')
    #algs = [scg_fnM]
    #algs = [gnM, lmqM, lmnM, doglegM, scg_qnM, scg_fnM]
    fnm = 'ex2_Tucker+LR1_conv_group_lmm_FMC.npz'
    methods_names = [
        "LM-Q", "LM-N", "TR-DL", "SCG-QN", "SCG-FN"
    ]
    algs = [lmqM, lmnM, doglegM, scg_qnM, scg_fnM]
    launcher_experiments1(
        N_jobs, dirname, n, algs, canonical_dict, lro_dict, tucker_dict, fnm, Nrun,
        methods_names=methods_names, projector=projector, constraints=constraints
    )
    print "Tucker+(Lr,1): convergence with Lag.mult. group constraint"
    tucker_dict = None
    lro_dict = {
        'L': L + [len(L)],
        'P': P
    }
    lro_dict['fullModesConfig'] = fmc
    constraints = group_constraint(n, [0], otype='projected')
    #algs = [doglegM, scg_qnM, scg_fnM]
    #algs = [alsM, gdM, gdrtM, gdrpM, cgfrM, cgprM, cghsM, cgdyM, gnM, lmqM, lmnM, doglegM, scg_qnM, scg_fnM]
    fnm = 'ex2_full-LR1_conv_group_projected_FMC.npz'
    #fnm = 'ex3_A_uconv.npz'
    '''
    methods_names = [
        "ALS", "GD", "CG-FR", "CG-PR", "CG-HS", "CG-DY", "GN", "LM-Q", "LM-N", "TR-DL", "SCG-QN", "SCG-FN"
    ]'''
    #algs = [alsM, gdM, cgfrM, cgprM, cghsM, cgdyM, gnM, lmqM, lmnM, doglegM, scg_qnM, scg_fnM]
    methods_names = [
        "ALS", "CG-FR", "CG-DY", "LM-Q", "LM-N", "TR-DL", "SCG-QN", "SCG-FN"
    ]
    algs = [alsM, cgfrM, cgdyM, lmqM, lmnM, doglegM, scg_qnM, scg_fnM]
    launcher_experiments1(
        N_jobs, dirname, n, algs, canonical_dict, lro_dict, tucker_dict, fnm, Nrun,
        methods_names=methods_names, projector=projector, constraints=constraints
    )
    print "Full (Lr,1): convergence with proj group constraint"
    #printStats(fnm)
    constraints = group_constraint(n, [0], otype='Lagrange')
    #algs = [doglegM, scg_qnM, scg_fnM]
    #algs = [gnM, lmqM, lmnM, doglegM, scg_qnM, scg_fnM]
    fnm = 'ex2_full-LR1_conv_group_lmm_FMC.npz'
    methods_names = [
        "LM-Q", "LM-N", "TR-DL", "SCG-QN", "SCG-FN"
    ]
    algs = [lmqM, lmnM, doglegM, scg_qnM, scg_fnM]
    launcher_experiments1(
        N_jobs, dirname, n, algs, canonical_dict, lro_dict, tucker_dict, fnm, Nrun,
        methods_names=methods_names, projector=projector, constraints=constraints
    )
    print "Full (Lr,1): convergence with Lag.mult. group constraint"

    
    
    
    
    
    
    
    
    
    
    
    
