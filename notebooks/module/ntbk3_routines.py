import numpy as np

def crosscorrelation_model(Va,Vb,sA,sB,Vab, loc, xx):
    dist = np.abs(xx - xx[loc])
    dist = np.minimum(dist, 0.5- (dist -0.5))
    output = 0.5 * (Vab[loc]/(Va[loc]*Vb[loc])**.5 + Vab/(Va*Vb)**.5) * \
            np.exp(-dist**2/( 0.5*(sA[loc]+sA +sB[loc]+sB) ))
    return output

def crosscovariance_model(Va,Vb,sA,sB,Vab, loc, x):
    dist = np.abs(xx - xx[loc])
    dist = np.minimum(dist, 0.5- (dist -0.5))
    output = 0.5 * (Vab[loc]/(Va[loc]*Vb[loc])**.5 + Vab/(Va*Vb)**.5) * \
            np.exp(-dist**2/( 0.5*(sA[loc]+sA+sB[loc]+sB) )) * \
            np.sqrt(Va[loc]*Vb)
    return output
    
def get_pkf_parameters(diag_ensemble):
    a = diag_ensemble['Mean concentration']['A'] 
    b = diag_ensemble['Mean concentration']['B']
    va = diag_ensemble['Std']['A']**2 
    vb = diag_ensemble['Std']['B']**2
    vab = diag_ensemble['Correlation']['A']['B']* (va*vb)**.5
    sa = diag_ensemble['Length-scale']['A']**2
    sb = diag_ensemble['Length-scale']['B']**2
    return np.array([a,b,va,vb,sa,sb,vab])
