import numpy as np

def trap(fun,bds,opt_d={},args=(),kwargs={}):

    prec=1.e-10
    if 'prec' in opt_d:
        prec = opt_d['prec']
    simpson=False
    if 'simpson' in opt_d:
        simpson = opt_d['simpson']
    h = (bds[1]-bds[0])/2.
    max_depth = 40 # minimum step size is 2**(max_depth+1)
    min_depth = 2
    prev_h_int = -1e20
    otsum = -1e20

    wrapfun = lambda x : fun(x,*args,**kwargs)
    tsum = 0.5*h*np.sum(wrapfun(np.asarray([bds[0],bds[1]])))

    for iter in range(max_depth):

        m_l = np.arange(bds[0]+h,bds[1],2*h)
        tsum += h*np.sum(wrapfun(m_l))

        if simpson:
            ttsum = tsum
            tsum = (4*ttsum - otsum)/3.0

        if abs(prev_h_int - tsum) < prec*abs(prev_h_int) and iter > min_depth-1:
            return tsum, {'code':1,'error': abs(prev_h_int - tsum) ,'step':h}
        else:
            l_err = abs(prev_h_int - tsum)
            prev_h_int = tsum
            if simpson:
                otsum = ttsum
                tsum = ttsum/2.0
            else:
                tsum /= 2.0 # reuse previous integrated value
            h/=2.0 # halve the step size

    if iter==max_depth:
        return tsum, {'code':0,'error': l_err, 'step': h }

def r(u):
    return 1. - u*( np.sign(u)*np.pi/2. - np.arctan(u) )

def g(u):
    return -1./(9.*(1. + u**2)**2)

def num_integrand(u):
    return r(u)*g(u)*np.log(r(u))

def den_integrand(u):
    return r(u)*g(u)

def find_cut():

    ul = np.linspace(0.,10.,1000)
    i1 = num_integrand(ul)
    i1max = np.abs(i1).max()
    for iu in range(1,ul.shape[0]):
        if abs(i1[iu]/i1max) < 1.e-3:
            i1_cut = ul[iu]
            break

    i2 = den_integrand(ul)
    i2max = np.abs(i2).max()
    for iu in range(1,ul.shape[0]):
        if abs(i2[iu]/i2max) < 1.e-3:
            #print(ul[iu],i2[iu])
            i2_cut = ul[iu]
            break

    return i1_cut, i2_cut

def integrate_funs():

    i1_cut, i2_cut = find_cut()

    prec = 1.e-10

    oldval = 1.e20
    for ibd in range(1,50):
        i1val, msg = trap(num_integrand,(0.,ibd*i1_cut),{'prec': prec/10.})
        if abs(i1val - oldval) < prec*abs(oldval):
            print('DONE numerator\n',i1val,ibd,i1_cut)
            print(msg)
            break
        oldval = i1val

    oldval = 1.e20
    for ibd in range(1,50):
        i2val, msg = trap(den_integrand,(0.,ibd*i2_cut),{'prec': prec/10.})
        if abs(i2val - oldval) < prec*abs(oldval):
            print('DONE denominator\n',i2val,ibd,i2_cut)
            print(msg)
            break
        oldval = i2val

    ravg = i1val / i2val

    return ravg

if __name__ == "__main__":

    ravg = integrate_funs()
    print(ravg)
