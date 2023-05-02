import numpy as np
from scipy.linalg import eigh_tridiagonal as tridiag
from os import path, system

qdir = './quad_grids/'

if not path.isdir(qdir):
    system('mkdir -p ' + qdir)

ceil = lambda x : int(np.ceil(x))
floor = lambda x : int(np.floor(x))

def GLQ(M):
    """
        Gauss-Legendre quadrature of order M
    """

    fname = qdir + '/GLQ_{:}.csv'.format(M)

     # algorithm from Golub and Welsch, Math. Comp. 23, 221 (1969)
    def beta_GL_udiag(n):# coefficients from NIST's DLMF, sec. 18.9
        an = (2*n+1.0)/(n+1.0)
        anp1 = (2*n+3.0)/(n+2.0)
        cnp1 = (n+1.0)/(n+2.0)
        return (cnp1/an/anp1)**(0.5)

    jac_diag = np.zeros(M)
    jac_udiag = np.zeros(M-1)
    jac_udiag = beta_GL_udiag(1.*np.arange(0,M-1,1))

    grid,v = tridiag(jac_diag,jac_udiag)

    wg =  2.0*v[0]**2

    np.savetxt(fname,np.transpose((grid,wg)),delimiter=',',\
        header='grid point, weight',fmt='%.16f')

    return grid, wg

def gauss_kronrod(n):
    # adapted from Dirk P. Laurie,
    # CALCULATION OF GAUSS-KRONROD QUADRATURE RULE
    # Mathematics of Computation 66, 1133 (1997).
    # doi:10.1090/S0025-5718-97-00861-2
    def coeff(n):
        an = (2*n+1.0)/(n+1.0)
        alp = 0.0
        anp1 = (2*n+3.0)/(n+2.0)
        cnp1 = (n+1.0)/(n+2.0)
        return alp,(cnp1/an/anp1)#**(0.5)
    a = np.zeros(2*n+1)
    b = np.zeros(2*n+1)
    b[0]=2.0
    for jn in range(ceil(3*n/2.0)+1):
        if jn < int(3*n/2.0):
            a[jn],b[jn+1] = coeff(jn)
        else:
            _,b[jn+1] = coeff(jn)
    gl_grid,gl_v = tridiag(a[:n],b[1:n]**(0.5))
    gl_wg=2.0*gl_v[0]**2

    t = np.zeros(floor(n/2.0)+2)
    s = np.zeros(floor(n/2.0)+2)

    t[1] = b[n+1]
    for m in range(n-1):
        u = 0.0
        for k in range(floor((m+1.0)/2.0),-1,-1):
            l = m-k
            u += (a[k+n+1] - a[l])*t[k+1] + b[k+n+1]*s[k] - b[l]*s[k+1]
            s[k+1] = u
        ts = s
        s = t
        t = ts
    for j in range(floor(n/2.0),-1,-1):
        s[j+1] = s[j]
    for m in range(n-1,2*n-2):
        u = 0.0
        for k in range(m+1-n,floor((m-1.0)/2.0)+1):
            l = m - k
            j = n - 1 -l
            u += -(a[k+n+1] - a[l])*t[j+1] - b[k+n+1]*s[j+1] + b[l]*s[j+2]
            s[j+1] = u
        if m%2 == 0:
            k = int(m/2)
            a[k+n+1] = a[k] + (s[j+1] - b[k+n+1]*s[j+2])/t[j+2]
        else:
            k = int((m+1)/2)
            b[k+n+1] = s[j+1]/s[j+2]
        ts = s
        s = t
        t = ts
    a[2*n] = a[n-1] - b[2*n]*s[1]/t[1]
    grid,v = tridiag(a,b[1:]**(0.5))#
    wg = b[0]*v[0]**2

    glwg = np.zeros(wg.shape)
    for ipt,pt in enumerate(grid):
        for jp,pp in enumerate(gl_grid):
            if abs(pp-pt)<1.e-12:
                glwg[ipt] = gl_wg[jp]

    np.savetxt(qdir+'GKQ_'+str(2*n+1)+'_pts.csv', \
        np.transpose((grid,wg,wg - glwg)), delimiter=',',\
        header='GK point, GK weight, GK - GL weight',fmt='%.16f')



    return

def switch_elts(arr,i1,i2):
    # switches the i1 and i2 elements of array arr
    tmp = arr[i1].copy()
    arr[i1] = arr[i2].copy()
    arr[i2] = tmp.copy()
    return arr

def GK_GA_PINF(fun,lbd,opt_d={},args=(),kwargs={}):
    # for integrating a function from lbd to infinity
    dopts = {'prec': 1.e-8, 'npts': 3, 'min_recur': 2, 'max_search': 1000}
    for anopt in opt_d:
        dopts[anopt] = opt_d[anopt]

    wfun = lambda x : fun(x,*args,**kwargs)
    if 'breakpoint' in dopts:
        bkpt = dopts['breakpoint']

    else:
        bkpt = max(1.e-3,lbd)
        tfun = wfun(bkpt)
        ofun = tfun
        tscp = 1.5
        for isrch in range(dopts['max_search']):

            bkpt *= tscp
            cfun = wfun(bkpt)
            if abs(cfun/tfun) < dopts['prec'] or \
                abs(cfun - ofun)/abs(cfun + ofun) < dopts['prec']:
                break
            ofun = cfun
            #tfun = cfun

    dopts['prec'] /= 2.
    igrl1, msg1 = GK_global_adap(wfun,(lbd,bkpt),opt_d=dopts)

    wifun = lambda x : fun(1./x, *args, **kwargs)/x**2
    igrl2, msg2 = GK_global_adap(wifun,(1./(2.*bkpt),1./bkpt),opt_d=dopts)
    igrl3, msg3 = GK_global_adap(wifun,(1./(4.*bkpt),1./bkpt),opt_d=dopts)
    #igrl4, msg4 = GK_global_adap(wifun,(dopts['prec']/10.,1./bkpt),opt_d=dopts)
    slope = 4*bkpt*(igrl2 - igrl3)
    icpt = igrl2 - slope/(2.*bkpt)
    if abs(icpt/max(1.e-12,igrl1)) > 1.e2:
        print('GK_GA_PINF FATAL!! Extrapolated improper integral much larger than lower range')
        print(igrl1,args,abs(icpt/max(1.e-12,igrl1)))
        exit()

    od = {'code_cut': msg1['code'], 'code_upper_2cut': msg2['code'],
        'code_upper_4cut': msg3['code'],
        'error_lower': msg1['error'], 'error_upper_2cut': msg2['error'],
        'error_upper_4cut': msg3['error'],
        'integral_lower': igrl1, 'integral_upper_2cut': igrl2,
        'integral_upper_4cut': igrl3,
        'extrap_integral_upper': icpt, 'extrap_slope': slope
    }

    return igrl1 + icpt, od


def GK_global_adap(fun,bds,opt_d={},args=(),kwargs={}):

    """
        error codes:
        > 0   successful integration
            1    absolutely no issues
        <= 0   unsucessful integration:
            0   exceeded maximum number of steps
           -1   NaN error (errors are NaN)
           -2   Bisection yielded regions smaller than machine precision
           -3   Result was below machine precision, estimating as zero
    """
    meps = abs(7/3-4/3-1) # machine precision

    wrapfun = lambda x : fun(x,*args,**kwargs)

    lbd,ubd = bds

    def_pts = 5
    prec = 1.0e-8
    if 'prec' in opt_d:
        prec = opt_d['prec']
    min_recur = 2
    if 'min_recur' in opt_d:
        min_recur = opt_d['min_recur']

    if 'max_recur' in opt_d:
        max_recur = opt_d['max_recur']
    else:
        # 2**max_recur bisections yields a region of width 10**(-60)
        max_recur = ceil((np.log(abs(bds[1]-bds[0])) + 60*np.log(10.0))/np.log(2.0))
        max_recur = max(max_recur,1000)

    if 'npts' in opt_d:
        npts = opt_d['npts']
    else:
        npts = def_pts

    if 'error monitoring' not in opt_d:
        opt_d['error monitoring'] = False
    if 'err_meas' not in opt_d:
        opt_d['err_meas'] = 'abs_diff'

    if 'rel_tol' in opt_d:
        rel_tol = opt_d['rel_tol']
    else:
        rel_tol = min(0.01,100*prec)

    def_grid = qdir + '/GKQ_'+str(2*npts+1)+'_pts.csv'
    if not path.isfile(def_grid) or path.getsize(def_grid)==0:
        gauss_kronrod(npts) # returns 2*N + 1 points
    mesh,wg,wg_err = np.transpose(np.genfromtxt(def_grid,delimiter=',',skip_header=1))

    if 'reg' in opt_d:
        working_regs = []
        for iareg,areg in enumerate(opt_d['reg']):
            if iareg == 0:
                working_regs.append([lbd,areg[1]])
            elif iareg == len(opt_d['reg'])-1:
                working_regs.append([areg[0],ubd])
            else:
                working_regs.append(areg)
    else:
        treg = np.linspace(lbd,ubd,min_recur+1)
        working_regs = []
        for ibord in range(len(treg)-1):
            working_regs.append([treg[ibord],treg[ibord+1]])

    reg_l = np.zeros((2*max_recur+1,2))
    err_l = np.zeros(2*max_recur+1)
    sum_l = np.zeros(2*max_recur+1)

    ipos = -1
    for irecur in range(max_recur):

        for ireg, areg in enumerate(working_regs):

            ipos += 1

            x_mesh = 0.5*(areg[1]-areg[0])*mesh + 0.5*(areg[1]+areg[0])
            x_wg = 0.5*(areg[1]-areg[0])*wg
            x_wg_err = 0.5*(areg[1]-areg[0])*wg_err
            tvar = wrapfun(x_mesh)

            tint = np.sum(x_wg*tvar)
            tint_err = abs(np.sum(x_wg_err*tvar))

            reg_l[ipos] = areg.copy() #np.vstack((reg_l,areg))
            sum_l[ipos] = tint#np.append(sum_l,tint)

            if opt_d['err_meas']=='quadpack':
                """
                empirical error measure from:
                R. Piessens, E. de Doncker-Kapenga,  C. W. Uberhuber, and D. K. Kahaner
                ``QUADPACK: A Subroutine Package for Automatic Integration''
                Springer-Verlag, Berlin, 1983.
                doi: 10.1007/978-3-642-61786-7
                """
                fac = np.sum(x_wg*np.abs(tvar - tint/(areg[1]-areg[0])))
                gk_err = tint_err
                if fac == 0.0:
                    cloc_err = 0.0
                else:
                    cloc_err = fac*min(1.0,(200*gk_err/fac)**(1.5))
                #err_l[irecur+ireg] np.append(err_l,lerr_meas)
            elif opt_d['err_meas']=='abs_diff' or opt_d['err_meas']=='global_rel':
                #err_l = np.append(err_l,tint_err)
                cloc_err = tint_err
            elif opt_d['err_meas']=='local_rel':
                #err_l = np.append(err_l,tint_err/max(meps,abs(tint)))
                cloc_err = tint_err/max(meps,abs(tint))

            err_l[ipos] = cloc_err

        csum = np.sum(sum_l[:ipos+1])
        cprec = max(meps,min(prec,abs(csum)/2))

        if opt_d['err_meas']=='global_rel':
            global_error = np.sum(err_l)/max(meps,csum)
        else:
            global_error = np.sum(err_l)

        if opt_d['error monitoring']:
            print(global_error,csum)

        if abs(csum)< meps:
            return 0.0,{'code':-3,'error':global_error}

        if global_error != global_error: # if the output is NaN, completely failed
            return csum,{'code':-1,'error':global_error}

        if global_error < cprec: # SUCCESS!!!!
            return csum,{'code':1,'error':global_error}
        else:
            #inds = np.argsort(err_l)
            ibad = np.argmax(err_l)
            bad_reg = reg_l[ibad].copy()
            bad_err = err_l[ibad].copy()

            err_l = switch_elts(err_l,ipos,ibad)
            reg_l = switch_elts(reg_l,ipos,ibad)
            sum_l = switch_elts(sum_l,ipos,ibad)
            ipos -= 1
            #err_l = err_l[inds][:-1]
            #reg_l = reg_l[inds][:-1]
            #sum_l = sum_l[inds][:-1]

            mid = (bad_reg[0] + bad_reg[1])/2.0 # recursive bisection of highest error region
            if abs(bad_reg[1]-bad_reg[0])< meps or abs(bad_reg[1]-mid)< meps \
                or abs(bad_reg[0]-mid)< meps:
                # bisection yields differences below machine precision, integration failed
                return csum,{'code':-2,'error':global_error}
            working_regs = [[bad_reg[0],mid],[mid,bad_reg[1]]]

    if irecur == max_recur-1:
        if abs(csum)<meps:
            return 0.0,{'code':-3,'error':global_error}
        else:
            return csum,{'code':0,'error':global_error}


if __name__ == "__main__":

    tfun = lambda x : np.exp(-x**2)

    igrl, msg = GK_GA_PINF(tfun,0.,opt_d={},args=(),kwargs={})

    #igrl,msg = GK_global_adap(tfun,(-1.,.7),opt_d={},args=(),kwargs={})
    print(igrl,igrl - (np.pi)**(0.5)/2.)
    print(msg)
