import numpy as np
from os import path,system

from AKCK_LFF.fitted_LFF import g_plus_new, g_plus_dlda
from AKCK_LFF.PW92 import ec_pw92
from AKCK_LFF.g_corradini import g_corradini
from AKCK_LFF.ra_lff import g_plus_ra
from AKCK_LFF.rMCP07 import g_rMCP07, GKI_im_freq
from AKCK_LFF.gauss_quad import GK_global_adap, GK_GA_PINF, qdir, GLQ
from AKCK_LFF.asymptotics import pade_g_plus

pi = np.pi
kf_to_rs = (9.*pi/4.)**(1./3.)
eH_to_eV = 27.211386245988

gopts = {'prec': 1.e-6, 'min_recur': 4, 'err_meas': 'quadpack', 'npts': 3}

ecdir = './ec_data/'
if not path.isdir(ecdir):
    system('mkdir -p ' + ecdir)

def freqcut(rs):
    c = [1.227277, 5.991171, 0.283892, 0.379981]
    bkpt = 40.
    if rs <= bkpt:
        ff = c[0] + c[1]*rs**c[2]
    else:
        ff = c[0] + c[1]*bkpt**c[2] + (rs - bkpt)**c[3]

    return ff

def qcut(rs):
    c = [3.928319, 0.540168, 0.042225, 0.001810, 2.501585]

    bkpt1 = 5.
    bkpt2 = 60.
    if rs <= bkpt1:
        f = c[0] + c[1]*rs
    elif rs <= bkpt2:
        f = c[0] + bkpt1*c[1] + c[2]*(rs - bkpt1) + c[3]*(rs - bkpt1)**2
    else:
        f = c[0] + bkpt1*c[1] + (bkpt2 - bkpt1)*(c[2] + (bkpt2 - bkpt1)*c[3]) \
            + c[4]*(rs - bkpt2)

    return f

def gen_grid(rs,Nq=100,Nlam=100,Nw=100):

    Nq += Nq%2
    Nw += Nw%2

    qfl = qdir + '/GLQ_{:}.csv'.format(Nq)
    if path.isfile(qfl):
        tq, twgq = np.transpose(np.genfromtxt(qfl,delimiter=',',skip_header=1))
    else:
        tq, twgq = GLQ(Nq)

    ql = np.zeros(Nq + Nq//2)
    qwg = np.zeros(Nq + Nq//2)

    qc = qcut(rs)
    ql[:Nq] = 0.5*qc*(tq + 1.)
    qwg[:Nq] = 0.5*qc*twgq

    qfl = qdir + '/GLQ_{:}.csv'.format(Nq//2)
    if path.isfile(qfl):
        tq, twgq = np.transpose(np.genfromtxt(qfl,delimiter=',',skip_header=1))
    else:
        tq, twgq = GLQ(Nq//2)

    tq = 0.5*(tq + 1.)/qc
    twgq *= 0.5/qc
    ql[Nq:] = 1./tq
    qwg[Nq:] = twgq/tq**2

    lfl = qdir + '/GLQ_{:}.csv'.format(Nlam)
    if path.isfile(lfl):
        tlam, twgl = np.transpose(np.genfromtxt(lfl,delimiter=',',skip_header=1))
    else:
        tlam, twgl = GLQ(Nlam)

    lam = 0.5*(1. + tlam)
    lamwg = 0.5*twgl

    wfl = qdir + '/GLQ_{:}.csv'.format(Nw)
    if path.isfile(wfl):
        tw, twgw = np.transpose(np.genfromtxt(wfl,delimiter=',',skip_header=1))
    else:
        tw, twgw = GLQ(Nw)

    wl = np.zeros(Nw + Nw//2)
    wwg = np.zeros(Nw + Nw//2)

    uc = freqcut(rs)
    wl[:Nw] = 0.5*uc*(tw + 1.)
    wwg[:Nw] = 0.5*uc*twgw

    wfl = qdir + '/GLQ_{:}.csv'.format(Nw//2)
    if path.isfile(wfl):
        tw, twgw = np.transpose(np.genfromtxt(wfl,delimiter=',',skip_header=1))
    else:
        tw, twgw = GLQ(Nw//2)

    tw = 0.5*(tw + 1.)
    twgw *= 0.5
    wl[Nw:] = uc - np.log(1. - tw)
    wwg[Nw:] = twgw/(1. - tw)

    grid = np.zeros((ql.shape[0]*lam.shape[0]*wl.shape[0],4))
    ipt = 0
    for iq in range(ql.shape[0]):
        for ilam in range(lam.shape[0]):
            for iw in range(wl.shape[0]):
                grid[ipt,0] = ql[iq]
                grid[ipt,1] = lam[ilam]
                grid[ipt,2] = wl[iw]
                grid[ipt,3] = qwg[iq]*lamwg[ilam]*wwg[iw]
                ipt += 1

    return grid

def ec_from_chi(rs,grid,fxc='RPA'):

    kf = kf_to_rs/rs

    q = kf*grid[:,0]
    qscl = q/grid[:,1]

    vcoul_scl = 4*pi*grid[:,1]/q**2

    rs_scl = rs*grid[:,1]
    if fxc == 'RPA':
        gplus = 0.
    elif fxc == 'NEW':
        gplus = g_plus_new(qscl,rs_scl)
    elif fxc == 'COR':
        gplus = g_corradini(qscl,\
            {'rs': rs_scl, 'rsh': rs_scl**(0.5),'kF': kf_to_rs/rs_scl, \
            'n': 3./(4.*pi*rs_scl**3)})

    fxch = vcoul_scl*(1. - gplus)

    chi0 = chi0_im_freq(kf,grid[:,0]/2.,grid[:,2]/grid[:,0])

    itgrd = chi0**2*fxch/(1. - chi0*fxch)
    ec = -3.*np.dot(grid[:,3],itgrd)

    return ec

def chi0_im_freq(kf,z,wt):

    # z = q / (2 * kf)
    # wt = Im (omega) / kf**2 * (kf / q)

    log_fac = np.log( (wt**2 + (z + 1.)**2)/(wt**2 + (z - 1.)**2) )
    chi0 = 1./(2.*pi**2) * ( (z**2 - wt**2 - 1.)/(4.*z) *log_fac  - 1. \
        + wt*np.arctan((1. + z)/wt) + wt*np.arctan((1. - z)/wt) )

    return kf*chi0

def regularize(arr,thresh):
    msk = np.abs(arr) < thresh
    arr[msk] = thresh*np.sign(arr[msk])
    arr[arr==0.] = thresh
    return arr

def ec_freq_integrand(u,x,lam,rs,fxc='RPA'):

    kf = kf_to_rs/rs

    q = kf*x
    qscl = q/lam

    vcoul_scl = 4*pi*lam/q**2

    rs_scl = rs*lam

    dv = {'rs': rs_scl, 'rsh': rs_scl**(0.5),'kF': kf_to_rs/rs_scl, \
    'n': 3./(4.*pi*rs_scl**3)}
    if fxc == 'RPA':
        gplus = 0.
    elif fxc == 'NEW':
        gplus = g_plus_new(qscl,rs_scl)
    elif fxc == 'COR':
        gplus = g_corradini(qscl,dv)
    elif fxc == 'RAS':
        gplus = g_plus_ra(qscl,0.,rs_scl)
    elif fxc == 'RAD':
        gplus = g_plus_ra(qscl,u*(kf/lam)**2,rs_scl)
    elif fxc == 'rMCP07':
        gplus = g_rMCP07(qscl,u*(kf/lam)**2,dv)
    elif fxc == 'NEWD':
        gplus = g_plus_dlda(qscl,u*(kf/lam)**2,rs_scl)
    elif fxc == 'PADE':
        gplus = pade_g_plus(qscl,rs_scl)

    fxch = vcoul_scl*(1. - gplus)

    chi0 = chi0_im_freq(kf,x/2.,u/x)

    itgrd = np.zeros(u.shape)
    #tmsk = (u > 1.e-3) & (x > 1.e-3)
    #itgrd[tmsk] = chi0[tmsk]**2*fxch/(1. - chi0[tmsk]*fxch)
    iden = regularize(1. - chi0*fxch,1.e-18)
    if hasattr(x,'__len__'):
        msk = x > 1.e-3
        itgrd[msk] = chi0[msk]**2*fxch[msk]/iden[msk]
    else:
        if x > 1.e-3:
            itgrd = chi0**2*fxch/iden

    return itgrd

def wvvctr_integrand(tx,lam,rs,fxc='RPA', uc=4., qc=2.):

    topts = gopts.copy()
    topts['prec'] = 1.e-7
    topts['breakpoint'] = freqcut(rs)
    i1l = np.zeros(tx.shape)
    for j in range(tx.shape[0]):
        i1l[j], msg = GK_global_adap(ec_freq_integrand, (0.,uc), opt_d=topts, \
            args=(tx[j],lam,rs), kwargs={'fxc':fxc})
        #i1l[j], msg = GK_GA_PINF(ec_freq_integrand,0., opt_d=topts, \
        #    args=(tx[j],lam,rs), kwargs={'fxc':fxc})
    return i1l

def lam_integrand(lam,rs,fxc='RPA',uc = 4., qc = 2.):
    topts = gopts.copy()
    #topts['prec'] = 1.e-7
    topts['breakpoint'] = qcut(rs)

    i2l = np.zeros(lam.shape)
    for j in range(lam.shape[0]):
        i2l[j], msg = GK_global_adap(wvvctr_integrand, (0.,qc), opt_d=gopts, \
            args=(lam[j],rs), kwargs={'fxc':fxc, 'uc': uc, 'qc': qc})
        #i2l[j], msg = GK_GA_PINF(wvvctr_integrand,0., opt_d=topts, \
        #    args=(lam[j],rs), kwargs={'fxc':fxc, 'uc': uc, 'qc': qc})
    return i2l

"""
def wvvctr_integrand(tx,lam,rs,fxc='RPA',uc = 4., qc = 2.):

    i1_a, msg_a = GK_global_adap(ec_freq_integrand, (0.,.5), opt_d=gopts, \
        args=(tx,lam,rs), kwargs={'fxc':fxc, 'uc': uc, 'qc': qc})

    #ifrec_inv = lambda v : ec_freq_integrand(1./v,tx,lam,rs,fxc=fxc)/v**2
    i1_b, msg_b = GK_global_adap(ec_freq_integrand, (.5,1.), opt_d=gopts, \
        args=(tx,lam,rs), kwargs={'fxc':fxc, 'uc': uc, 'qc': qc})

    return i1_a + i1_b

def lam_integrand(lam,rs,fxc='RPA',uc = 6., qc = 10.):
    i2_a, msg_a = GK_global_adap(wvvctr_integrand, (0.,.5), opt_d=gopts, \
        args=(lam,rs), kwargs={'fxc':fxc, 'uc': uc, 'qc': qc})

    #iwv_inv = lambda p : wvvctr_integrand(1./p,lam,rs,fxc=fxc)/p**2
    #i2_b, msg_b = GK_global_adap(iwv_inv,(0.,1./qc),opt_d=gopts)
    i2_b, msg_b = GK_global_adap(wvvctr_integrand, (.5,1.), opt_d=gopts, \
        args=(lam,rs), kwargs={'fxc':fxc, 'uc': uc, 'qc': qc})

    return i2_a + i2_b
#"""

def get_ec(rs,fxc='RPA'):
    tgrid = gen_grid(rs)
    ecden = ec_freq_integrand(tgrid[:,2],tgrid[:,0],tgrid[:,1],rs,fxc=fxc)
    return -3.*np.dot(tgrid[:,3],ecden)

def get_ec_GK(rs,fxc='RPA',uc = 1., qc = 2.5):

    topts = gopts.copy()
    topts['prec'] = 1.e-7
    i3,msg = GK_global_adap(lam_integrand,(0.,1.),opt_d=topts, \
        args =(rs,), kwargs={'fxc':fxc, 'uc': uc, 'qc': qc})
    if msg['code'] < 1 or np.isnan(i3) or np.isinf(i3):
        td = gopts.copy()
        for tprec in [1.e-7,1.e-6,1.e-5]:
            td['prec'] = tprec
            i3,msg = GK_global_adap(lam_integrand,(0.,1.),opt_d=td, \
                args =(rs,), kwargs={'fxc':fxc, 'uc': uc, 'qc': qc})
            if msg['code'] == 1 and not np.isnan(i3) and not np.isinf(i3):
                break
    return -3.*i3

def ec_rpa_unp(rs):
    # J. P. Perdew and Y. Wang, PRB 45, 13244 (1992).
    # doi: 10.1103/PhysRevB.45.13244
    def g(v,rs):
        q0 = -2.0*v[0]*(1.0 + v[1]*rs)
        q1 = 2.0*v[0]*(v[2]*rs**(0.5) + v[3]*rs + v[4]*rs**(1.5) + v[5]*rs**(1.75))
        return q0*np.log(1.0 + 1.0/q1)
    return g([0.031091,0.082477,5.1486,1.6483,0.23647,0.20614],rs)

def gen_dat_files(gpl):

    rs_min = 1.e-1
    rs_max = 120.
    Nrs = 100
    rsl = np.exp(np.linspace(np.log(rs_min),np.log(rs_max),Nrs))

    for gpstr in gpl:
        ec = np.zeros(Nrs)

        for irs, rs in enumerate(rsl):
            ec[irs] = get_ec(rs,fxc=gpstr)
            #print(rs,ec[irs])

        np.savetxt(ecdir + '/eps_c_{:}.csv'.format(gpstr), \
            np.transpose((rsl,ec)), delimiter=',', header='rs, eps_c')

    return

def plots(gpl):

    import matplotlib.pyplot as plt
    from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

    plt.rcParams.update({'text.usetex': True, 'font.family': 'dejavu'})

    label_d = {
        'RPA': 'RPA', 'NEW': 'This work', 'COR': 'Corradini $\mathrm{\it et \, al.}$',
        'RAS': 'RA, static', 'RAD': 'RA', 'rMCP07': 'rMCP07',
        'PADE': 'Padé', 'NEWD': 'NEWD'}
    color_d = {
        'RPA': 'darkorange', 'NEW': 'darkblue', 'COR': 'darkslategray',
        'RAS': 'darkred', 'RAD': 'darkred', 'rMCP07': 'tab:orange', 'PADE': 'teal',  'NEWD': 'cyan'}

    style_d = {'RPA': '--', 'NEW': '-', 'COR': '-.',
        'RAS': '--', 'RAD': ':', 'rMCP07': '-', 'PADE': '--', 'NEWD': '-'}

    lim_d = {'NEW': 33. , 'RAS': 38., 'RAD': 38.}

    missing_dat = []
    for gp in gpl:
        if not path.isfile(ecdir + '/eps_c_{:}.csv'.format(gp)):
            missing_dat.append(gp)
    gen_dat_files(missing_dat)

    fig, ax = plt.subplots(figsize=(6,4))

    xbds = [1e20,-1e20]
    ybds = [1e20,-1e20]
    for gp in gpl:
        tdat = np.genfromtxt(ecdir + '/eps_c_{:}.csv'.format(gp), \
            delimiter=',', skip_header=1)
        tdat2 = []
        for i in range(tdat.shape[0]):
            if np.isinf(tdat[i,1]) or np.isnan(tdat[i,1]):
                continue
            tdat2.append([tdat[i,0],tdat[i,1]])
        tdat = np.array(tdat2).copy()

        pw92, _, _, _ = ec_pw92(tdat[:,0],0.)
        if gp in lim_d:
            msk = tdat[:,0] <= lim_d[gp]
        else:
            msk = np.ones(tdat.shape[0],dtype=bool)
        #tfun = (tdat[msk,1]-pw92[msk])*eH_to_eV
        tfun = 100*(1. - tdat[msk,1]/pw92[msk])
        ax.plot(tdat[msk,0],tfun,color=color_d[gp],label=label_d[gp],\
            linestyle=style_d[gp])
        xbds = [min(xbds[0],tdat[:,0].min()),max(xbds[1],tdat[:,0].max() )]
        ybds = [min(ybds[0],tfun.min()),max(ybds[1],tfun.max() )]

    #rsl = np.linspace(xbds[0],xbds[1],2000)
    #pw92, _, _, _ = ec_pw92(rsl,0.)
    #ax.plot(rsl,pw92,color='k')
    #ax.plot(rsl,ec_rpa_unp(rsl),color='tab:green',linestyle=':')
    #ax.set_ylim(1.02*ybds[0],1.02*ybds[1])
    ax.set_ylim(-10.,20.)
    ax.set_xlim(*xbds)
    ax.hlines(0.,*xbds,color='k',linestyle=':',linewidth=1)
    ax.set_xscale('log')
    ax.set_xlabel('$r_\\mathrm{s}$ (bohr)',fontsize=14)
    #ax.set_ylabel(r'$\varepsilon_\mathrm{c}(r_\mathrm{s},0) - \varepsilon_\mathrm{c}^\mathrm{PW92}(r_\mathrm{s},0)$ (eV/elec.)',fontsize=12)
    ax.set_ylabel(r'$\varepsilon_\mathrm{c}(r_\mathrm{s},0)$ PD (\%)',fontsize=12)

    ax.yaxis.set_minor_locator(MultipleLocator(1.))
    ax.yaxis.set_major_locator(MultipleLocator(5.))

    ax.legend(fontsize=12,frameon=False,ncol=1)

    #plt.show();exit()
    plt.savefig('./ec_data/eps_c_err.pdf',dpi=600,bbox_inches='tight')

    return

def RPA_sanity_check():
    rsl = [0.1, 0.5, 1.,2.,3.,4.,5.,10.,20.,40.,60.,80.,100.,120.]
    tstr = r'$\rs$ & $\varepsilon\suc^\mathrm{RPA}(\rs)$ & $\varepsilon\suc^\mathrm{PW-RPA}(\rs)$ & Percent Deviation (\%) \\ \hline' + ' \n'
    print('  rs      eps_c' + ' '*11 + 'PE from PW92 (%)')
    for rs in rsl:
        ec_rpa = get_ec(rs,fxc='RPA')
        ec_pw_rpa = ec_rpa_unp(rs)
        tpe = 100*(1. - ec_rpa/ec_pw_rpa)
        print('{:}       {:.6f}      {:.2f}'.format(rs,ec_rpa,tpe))
        tstr += '{:} & {:.6f} & {:.6f} & {:.2f} \\\\ \n'.format(rs,ec_rpa,ec_pw_rpa,tpe )
    with open('./ec_data/rpa_sanity.tex','w+') as tfl:
        tfl.write(tstr)

    return

if __name__ == "__main__":


    #RPA_sanity_check() ; exit()

    plots(['NEW','COR','RAS','RAD','rMCP07'])#['RPA','NEW','COR','RAS','RAD'])
    exit()
    #"""

    for rs in [1.,2.,3.,4.,5.,10.,100.]:
        ec_rpa = ec_from_chi(rs,pts)
        print(rs,ec_rpa,100*(1. - ec_rpa/ec_rpa_unp(rs)) )
    exit()

    w = [1.e-6,0.5,1.,5.]
    ql = np.linspace(1.e-6,5.,5000)

    rs = 1.
    kf = kf_to_rs/rs

    for aw in w:
        plt.plot(ql,chi0_im_freq(kf,ql,aw))
    plt.show()
