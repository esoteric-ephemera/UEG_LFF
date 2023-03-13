import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from scipy.optimize import least_squares, bisect
from itertools import product

from PZ81 import chi_enh_pz81
from PW92 import ec_pw92

from QMC_data import get_ck_chi_enh, get_HM_QMC_dat, get_AD_DMC_dat, get_CA_ec

pi = np.pi

"""
    Eqs. 4.9 - 4.10 of
    S.H. Vosko, L. Wilk, and M. Nusair, Can. J. Phys. 58, 1200 (1980);
    doi: 10.1139/p80-159
"""
kf_to_rs = (9.*pi/4.)**(1./3.)


def spinf(z,pow):
    opz = np.minimum(2,np.maximum(0.0,1+z))
    omz = np.minimum(2,np.maximum(0.0,1-z))
    return (opz**pow + omz**pow)/2.0

def ts(rs,z):
    # kinetic energy per electron
    ts0 = 3./10.*(kf_to_rs/rs)**2
    ds = spinf(z,5./3.)
    return ts0*ds

def epsx(rs,z):
    # exchange energy per electron
    ex0 = -3./(4.*pi)*kf_to_rs/rs
    dx = spinf(z,4./3.)
    return ex0*dx

def gPW92(rs,v):
    q0 = -2.0*v[0]*(1.0 + v[1]*rs)
    rsh = rs**(0.5)
    q1 = 2.0*v[0]*( rsh* (v[2] + rsh*( v[3] + rsh*( v[4] + rsh*v[5]))) )
    return q0*np.log(1.0 + 1.0/q1)

def dgPW92(rs,v):
    q0 = -2.0*v[0]*(1.0 + v[1]*rs)
    q0p = -2.0*v[0]*v[1]

    rsh = rs**(0.5)
    q1 = 2.0*v[0]*( rsh* (v[2] + rsh*( v[3] + rsh*( v[4] + rsh*v[5]))) )
    q1p = v[0]*( v[2]/rsh + 2.*v[3] + rsh*( 3.*v[4] + 4.*rsh*v[5] ) )

    dg = q0p*np.log(1. + 1./q1) - q0*q1p/(q1*(1. + q1))
    return dg

def get_PW92_pars(cfit,var):

    if var == 'unp':
        c0 = (1. - np.log(2.))/pi**2
        c1 = 0.046644

    elif var == 'pol':
        c0 = (1. - np.log(2.))/(2.*pi**2)
        c1 = 0.025599

    elif var == 'alp':
        c0 = 1./(6.*pi**2)
        PT_integral = 0.5315045266#0.531504
        c1 = (np.log(16.*pi*kf_to_rs) - 3. + PT_integral )/(6.*pi**2)

    ps = np.zeros(6)
    ps[0] = c0
    ps[1] = cfit[0]
    ps[2] = np.exp(-c1/(2.*c0))/(2.*c0)
    ps[3] = 2.*ps[0]*ps[2]**2
    ps[4] = cfit[1]
    ps[5] = cfit[2]

    return ps

def epsc_PW92_rev(rs,z,ps):

    unp_pars = get_PW92_pars(ps[:3],'unp')
    pol_pars = get_PW92_pars(ps[3:6],'pol')
    mac_pars = get_PW92_pars(ps[6:],'alp')

    ec0 = gPW92(rs,unp_pars)
    ec1 = gPW92(rs,pol_pars)
    mac = gPW92(rs,mac_pars)

    fz_den = (2.**(1./3.)-1.)
    fdd0 = 4./9./fz_den
    dx_z = spinf(z,4./3.)
    fz = (dx_z - 1.)/fz_den

    z4 = z**4
    fzz4 = fz*z4

    ec = ec0 - mac/fdd0*(fz - fzz4) + (ec1 - ec0)*fzz4

    return ec

def chi_enh_pw92(rs):
    mac_pw92 = gPW92(rs,[0.016887,0.11125,10.357,3.6231,0.88026,0.49671])
    return 1./(1. - rs/(pi*kf_to_rs) - 3.*(rs/kf_to_rs)**2*mac_pw92)

def chi_enh(rs,ps):
    malpha_c = gPW92(rs,ps)
    chi_s_chi_p = 1. - rs/(pi*kf_to_rs) - 3.*(rs/kf_to_rs)**2*malpha_c
    return 1./chi_s_chi_p

def d_chi_enh(rs,ps):
    malpha_c = gPW92(rs,ps)
    d_malpha_c = dgPW92(rs,ps)
    chi_sp = 1. - rs/(pi*kf_to_rs) - 3.*(rs/kf_to_rs)**2*malpha_c
    d_chi_sp_drs = -1./(pi*kf_to_rs) - 3.*rs/kf_to_rs**2 *(2.*malpha_c \
        + rs*d_malpha_c )

    return -d_chi_sp_drs/chi_sp**2

def refit_PW92():

    rs_fit, echi, uchi = get_ck_chi_enh()
    Nchi = rs_fit.shape[0]
    ec_HM_ld, NLD = get_HM_QMC_dat()
    ec_AD_ld, NLD2 = get_AD_DMC_dat()
    ec_CA, NCA = get_CA_ec()
    Nfpts = Nchi + NLD + NLD2 + NCA

    ips = [0.21370,1.6382,0.49294,0.20548,3.3662,0.62517,0.11125, 0.88026,0.49671]
    Nps = len(ips)
    bdsl = [0. for i in range(Nps)]
    bdsu = [np.inf for i in range(Nps)]

    bps = ips.copy()
    bobj = 1.e20

    twgts = [.01,.05,.1,.5,1.,5.]

    for twgt in product(np.zeros(1),np.ones(1),np.ones(1),np.ones(1)):#twgts,twgts,twgts,np.zeros(1)):#twgts):

        tsum = sum(twgt)
        upsilon = [twgt[i]/tsum for i in range(len(twgt))]

        def obj(c):
            res = np.zeros(Nfpts)
            macps = get_PW92_pars(c[6:],'alp')

            res[:Nchi] = upsilon[0]*(chi_enh(rs_fit, macps) - echi)/uchi
            #res[:Nchi] = -upsilon[0]*(gPW92(rs_fit,macps)+echi)/uchi
            i = Nchi
            for idict, adict in enumerate([ec_HM_ld,ec_AD_ld]):
                for trs in adict:
                    tec = epsc_PW92_rev(trs,adict[trs][:,0],c)
                    j = i+tec.shape[0]
                    res[i:j] = upsilon[1+idict]*(tec - adict[trs][:,1])/adict[trs][:,2]
                    i = j

            for az in ec_CA:
                tec = epsc_PW92_rev(ec_CA[az][:,0],1.*az,c)
                j = i + tec.shape[0]
                res[i:j] = upsilon[3]*(tec - ec_CA[az][:,1])/ec_CA[az][:,2]
                i = j

            return res

        res = least_squares(obj,bps,bounds = (bdsl,bdsu))
        tobj = np.sum(res.fun**2)
        if tobj < bobj:
            bobj = tobj
            bps = res.x.copy()
            bups = upsilon.copy()

    print(bups)
    unp_pars = get_PW92_pars(bps[:3],'unp')
    pol_pars = get_PW92_pars(bps[3:6],'pol')
    mac_pars = get_PW92_pars(bps[6:],'alp')

    parnms = ['A','\\alpha_1','\\beta_1','\\beta_2','\\beta_3','\\beta_4']
    tstr = r'& \varepsilon_\mathrm{c}^{(0)} & \varepsilon_\mathrm{c}^{(1)} & \alpha_\mathrm{c} \\ \hline' + '\n'
    for ipar in range(len(parnms)):
        tstr += '{:} & {:.9f} & {:.9f} & {:.9f} \\\\ \n'.format(parnms[ipar],\
            unp_pars[ipar],pol_pars[ipar],mac_pars[ipar])
    print(tstr)

    tstr = r' & QMC \cite{chen2019,kukkonen2021} & \multicolumn{2}{c}{PW92} & \multicolumn{2}{c}{This work} \\' + '\n'
    tstr += r' $r_\mathrm{s}$ & & $\chi_S/\chi_P$ & PD (\%) & $\chi_S/\chi_P$ & PD (\%) \\ \hline' + ' \n'
    for irs, rs in enumerate(rs_fit):

        echi_pw92 = chi_enh_pw92(rs)
        echi_new = chi_enh(rs,mac_pars)

        pd_pw92 = 200.*abs(echi[irs] - echi_pw92)/(echi[irs] + echi_pw92)
        pd_new = 200.*abs(echi[irs] - echi_new)/(echi[irs] + echi_new)
        tprec = len(str(echi[irs]).split('.')[-1])
        tstr += '{:} & {:}({:.0f}) & {:.6f} & {:.2f} & {:.6f} & {:.2f} \\\\ \n'.format(\
            int(rs),echi[irs],uchi[irs]*10.**tprec,echi_pw92,pd_pw92,echi_new,pd_new)
    with open('./chi_enhance.tex','w+') as tfl:
        tfl.write(tstr)

    rs_min = 1.e-1
    rs_max = 1.e3
    Nrs = 5000
    rsl_log = np.linspace(np.log(rs_min),np.log(rs_max),Nrs)
    rsl = np.exp(rsl_log)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.errorbar(rs_fit,echi,yerr=uchi,color='k',\
        markersize=3,marker='o',linewidth=0,elinewidth=1.5)
    #plt.plot(rsl,chi_enh(rsl,c0_alpha,c1_alpha,get_gam(res2.x[0]),*res2.x))
    nchi = chi_enh(rsl,mac_pars)
    ax.plot(rsl,nchi,color='darkblue',label='This work')
    ax.annotate('This work',(80.,0.12),color='darkblue',fontsize=12)
    #plt.plot(rsl,chi_enh(rsl,c0_alpha,*res3.x))

    echi_pw92 = chi_enh_pw92(rsl)
    echi_pz81 = chi_enh_pz81(rsl)
    ax.plot(rsl,echi_pw92,color='darkorange',linestyle='--',label='PW92')
    ax.annotate('PW92',(94.,230.),color='darkorange',fontsize=14)

    ax.plot(rsl,echi_pz81,color='tab:green',linestyle='-.',label='PZ81')
    ax.annotate('PZ81',(8.56,114.6),color='tab:green',fontsize=14)

    axins = inset_axes(ax, width=1.7, height=1.3, loc='lower left', \
        bbox_to_anchor=(.055,.5), bbox_transform=ax.transAxes)

    axins.errorbar(rs_fit,echi,yerr=uchi,color='k',\
        markersize=3,marker='o',linewidth=0,elinewidth=1.5)
    axins.plot(rsl,nchi,color='darkblue',label='This work')

    axins.plot(rsl,echi_pw92,color='darkorange',linestyle='--',label='PW92')
    axins.plot(rsl,echi_pz81,color='tab:green',linestyle='-.',label='PZ81')
    axins.set_xlim(0.5,6.)
    axins.set_ylim(1.,1.8)

    axins.xaxis.set_minor_locator(MultipleLocator(0.5))
    axins.xaxis.set_major_locator(MultipleLocator(1.))

    axins.yaxis.set_minor_locator(MultipleLocator(0.25))
    axins.yaxis.set_major_locator(MultipleLocator(0.5))

    ax.set_xlim(rs_min,rs_max)
    ax.set_ylim(1.e-2,1.5e3)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('$r_\\mathrm{s}$ (bohr)',fontsize=14)
    ax.set_ylabel(r'$\chi_S/\chi_P$',fontsize=14)

    #ax.legend(fontsize=14)

    #plt.show() ; exit()
    plt.savefig('./suscep_enhance.pdf',dpi=600,bbox_inches='tight')

    plt.cla()
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(figsize=(5,5.5))
    zl = np.linspace(0.,1.,2000)

    ybds = [1.e20,-1.e20]

    colors = ['darkblue','darkorange','tab:green','darkred','gray','m','k']
    mrks = ['o','s']
    mHa = 1.e3

    rs_l = []
    for adict in [ec_HM_ld, ec_AD_ld]:
        for trs in adict:
            if trs not in rs_l:
                rs_l.append(trs)
    rs_l = np.sort(rs_l)

    for irs, trs in enumerate(rs_l):

        nmtch = 0
        for idict, adict in enumerate([ec_HM_ld,ec_AD_ld]):

            if trs in adict:

                lstr = None
                if nmtch == 0:
                    lstr = '$r_\\mathrm{s}='+'{:}$'.format(trs)
                    ax.annotate(lstr,(.01,adict[trs][0,1]*mHa+.15),color=colors[irs],\
                        fontsize=12)

                #mfill = colors[irs]
                #if idict == 1:
                #    mfill = 'none'
                ax.errorbar(adict[trs][:,0],adict[trs][:,1]*mHa,\
                    yerr=adict[trs][:,2]*mHa,\
                    markersize=6,marker=mrks[idict],linewidth=0,\
                    elinewidth=1.5,color=colors[irs],\
                    label=lstr)#,markerfacecolor=mfill)

                nmtch += 1

        eps_c_new = epsc_PW92_rev(trs,zl,bps)
        eps_c_pw92, _, _, _ = ec_pw92(trs,zl)

        eps_c_new *= mHa
        eps_c_pw92 *= mHa

        ybds = [min(ybds[0],eps_c_new.min(),eps_c_pw92.min()),\
            max(ybds[1],eps_c_new.max(),eps_c_pw92.max())
        ]
        ax.plot(zl,eps_c_new,color=colors[irs],linestyle='-')
        ax.plot(zl,eps_c_pw92,color=colors[irs],linestyle='-.')


    teps = 1.e-2
    ax.set_xlim(0.-teps,1.+teps)
    ax.set_ylim(1.02*ybds[0],0.98*ybds[1])

    ax.set_xlabel(r'$\zeta = (n_\uparrow - n_\downarrow)/(n_\uparrow + n_\downarrow)$',fontsize=14)
    ax.set_ylabel(r'$\varepsilon_\mathrm{c}(r_\mathrm{s},\zeta)$ (mHa)',fontsize=14)

    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_locator(MultipleLocator(0.25))

    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(1.))

    #ax.legend(fontsize=12)

    #plt.show() ;exit()
    plt.savefig('./ec_ld_refit.pdf',dpi=600,bbox_inches='tight')

    plt.cla()
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(figsize=(6,4))

    rsl = np.linspace(0.1,120.,5000)
    ybds = [1e20,-1e20]

    for iz, az in enumerate(ec_CA):
        ax.errorbar(ec_CA[az][:,0],ec_CA[az][:,1],yerr=ec_CA[az][:,2],\
            color=colors[iz], markersize=3,marker='o',linewidth=0,elinewidth=1.5)

        eps_c_new = epsc_PW92_rev(rsl,1.*az,bps)
        ax.plot(rsl,eps_c_new,color=colors[iz])
        eps_c_pw92, _, _, _ = ec_pw92(rsl,1.*az)
        ax.plot(rsl,eps_c_pw92,color=colors[iz],linestyle=':')

        ybds = [
            min(ybds[0],eps_c_new.min(),eps_c_pw92.min()),\
            max(ybds[1],eps_c_new.max(),eps_c_pw92.max())
        ]

    ax.set_xlim(rsl[0],rsl[-1])
    ax.set_ylim(1.02*ybds[0],0.98*ybds[1])

    ax.set_xscale('log')

    ax.set_xlabel(r'$r_\mathrm{s}$',fontsize=14)
    ax.set_ylabel(r'$\varepsilon_\mathrm{c}(r_\mathrm{s},\zeta)$ (Ha)',fontsize=14)

    plt.show()

    return

if __name__ == "__main__":

    refit_PW92()
