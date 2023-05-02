import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from scipy.optimize import least_squares, bisect

from PZ81 import chi_enh_pz81
from PW92 import ec_pw92

from QMC_data import get_ck_chi_enh, get_HM_QMC_dat, get_AD_DMC_dat

pi = np.pi

"""
    Eqs. 4.9 - 4.10 of
    S.H. Vosko, L. Wilk, and M. Nusair, Can. J. Phys. 58, 1200 (1980);
    doi: 10.1139/p80-159
"""
kf_to_rs = (9.*pi/4.)**(1./3.)
c0_alpha = -1./(6.*pi**2)
PT_integral = 0.5315045266#0.531504
c1_alpha = (np.log(16.*pi*kf_to_rs) - 3. + PT_integral )/(6.*pi**2)


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

def epsc_PW92_rev(rs,z,ps):

    ec0 = gPW92(rs,[0.031091,0.21370,7.5957,3.5876,1.6382,0.49294])
    ec1 = gPW92(rs,[0.015545,0.20548,14.1189,6.1977,3.3662,0.62517])
    mac = gPW92(rs,ps)

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

def fit_alpha_c_new():

    rs_fit, echi, uchi = get_ck_chi_enh()
    Nchi = rs_fit.shape[0]
    ec_HM_ld, NLD = get_HM_QMC_dat()
    ec_AD_ld, NLD2 = get_AD_DMC_dat()

    def get_alpha_1(a,b1,b2,b3,b4,rs1 = 77.5, rs2 = 75.):

        """
            The Perdew and Wang work states that alpha_c is fitted, at rs = 75, to
            a value which makes the spin-susceptibilty enhancement diverge at rs = 77.5

            This sentence is perplexing for many reasons, one of which being that
            this is not true, unless I am truly missing their intent

            Either way, we do this process here
        """

        """
        tobj = lambda a1 : d_chi_enh(rs1,[a,a1,b1,b2,b3,b4])
        a1l = np.arange(0.,20.,0.1)
        Na1 = a1l.shape[0]
        stjl = np.sign(tobj(a1l))
        s0 = stjl[0]
        for i in range(1,Na1):
            if s0*stjl[i] < 0.:
                break

        alpha1, msg = bisect(tobj,a1l[max(0,i-1)],a1l[min(Na1-1,i+1)],full_output=True)
        if not msg.converged:
            print(msg)

        """
        t1 = (1. - rs1/(pi*kf_to_rs))/(3.*(rs1/kf_to_rs)**2)

        rs2h = rs2**(0.5)
        q1 = 2.*a*(rs2h*( b1 + rs2h*(  b2 + rs2h*(   b3 + rs2h*b4   )  ) ))
        logfac = np.log(1. + 1./q1)
        alpha1 = -(t1/(2.*a*logfac) + 1.)/(rs2)
        #"""

        """
        rsl = np.linspace(60.,100.,2000)
        plt.plot(rsl,chi_enh(rsl,[a,alpha1,b1,b2,b3,b4]))
        plt.plot(rsl,d_chi_enh(rsl,[a,alpha1,b1,b2,b3,b4]))
        plt.show()
        exit()
        """

        return alpha1

    rsf = [77.5]#np.arange(70.,80.01,0.1)

    bres = 1.e20
    bps = [0.,0.]
    frs = [0.,0.]

    c1_pw92 = 1./(2.*abs(c0_alpha))*np.exp(-c1_alpha/(2.*abs(c0_alpha)))
    #ips = [0.88026,0.49671]
    ips = [0.11125, 0.88026,0.49671]
    bdsl = [0.,-np.inf,0.]#[0. for i in range(len(ips))]
    bdsu = [np.inf for i in range(len(ips))]

    def get_PW92_pars(c,rs1=77.5,rs2=75.):
        ps = np.zeros(6)
        ps[0] = -c0_alpha
        ps[1] = c[0]
        ps[2] = c1_pw92
        ps[3] = 2.*ps[0]*ps[2]**2
        ps[4] = c[1]
        ps[5] = c[2]
        #ps[1] = get_alpha_1(ps[0],ps[2],ps[3],ps[4],ps[5],rs1 = rs1, rs2 = rs2)
        return ps

    upsilon = 1.e-2

    def obj(c):
        res = np.zeros(Nchi + NLD + NLD2)
        tps = get_PW92_pars(c)#,rs1=ars,rs2=brs)

        res[:Nchi] = (chi_enh(rs_fit, tps) - echi)/uchi
        i = Nchi
        for adict in [ec_HM_ld]:#,ec_AD_ld]:
            for trs in adict:
                tec = epsc_PW92_rev(trs,adict[trs][:,0],tps)
                j = i+tec.shape[0]
                res[i:j] = upsilon*(tec - adict[trs][:,1])/adict[trs][:,2]
                i = j

        return res

    res = least_squares(obj,ips,bounds = (bdsl,bdsu))
    tobj = np.sum(res.fun**2)
    if tobj < bres:
        bres = tobj
        bps = (res.x).copy()
        #frs = [ars,brs]

    opars = get_PW92_pars(bps)
    print(bres)
    parnms = ['A','\\alpha_1','\\beta_1','\\beta_2','\\beta_3','\\beta_4']
    tstr = ''
    for ipar in range(len(parnms)):
        tstr += '{:} &= {:.9f} \\\\ \n'.format(parnms[ipar],opars[ipar])
    print(tstr)

    c1_extrap = -2*opars[0]*np.log(2*opars[0]*opars[2])
    print(c1_extrap,c1_alpha,6.*pi**2*c1_extrap + 3. - np.log(16.*pi*kf_to_rs),PT_integral)
    #print(np.sum(((chi_enh(rs_fit,opars) - echi)/uchi)**2))

    tstr = r' & QMC \cite{chen2019,kukkonen2021} & \multicolumn{2}{c}{PW92} & \multicolumn{2}{c}{This work} \\' + '\n'
    tstr += r' $r_\mathrm{s}$ & & $\chi_S/\chi_P$ & PD (\%) & $\chi_S/\chi_P$ & PD (\%) \\ \hline' + ' \n'
    for irs, rs in enumerate(rs_fit):

        echi_pw92 = chi_enh_pw92(rs)
        echi_new = chi_enh(rs,opars)

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

    #plt.plot(rsl,gPW92(rsl,opars)-gPW92(rsl,[0.016887,0.11125,10.357,3.6231,0.88026,0.49671]))
    #plt.show();exit()

    fig, ax = plt.subplots(figsize=(6,4))
    ax.errorbar(rs_fit,echi,yerr=uchi,color='k',\
        markersize=3,marker='o',linewidth=0,elinewidth=1.5)
    #plt.plot(rsl,chi_enh(rsl,c0_alpha,c1_alpha,get_gam(res2.x[0]),*res2.x))
    nchi = chi_enh(rsl,opars)
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

        eps_c_new = epsc_PW92_rev(trs,zl,opars)
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

    return

if __name__ == "__main__":

    fit_alpha_c_new()
