import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from scipy.optimize import least_squares, bisect
from os import path,system

from AKCK_LFF.PZ81 import chi_enh_pz81
from AKCK_LFF.PW92 import ec_pw92

from AKCK_LFF.QMC_data import get_ck_chi_enh, ke_ex

bdir = './stiffness_refit/'
if not path.isdir(bdir):
    system('mkdir ' + bdir)

pi = np.pi

plt.rcParams.update({'text.usetex': True, 'font.family': 'dejavu'})

"""
    Eqs. 4.9 - 4.10 of
    S.H. Vosko, L. Wilk, and M. Nusair, Can. J. Phys. 58, 1200 (1980);
    doi: 10.1139/p80-159
"""
kf_to_rs = (9.*pi/4.)**(1./3.)
c0_alpha = -1./(6.*pi**2)
PT_integral = 0.5315045266#0.531504
c1_alpha = (np.log(16.*pi*kf_to_rs) - 3. + PT_integral )/(6.*pi**2)

def get_exp_pars(A,alpha1,beta1,beta2,beta3,beta4):
    c0 = A
    c1 = -2*c0*np.log(2.*c0*beta1)
    c2 = A*alpha1
    c3 = -2*A*(alpha1*np.log(2*A*beta1) - (beta2/beta1)**2 + beta3/beta1)
    d0 = alpha1/beta4
    d1 = alpha1*beta3/beta4**2
    return c0, c1, c2, c3, d0, d1

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

def alpha_suc_pz(rs,c,ret_coeff=False):
    alpha_c = np.zeros(rs.shape)

    tden = 1. + c[1] + c[2]
    tden2 = tden*tden
    g1 = -c[0]/tden
    g2 = c[0]*(c[1]/2. + c[2])/tden2
    g3 = -c[0]* ( 2.*(c[1]/2. + c[2])**2/tden + 0.25*c[1])/tden2

    ca = c0_alpha
    cb = -c1_alpha

    cd = g1 - cb
    cc =(-3*g1 + 3*g2 - g3 - 4*ca + 3*cb)/2.
    ce = (g1 - g2 + g3 + 2*ca - cb)/2.
    if ret_coeff:
        return {'A': ca, 'B': cb, 'C': cc, 'D': cd, 'E': ce,\
            'gamma': c[0], 'beta1': c[1], 'beta2': c[2]}

    tmsk = rs < 1.
    rsm = rs[tmsk]
    lrsm = np.log(rsm)
    alpha_c[tmsk] = ca*lrsm + cb + cc*rsm*lrsm + cd*rsm + ce*rsm**2*lrsm

    tmsk = rs >= 1.
    rsm = rs[tmsk]
    alpha_c[tmsk] = -c[0]/(1. + c[1]*rsm**(0.5) + c[2]*rsm)

    return alpha_c

def chi_enh_new_pz(rs,c):
    ac = alpha_suc_pz(rs,c)
    chi_s_chi_p = 1. - rs/(pi*kf_to_rs) + 3.*(rs/kf_to_rs)**2*ac
    return 1./chi_s_chi_p

def get_dev_from_PW92(newpars,rsmin = 1.e-2, rsmax = 5.e2, Nrs = 100000):

    rsl = np.exp(np.linspace(np.log(rsmin),np.log(rsmax),Nrs))

    mac_pw92 = gPW92(rsl,[0.016887,0.11125,10.357,3.6231,0.88026,0.49671])
    mac_new = gPW92(rsl,newpars)

    pdiff = 200.*abs(mac_new - mac_pw92)/abs(mac_new + mac_pw92)

    imax = np.argmax(pdiff)
    return rsl[imax], -mac_pw92[imax], -mac_new[imax], pdiff[imax]

def fit_alpha_c_new():

    rs_fit, echi, uchi = get_ck_chi_enh()
    Nchi = rs_fit.shape[0]

    #pzobj = lambda c : (chi_enh_new_pz(rs_fit,c) - echi)/uchi
    #pzres = least_squares(pzobj,[3.4787, -84.4585, 4.09087])
    #print(pzres.x)

    # Table VI
    unp_fluid = np.transpose(np.array([
        [30.,40.,60.,80.,100.],
        [22.6191,17.6143,12.2556,9.4259,7.6709],
        [7.e-4,3.e-4,3.e-4,4.e-4,3.e-4]
    ]))

    pol_fluid = np.transpose(np.array([
        [30.,40.,60.,80.,100.],
        [22.4819,17.5558,12.2418,9.4246,7.6720],
        [7.e-4,7.e-4,5.e-4,3.e-4,4.e-4]
    ]))

    unp_fluid[:,1] = -1.e-3*unp_fluid[:,1] - ke_ex(unp_fluid[:,0],0.)
    unp_fluid[:,2] *= 1.e-3

    pol_fluid[:,1] = -1.e-3*pol_fluid[:,1] - ke_ex(pol_fluid[:,0],1.)
    pol_fluid[:,2] *= 1.e-3


    NADRS = unp_fluid.shape[0]
    AD_rs = np.zeros(NADRS)
    AD_ac = np.zeros(NADRS)
    AD_ac_ucrt = np.zeros(NADRS)

    PZ_fdd0 = 4./(9.*(2.**(1./3.)-1.))
    for irs in range(NADRS):
        AD_rs[irs] = unp_fluid[irs,0]
        AD_ac[irs] = PZ_fdd0*(pol_fluid[irs,1] - unp_fluid[irs,1])
        AD_ac_ucrt[irs] = PZ_fdd0*(pol_fluid[irs,2]**2 + unp_fluid[irs,2]**2)**(0.5)

    c1_pw92 = 1./(2.*abs(c0_alpha))*np.exp(-c1_alpha/(2.*abs(c0_alpha)))

    ips = [0.11125, 0.88026,0.49671]
    bdsl = [0.,-np.inf,0.]#[0. for i in range(len(ips))]
    bdsu = [np.inf for i in range(len(ips))]

    def get_PW92_pars(c):
        ps = np.zeros(6)
        ps[0] = abs(c0_alpha)
        ps[1] = c[0]
        ps[2] = c1_pw92
        ps[3] = 2.*ps[0]*ps[2]**2
        ps[4] = c[1]
        ps[5] = c[2]
        return ps

    def obj(c):
        res = np.zeros(Nchi + NADRS)
        tps = get_PW92_pars(c)

        res[:Nchi] = (chi_enh(rs_fit, tps) - echi)/uchi
        i = Nchi

        ac = -gPW92(AD_rs,tps)
        res[Nchi:] = (ac - AD_ac)/AD_ac_ucrt

        return res

    res = least_squares(obj,ips,bounds = (bdsl,bdsu))
    tobj = np.sum(res.fun**2)

    opars = get_PW92_pars(res.x)
    mac_exps = get_exp_pars(*opars)

    parnms = ['A','\\alpha_1','\\beta_1','\\beta_2','\\beta_3','\\beta_4']
    expnms = ['$c_0$','$c_1$','$c_2$','$c_3$','$d_0$','$d_1$']
    tstr = ''
    for ipar in range(len(parnms)):
        tstr += '${:}$ & {:.9f} & {:} & {:.9f} \\\\ \n'.format(parnms[ipar],opars[ipar],expnms[ipar],mac_exps[ipar])
    with open(bdir + 'alphac_pars_rev.tex','w+') as tfl:
        tfl.write(tstr)

    tstr = r' & QMC \cite{chen2019,kukkonen2021} & \multicolumn{2}{c}{PW92} & \multicolumn{2}{c}{This work} \\' + '\n'
    tstr += r' $r_\mathrm{s}$ & & $\chi_s/\chi_s^{(0)}$ & PD (\%) & $\chi_s/\chi_s^{(0)}$ & PD (\%) \\ \hline' + ' \n'
    for irs, rs in enumerate(rs_fit):

        echi_pw92 = chi_enh_pw92(rs)
        echi_new = chi_enh(rs,opars)

        pd_pw92 = 200.*(echi_pw92 - echi[irs])/(echi[irs] + echi_pw92)
        pd_new = 200.*(echi_new - echi[irs])/(echi[irs] + echi_new)
        tprec = len(str(echi[irs]).split('.')[-1])
        tstr += '{:} & {:}({:.0f}) & {:.6f} & {:.2f} & {:.6f} & {:.2f} \\\\ \n'.format(\
            int(rs),echi[irs],uchi[irs]*10.**tprec,echi_pw92,pd_pw92,echi_new,pd_new)
    with open(bdir + 'chi_enhance.tex','w+') as tfl:
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
    ax.annotate('This work',(150.,80.),color='darkblue',fontsize=14)
    #plt.plot(rsl,chi_enh(rsl,c0_alpha,*res3.x))

    echi_pw92 = chi_enh_pw92(rsl)
    echi_pz81 = chi_enh_pz81(rsl)
    ax.plot(rsl,echi_pw92,color='darkorange',linestyle='--',label='PW92')
    ax.annotate('PW92',(94.,518.),color='darkorange',fontsize=14)

    ax.plot(rsl,echi_pz81,color='tab:green',linestyle='-.',label='PZ81')
    ax.annotate('PZ81',(4.4,114.6),color='tab:green',fontsize=14)

    #"""
    axins = inset_axes(ax, width=1.7, height=1.,\
        loc='lower left', bbox_to_anchor=(.46,.06), \
        bbox_transform=ax.transAxes)

    axins.errorbar(rs_fit,echi,yerr=uchi,color='k',\
        markersize=3,marker='o',linewidth=0,elinewidth=1.5)
    axins.plot(rsl,nchi,color='darkblue',label='This work')

    axins.plot(rsl,echi_pw92,color='darkorange',linestyle='--',label='PW92')
    axins.plot(rsl,echi_pz81,color='tab:green',linestyle='-.',label='PZ81')

    #ax.plot(rsl,chi_enh_new_pz(rsl,pzres.x),color='red')
    #axins.plot(rsl,chi_enh_new_pz(rsl,pzres.x),color='red')

    axins.set_xlim(0.5,6.)
    axins.set_ylim(1.,1.8)

    axins.xaxis.set_minor_locator(MultipleLocator(0.5))
    axins.xaxis.set_major_locator(MultipleLocator(1.))

    axins.yaxis.set_minor_locator(MultipleLocator(0.25))
    axins.yaxis.set_major_locator(MultipleLocator(0.5))
    #"""
    ax.set_xlim(rs_min,rs_max)
    ax.set_ylim(1.e-2,1.5e3)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('$r_\\mathrm{s}$ (bohr)',fontsize=14)
    ax.set_ylabel(r'$\chi_s/\chi_s^{(0)}$',fontsize=14)

    #ax.legend(fontsize=14)

    #plt.show() ; exit()
    plt.savefig(bdir + 'suscep_enhance.pdf',dpi=600, \
        bbox_inches='tight')

    plt.cla()
    plt.clf()
    plt.close()

    mx_rs_dev, mx_alp_pw92, mx_alp_new, mx_pdiff = get_dev_from_PW92(opars)
    print('Max percent deviation between PW92 ({:.2e}) and this work ({:.2e})'.format(mx_alp_pw92,mx_alp_new))
    print('  at rs = {:.2f} ({:.6f}%)'.format(mx_rs_dev,mx_pdiff))

    return

if __name__ == "__main__":

    fit_alpha_c_new()
