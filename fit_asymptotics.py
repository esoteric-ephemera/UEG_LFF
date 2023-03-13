from os import path, system
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from asymptotics import get_g_plus_pars, get_g_minus_pars

pi = np.pi
kf_to_rs = (9*pi/4.)**(1./3.)

bdir = './approx_asy/'
if not path.isdir(bdir):
    system('mkdir ' + bdir)

def a_pos_ffn(rs,c,ret_coeff=False):

    ncoeff = [0.,5.*c[1], 7.*c[1]**2 + 8.*c[2], 12.*c[1]*c[2],\
        16.*c[2]**2]
    dcoeff = [1.,c[1],c[2]]
    if ret_coeff:
        return ncoeff, dcoeff

    rsh = rs**(0.5)
    Nn = len(ncoeff)
    Nd = len(dcoeff)

    num = np.zeros(rs.shape)
    for i in range(Nn):
        num = num*rsh + ncoeff[Nn-1-i]

    den = np.zeros(rs.shape)
    for i in range(Nd):
        den = den*rsh + dcoeff[Nd-1-i]

    pfc = -c[0]*kf_to_rs**2*rs/108.

    return 1./4. + pfc*num/den**3

def a_min_ffn(rs,c):
    pfc = -3.*pi*c[0]*rs/(4.*kf_to_rs)
    return 0.25 + pfc/(1. + c[1]*rs**(0.5) + c[2]*rs)

def c_pos_ffn(rs,c,ret_coeff=False):

    if ret_coeff:
        return [-c[0],-c[0]*c[1]/2.], [1., c[1],c[2]]

    rsh = rs**(0.5)
    num = -c[0]*(1. + c[1]*rsh/2.)
    den = 1. + c[1]*rsh + c[2]*rs
    pfc = -pi*kf_to_rs*rs/2.
    return pfc*num/den**2

"""
def polyfun(rs,c):
    Nc = len(c)
    rsh = rs**(0.5)
    pfn = np.zeros(rs.shape)
    for i in range(Nc):
        pfn = pfn*rsh + c[Nc-1-i]
    return pfn
"""
def polyfun(rs,c):
    rsh = rs**(0.5)
    pfn = c[0]*rs + c[1]*rsh + c[2] + c[3]/rsh + c[4]/rs
    if len(c) == 6:
        pfn += c[5]/rs/rsh
    return pfn

def fit_pos_pars():

    pzps = ['\\gamma','\\beta_1','\\beta_2']
    NPZ = len(pzps)

    rmin = 1.
    wrmax = 50.
    srmax = 10.
    Nr = 5000

    wrs_l = np.exp(np.linspace(np.log(rmin),np.log(wrmax),Nr))
    wkf_l = kf_to_rs/wrs_l

    srs_l = np.linspace(rmin,srmax,Nr+1)
    skf_l = kf_to_rs/srs_l

    wap,_,wcp = get_g_plus_pars(wrs_l)
    #sap,_,scp = get_g_plus_pars(srs_l)

    tres = lambda c : a_pos_ffn(wrs_l,c) - wap
    wa = least_squares(tres,[0.1423,1.0529,0.3334])
    cwa = [round(y,9) for y in wa.x]
    cnwa, cdwa = a_pos_ffn(None,wa.x,ret_coeff=True)
    cnwa = [round(y,9) for y in cnwa]
    cdwa = [round(y,9) for y in cdwa]
    #print(cnwa,cdwa)

    wa_ffn = a_pos_ffn(wrs_l,cwa)
    pd_wa = 100.*(1. - wa_ffn/wap)
    iworst = np.argmax(np.abs(pd_wa))
    print('MAX A+ RP error at rs = {:.2f} is {:.2f} '.format(wrs_l[iworst],pd_wa[iworst]))

    tres = lambda c : polyfun(wrs_l,c) - wap
    sa = least_squares(tres,np.ones(5))
    csa = [round(y,9) for y in sa.x]

    sa_ffn = polyfun(wrs_l,csa)
    pd_sa = 100.*(1. - sa_ffn/wap)
    iworst = np.argmax(np.abs(pd_sa))
    print('MAX A+ SP error at rs = {:.2f} is {:.2f} '.format(wrs_l[iworst],pd_sa[iworst]))


    fig, ax = plt.subplots(figsize=(6,4))
    #ax.plot(wrs_l,wcp,color='k')
    ax.plot(wrs_l,pd_wa, color='darkblue',\
        linestyle='-',label = 'Rational polynomial')
    ax.plot(wrs_l,pd_sa, color='darkorange',\
        linestyle='--',label= 'Simple polynomial')

    ybds = [ 1.02*min(pd_wa.min(),pd_sa.min()),\
        1.02*max(pd_wa.max(),pd_sa.max()) ]
    ax.set_ylim(*ybds)

    ax.set_xlim(rmin,wrmax)
    ax.hlines(0.,rmin,wrmax,color='k',linewidth=1)

    tmks = []
    if ybds[0] <= -1.:
        tmks.append(-1.)
    if ybds[1] >= 1.:
        tmks.append(1.)

    for val in tmks:
        ax.hlines(val,rmin,wrmax,color='gray',linewidth=1,linestyle=':')

    ax.set_xscale('log')
    ax.set_xlabel('$r_\\mathrm{s}$ (bohr)',fontsize=14)

    ax.set_ylabel('PE in $A_+(r_\\mathrm{s})$ (%)',fontsize=14)

    ax.legend(fontsize=12,frameon=False)
    #plt.show() ; exit()
    plt.savefig(bdir + 'app_a_plus.pdf',dpi=600,bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()

    tres = lambda c: c_pos_ffn(wrs_l,c) - wcp
    wres = least_squares(tres,[0.1423,1.0529,0.3334])
    cwc = [round(y,9) for y in wres.x]
    #print(cwc)
    cnwc, cdwc = c_pos_ffn(None,wres.x,ret_coeff=True)
    cnwc = [round(y,9) for y in cnwc]
    cdwc = [round(y,9) for y in cdwc]
    #print(cnwc,cdwc)

    wc_ffn = c_pos_ffn(wrs_l,cwc)
    pd_wc = 100.*(1. - wc_ffn/wcp)
    iworst = np.argmax(np.abs(pd_wc))
    print('MAX C+ RP error at rs = {:.2f} is {:.2f} '.format(wrs_l[iworst],pd_wc[iworst]))

    tres = lambda c : polyfun(wrs_l,c) - wcp
    sres = least_squares(tres,np.ones(5))
    csc = [round(y,9) for y in sres.x]
    #print(csc)

    sc_ffn = polyfun(wrs_l,csc)
    pd_sc = 100.*(1. - sc_ffn/wcp)
    iworst = np.argmax(np.abs(pd_sc))
    print('MAX C+ SP error at rs = {:.2f} is {:.2f} '.format(wrs_l[iworst],pd_sc[iworst]))

    fig, ax = plt.subplots(figsize=(6,4))
    #ax.plot(wrs_l,wcp,color='k')
    ax.plot(wrs_l,pd_wc, color='darkblue',\
        linestyle='-',label = 'Rational polynomial')
    ax.plot(wrs_l,pd_sc, color='darkorange',\
        linestyle='--',label= 'Simple polynomial')
    ybds = [ 1.02*min(pd_wc.min(),pd_sc.min()),\
        1.02*max(pd_wc.max(),pd_sc.max()) ]
    ax.set_ylim(*ybds)

    ax.set_xlim(rmin,wrmax)
    ax.hlines(0.,rmin,wrmax,color='k',linewidth=1)

    tmks = []
    if ybds[0] <= -1.:
        tmks.append(-1.)
    if ybds[1] >= 1.:
        tmks.append(1.)

    for val in tmks:
        ax.hlines(val,rmin,wrmax,color='gray',linewidth=1,linestyle=':')
    ax.set_xscale('log')
    ax.set_xlabel('$r_\\mathrm{s}$ (bohr)',fontsize=14)

    ax.set_ylabel('PE in $C_+(r_\\mathrm{s})$ (%)',fontsize=14)

    ax.legend(fontsize=12,frameon=False)
    #plt.show() ; exit()
    plt.savefig(bdir + 'app_c_plus.pdf',dpi=600,bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()


    wam,_,_ = get_g_minus_pars(wrs_l,0.)
    np.savetxt('./tmp.csv',np.transpose((wrs_l,wam)),delimiter=',')
    #sam,_,_ = get_g_minus_pars(srs_l,0.)

    tres = lambda c: a_min_ffn(wrs_l,c) - wam
    am_rp = least_squares(tres,[0.0594229,0.259, 0.272])
    camw = [round(y,9) for y in am_rp.x]

    wam_ffn = a_min_ffn(wrs_l,camw)
    pd_wam = 100.*(1. - wam_ffn/wam)
    iworst = np.argmax(np.abs(pd_wam))
    print('MAX A- RP error at rs = {:.2f} is {:.2f} '.format(wrs_l[iworst],pd_wam[iworst]))

    tres = lambda c : polyfun(wrs_l,c) - wam
    am_sp = least_squares(tres,[0.,0.,-0.031,0.448,-0.221])
    cams = [round(y,9) for y in am_sp.x]
    #print(csc)

    sam_ffn = polyfun(wrs_l,cams)
    pd_sam = 100.*(1. - sam_ffn/wam)
    iworst = np.argmax(np.abs(pd_sam))
    print('MAX A- SP error at rs = {:.2f} is {:.2f} '.format(srs_l[iworst],pd_sam[iworst]))

    fig, ax = plt.subplots(figsize=(6,4))
    #ax.plot(wrs_l,wcp,color='k')
    ax.plot(wrs_l,pd_wam, color='darkblue',\
        linestyle='-',label = 'Rational polynomial')
    ax.plot(wrs_l,pd_sam, color='darkorange',\
        linestyle='--',label= 'Simple polynomial')

    ybds = [ 1.02*min(pd_wam.min(),pd_sam.min()),\
        1.02*max(pd_wam.max(),pd_sam.max()) ]
    ax.set_ylim(*ybds)

    ax.set_xlim(rmin,wrmax)
    ax.hlines(0.,rmin,wrmax,color='k',linewidth=1)

    tmks = []
    if ybds[0] <= -1.:
        tmks.append(-1.)
    if ybds[1] >= 1.:
        tmks.append(1.)

    for val in tmks:
        ax.hlines(val,rmin,wrmax,color='gray',linewidth=1,linestyle=':')

    ax.set_xscale('log')
    ax.set_xlabel('$r_\\mathrm{s}$ (bohr)',fontsize=14)

    ax.set_ylabel('PE in $A_-(r_\\mathrm{s})$ (%)',fontsize=14)

    ax.legend(fontsize=12,frameon=False)
    #plt.show() ; exit()
    plt.savefig(bdir + 'app_a_minus.pdf',dpi=600,bbox_inches='tight')


    tstr = ' & $A_+(r_\\mathrm{s})$ & $C_+(r_\\mathrm{s})$ & $A_-(r_\\mathrm{s})$ \\\\ \\hline \n'
    for i in range(NPZ):
        tstr += '${:}$ & {:} & {:} & {:} \\\\ \n'.format(pzps[i],cwa[i],cwc[i],camw[i])
    with open(bdir + 'A_C_PZ.tex','w+') as tfl:
        tfl.write(tstr)

    tstr = ' & $A_+(r_\\mathrm{s})$ & $C_+(r_\\mathrm{s})$ & $A_-(r_\\mathrm{s})$ \\\\ \\hline \n'
    NSA = len(csa)
    NSC = len(csc)
    NAM = len(cams)
    for i in range(max(NSA,NSC,NAM)):
        tstr += '$w_{'+'{:}'.format(i)+'j}$'
        if i < NSA:
            tstr += ' & {:}'.format(csa[i])
        else:
            tstr += ' & '

        if i < NSC:
            tstr += ' & {:}'.format(csc[i])
        else:
            tstr += ' & '

        if i < NAM:
            tstr += ' & {:}'.format(cams[i])
        else:
            tstr += ' & '

        tstr += ' \\\\ \n'

    with open(bdir + 'A_C_poly.tex','w+') as tfl:
        tfl.write(tstr)

    return

if __name__ == "__main__":

    fit_pos_pars()
