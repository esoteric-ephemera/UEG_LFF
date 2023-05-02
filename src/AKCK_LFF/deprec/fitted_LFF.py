import numpy as np
import matplotlib.pyplot as plt
from os import path, system

from asymptotics import get_g_minus_pars, get_g_plus_pars, gexc2

pi = np.pi
rs_to_kf = (9*pi/4.)**(1./3.)

def gplus_CK(q,rs):

    c = [0.0345954, 0.593529, 0.288928, 1.365886, 0.664103, 0.331411, \
        1.078443, 0.998886 ]

    kf = rs_to_kf/rs

    q2 = (q/kf)**2
    q4 = q2*q2
    q8 = q4*q4
    Apos, Bpos, Cpos = get_g_plus_pars(rs)

    alpha = c[0]*np.exp(-abs(c[1])*rs)
    beta = c[2] + c[3]*np.exp(-abs(c[4])*rs)
    gamma = c[5] + c[6]*np.exp(-abs(c[7])*rs)

    Dpos = -(3.*pi**2)**(4./3.)*gexc2(rs)/(2.*pi)

    interp1 = np.exp(-beta*q8/256.)
    interp2 = -np.expm1(-gamma*q8/256.)

    asymp1 = q2*(Apos + Dpos*q2 + alpha*q4)
    asymp2 = (Bpos + Cpos*q2)
    gplus = asymp1*interp1 + asymp2*interp2

    return gplus

def smoothstep(x,a,b):
    ff = np.zeros(x.shape)
    exparg = a*(x - b)
    xmsk = exparg < 100.
    tfac = np.exp(-a*b)
    ff[xmsk] = (1. - tfac)/(1. - 2*tfac + np.exp(a*(x[xmsk] - b)))
    return ff

def gplus_no_GEA(q,rs):

    kf = rs_to_kf/rs
    q2 = (q/kf)**2
    q4 = q2*q2
    q8 = q4*q4
    Apos, Bpos, Cpos = get_g_plus_pars(rs)

    c = [0.0363454, 1.704220, 0.0931081, 1.889440, 0.948436]
    alpha = c[0]*np.exp(-c[1]*rs)
    beta = c[2] + c[3]*np.exp(-c[4]*rs)

    interp1 = np.exp(-beta*q8/256.)
    interp2 = 1. - interp1

    asymp1 = q2*(Apos + alpha*q4)
    asymp2 = Bpos + Cpos*q2
    gplus = asymp1*interp1 + asymp2*interp2
    """
    c = [0.0332710, 0.702075, 2.965026, 0.781172]
    alpha = c[0]*np.exp(-abs(c[1])*rs)
    beta = c[2]
    gamma = c[3]

    interp1 = smoothstep(q2**2/16.,beta,gamma)
    interp2 = 1. - interp1
    Dpos = -(3.*pi**2)**(4./3.)*gexc2(rs)/(2.*pi)

    asymp1 = q2*(Apos + Dpos*q2 + alpha*q4)
    asymp2 = Bpos + Cpos*q2
    gplus = asymp1*interp1 + asymp2*interp2
    """
    return gplus


def gminus_CK(q,rs,z):

    c = [0.0221214, 0.637241, 0.883245, -0.595246, 1.149315, 1.641103, \
        3.550855, -1.444974 ]

    kf = rs_to_kf/rs
    q2 = (q/kf)**2
    q8 = q2**4
    Amin, Bmin, Cmin = get_g_minus_pars(rs,z)

    alpha = c[0]*np.exp(-c[1]*rs)
    beta = c[2] + c[3]*rs**2*np.exp(-abs(c[4])*rs)
    gamma = c[5] + c[6]*rs**2*np.exp(-abs(c[7])*rs)

    interp1 = np.exp(-beta*q8/256.)
    interp2 = -np.expm1(-gamma*q8/256.)

    asymp1 = q2*(Amin + alpha*q2)
    asymp2 = (Bmin + Cmin*q2)
    gmin = asymp1*interp1 + asymp2*interp2

    return gmin

def gplus_plots():

    from ra_lff import  g_plus_ra
    from g_corradini import g_corradini
    from mcp07_static import mcp07_static

    if not path.isdir('./figs/'):
        system('mkdir ./figs')

    colors = ['darkblue','darkorange','tab:green','darkred','darkslategray']

    xl = np.linspace(0.0,4.0,5000)
    xlp = xl[1:]

    rs_l = [0.1,1,2,5,10,100]

    for irs, rs in enumerate(rs_l):

        fig, ax = plt.subplots(2,1,figsize=(5,7.5))

        kf = rs_to_kf/rs

        ckgpf = './data_files/CK_Gplus_rs_{:}.csv'.format(int(rs))
        if path.isfile(ckgpf):

            tdat_CK = np.genfromtxt(ckgpf,delimiter=',',skip_header=1)

            tq2_CK = tdat_CK[:,0]**2#(tdat_CK[:,0]*kf)**2
            ax[0].errorbar(tdat_CK[:,0],tdat_CK[:,1],yerr=tdat_CK[:,2],color='k',\
                markersize=3,marker='o',linewidth=0,elinewidth=1.5)

            ax[1].errorbar(tdat_CK[:,0],4.*pi*tdat_CK[:,1]/tq2_CK, \
                yerr=4.*pi*tdat_CK[:,2]/tq2_CK, color='k',\
                markersize=3,marker='o',linewidth=0,elinewidth=1.5)

        mcsgpf = './data_files/MCS_Gplus_rs_{:}.csv'.format(int(rs))
        if path.isfile(mcsgpf):

            tdat_MCS = np.genfromtxt(mcsgpf,delimiter=',',skip_header=1)

            tq2_MCS = tdat_MCS[:,0]**2#(tdat_MCS[:,0]*kf)**2

            ax[0].errorbar(tdat_MCS[:,0],tdat_MCS[:,1], \
                yerr=tdat_MCS[:,2],color='m',\
                markersize=3,marker='o',linewidth=0,elinewidth=1.5)

            ax[1].errorbar(tdat_MCS[:,0],4.*pi*tdat_MCS[:,1]/tq2_MCS, \
                yerr=4.*pi*tdat_MCS[:,2]/tq2_MCS, color='m', \
                markersize=3,marker='o',linewidth=0,elinewidth=1.5)

        kf2 = kf*kf
        kl2 = xlp**2#(xlp*kf)**2
        a,b,c = get_g_plus_pars(rs)

        ax[0].plot(xl,a*xl**2,color=colors[1],linestyle=':',\
            label='SQE')#'$A_+(r_\\mathrm{s})(q/k_\\mathrm{F})^2$')
        ax[1].plot(xlp,4.*pi*a*np.ones_like(xlp),color=colors[1],linestyle=':',\
            label='SQE')#'$A_+(r_\\mathrm{s})(q/k_\\mathrm{F})^2$')

        ax[0].plot(xl,c*xl**2 + b,color=colors[2],linestyle=':',\
            label='LQE')#'$C(r_\\mathrm{s})(q/k_\\mathrm{F})^2 + B_+(r_\\mathrm{s})$')
        ax[1].plot(xlp,4.*pi*(c + b/kl2**2),color=colors[2],linestyle=':',\
            label='LQE')#'$C(r_\\mathrm{s})(q/k_\\mathrm{F})^2 + B_+(r_\\mathrm{s})$')

        #gpapp = gplus_CK(xl*kf,rs)
        gpapp = gplus_no_GEA(xl*kf,rs)
        ax[0].plot(xl,gpapp,color=colors[0],\
            label='This work')
        #gpapp_oq2 = 4.*pi*gplus_CK(xlp*kf,rs)/kl2
        gpapp_oq2 = 4.*pi*gplus_no_GEA(xlp*kf,rs)/kl2
        ax[1].plot(xlp,gpapp_oq2,color=colors[0],\
            label='This work')

        ax[0].plot(xl,g_plus_ra(xl*kf,0.,rs),color=colors[3],linestyle='-',\
            label='Richardson-Ashcroft')
        ax[1].plot(xlp,4.*pi*g_plus_ra(xlp*kf,0.,rs)/kl2,color=colors[3],linestyle='-',\
            label='Richardson-Ashcroft')

        dens_d = {'rs': rs, 'kF': kf, 'n': 3./(4.*pi*rs**3), 'rsh': rs**(0.5)}
        gcorr = g_corradini(xl*kf,dens_d)
        ax[0].plot(xl,gcorr, color=colors[4],linestyle='-',\
            label=r'Corradini et al.')
        ax[1].plot(xlp,4.*pi*g_corradini(xlp*kf,dens_d)/kl2, color=colors[4],\
            linestyle='-', label=r'Corradini et al.')

        """
        fxc_mcp07 = mcp07_static(xl*kf,dens_d,param='PW92')
        g_mcp07 = -fxc_mcp07*(xl*kf)**2/(4.*pi)
        ax[0].plot(xl,g_mcp07,color='cyan',linestyle='-.',label=r'MCP07')
        fxc_mcp07 = mcp07_static(xlp*kf,dens_d,param='PW92')
        g_mcp07 = -fxc_mcp07*(xlp*kf)**2/(4.*pi)
        ax[1].plot(xlp,g_mcp07/kl2,color='cyan',linestyle='-.',label=r'MCP07')
        """

        for iplt in range(2):
            ax[iplt].set_xlim(xl.min(),xl.max())
        ax[1].set_xlabel('$q/k_\\mathrm{F}$',fontsize=12)
        ax[0].set_ylabel('$G_+(q)$',fontsize=12)
        ax[1].set_ylabel('$4\\pi \\, G_+(q) (k_\\mathrm{F}/q)^2$',fontsize=12)
        ax[0].set_ylim(0.,1.1*gpapp.max())
        ax[1].set_ylim(0.,1.1*gpapp_oq2.max())

        if rs in [1,2]:
            ileg = 0
            tcoord = (0.01,0.05)
        elif rs in [0.1]:
            ileg = 1
            tcoord = (0.9,0.9)
        elif rs in [100]:
            ileg = 1
            tcoord = (0.9, 0.02)
        else:
            ileg = 1
            tcoord = (0.9,0.05)
        ax[ileg].legend(fontsize=10,title='$r_\\mathrm{s}'+'={:}$'.format(rs),\
            title_fontsize=12,frameon=False)#,ncol = 3,loc=(0.33,1.01))

        ax[0].annotate('(a)',(0.01,0.9),fontsize=16,xycoords='axes fraction')
        ax[1].annotate('(b)',tcoord,fontsize=16,xycoords='axes fraction')

        #plt.show() ; exit()
        plt.savefig('./figs/gplus_rs_{:}_2p.pdf'.format(rs), dpi=600, \
            bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()

    return


def gminus_plots():

    from ra_lff import g_minus_ra

    if not path.isdir('./figs/'):
        system('mkdir ./figs')

    colors = ['darkblue','darkorange','tab:green','darkred','k','gray']

    xl = np.linspace(0.0,4.0,5000)
    xlp = xl[1:]

    rs_l = [0.1,1,2,3,4,5,100]

    for irs, rs in enumerate(rs_l):

        got_QMC_dat = False
        ckgmf = './data_files/CK_Gmin_rs_{:}.csv'.format(int(rs))
        if path.isfile(ckgmf):
            tdat = np.genfromtxt(ckgmf,delimiter=',',skip_header=1)
            got_QMC_dat = True

        fig, ax = plt.subplots(2,1,figsize=(5,7.5))

        kf = rs_to_kf/rs
        kf2 = kf*kf
        kl2 = xlp**2#(xlp*kf)**2
        a,b,c = get_g_minus_pars(rs,0.)

        if got_QMC_dat:
            tq = tdat[:,0]#*kf
            ax[0].errorbar(tdat[:,0],tdat[:,1],yerr=tdat[:,2],color='k',\
                markersize=3,marker='o',linewidth=0,elinewidth=1.5)

            ax[1].errorbar(tdat[:,0],4.*pi*tdat[:,1]/tq**2, \
                yerr=4.*pi*tdat[:,2]/tq**2, color='k',\
                markersize=3,marker='o',linewidth=0,elinewidth=1.5)

        ax[0].plot(xl,a*xl**2,color=colors[1],linestyle=':',\
            label='SQE')#'$A_-(r_\\mathrm{s})(q/k_\\mathrm{F})^2$')
        ax[1].plot(xlp,4.*pi*a*np.ones_like(xlp),color=colors[1],\
            linestyle=':', label='SQE')#'$A_-(r_\\mathrm{s})(q/k_\\mathrm{F})^2$')

        ax[0].plot(xl,c*xl**2 + b,color=colors[2],linestyle=':',\
            label='LQE')#'$C(r_\\mathrm{s})(q/k_\\mathrm{F})^2 + B_-(r_\\mathrm{s})$')
        ax[1].plot(xlp,4.*pi*(c/kf2 + b/kl2**2),color=colors[2],linestyle=':',\
            label='LQE')#'$C(r_\\mathrm{s})(q/k_\\mathrm{F})^2 + B_-(r_\\mathrm{s})$')

        gmapp = gminus_CK(xl*kf,rs,0.)
        ax[0].plot(xl,gmapp,color=colors[0], label='This work')
        gmapp_oq2 = 4.*pi*gminus_CK(xlp*kf,rs,0.)/kl2
        ax[1].plot(xlp,gmapp_oq2,color=colors[0],label='This work')

        ax[0].plot(xl,g_minus_ra(xl*kf,0.,rs),color=colors[3],linestyle='-',\
            label='Richardson-Ashcroft')
        ax[1].plot(xlp,4.*pi*g_minus_ra(xlp*kf,0.,rs)/kl2, color=colors[3], \
            linestyle='-', label='Richardson-Ashcroft')

        for iplt in range(2):
            ax[iplt].set_xlim(xl.min(),xl.max())
        ax[1].set_xlabel('$q/k_\\mathrm{F}$',fontsize=12)
        ax[0].set_ylabel('$G_-(q)$',fontsize=12)
        ax[1].set_ylabel('$ 4 \\pi \\, G_-(q) (k_\\mathrm{F}/q)^2$', fontsize=12)
        ax[0].set_ylim(0.,1.1*gmapp.max())
        ax[1].set_ylim(0.,1.1*gmapp_oq2.max())

        #ax[0].legend(fontsize=10,title='$r_\\mathrm{s}'+'={:}$'.format(rs),\
        #    title_fontsize=18,ncol = 4,loc=(0.5,1.01))
        if rs in [0.1,1,2,100]:
            ileg = 0
            tcoord = (0.01,0.05)
        else:
            ileg = 1
            tcoord = (0.9,0.05)
        ax[ileg].legend(fontsize=10,title='$r_\\mathrm{s}'+'={:}$'.format(rs),\
            title_fontsize=12,frameon=False)

        ax[0].annotate('(a)',(0.01,0.9),fontsize=16,xycoords='axes fraction')
        ax[1].annotate('(b)',tcoord,fontsize=16,xycoords='axes fraction')

        #plt.show() ; exit()
        plt.savefig('./figs/gminus_rs_{:}_2p.pdf'.format(rs), dpi=600,\
            bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()

    return

if __name__ == "__main__":

    gplus_plots()
    #gminus_plots()
