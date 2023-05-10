import numpy as np
import matplotlib.pyplot as plt
from os import path, system

from AKCK_LFF.fitted_LFF import g_plus_new, g_minus_new
from AKCK_LFF.asymptotics import get_g_minus_pars, get_g_plus_pars
from AKCK_LFF.ra_lff import  g_plus_ra, g_minus_ra
from AKCK_LFF.g_corradini import g_corradini
#from AKCK_LFF.mcp07_static import mcp07_static

from AKCK_LFF.global_plot_pars import colors, lsls

pi = np.pi
rs_to_kf = (9*pi/4.)**(1./3.)


rdir = path.dirname(path.realpath(__file__)) + '/'

if not path.isdir('./figs/'):
    system('mkdir ./figs')

def gplus_plots():

    xl = np.linspace(0.0,4.0,5000)
    xlp = xl[1:]

    rs_l = [0.1,1,2,5,10,100]

    for irs, rs in enumerate(rs_l):

        fig, ax = plt.subplots(2,1,figsize=(5,7.5))

        kf = rs_to_kf/rs

        ckgpf = rdir + '/data_files/CK_Gplus_rs_{:}.csv'.format(int(rs))
        if path.isfile(ckgpf):

            tdat_CK = np.genfromtxt(ckgpf,delimiter=',',skip_header=1)

            tq2_CK = tdat_CK[:,0]**2#(tdat_CK[:,0]*kf)**2
            ax[0].errorbar(tdat_CK[:,0],tdat_CK[:,1],yerr=tdat_CK[:,2],\
                color=colors['CKKC'],\
                markersize=3,marker='o',linewidth=0,elinewidth=1.5)

            ax[1].errorbar(tdat_CK[:,0],4.*pi*tdat_CK[:,1]/tq2_CK, \
                yerr=4.*pi*tdat_CK[:,2]/tq2_CK, color=colors['CKKC'],\
                markersize=3,marker='o',linewidth=0,elinewidth=1.5)

        mcsgpf = rdir + '/data_files/MCS_Gplus_rs_{:}.csv'.format(int(rs))
        if path.isfile(mcsgpf):

            tdat_MCS = np.genfromtxt(mcsgpf,delimiter=',',skip_header=1)

            tq2_MCS = tdat_MCS[:,0]**2#(tdat_MCS[:,0]*kf)**2

            ax[0].errorbar(tdat_MCS[:,0],tdat_MCS[:,1], \
                yerr=tdat_MCS[:,2],color=colors['MCS'],\
                markersize=3,marker='s',linewidth=0,elinewidth=1.5)

            ax[1].errorbar(tdat_MCS[:,0],4.*pi*tdat_MCS[:,1]/tq2_MCS, \
                yerr=4.*pi*tdat_MCS[:,2]/tq2_MCS, color=colors['MCS'], \
                markersize=3,marker='s',linewidth=0,elinewidth=1.5)

        kf2 = kf*kf
        kl2 = xlp**2#(xlp*kf)**2
        a,b,c = get_g_plus_pars(rs)

        ax[0].plot(xl,a*xl**2,color=colors['SQE'],linestyle=lsls['SQE'],\
            label='SQE')
        ax[1].plot(xlp,4.*pi*a*np.ones_like(xlp),\
            color=colors['SQE'],linestyle=lsls['SQE'], label='SQE')

        ax[0].plot(xl,c*xl**2 + b, \
            color=colors['LQE'], linestyle=lsls['LQE'], label='LQE')
        ax[1].plot(xlp,4.*pi*(c + b/kl2),\
            color=colors['LQE'],linestyle=lsls['LQE'],label='LQE')

        gpapp = g_plus_new(xl*kf,rs)
        ax[0].plot(xl,gpapp,color=colors['NEW'], label='This work',\
            linestyle=lsls['NEW'])

        gpapp_oq2 = 4.*pi*g_plus_new(xlp*kf,rs)/kl2
        ax[1].plot(xlp,gpapp_oq2,color=colors['NEW'],\
            label='This work',linestyle=lsls['NEW'])

        gp_ra = g_plus_ra(xl*kf,0.,rs)
        ax[0].plot(xl,gp_ra,color=colors['RA'],linestyle=lsls['RA'],\
            label='Richardson-Ashcroft')

        gp_ra_oq2 = 4.*pi*g_plus_ra(xlp*kf,0.,rs)/kl2
        ax[1].plot(xlp,gp_ra_oq2,color=colors['RA'],linestyle=lsls['RA'],\
            label='Richardson-Ashcroft')

        dens_d = {'rs': rs, 'kF': kf, 'n': 3./(4.*pi*rs**3), 'rsh': rs**(0.5)}
        gcorr = g_corradini(xl*kf,dens_d)
        ax[0].plot(xl,gcorr, color=colors['COR'],linestyle=lsls['COR'],\
            label=r'Corradini $\mathrm{\it et \, al.}$')
        ax[1].plot(xlp,4.*pi*g_corradini(xlp*kf,dens_d)/kl2, color=colors['COR'],\
            linestyle=lsls['COR'], label=r'Corradini $\mathrm{\it et \, al.}$')

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

        if abs(rs - 0.1) < 1.e-15:
            ymax0 = 1.1*max(gpapp.max(),gp_ra.max())
            ymax1 = 5.
        elif abs(rs - 100.) < 1.e-15:
            ymax0 = 1.1*gpapp.max()
            ymax1 = 5.6
        elif rs <= 10.:
            ymax0 = 1.1*max(gpapp.max(),gp_ra.max())
            ymax1 = 1.1*max(gpapp_oq2.max(),gp_ra_oq2.max())
        else:
            ymax0 = 1.1*gpapp.max()
            ymax1 = 1.1*gpapp_oq2.max()

        ax[0].set_ylim(0.,ymax0)
        ax[1].set_ylim(0.,ymax1)

        if rs in [1,2]:
            ileg = 0
            tcoord = (0.01,0.05)
        elif rs in [0.1]:
            ileg = 1
            tcoord = (0.9,0.9)
        #elif rs in [100]:
        #    ileg = 1
        #    tcoord = (0.9, 0.05)
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


def gminus_plots(acpars='PW92'):

    xl = np.linspace(0.0,4.0,5000)
    xlp = xl[1:]

    rs_l = [0.1,1,2,3,4,5,100]

    for irs, rs in enumerate(rs_l):

        got_QMC_dat = False
        ckgmf = rdir + '/data_files/CK_Gminus_rs_{:}.csv'.format(int(rs))
        if path.isfile(ckgmf):
            tdat = np.genfromtxt(ckgmf,delimiter=',',skip_header=1)
            got_QMC_dat = True

        fig, ax = plt.subplots(2,1,figsize=(5,7.5))

        kf = rs_to_kf/rs
        kf2 = kf*kf
        kl2 = xlp**2#(xlp*kf)**2
        a,b,c = get_g_minus_pars(rs,acpars=acpars)

        if got_QMC_dat:
            tq = tdat[:,0]#*kf
            ax[0].errorbar(tdat[:,0],tdat[:,1],yerr=tdat[:,2],\
                color=colors['CKKC'],\
                markersize=3,marker='o',linewidth=0,elinewidth=1.5)

            ax[1].errorbar(tdat[:,0],4.*pi*tdat[:,1]/tq**2, \
                yerr=4.*pi*tdat[:,2]/tq**2, color=colors['CKKC'],\
                markersize=3,marker='o',linewidth=0,elinewidth=1.5)

        ax[0].plot(xl,a*xl**2,color=colors['SQE'],linestyle=lsls['SQE'],\
            label='SQE')
        ax[1].plot(xlp,4.*pi*a*np.ones_like(xlp),color=colors['SQE'],\
            linestyle=lsls['SQE'], label='SQE')

        ax[0].plot(xl,c*xl**2 + b,color=colors['LQE'],linestyle=lsls['LQE'],\
            label='LQE')
        ax[1].plot(xlp,4.*pi*(c + b/kl2), color=colors['LQE'],\
            linestyle=lsls['LQE'], label='LQE')

        gmapp = g_minus_new(xl*kf,rs,acpars=acpars)
        ax[0].plot(xl,gmapp,color=colors['NEW'], label='This work',\
            linestyle = lsls['NEW'])
        gmapp_oq2 = 4.*pi*g_minus_new(xlp*kf,rs,acpars=acpars)/kl2
        ax[1].plot(xlp,gmapp_oq2,color=colors['NEW'], label='This work',\
            linestyle = lsls['NEW'])

        gm_ra = g_minus_ra(xl*kf,0.,rs)
        ax[0].plot(xl,gm_ra,color=colors['RA'],linestyle=lsls['RA'],\
            label='Richardson-Ashcroft')

        gm_ra_oq2 = 4.*pi*g_minus_ra(xlp*kf,0.,rs)/kl2
        ax[1].plot(xlp,gm_ra_oq2, color=colors['RA'], \
            linestyle=lsls['RA'], label='Richardson-Ashcroft')

        for iplt in range(2):
            ax[iplt].set_xlim(xl.min(),xl.max())
        ax[1].set_xlabel('$q/k_\\mathrm{F}$',fontsize=12)
        ax[0].set_ylabel('$G_-(q)$',fontsize=12)
        ax[1].set_ylabel('$ 4 \\pi \\, G_-(q) (k_\\mathrm{F}/q)^2$', fontsize=12)

        if rs <= 10.:
            ymax0 = 1.1*max(gmapp.max(),gm_ra.max())
            ymax1 = 1.1*max(gmapp_oq2.max(),gm_ra_oq2.max())
        else:
            ymax0 = 1.1*gmapp.max()
            ymax1 = 1.1*gmapp_oq2.max()

        ax[0].set_ylim(0.,ymax0)
        ax[1].set_ylim(0.,ymax1)

        #ax[0].legend(fontsize=10,title='$r_\\mathrm{s}'+'={:}$'.format(rs),\
        #    title_fontsize=18,ncol = 4,loc=(0.5,1.01))
        if rs in [1,100]:
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
        lstr = ''
        if acpars == 'AKCK':
            lstr = '_AKCK_ac'
        plt.savefig('./figs/gminus_rs_{:}_2p{:}.pdf'.format(rs,lstr), dpi=600,\
            bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()

    return


if __name__ == "__main__":

    gplus_plots()
    gminus_plots()
