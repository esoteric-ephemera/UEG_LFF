import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from AKCK_LFF.g_corradini import g_corradini
from AKCK_LFF.fitted_LFF import g_plus_new, g_minus_new
from AKCK_LFF.ra_lff import g_plus_ra, g_minus_ra

plt.rcParams.update({'text.usetex': True, 'font.family': 'dejavu'})

def surf_plots():
    rs_l = np.linspace(1.,10.,5000)
    kf_l = (9*np.pi/4.)**(1./3.)/rs_l
    x_l = np.linspace(0.01,4.,2001)

    x, rs = np.meshgrid(x_l, rs_l)
    q = x.copy()
    dv = {'rs': rs.copy(), 'rsh': rs**(0.5), 'kF': (9*np.pi/4.)**(1./3.)/rs,
        'n' : 3./(4.*np.pi*rs**3)}
    for irs in range(rs_l.shape[0]):
        q[irs,:] *= kf_l[irs]

    for ig in range(3):
        fig, ax = plt.subplots(1,2,figsize=(8,4),subplot_kw={"projection": "3d"})

        if ig == 2:
            gfn = g_minus_new(q,rs)
            gsymb = '-'
            fsymb = 'm'
            ym = 2.5
        else:
            gfn = g_plus_new(q,rs)
            gsymb = '+'
            fsymb = 'p'
            ym = 4.

        tfac = 4*np.pi/x**2
        ax[0].plot_surface(x, rs, tfac*gfn, cmap=cm.viridis,\
            linewidth=0, antialiased=False,rasterized=True)

        ax[0].text(-.6,6.2,1.25*ym,'(a) This work',fontsize=14)


        if ig == 0:
            gfn = g_corradini(q,dv)
            tlab = '(b) Corradini et al.'
            modstr = 'corr'
        elif ig == 1:
            gfn = g_plus_ra(q,0.,rs)
            tlab = '(b) Static RA'
            modstr = 'RAS'
        elif ig == 2:
            gfn = g_minus_ra(q,0.,rs)
            tlab = '(b) Static RA'
            modstr = 'RAS'

        ax[1].plot_surface(x, rs, tfac*gfn, cmap=cm.viridis,\
            linewidth=0, antialiased=False,rasterized=True)
        ax[1].text(-.6,6.2,1.25*ym, tlab,fontsize=14)

        for iax in range(2):
            ax[iax].set_xlabel('$q/k_\\mathrm{F}$',fontsize=14)
            ax[iax].set_ylabel('$r_\\mathrm{s}$',fontsize=14)
            #ax[iax].set_zlabel('$4\\pi \\, G_'+gsymb+'(r_\\mathrm{s},q)(k_\\mathrm{F}/q)^2$',fontsize=14)
            ax[iax].set_zlabel('$4\\pi (k_\\mathrm{F}/q)^2 G_'+gsymb+'$',fontsize=14)
            ax[iax].view_init(elev=20,azim=-70)
            ax[iax].set_zlim(0.,ym)

        #plt.show() ; exit()

        plt.savefig('./figs/g{:}_{:}.pdf'.format(fsymb,modstr), \
            dpi=600, bbox_inches='tight')

        plt.cla()
        plt.clf()
        plt.close()

    return

if __name__ == "__main__":

    surf_plots()
