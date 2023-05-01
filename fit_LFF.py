import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from os import path, system, sys

from asymptotics import get_g_plus_pars, get_g_minus_pars
from ra_lff import g_plus_ra, g_minus_ra
from g_corradini import g_corradini

#plt.rcParams.update({'text.usetex': True, 'font.family': 'dejavu'})
pi = np.pi
rs_to_kf = (9*pi/4.)**(1./3.)

vtow = {'+': 'plus', '-': 'minus'}

def smooth_step(x,a,b):
    f1 = np.exp(a*b)
    f2 = np.exp(-a*x)
    f = (f1 - 1.)*f2/(1. + (f1 - 2.)*f2)
    return f

def simple_LFF(q,rs,c,var,init=False):

    kf = rs_to_kf/rs
    q2 = (q/kf)**2
    q4 = q2*q2

    if var == '+':
        CA, CB, CC = get_g_plus_pars(rs)
    elif var == '-':
        CA, CB, CC = get_g_minus_pars(rs,0.)

    if init:
        alpha = c[0]
        beta = c[1]
        gamma = c[2]
    else:
        alpha = c[0] + c[1]*np.exp(-abs(c[2])*rs)
        beta = c[3]
        gamma = c[4]

    interp1 = smooth_step(q4/16.,beta,gamma)
    interp2 = 1. - interp1

    asymp1 = q2*(CA + alpha*q4)
    asymp2 = CB + CC*q2
    LFF = asymp1*interp1 + asymp2*interp2

    return LFF

figdir = './figs_from_fit/'
pardir = './fitted_LFF_pars/'
for tdir in [figdir,pardir]:
    if not path.isdir(tdir):
        system('mkdir ' + tdir)


def main_fit(rs_l,ips0,var):

    Nps = len(ips0)

    tdat = {}
    tdat_CK = {}
    tdat_MCS = {}
    npts = 0

    sgnstr = vtow[var]

    for irs, rs in enumerate(rs_l):
        CKLFF = './data_files/CK_G{:}_rs_{:}.csv'.format(sgnstr,int(rs))

        if path.isfile(CKLFF):
            tdat_CK[rs] = np.genfromtxt(CKLFF,delimiter=',',skip_header=1)
            if rs in tdat:
                tdat[rs] = np.vstack((tdat[rs],tdat_CK[rs]))
            else:
                tdat[rs] = tdat_CK[rs].copy()
            npts += tdat_CK[rs].shape[0]

        if var == '+':
            mcsgpf = './data_files/MCS_Gplus_rs_{:}.csv'.format(int(rs))
            if path.isfile(mcsgpf):
                tdat_MCS[rs] = np.genfromtxt(mcsgpf,delimiter=',',skip_header=1)
                if rs in tdat:
                    continue
                    tdat[rs] = np.vstack((tdat[rs],tdat_MCS[rs]))
                else:
                    tdat[rs] = tdat_MCS[rs].copy()
                    npts += tdat_MCS[rs].shape[0]

    zl = np.linspace(0.0,4.0,1000)

    def fobj(c):
        fres = np.zeros(npts+1)
        tpts = 0
        for rs in rs_l:
            kf = rs_to_kf/rs
            if rs in tdat:
                LFF = simple_LFF(tdat[rs][:,0]*kf, rs, c, var, init=False)
                fres[tpts:tpts+LFF.shape[0]] = (LFF - tdat[rs][:,1])/tdat[rs][:,2]
                tpts += LFF.shape[0]
            else:
                LFF = simple_LFF(zl*kf,rs,c,var)
            fres[-1] += len(LFF[LFF<0.])
        return fres

    ips = ips0.copy()
    #for i in range(5):
    res = least_squares(fobj,ips)
    ips = (res.x).copy()

    tstr = ''
    for ipar in range(Nps):
        lchar = ', '
        if ipar == Nps - 1:
            lchar = ' \n'
        tstr += 'c{:}{:}'.format(ipar,lchar)

    tstr_tex = ''
    for ipar, apar in enumerate(ips):
        #tstr += 'c_{:}, {:.6e} \n'.format(ipar,apar)
        tmpstr = '{:.6e}'.format(apar)
        fac, exp = tmpstr.split('e')
        iexp = int(exp)
        nfac = 6
        if iexp < -1:
            nfac -= iexp + 1
        tstr_tex += 'c_{:}'.format(ipar)
        tmpstr = ('{:.' + '{:}'.format(nfac) + 'f}').format(apar)

        lchar = ', '
        if ipar == Nps - 1:
            lchar = ' \n'
        tstr += tmpstr + lchar

        tstr_tex += ' &= ' + tmpstr + ' \\\\ \n'

    #print(tstr)
    with open(pardir + 'g{:}_pars.csv'.format(sgnstr),'w+') as tfl:
        tfl.write(tstr)

    with open(pardir + 'g{:}_pars.tex'.format(sgnstr),'w+') as tfl:
        tfl.write(tstr_tex)


    print('SSR = {:}'.format(np.sum(fobj(ips)**2)))

    colors = ['darkblue','darkorange','tab:green','darkred','darkslategray']

    for irs, rs in enumerate(rs_l):

        fig, ax = plt.subplots(figsize=(6,4))

        kf = rs_to_kf/rs
        if var == '+':
            a,b,c = get_g_plus_pars(rs)
        elif var == '-':
            a,b,c = get_g_minus_pars(rs,0.)

        if rs in tdat_CK:
            ax.errorbar(tdat_CK[rs][:,0],tdat_CK[rs][:,1],yerr=tdat_CK[rs][:,2],color='k',\
                markersize=3,marker='o',linewidth=0,elinewidth=1.5)
        if rs in tdat_MCS:
            ax.errorbar(tdat_MCS[rs][:,0],tdat_MCS[rs][:,1],yerr=tdat_MCS[rs][:,2],color='m',\
                markersize=3,marker='o',linewidth=0,elinewidth=1.5)

        ax.plot(zl,a*zl**2,color=colors[1],linestyle=':',\
            label='SQE')
        ax.plot(zl,c*zl**2+b,color=colors[2],linestyle=':',\
            label='LQE')

        new_LFF = simple_LFF(zl*kf,rs,ips,var)
        ax.plot(zl,new_LFF,color=colors[0],label='This work')

        if var == '+':
            RA_LFF = g_plus_ra(zl*kf,0.,rs)

            gcorr = g_corradini(zl*kf,\
                {'rs': rs, 'kF': kf, 'n': 3./(4.*pi*rs**3), 'rsh': rs**(0.5)})
            ax.plot(zl,gcorr,\
                color=colors[4],linestyle='-.',\
                label=r'Corradini $\mathrm{\it et \, al.}$')

        elif var == '-':
            RA_LFF = g_minus_ra(zl*kf,0.,rs)

        ax.plot(zl,RA_LFF,color=colors[3],linestyle='--',\
            label='Richardson-Ashcroft')

        ax.set_xlim(zl.min(),zl.max())
        ax.set_xlabel('$q/k_\\mathrm{F}$',fontsize=12)
        ax.set_ylabel('$G_'+var+'(q)$',fontsize=12)
        ax.set_ylim(0.,1.1*new_LFF.max())

        ax.legend(fontsize=10,\
            title='$r_\\mathrm{s}'+'={:}$'.format(rs),\
            title_fontsize=18,frameon=False)

        plt.savefig(figdir + 'g{:}_rs_{:}.pdf'.format(\
            sgnstr, rs), dpi=600,bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()

    return

def init_fit(rs_l,var):

    tdat = {}
    tdat_CK = {}
    tdat_MCS = {}
    npts = 0

    tstr = 'rs, c0, c1, c2 \n'
    zl = np.linspace(0.0,4.0,1000)

    sgnstr = vtow[var]

    for irs, rs in enumerate(rs_l):
        CKLFF = './data_files/CK_G{:}_rs_{:}.csv'.format(sgnstr,int(rs))

        if path.isfile(CKLFF):
            tdat_CK[rs] = np.genfromtxt(CKLFF,delimiter=',',skip_header=1)
            if rs in tdat:
                tdat[rs] = np.vstack((tdat[rs],tdat_CK[rs]))
            else:
                tdat[rs] = tdat_CK[rs].copy()
            npts += tdat_CK[rs].shape[0]

        if var == '+':
            mcsgpf = './data_files/MCS_Gplus_rs_{:}.csv'.format(int(rs))
            if path.isfile(mcsgpf):
                tdat_MCS[rs] = np.genfromtxt(mcsgpf,delimiter=',',skip_header=1)
                if rs in tdat:
                    tdat[rs] = np.vstack((tdat[rs],tdat_MCS[rs]))
                else:
                    tdat[rs] = tdat_MCS[rs].copy()
                npts += tdat_MCS[rs].shape[0]

        def fobj(c):
            fres = np.zeros(npts+1)
            tpts = 0
            for rs in rs_l:#tdat:
                kf = (9*pi/4.)**(1./3.)/rs
                if rs in tdat:
                    LFF = simple_LFF(tdat[rs][:,0]*kf,rs,c,var,init=True)
                    fres[tpts:tpts+LFF.shape[0]] = (LFF - tdat[rs][:,1])/tdat[rs][:,2]
                    tpts += LFF.shape[0]
                else:
                    LFF = simple_LFF(zl*kf,rs,c,var,init=True)
                fres[-1] += len(LFF[LFF<0.])
            return fres

        res = least_squares(fobj,[.8,1.7,0.])
        tstr += ('{:}, '*3 + '{:}\n').format(rs,*res.x)

    with open(pardir + 'optpars_g'+sgnstr+'.csv','w+') as tfl:
        tfl.write(tstr)

    return

def manip(rs,var):

    from matplotlib.widgets import Slider

    zl = np.linspace(0.0,4.0,1000)

    tdat_CK = {}
    tdat_MCS = {}

    sgnstr = vtow[var]
    CKLFF = './data_files/CK_G{:}_rs_{:}.csv'.format(sgnstr,int(rs))
    if path.isfile(CKLFF):
        tdat_CK[rs] = np.genfromtxt(CKLFF,delimiter=',',skip_header=1)

    if var == '+':
        mcsgpf = './data_files/MCS_Gplus_rs_{:}.csv'.format(int(rs))
        if path.isfile(mcsgpf):
            tdat_MCS[rs] = np.genfromtxt(mcsgpf,delimiter=',',skip_header=1)

    fig, ax = plt.subplots(figsize=(6,6))

    fig.subplots_adjust(bottom=0.25)

    kf = rs_to_kf/rs

    if var == '+':
        a,b,c = get_g_plus_pars(rs)
    elif var == '-':
        a,b,c = get_g_minus_pars(rs,0.)

    if rs in tdat_CK:
        ax.errorbar(tdat_CK[rs][:,0],tdat_CK[rs][:,1],yerr=tdat_CK[rs][:,2],color='k',\
            markersize=3,marker='o',linewidth=0,elinewidth=1.5)
    if rs in tdat_MCS:
        ax.errorbar(tdat_MCS[rs][:,0],tdat_MCS[rs][:,1],yerr=tdat_MCS[rs][:,2],color='m',\
            markersize=3,marker='o',linewidth=0,elinewidth=1.5)

    ax.plot(zl,a*zl**2,color='darkorange',linestyle='--')
    ax.plot(zl,c*zl**2+b,color='tab:green',linestyle='-.')

    a0 = 0.05
    b0 = 0.75
    g0 = 2.58
    twrap = lambda ps : simple_LFF(zl*kf,rs,ps,var,init=True)
    line, = ax.plot(zl,twrap([a0,b0,g0]),color='darkblue')
    #line, = ax.plot(zl,gplus_zeropar(zl*kf,rs,gamma=a0),color='darkorange')

    ax.set_xlim(zl.min(),zl.max())
    ax.set_xlabel('$q/k_\\mathrm{F}$',fontsize=12)
    ax.set_ylabel('$G_'+var+'(q)$',fontsize=12)
    ax.set_ylim(0.,2.0)

    a_ax = fig.add_axes([0.15, 0.12, 0.65, 0.03])
    a_adj = Slider(
        ax=a_ax,
        label='$\\alpha$',
        valmin=-6.0,
        valmax=6.0,
        valinit=a0
    )

    b_ax = fig.add_axes([0.15, 0.08, 0.65, 0.03])
    b_adj = Slider(
        ax=b_ax,
        label='$\\beta$',
        valmin=0.0,
        valmax=6.0,
        valinit=b0
    )

    g_ax = fig.add_axes([0.15, 0.04, 0.65, 0.03])
    g_adj = Slider(
        ax=g_ax,
        label='$\\gamma$',
        valmin=0.0,
        valmax=6.0,
        valinit=g0
    )

    def update_plot(val):
        line.set_ydata(twrap([a_adj.val,b_adj.val,g_adj.val]))
        fig.canvas.draw_idle()

    a_adj.on_changed(update_plot)
    b_adj.on_changed(update_plot)
    g_adj.on_changed(update_plot)


    plt.show() ; exit()


if __name__ == "__main__":

    uargs = {'routine': None, 'var': None}
    if len(sys.argv) < 1 + len(uargs.keys()):
        qstr = 'Need to specify:\n'
        for akey in uargs:
            qstr += '  ' + akey + '\n'
        raise SystemExit(qstr)

    for anarg in sys.argv[1:]:
        tkey, tval = anarg.split('=')
        uargs[tkey.lower()] = tval.lower()

    if uargs['routine'] == 'main':
        rs_l = [1.e-6,0.01,0.1,1,2,3,4,5,10,69,100]
        if uargs['var'] == '+':
            ips = [-0.00365479, 0.0215642, 0.182898, 4.5, 1.2]
        elif uargs['var'] == '-':
            ips = [-0.00456264, 0.0261967, 0.338185, 0.65, 1.8]
        main_fit(rs_l,ips,uargs['var'])

    elif uargs['routine'] == 'init':
        if uargs['var'] == '+':
            rs_l = [1,2,5,10]
        elif uargs['var'] == '-':
            rs_l = [1,2,3,4,5]
        init_fit(rs_l,uargs['var'])

    elif uargs['routine'] == 'manip':

        manip(float(uargs['rs']))
