import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from os import path, system, sys

from asymptotics import get_g_plus_pars, gexc2
from ra_lff import g_plus_ra
from g_corradini import g_corradini
from mcp07_static import mcp07_static

#plt.rcParams.update({'text.usetex': True, 'font.family': 'dejavu'})
pi = np.pi

rs_to_kf = (9*pi/4.)**(1./3.)

def gplus_corradini(q,rs,z,c):

    kf = rs_to_kf/rs
    rsh = rs**(0.5)
    Q2 = (q/kf)**2

    Apos, Bpos, C = get_g_plus_pars(rs)

    g = Bpos/(Apos - C)
    alpha = c[0]#*Apos/(rsh**(0.5)*Bpos*g)
    beta = c[1]#/(Bpos*g)

    Gplus = C*Q2 + Bpos*Q2/(g + Q2) + alpha*Q2**2 *np.exp(-beta*Q2)

    return Gplus

def log_func(z):
    return (z**2 - 1.)/(2.*z)*np.log(np.abs((1. - z)/(1. + z)))

def gplus_UI_new(q,rs,z,c):

    kf = rs_to_kf/rs
    q2 = (q/kf)**2
    q4 = q2*q2
    q6 = q4*q2

    Apos, Bpos, Cpos = get_g_plus_pars(rs)

    cc = c[0] + c[1]*np.exp(-c[2]*rs)
    ct = 1./64. + c[3]*np.exp(-c[4]*rs)
    bt = 1./4. + c[5]*np.exp(-c[4]*rs)
    #ct = (1./64. + c[3]*rs + c[5]/64.*rs**2)/(1. + c[4]*rs + c[5]*rs**2)
    #bt = (-1./4. + c[6]*rs - c[8]/4.*rs**2)/(1. + c[7]*rs + c[8]*rs**2)
    a = (Bpos*ct - bt*cc)/(2.*ct)
    b = (Apos + Cpos - a*bt)/2.

    alp = -a
    bet = b - Cpos

    interp = (1. - ct*q6)/(1. + bt*q2 + ct*q6)
    #interp = (1. - (q/(2.*kf))**6)/(1. - (q/(2.*kf))**2 + (q/(2.*kf))**6)

    gmin = a + b*q2 + cc*q4 + (alp + bet*q2 + cc*q4)*interp

    return gmin

def gplus_UI(q,rs,z,c):

    kf = rs_to_kf/rs
    x = q/(2*kf)
    x2 = x*x
    x4 = x2*x2
    x6 = x4*x2

    Apos, Bpos, C = get_g_plus_pars(rs)

    ca = c[0] + c[1]*np.exp(-c[2]*rs**2)
    cb = -4.*ca/15 + 9.*Apos/4. - 3.*Bpos/16. + 7.*C/4.
    cc = -ca/5. - 3.*Apos/4. + 9.*Bpos/16. + 3.*C/4.
    calp = ca
    cbet = 2.*ca/5. + 9.*Apos/4. - 3.*Bpos/16. - 9.*C/4.
    cgam = -cc

    logfun = np.zeros_like(x)
    logfun[x == 0.] = 1.
    xmsk = (0. < x) & (x < 1.)
    logfun[xmsk] = log_func(x[xmsk])

    xmsk = 1. < x
    logfun[xmsk] = log_func(x[xmsk])

    gmin = ca*x4 + cb*x2 + cc + (calp*x4 + cbet*x2 + cgam)*logfun
    return gmin

def gplus_CK(q,rs,z,c,init=False):
    kf = rs_to_kf/rs
    q2 = (q/kf)**2
    q4 = q2**2
    q6 = q2**3
    q8 = q2**4
    Apos, Bpos, Cpos = get_g_plus_pars(rs)

    if init:
        alpha = c[0]
        beta = c[1]
        gamma = c[2]

    else:
        alpha = c[0]*np.exp(-abs(c[1])*rs)# + c[1]*np.exp(-abs(c[2])*rs)
        beta = c[2] + c[3]*np.exp(-abs(c[4])*rs)
        gamma = c[5] + c[6]*np.exp(-abs(c[7])*rs)

    Dpos = -(3.*pi**2)**(4./3.)*gexc2(rs)/(2.*pi)

    interp1 = np.exp(-beta*q8/256.)
    interp2 = -np.expm1(-gamma*q8/256.)

    asymp1 = q2*(Apos + Dpos*q2 + alpha*q2*q2)
    asymp2 = (Bpos + Cpos*q2)
    gplus = asymp1*interp1 + asymp2*interp2

    return gplus

"""
def gplus_zeropar(q,rs):

    kf = rs_to_kf/rs
    q2 = (q/kf)**2

    Apos, Bpos, C = get_g_plus_pars(rs)
    Dpos = -(3.*pi**2)**(4./3.)*gexc2(rs)/(2.*pi)

    beta = ((Apos**2 - C**2 + ((Apos**2 - C**2)**2 + 4*Apos*Bpos*C*Dpos)**(0.5))/(2.*Apos*Bpos))**2
    betah = beta**0.5
    alpha = 2./Apos*(betah*C - Dpos)

    gplus = q2*(Apos + betah*C*q2)/(1. + q2*(alpha + q2*beta))**(0.5)
    return gplus
"""
def gplus_zeropar(q,rs,gamma=1.):

    kf = rs_to_kf/rs
    q2 = (q/kf)**2

    Apos, Bpos, C = get_g_plus_pars(rs)
    Dpos = -(3.*pi**2)**(4./3.)*gexc2(rs)/(2.*pi)
    #print(Bpos/())
    beta = (Apos - C)/Bpos
    alpha = Apos - beta*gamma
    delta = C - alpha

    #gplus = Apos*q2 + beta*q2/(1. + beta*q2)*(gamma + delta*q2)
    return q2*(Apos + q2*(Dpos + q2*gamma))

def check_poles(c,ubd=None):
    # simple routine to check for poles in denominator of rational polynomial
    # assumes c are the coefficients of a polynomial P_n(x), where n=len(c)
    # P_n(x) = Sum_{i=0}^n c[i]*x^i
    tmp_rts = np.polynomial.polynomial.polyroots(c)
    no_pos_roots = True

    rmax = np.inf
    if ubd is not None:
        rmax = ubd

    for rt in tmp_rts:
        if rt.imag == 0.0 and rt > 0.0 and rt <= rmax:
            no_pos_roots = False
            break
    return no_pos_roots

if not path.isdir('./figs/'):
    system('mkdir ./figs')

def main_fit(rs_l,ips0):

    Nps = len(ips0)

    tdat = {}
    tdat_CK = {}
    tdat_MCS = {}
    npts = 0

    for irs, rs in enumerate(rs_l):
        ckgpf = './data_files/CK_Gplus_rs_{:}.csv'.format(int(rs))
        mcsgpf = './data_files/MCS_Gplus_rs_{:}.csv'.format(int(rs))
        if path.isfile(ckgpf):
            tdat_CK[rs] = np.genfromtxt(ckgpf,delimiter=',',skip_header=1)
            if rs in tdat:
                tdat[rs] = np.vstack((tdat[rs],tdat_CK[rs]))
            else:
                tdat[rs] = tdat_CK[rs].copy()
            npts += tdat_CK[rs].shape[0]

        if path.isfile(mcsgpf):
            tdat_MCS[rs] = np.genfromtxt(mcsgpf,delimiter=',',skip_header=1)
            """
            tdat_wgt = tdat_MCS[rs].copy()
            #tdat_wgt[:,2] *= 4
            if rs in tdat:
                continue
                #tdat[rs] = np.vstack((tdat[rs],tdat_MCS[rs]))
                tdat[rs] = np.vstack((tdat[rs],tdat_wgt))
            else:
                tdat[rs] = tdat_wgt.copy()
                #tdat[rs] = tdat_MCS[rs].copy()
            npts += tdat_MCS[rs].shape[0]
            #"""


    zl = np.linspace(0.0,4.0,1000)

    def fobj(c):
        fres = np.zeros(npts+1)
        tpts = 0
        for rs in rs_l:
            kf = (9*pi/4.)**(1./3.)/rs
            if rs in tdat:
                gp = gplus_CK(tdat[rs][:,0]*kf,rs,0.,c)
                #gm = gminus_ra_new(tdat[rs][:,0]*kf,rs,0.,c)
                fres[tpts:tpts+gp.shape[0]] = (gp - tdat[rs][:,1])/tdat[rs][:,2]
                #fres[tpts+gm.shape[0]:tpts+2*gm.shape[0]] = \
                #    fres[tpts:tpts+gm.shape[0]]/tdat[rs][:,0]**2
                tpts += gp.shape[0]
            else:
                gp = gplus_CK(zl*kf,rs,0.,c)
            fres[-1] += len(gp[gp<0.])
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
    with open('gplus_pars.csv','w+') as tfl:
        tfl.write(tstr)

    with open('gplus_pars.tex','w+') as tfl:
        tfl.write(tstr_tex)


    print('SSR = {:}'.format(np.sum(fobj(ips)**2)))

    colors = ['darkblue','darkorange','tab:green','darkred','darkslategray']

    for irs, rs in enumerate(rs_l):

        fig, ax = plt.subplots(figsize=(6,4))

        kf = rs_to_kf/rs
        a,b,c = get_g_plus_pars(rs)
        if rs in tdat_CK:
            ax.errorbar(tdat_CK[rs][:,0],tdat_CK[rs][:,1],yerr=tdat_CK[rs][:,2],color='k',\
                markersize=3,marker='o',linewidth=0,elinewidth=1.5)
        if rs in tdat_MCS:
            ax.errorbar(tdat_MCS[rs][:,0],tdat_MCS[rs][:,1],yerr=tdat_MCS[rs][:,2],color='m',\
                markersize=3,marker='o',linewidth=0,elinewidth=1.5)

        ax.plot(zl,a*zl**2,color=colors[1],linestyle='--',\
            label='$A_+(r_\\mathrm{s})(q/k_\\mathrm{F})^2$')
        ax.plot(zl,c*zl**2+b,color=colors[2],linestyle='-.',\
            label='$C(r_\\mathrm{s})(q/k_\\mathrm{F})^2 + B_+(r_\\mathrm{s})$')

        gpnew = gplus_CK(zl*kf,rs,0.,ips)
        ax.plot(zl,gpnew,color=colors[0],\
            label='This work')

        ax.plot(zl,g_plus_ra(zl*kf,0.,rs),color=colors[3],linestyle=':',\
            label='Richardson-Ashcroft')

        gcorr = g_corradini(zl*kf,\
            {'rs': rs, 'kF': kf, 'n': 3./(4.*pi*rs**3), 'rsh': rs**(0.5)})
        ax.plot(zl,gcorr,\
            color=colors[4],linestyle='-.',\
            label=r'Corradini et al.')

        """
        fxc_mcp07 = mcp07_static(zl*kf,{'rs': rs, 'kF': kf, 'n': 3./(4.*pi*rs**3), 'rsh': rs**(0.5)},param='PW92')
        g_mcp07 = -fxc_mcp07*(zl*kf)**2/(4.*pi)
        ax.plot(zl,g_mcp07,\
            color='cyan',linestyle='-.',\
            label=r'MCP07')
        """

        ax.set_xlim(zl.min(),zl.max())
        ax.set_xlabel('$q/k_\\mathrm{F}$',fontsize=12)
        ax.set_ylabel('$G_+(q)$',fontsize=12)
        ax.set_ylim(0.,max(gpnew.max(),gcorr.max()))
        #ax.annotate('$r_\\mathrm{s}'+'={:}$'.format(rs),(0.7,0.03),\
        #    xycoords='axes fraction',fontsize=24)

        ax.legend(fontsize=10,title='$r_\\mathrm{s}'+'={:}$'.format(rs),\
            title_fontsize=18)

        #plt.show() ; exit()
        plt.savefig('./figs/gplus_rs_{:}.pdf'.format(rs),dpi=600,bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()

    return

def init_fit(rs_l):

    tdat = {}
    tdat_CK = {}
    tdat_MCS = {}
    npts = 0

    tstr = 'rs, c0, c1, c2 \n'
    zl = np.linspace(0.0,4.0,1000)

    for irs, rs in enumerate(rs_l):
        ckgpf = './data_files/CK_Gplus_rs_{:}.csv'.format(int(rs))
        mcsgpf = './data_files/MCS_Gplus_rs_{:}.csv'.format(int(rs))
        if path.isfile(ckgpf):
            tdat_CK[rs] = np.genfromtxt(ckgpf,delimiter=',',skip_header=1)
            if rs in tdat:
                tdat[rs] = np.vstack((tdat[rs],tdat_CK[rs]))
            else:
                tdat[rs] = tdat_CK[rs].copy()
            npts += tdat_CK[rs].shape[0]

        elif path.isfile(mcsgpf):
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
                    gp = gplus_CK(tdat[rs][:,0]*kf,rs,0.,c,init=True)
                    #gp = gplus_corradini(tdat[rs][:,0]*kf,rs,0.,c)
                    #gm = gminus_ra_new(tdat[rs][:,0]*kf,rs,0.,c)
                    fres[tpts:tpts+gp.shape[0]] = (gp - tdat[rs][:,1])/tdat[rs][:,2]
                    #fres[tpts+gm.shape[0]:tpts+2*gm.shape[0]] = \
                    #    fres[tpts:tpts+gm.shape[0]]/tdat[rs][:,0]**2
                    tpts += gp.shape[0]
                else:
                    gp = gplus_CK(zl*kf,rs,0.,c,init=True)
                    #gp = gplus_corradini(zl*kf,rs,0.,c)
                fres[-1] += len(gp[gp<0.])
            return fres

        res = least_squares(fobj,[.8,1.7,0.])
        tstr += ('{:}, '*3 + '{:}\n').format(rs,*res.x)

    with open('./optpars_gplus.csv','w+') as tfl:
        tfl.write(tstr)

    return

def manip(rs):

    from matplotlib.widgets import Slider

    zl = np.linspace(0.0,4.0,1000)

    tdat_CK = {}
    tdat_MCS = {}
    ckgpf = './data_files/CK_Gplus_rs_{:}.csv'.format(int(rs))
    mcsgpf = './data_files/MCS_Gplus_rs_{:}.csv'.format(int(rs))
    if path.isfile(ckgpf):
        tdat_CK[rs] = np.genfromtxt(ckgpf,delimiter=',',skip_header=1)

    if path.isfile(mcsgpf):
        tdat_MCS[rs] = np.genfromtxt(mcsgpf,delimiter=',',skip_header=1)

    fig, ax = plt.subplots(figsize=(6,6))

    fig.subplots_adjust(bottom=0.25)

    kf = rs_to_kf/rs
    a,b,c = get_g_plus_pars(rs)
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
    twrap = lambda ps : gplus_CK(zl*kf,rs,0.,ps,init=True)
    line, = ax.plot(zl,twrap([a0,b0,g0]),color='darkblue')
    #line, = ax.plot(zl,gplus_zeropar(zl*kf,rs,gamma=a0),color='darkorange')

    ax.set_xlim(zl.min(),zl.max())
    ax.set_xlabel('$q/k_\\mathrm{F}$',fontsize=12)
    ax.set_ylabel('$G_+(q)$',fontsize=12)
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

    if len(sys.argv) == 1 or sys.argv[1].lower() == 'main':
        rs_l = [1.e-6,0.01,0.1,1,2,3,4,5,10,69,100]
        ips = [0.0216503, 0.120479, #0.00272079, 0.0194754, 0.157673,
            0.324648, 0.869809, 0.28963,
            0.256919, 1.22367, 0.960642]
        main_fit(rs_l,ips)

    elif sys.argv[1].lower() == 'init':
        init_fit([1,2,5,10])

    elif sys.argv[1].lower() == 'manip':

        manip(float(sys.argv[2]))
