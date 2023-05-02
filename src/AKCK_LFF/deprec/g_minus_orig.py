import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from os import path, system

from PW92 import g0_unp_pw92_pade
from asymptotics import get_g_minus_pars
from ra_lff import g_minus_ra

from g_plus_no_GEA import ifunc

pi = np.pi

rs_to_kf = (9*pi/4.)**(1./3.)

def gminus_sg(q,rs,z):

    # G.E. Simion and G.F. Giuliani, Phys. Rev. B 77, 035131 (2008).
    # DOI: 10.1103/PhysRevB.77.035131

    kf = rs_to_kf/rs
    Q2 = (q/kf)**2

    Amin, Bmin, C = get_g_minus_pars(rs,z)

    g = Bmin/(Amin - C)

    Gmin = C*Q2 + Bmin*Q2/(g + Q2)
    return Gmin

def gminus_mcs(q,rs,z):
    kf = rs_to_kf/rs
    Q2 = (q/kf)**2

    Amin, Bmin, C = get_g_minus_pars(rs,z)

    nn = 2.
    Gmin = Q2*(C + 1./( 1./(Amin - C)**nn + (Q2/Bmin)**nn )**(1./nn) )

    return Gmin


def gminus_corradini(q,rs,z,c):

    kf = rs_to_kf/rs
    Q2 = (q/kf)**2

    Amin, Bmin, C = get_g_minus_pars(rs,z)

    g = Bmin/(Amin - C)
    alpha = c[0]*Amin/(rsh**(0.5)*Bmin*g)
    beta = c[1]/(Bmin*g)

    Gmin = C*Q2 + Bmin*Q2/(g + Q2) + alpha*Q2**2 *np.exp(-beta*Q2)

    return Gmin

def gminus(q,rs,z,c,cpoles=False):

    kf = rs_to_kf/rs
    Q2 = (q/kf)**2

    Amin, Bmin, C = get_g_minus_pars(rs,z)

    beta = C*c[2]**(2./3.)
    alpha = (c[2]*Bmin + 2*c[1]*C)/(3.*c[2]**(1./3.))
    Gmin = Q2*(Amin + Q2*(alpha + Q2*beta))/(1. + Q2*(c[0] + Q2*(c[1] + Q2*c[2])))**(2./3.)
    #Gmin = Q2*(Amin + Q2*(c1 + Q2*c2))/(1. + Q2*(d1 + Q2*d2) )
    #fac = (Amin - C)/Bmin
    #Gmin = Q2*(Amin + fac*C*Q2)/(1. + fac*Q2)

    #beta = c[0]
    #gamma = Amin*beta/C
    #alpha = Amin**2*Bmin/(Amin*(gamma - C))
    #Gmin_low_q = Amin*Q2*(1. + c[0]*Q2)*np.exp(-c[1]*Q2**4/256.)
    #Gmin_high_q = (C*Q2 + Bmin)*(1. - np.exp(-c[2]*Q2**4/256.))
    #Gmin = Gmin_low_q + Gmin_high_q
    """
    if cpoles:
        lpass = check_poles([alpha,gamma])
        if not lpass:
            return 1.e20*np.ones(Q2.shape)
    #"""

    return Gmin

def log_func(z):
    return (z**2 - 1.)/(2.*z)*np.log(np.abs((1. - z)/(1. + z)))

def gminus_ra_new(q,rs,z,c):

    kf = rs_to_kf/rs
    q2 = (q/kf)**2
    q4 = q2*q2

    Amin, Bmin, C = get_g_minus_pars(rs,z)

    f = c[0]#*rs/(1. + c[1]*rs)
    ga0 = f*Amin
    gn0 = (1. - f)*Amin

    gainf = (4.*g0_unp_pw92_pade(rs) - 1.)/3.
    gninf = C

    ca = abs(c[1])
    ba = c[3]
    #dn = abs(c[2])
    cna = 1./4.
    cnb = gninf/(Bmin - gainf)
    cnc = abs(c[2])
    rcnd = cnb**2 - 4*cna*cnc
    if rcnd < 0.:
        return 1.e20*np.ones(q.shape)
    cn1 = -(cnb + rcnd**(0.5))/(2.*cna)
    cn2 = (-cnb + rcnd**(0.5))/(2.*cna)
    if cn1 > 0.:
        cn = cn1
    elif cn2 > 0.:
        cn = cn2
    elif cn1 > cn2:
        cn = cn1
    else:
        cn = cn2
    dn = cn**2/4. + abs(c[2])

    #cn = (gainf - Bmin)*dn/gninf
    #dn = cn**2/abs(c[2])

    ga = q2*(ga0 + gainf*ca*q4)/(1. + ba*q2 + ca*q4*q2 )
    gn = q2*(gn0 + gninf*dn*q4)/(1. + cn*q2 + dn*q4)

    return ga + gn

def gminus_UI_new(q,rs,z,c):

    kf = rs_to_kf/rs
    q2 = (q/kf)**2
    q4 = q2*q2
    q6 = q4*q2

    Amin, Bmin, Cmin = get_g_minus_pars(rs,z)

    cc = c[0] + c[1]*np.exp(-c[2]*rs)
    ct = 1./64. + c[3]*np.exp(-c[4]*rs)
    bt = 1./4. + c[5]*np.exp(-c[4]*rs)
    #ct = (1./64. + c[3]*rs + c[5]/64.*rs**2)/(1. + c[4]*rs + c[5]*rs**2)
    #bt = (-1./4. + c[6]*rs - c[8]/4.*rs**2)/(1. + c[7]*rs + c[8]*rs**2)
    a = (Bmin*ct - bt*cc)/(2.*ct)
    b = (Amin + Cmin - a*bt)/2.

    alp = -a
    bet = b - Cmin

    interp = (1. - ct*q6)/(1. + bt*q2 + ct*q6)
    #interp = (1. - (q/(2.*kf))**6)/(1. - (q/(2.*kf))**2 + (q/(2.*kf))**6)

    gmin = a + b*q2 + cc*q4 + (alp + bet*q2 + cc*q4)*interp

    return gmin

def gminus_UI(q,rs,z,c):

    kf = rs_to_kf/rs
    x = q/(2*kf)
    x2 = x*x
    x4 = x2*x2
    x6 = x4*x2

    Amin, Bmin, C = get_g_minus_pars(rs,z)

    ca = c[0] + c[1]*np.exp(-c[2]*rs**2)
    cb = -4.*ca/15 + 9.*Amin/4. - 3.*Bmin/16. + 7.*C/4.
    cc = -ca/5. - 3.*Amin/4. + 9.*Bmin/16. + 3.*C/4.
    calp = ca
    cbet = 2.*ca/5. + 9.*Amin/4. - 3.*Bmin/16. - 9.*C/4.
    cgam = -cc

    logfun = np.zeros_like(x)
    logfun[x == 0.] = 1.
    xmsk = (0. < x) & (x < 1.)
    logfun[xmsk] = log_func(x[xmsk])

    xmsk = 1. < x
    logfun[xmsk] = log_func(x[xmsk])

    gmin = ca*x4 + cb*x2 + cc + (calp*x4 + cbet*x2 + cgam)*logfun
    return gmin

def gminus_CK(q,rs,z,c):
    kf = rs_to_kf/rs
    q2 = (q/kf)**2
    q4 = q2*q2
    q8 = q4*q4
    Amin, Bmin, Cmin = get_g_minus_pars(rs,z)

    alpha = c[0] + c[1]*np.exp(-abs(c[2])*rs)
    beta = c[3]
    gamma = c[4]

    #interp1 = np.exp(-beta*q8/256.)
    #interp2 = -np.expm1(-gamma*q8/256.)
    interp1 = ifunc(q4/16.,beta,gamma)
    interp2 = 1. - interp1

    asymp1 = q2*(Amin + alpha*q2*q2)
    asymp2 = (Bmin + Cmin*q2)
    gmin = asymp1*interp1 + asymp2*interp2

    return gmin

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

rs_l = [1.e-6,0.01,0.1,1,2,3,4,5,10,69,100]

tdat = {}
npts = 0

for irs, rs in enumerate(rs_l):
    ckgmf = './data_files/CK_Gmin_rs_{:}.csv'.format(int(rs))
    if path.isfile(ckgmf):
        tdat[rs] = np.genfromtxt(ckgmf,delimiter=',',skip_header=1)
        npts += tdat[rs].shape[0]

#"""
zl = np.linspace(0.0,4.0,1000)

def fobj(c):
    fres = np.zeros(npts+1)
    tpts = 0
    for rs in rs_l:#tdat:
        kf = (9*pi/4.)**(1./3.)/rs
        if rs in tdat:
            gm = gminus_CK(tdat[rs][:,0]*kf,rs,0.,c)
            #gm = gminus_ra_new(tdat[rs][:,0]*kf,rs,0.,c)
            fres[tpts:tpts+gm.shape[0]] = (gm - tdat[rs][:,1])/tdat[rs][:,2]
            #fres[tpts+gm.shape[0]:tpts+2*gm.shape[0]] = \
            #    fres[tpts:tpts+gm.shape[0]]/tdat[rs][:,0]**2
            tpts += gm.shape[0]
        else:
            gm = gminus_CK(zl*kf,rs,0.,c)
            fres[-1] += len(gm[gm<0.])
    return fres
#"""
#ips = [8.935746e-04, -7.050533e-03, 8.529516e-02, -1.063034e-02, 5.379620e-03, -4.959856e-01]
#ips = [8.570115e-03, -9.491117e-04, 2.031323e-02, 7.839506e-03, 1.990629e-03, -9.818464e-03, 1.572911e-02]
ips = [0.0230829, -0.00449486,0.983665, -0.81567, 0.984994,1.4431, 22.0007, 3.66798]
Nps = len(ips)
for i in range(5):
    res = least_squares(fobj,ips)
    ips = (res.x).copy()
#"""
#ips = [-0.00430811, -0.0013902, 0.189778,0.0152401, 4.31559, -0.0570435,0.104257, -0.405696, 0.0562809]

#print(res.x)
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
with open('gmin_pars.csv','w+') as tfl:
    tfl.write(tstr)

with open('gmin_pars.tex','w+') as tfl:
    tfl.write(tstr_tex)

print(np.sum(fobj(ips)**2))
#print( 0.5*(-res.x[1] + (0.j + res.x[1]**2 - 4.*res.x[2])**(0.5)), 0.5*(-res.x[1] - (0.j + res.x[1]**2 - 4.*res.x[2])**(0.5)) )

colors = ['darkblue','darkorange','tab:green','darkred','k','gray']

for irs, rs in enumerate(rs_l):

    fig, ax = plt.subplots(figsize=(6,4))

    kf = rs_to_kf/rs
    a,b,c = get_g_minus_pars(rs,0.)
    if rs in tdat:
        ax.errorbar(tdat[rs][:,0],tdat[rs][:,1],yerr=tdat[rs][:,2],color='k',\
            markersize=3,marker='o',linewidth=0,elinewidth=1.5)
    ax.plot(zl,a*zl**2,color=colors[1],linestyle='--',\
        label='$A_-(r_\\mathrm{s})(q/k_\\mathrm{F})^2$')
    ax.plot(zl,c*zl**2+b,color=colors[2],linestyle='-.',\
        label='$C(r_\\mathrm{s})(q/k_\\mathrm{F})^2 + B_-(r_\\mathrm{s})$')
    ax.plot(zl,gminus_CK(zl*kf,rs,0.,ips),color=colors[0],\
        label='This work')
    ax.plot(zl,g_minus_ra(zl*kf,0.,rs),color=colors[3],linestyle=':',\
        label='Richardson-Ashcroft')

    ax.set_xlim(zl.min(),zl.max())
    ax.set_xlabel('$q/k_\\mathrm{F}$',fontsize=12)
    ax.set_ylabel('$G_-(q)$',fontsize=12)
    ax.set_ylim(0.,1.2)
    #ax.annotate('$r_\\mathrm{s}'+'={:}$'.format(rs),(0.7,0.03),\
    #    xycoords='axes fraction',fontsize=24)

    ax.legend(fontsize=10,title='$r_\\mathrm{s}'+'={:}$'.format(rs),\
        title_fontsize=18)

    #plt.show() ; exit()
    plt.savefig('./figs/gminus_rs_{:}.pdf'.format(rs),dpi=600,bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()
