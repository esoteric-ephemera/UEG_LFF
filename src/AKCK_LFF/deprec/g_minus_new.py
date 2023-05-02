import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from os import path, system

from ra_lff import g_minus_ra

pi = np.pi

rs_to_kf = (9*pi/4.)**(1./3.)

def g0_unp_pw92_pade(rs):
    """
    see Eq. 29 of
      J. P. Perdew and Y. Wang,
        Phys. Rev. B 46, 12947 (1992),
        https://doi.org/10.1103/PhysRevB.46.12947
        and erratum Phys. Rev. B 56, 7018 (1997)
        https://doi.org/10.1103/PhysRevB.56.7018
        NB the erratum only corrects the value of the a3
        parameter in gc(rs, zeta, kf R)
    """

    alpha = 0.193
    beta = 0.525
    return 0.5*(1 + 2*alpha*rs)/(1 + rs*(beta + rs*alpha*beta))**2

def ec_pw92(rs,z):

    """
        Richardson-Ashcroft LFF needs some special derivatives of epsc, and moreover, needs them in
        Rydbergs, instead of Hartree.
        This routine gives those special derivatives in Rydberg

        J.P. Perdew and Y. Wang,
        ``Accurate and simple analytic representation of the electron-gas correlation energy'',
        Phys. Rev. B 45, 13244 (1992).
        https://doi.org/10.1103/PhysRevB.45.13244
    """

    rsh = rs**(0.5)
    def g(v):

        q0 = -2*v[0]*(1 + v[1]*rs)
        dq0 = -2*v[0]*v[1]

        q1 = 2*v[0]*(v[2]*rsh + v[3]*rs + v[4]*rs*rsh + v[5]*rs*rs)
        dq1 = v[0]*(v[2]/rsh + 2*v[3] + 3*v[4]*rsh + 4*v[5]*rs)
        ddq1 = v[0]*(-0.5*v[2]/rsh**3 + 3/2*v[4]/rsh + 4*v[5])

        q2 = np.log(1 + 1/q1)
        dq2 = -dq1/(q1**2 + q1)
        ddq2 = (dq1**2*(1 + 2*q1)/(q1**2 + q1) - ddq1)/(q1**2 + q1)

        g = q0*q2
        dg = dq0*q2 + q0*dq2
        ddg = 2*dq0*dq2 + q0*ddq2

        return g,dg,ddg

    unp_pars = [0.031091,0.21370,7.5957,3.5876,1.6382,0.49294]
    pol_pars = [0.015545,0.20548,14.1189,6.1977,3.3662,0.62517]
    alp_pars = [0.016887,0.11125,10.357,3.6231,0.88026,0.49671]

    fz_den = 0.5198420997897464#(2**(4/3)-2)
    fdd0 = 1.7099209341613653#8/9/fz_den

    opz = np.minimum(2,np.maximum(0.0,1+z))
    omz = np.minimum(2,np.maximum(0.0,1-z))
    dxz = (opz**(4/3) + omz**(4/3))/2.0
    d_dxz_dz = 2/3*(opz**(1/3) - omz**(1/3))
    d2_dxz_dz2 = 2/9*(opz**(-2/3) + omz**(-2/3))

    fz = 2*(dxz - 1)/fz_den
    d_fz_dz = 2*d_dxz_dz/fz_den
    d2_fz_dz2 = 2*d2_dxz_dz2/fz_den

    ec0,d_ec0_drs,d_ec0_drs2 = g(unp_pars)
    ec1,d_ec1_drs,d_ec1_drs2 = g(pol_pars)
    ac,d_ac_drs,d_ac_drs2 = g(alp_pars)
    z4 = z**4
    fzz4 = fz*z4

    ec = ec0 - ac/fdd0*(fz - fzz4) + (ec1 - ec0)*fzz4

    d_ec_drs = d_ec0_drs*(1 - fzz4) + d_ec1_drs*fzz4 - d_ac_drs/fdd0*(fz - fzz4)
    d_ec_dz = -ac*d_fz_dz/fdd0 + (4*fz*z**3 + d_fz_dz*z4)*(ac/fdd0 + ec1 - ec0)

    d_ec_drs2 = d_ec0_drs2*(1 - fzz4) + d_ec1_drs2*fzz4 - d_ac_drs2/fdd0*(fz - fzz4)
    d_ec_dz2 = -ac*d2_fz_dz2/fdd0 + (12*fz*z**2 + 8*d_fz_dz*z**3 + d2_fz_dz2*z4) \
        *(ac/fdd0 + ec1 - ec0)

    return ec, d_ec_drs, d_ec_drs2, d_ec_dz2

def Bpos(rs):
    a1 = 2.15
    a2 = 0.435
    b1 = 1.57
    b2 = 0.409
    rsh = rs**(0.5)
    B = (1. + rsh*(a1 + a2*rs))/(3. + rsh*(b1 + b2*rs))

    return B

def get_g_minus_pars(rs,z):

    # Eq. 2.59 and Table 2.1 of Quantum Theory of Electron Liquid
    rss3 = pi*rs_to_kf

    kf = rs_to_kf/rs

    ef = kf**2/2.
    # square of Thomas-Fermi screening wavevector
    ks2 = 4*kf/pi

    ec, d_ec_drs, d_ec_drs2, d_ec_dz2 = ec_pw92(rs,z)
    # Eq. 5.113
    one_m_chi = rs/rss3 - 3.*d_ec_dz2/(2*ef)
    # Eq. 5.167
    gm0 = one_m_chi/ks2

    Amin = kf**2 * gm0

    # Table 5.1
    Bmin = Bpos(rs) + 2*g0_unp_pw92_pade(rs) - 1.

    d_rs_ec_drs = ec + rs*d_ec_drs
    C = -pi*d_rs_ec_drs/(2.*kf)

    return Amin, Bmin, C

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

def gminus_CK(q,rs,z,c):
    kf = rs_to_kf/rs
    q2 = (q/kf)**2
    q4 = q2*q2
    q6 = q4*q2
    q8 = q6*q2
    Amin, Bmin, Cmin = get_g_minus_pars(rs,z)

    interp1 = np.exp(-c[1]*q8/256.)
    interp2 = -np.expm1(-c[2]*q8/256.)
    asymp1 = q2*(Amin + c[0]*q2)
    asymp2 = (Bmin + Cmin*q2)
    #gmin = asymp2 + (asymp1 - asymp2)*interp1
    gmin = asymp1*interp1 + asymp2*interp2
    """
    a = Bmin/2.
    b = (Amin + Cmin)/2.
    alp = -a
    bet = (Amin - Cmin)/2.
    gam = c[0]

    interp = 2.*np.exp(-c[1]*q**8) - 1.
    gmin = a + b*q2 + c[0]*q4 + (alp + bet*q2 + gam*q4)*interp
    """

    return gmin

def gminus_UI_new(q,rs,z,c):

    kf = rs_to_kf/rs
    q2 = (q/kf)**2
    q4 = q2*q2
    q6 = q4*q2

    Amin, Bmin, Cmin = get_g_minus_pars(rs,z)
    a = (Bmin*c[2] - c[1]*c[0])/(2.*c[2])
    b = (Amin + Cmin - a*c[1])/2.

    alp = -a
    bet = b - Cmin

    interp = (1. - c[2]*q6)/(1. + c[1]*q2 + c[2]*q6)

    gmin = a + b*q2 + c[0]*q4 + (alp + bet*q2 + c[0]*q4)*interp

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

rs_l = [1,2,3,4,5]

tstr = 'rs, c0, c1, c2 \n'

pmat = np.zeros((len(rs_l),3))

zl = np.linspace(0.0,4.0,1000)

for irs, rs in enumerate(rs_l):
    tdat = {}
    npts = 0
    ckgmf = './data_files/CK_Gmin_rs_{:}.csv'.format(int(rs))
    tdat[rs] = np.genfromtxt(ckgmf,delimiter=',',skip_header=1)
    npts += tdat[rs].shape[0]


#"""
    def fobj(c):
        fres = np.zeros(npts)
        tpts = 0
        for rs in tdat:
            kf = (9*pi/4.)**(1./3.)/rs
            #gm = gminus_UI_new(tdat[rs][:,0]*kf,rs,0.,c)
            gm = gminus_CK(tdat[rs][:,0]*kf,rs,0.,c)
            #gm = gminus_ra_new(tdat[rs][:,0]*kf,rs,0.,c)
            fres[tpts:tpts+gm.shape[0]] = (gm - tdat[rs][:,1])/tdat[rs][:,2]
            #fres[tpts+gm.shape[0]:tpts+2*gm.shape[0]] = \
            #    fres[tpts:tpts+gm.shape[0]]/tdat[rs][:,0]**2
            tpts += gm.shape[0]
        return fres

    res = least_squares(fobj,[0.05, 0.75,2.58])
    tstr += ('{:}, '*3 + '{:}\n').format(rs,*res.x)
    pmat[irs,:] = res.x
    print(rs,np.sum(fobj(res.x)**2))
    #kf = (9*pi/4.)**(1./3.)/rs
    #plt.plot(zl,gminus_CK(zl*kf,rs,0.,res.x))
    #plt.errorbar(tdat[rs][:,0],tdat[rs][:,1],yerr=tdat[rs][:,2],color='k',\
    #    markersize=3,marker='o',linewidth=0,elinewidth=1.5)

with open('./optpars_gminus.csv','w+') as tfl:
    tfl.write(tstr)

plt.scatter(rs_l,pmat[:,0])
plt.scatter(rs_l,pmat[:,1])
plt.scatter(rs_l,pmat[:,2])
plt.show()
