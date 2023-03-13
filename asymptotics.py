import numpy as np
from PW92 import g0_unp_pw92_pade, ec_pw92
from alda import alda

pi = np.pi

rs_to_kf = (9*pi/4.)**(1./3.)
gex2 = -5./(216.*pi*(3.*pi**2)**(1./3.))

def gec2(rs):
    beta0 = 0.066725
    ca = 3.0
    cb = 1.046
    cc = 0.100
    cd = 1.778*cc
    beta_acgga = beta0*(1 + ca*rs*(cb + cc*rs))/(1 +  ca*rs*(1. + cd*rs))
    return beta_acgga/16.*(pi/3.)**(1./3.)

def gexc2(rs):
    return gex2 + gec2(rs)

def Bpos(rs):
    a1 = 2.15
    a2 = 0.435
    b1 = 1.57
    b2 = 0.409
    rsh = rs**(0.5)
    B = (1. + rsh*(a1 + a2*rs))/(3. + rsh*(b1 + b2*rs))

    return B

def get_g_plus_pars(rs):

    kf = rs_to_kf/rs

    ec, d_ec_drs, _, _ = ec_pw92(rs,0.0)

    fxc_alda = alda({'rs': rs, 'kF': kf, 'n': 3./(4.*pi*rs**3), 'rsh': rs**(0.5)},\
        x_only=False, param='PW92')
    Apos = -kf**2*fxc_alda/(4.*pi)

    d_rs_ec_drs = ec + rs*d_ec_drs
    C = -pi*d_rs_ec_drs/(2.*kf)

    return Apos, Bpos(rs), C


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

def chi_enh(rs):

    # Eq. 2.59 and Table 2.1 of Quantum Theory of Electron Liquid
    rss3 = pi*rs_to_kf

    kf = rs_to_kf/rs

    ef = kf**2/2.
    # square of Thomas-Fermi screening wavevector
    ks2 = 4*kf/pi

    ec, d_ec_drs, d_ec_drs2, d_ec_dz2 = ec_pw92(rs,0.)
    # Eq. 5.113
    return 1./(1. - rs/rss3 + 3.*d_ec_dz2/(2*ef))

def pade_g_plus(q,rs):
    Ap, Bp, Cp = get_g_plus_pars(rs)
    kf = rs_to_kf/rs
    x2 = (q/kf)**2

    Dp = -(3.*pi**2)**(4./3.)*gexc2(rs)/(2.*pi)
    alp = 2.*Dp/(Ap - Cp)
    bet = ( (Ap - Cp)/Bp)**2
    enh = Cp + (Ap - Cp)/(1. + alp*x2 + bet*x2**2)**(0.5)
    gplus = enh*x2
    return gplus

if __name__ == "__main__":


    rs_l = [1,2,3,4,5]
    g_vmc = [1.152,1.296,1.438,1.576,1.683]
    g_vmc_ucrt = [2,6,9,9,15]

    from scipy.optimize import least_squares
    import matplotlib.pyplot as plt

    rsl = np.linspace(1.,100.,5000)
    #apar, bpar, cpar = get_g_plus_pars(rsl)
    apar, bpar, cpar = get_g_minus_pars(rsl,0.)
    plt.plot(rsl,(apar - cpar)/bpar)
    plt.show(); exit()

    fchi = chi_enh(rsl)

    imax = np.argmax(fchi)
    rsmax = rsl[imax]
    hmax = fchi[imax]/2.
    find_right = False
    for irs in range(rsl.shape[0]):
        tdiff = fchi[irs] - hmax
        if tdiff > 0. and (not find_right):
            ileft = irs
            find_right = True
        elif tdiff < 0. and find_right:
            iright = irs
            break
    hwhm = (rsl[iright] - rsl[ileft])/2.

    ffn = lambda c, x : c[0]/(1. + ((x - c[1])/c[2])**2)
    def obj(c):
        return ffn(c,rsl) - fchi
    res = least_squares(obj,[fchi[imax],rsmax,hwhm])
    print(res)

    plt.plot(rsl,fchi)
    plt.plot(rsl,ffn(res.x,rsl))
    #plt.scatter(rs_l,g_vmc)
    plt.show()
    exit()

    tstr = ''
    for irs, rs in enumerate(rs_l):
        enh = chi_enh(rs)
        pdiff = 200*abs(enh - g_vmc[irs])/(enh + g_vmc[irs])
        tstr += '{:} & {:}({:}) & {:.6f} & {:.2f} \\\\ \n'.format(rs,\
            g_vmc[irs], g_vmc_ucrt[irs], enh, pdiff)
    print(tstr)
