import numpy as np

from PW92 import g0_unp_pw92_pade, ec_pw92

pi = np.pi
alpha = (4./(9*pi))**(1./3.)

def get_lambdas(rs):

    ec, d_ec_drs, d_ec_drs2, d_ec_dz2 = ec_pw92(rs,0.0)
    # conversion to Rydberg
    ec *= 2.
    d_ec_drs *= 2.
    d_ec_drs2 *= 2.
    d_ec_dz2 *= 2.

    # Eq. 40, corrected by Lein, Gross, and Perdew
    lam_s_inf = 3./5. - 2*pi*alpha*rs/5.*(rs*d_ec_drs + 2*ec)

    # Eq. 44
    lam_pade = -0.11*rs/(1 + 0.33*rs)
    # Eq. 39, corrected by Lein, Gross, and Perdew
    sfac = 1. - 3*(2.*pi/3.)**(2./3.)*rs*d_ec_dz2/2.
    lam_n_0 = lam_pade*sfac
    lam_a_0 = sfac - lam_n_0

    # Eq. 38
    lam_s_0 = 1. + pi/3*alpha*rs**2*(d_ec_drs - rs*d_ec_drs2/2.) - lam_n_0

    lam_n_inf = 3*pi*alpha*rs*(ec + rs*d_ec_drs)

    # just below Eq. 39
    g0 = g0_unp_pw92_pade(rs)
    lam_a_inf = (2*g0 - 1.)/3.

    return lam_s_0, lam_s_inf, lam_n_0, lam_n_inf, lam_a_0, lam_a_inf

def lff_ra_symm(q,w,rs):

    """
        NB: q = (wavevector in a.u.)/(2*kf), w = (frequency in a.u.)/(2*kf**2)

        There are at least three alpha's in the RA paper
        alpha is determined from exact constraints, and is used in the lambdas (lam_)
        alp is a parameter used to control the parameterization, and is used in a, b and c
    """
    alp = 0.9

    g0 = g0_unp_pw92_pade(rs)
    omg0 = 1. - g0

    lam_s_0, lam_s_inf, _, _, _, _ = get_lambdas(rs)
    """
        p. 57 of RA work: per Phys Rev style (https://journals.aps.org/prl/authors):
            "Note that the solidus (/) in fractions, for example 1/2a, means 1/(2a) and not (1/2)a."
        which to me implies that what is meant is 9/(16*[1 - g(0)]) and not 9[1-g(0)]/16.
    """
    gam_s = 9*lam_s_inf/(16*omg0) + (4.*alp - 3.)/(4.*alp)

    # Eq. 56
    a_s = lam_s_inf + (lam_s_0 - lam_s_inf)/(1. + (gam_s*w)**2)
    # Eq. 55
    c_s = 3*lam_s_inf/(4*omg0) - (4/3 - 1./alp + \
        3*lam_s_inf/(4*omg0))/(1 + gam_s*w)
    # Eq. 54
    b_s = a_s/( ( (3*a_s - 2*c_s*omg0)*(1. + w) - 8*omg0/3. )*(1. + w)**3 )

    q2 = q**2
    q6 = q2**3
    # Eq. 53
    g_s = q2*(a_s + 2*b_s*omg0*q6/3.)/(1. + q2*(c_s + b_s*q6))

    return g_s

def lff_ra_occ(q,w,rs):

    """
        NB: q = (wavevector in a.u.)/(2*kf), w = (frequency in a.u.)/(2*kf**2)
    """

    gam_n = 0.68
    gnw = gam_n*w
    gnw2 = gnw*gnw
    opgnw = 1. + gnw

    _, _, lam_n_0, lam_n_inf, _, _ = get_lambdas(rs)
    """
    Eq. 65. Note that there is a "gamma" instead of "gamma_n" in the printed version of a_n
    assuming this just means gamma_n
    """
    a_n = lam_n_inf + (lam_n_0 - lam_n_inf)/(1. + gnw2)
    """
    Eq. 64
    in this equation, "gam_n(w)" is printed twice. I'm assuming this just means
    gam_n, since this is constant. That seems to give OK agreement with their figure
    """
    c_n = 3*gam_n/(1.18*opgnw) - ( (lam_n_0 + lam_n_inf/3)/(lam_n_0 + 2*lam_n_inf/3) \
        + 3*gam_n/(1.18*opgnw))/(1. + gnw2)
    # Eq. 63
    bt = a_n + lam_n_inf*(1. + 2/3*c_n*opgnw)
    b_n = -3/(2*lam_n_inf*opgnw**2)*( bt + (bt**2 + 4/3*a_n*lam_n_inf)**(0.5) )

    q2 = q**2
    q4 = q2*q2
    # Eq. 62
    g_n = q2*(a_n - lam_n_inf * b_n*q4/3.)/(1. + q2*(c_n + q2*b_n))

    return g_n

def lff_ra_asymm(q,w,rs):

    """
        NB: q = (wavevector in a.u.)/(2*kf), w = (frequency in a.u.)/(2*kf**2)
    """
    # NB symmetric and occupation G's include factor of 1/(2*kF)**2
    # because they are multiplied by a factor of 1/q**2 to get kernel
    # not needed here, because G_a(q-->0) = constant(i w)

    g0 = g0_unp_pw92_pade(rs)
    _, _, _, _, lam_a_0, lam_a_inf = get_lambdas(rs)
    # below 61
    gam_a = 9.*lam_a_inf/8. + 1./4.
    gaw2 = (gam_a*w)**2

    # 61
    beta = (4*g0 - 1.)/3. -lam_a_inf*gaw2/(3.*(1. + gaw2))

    # 60
    a_a = lam_a_inf + (lam_a_0 - lam_a_inf)/(1. + gaw2)

    # 59, not sure if gamma_a**2 * w is a typo
    # should be (gamma_a * w)**2
    c_a = 3.*lam_a_inf/2. - (1./3. + 3.*lam_a_inf/2.)/(1. + gaw2)

    # 58, assuming c_s is a typo, means c_a
    opw = 1. + w
    opw3 = opw**3
    opw4 = opw3*opw
    b_a = a_a/(3.*a_a*opw4 - 4*beta*opw3 - 3*c_a*beta*opw4)

    q2 = q*q
    q8 = q2**4
    ga = lam_a_inf*gaw2/(1. + gaw2) + (a_a*q2 + b_a*beta*q8)/(1. + c_a*q2 + b_a*q8)

    return ga

def g_minus_ra(q,w,rs):
    """
        NB: q = (wavevector in a.u.)/(2*kf), w = (frequency in a.u.)/(2*kf**2)

        C.F. Richardson and N.W. Ashcroft,
            Phys. Rev. B 50, 8170 (1994),

        and

        Eq. 32 of M. Lein, E.K.U. Gross, and J.P. Perdew,
            Phys. Rev. B 61, 13431 (2000)
    """
    kf = (9*pi/4.)**(1./3.)/rs
    z = q/(2.*kf)
    u = w/(2.*kf**2)
    ga = lff_ra_asymm(z,u,rs)
    gn = lff_ra_occ(z,u,rs)
    return ga + gn

def g_plus_ra(q,w,rs):
    """
        NB: q = (wavevector in a.u.)/(2*kf), w = (frequency in a.u.)/(2*kf**2)

        C.F. Richardson and N.W. Ashcroft,
            Phys. Rev. B 50, 8170 (1994),

        and

        Eq. 32 of M. Lein, E.K.U. Gross, and J.P. Perdew,
            Phys. Rev. B 61, 13431 (2000)
    """
    kf = (9*pi/4.)**(1./3.)/rs
    z = q/(2.*kf)
    u = w/(2.*kf**2)
    gs = lff_ra_symm(z,u,rs)
    gn = lff_ra_occ(z,u,rs)
    return gs + gn


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    rs = 2.0

    kf = (9.*pi/4.)**(1./3.)/rs
    ql = np.linspace(0.0,4.0,2000)
    tfac = ql**2
    """
    fxcst = fxc_ra(ql/(2*kf),0.0,rs)
    plt.plot(ql,fxcst)
    plt.ylim(-4.5,0)
    plt.xlim(0,9)
    plt.show()
    exit()
    """
    lsls = ['-','--']
    for iw,w in enumerate([2.]):#[0.5, 2]):
        gs = lff_ra_symm(ql,w/(2.*kf**2),rs)
        gn = lff_ra_occ(ql,w/(2.*kf**2),rs)
        ga = lff_ra_asymm(ql,w/(2.*kf**2),rs)
        #plt.plot(ql,(gs+gn)*tfac,label="$G_s(q,iw), w= {:}$".format(w),color='darkblue', \
        #    linestyle=lsls[iw])
        plt.plot(ql,gs,label="$G_s(q,iw), w= {:}$".format(w),color='darkblue', \
            linestyle=lsls[iw])
        plt.plot(ql,gn,label="$G_n(q,iw), w= {:}$".format(w),color='darkorange', \
            linestyle=lsls[iw])
        plt.plot(ql,ga,label="$G_a(q,iw), w= {:}$".format(w),color='darkgreen', \
            linestyle=lsls[iw])
    plt.legend(title='$r_s={:}$'.format(rs))
    plt.xlabel('$q/(2k_F)$')
    plt.ylabel('$G(q,iw)$')
    plt.ylim([-1.0,4.0])
    plt.show()
    exit()

    rsl = [0.01, 0.1, 0.5, 1,2,5,10,20,100]
    for rs in rsl:
        gxc = g0_unp_pw92_pade(rs)
        print('{:}, {:.3f}, {:.3f}'.format(rs,gxc,g0_unp_yasuhara(rs)))
