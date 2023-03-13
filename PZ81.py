import numpy as np

def small_rs_exp(rs,A,B,C,D):
    return A*np.log(rs) + B + C*rs*np.log(rs) + D*rs

def large_rs_exp(rs,g,b1,b2):
    rsh = rs**(0.5)
    return g/(1. + rsh*(b1 + rsh*b2))

def eps_c_pz81(rs):

    gamma = [-0.1423, -0.0843]
    beta1 = [1.0529, 1.3981]
    beta2 = [0.3334, 0.2611]
    A = [0.0311, 0.01555]
    B = [-0.048, -0.0269]
    C = [0.0020, 0.0007]
    D = [-0.0116, -0.0048]

    lmask = rs < 1.0
    umask = rs >= 1.0
    rsl = rs[lmask]
    rsu = rs[umask]
    ecv = np.zeros((2,rs.shape[0]))

    for i in range(2):
        ecv[i,umask] = large_rs_exp(rsu,gamma[i],beta1[i],beta2[i])
        ecv[i,lmask] = small_rs_exp(rsl,A[i],B[i],C[i],D[i])

    return ecv

def ec_pz81(rs,z):
    """
        J. P. Perdew and A. Zunger,
        ``Self-interaction correction to density-functional approximations for many-electron systems''
        Phys. Rev. B 23, 5048 (1981).
        https://doi.org/10.1103/PhysRevB.23.5048
    """

    fz = ((1.0+z)**(4.0/3.0) + (1.0-z)**(4.0/3.0) -2.0)/(2.0**(4.0/3.0)-2.0)

    # first entry is for spin-unpolarized, second for spin-polarized
    ecv = eps_c_pz81(rs)
    ec = ecv[0] + fz*(ecv[1] - ecv[0])

    return ec

def alpha_c_pz81(rs):
    """
        J. P. Perdew and A. Zunger,
        ``Self-interaction correction to density-functional approximations for many-electron systems''
        Phys. Rev. B 23, 5048 (1981).
        https://doi.org/10.1103/PhysRevB.23.5048
    """

    fz_den = (2.0**(4.0/3.0)-2.0)
    fdd0 = 8.0/9.0/fz_den

    ecv = eps_c_pz81(rs)
    alpha_c = fdd0*(ecv[1] - ecv[0])
    return alpha_c

def chi_enh_pz81(rs):
    alpha = (4./(9.*np.pi))**(1./3.)
    ac_pz81 = alpha_c_pz81(rs)
    return 1./(1. - alpha*rs/np.pi - 3.*(alpha*rs)**2*ac_pz81)
