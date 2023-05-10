import numpy as np

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


def gPW92(rs,v):
    q0 = -2.0*v[0]*(1.0 + v[1]*rs)
    rsh = rs**(0.5)
    q1 = 2.0*v[0]*( rsh* (v[2] + rsh*( v[3] + rsh*( v[4] + rsh*v[5]))) )
    return q0*np.log(1.0 + 1.0/q1)

def dgPW92(rs,v):
    q0 = -2.0*v[0]*(1.0 + v[1]*rs)
    q0p = -2.0*v[0]*v[1]

    rsh = rs**(0.5)
    q1 = 2.0*v[0]*( rsh* (v[2] + rsh*( v[3] + rsh*( v[4] + rsh*v[5]))) )
    q1p = v[0]*( v[2]/rsh + 2.*v[3] + rsh*( 3.*v[4] + 4.*rsh*v[5] ) )

    dg = q0p*np.log(1. + 1./q1) - q0*q1p/(q1*(1. + q1))
    return dg

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

    opz = np.minimum(2.,np.maximum(0.0,1.+z))
    omz = np.minimum(2.,np.maximum(0.0,1.-z))
    dxz = 0.5*(opz**(4./3.) + omz**(4./3.))
    d_dxz_dz = 2./3.*(opz**(1./3.) - omz**(1./3.))

    inftsml = 1.e-12
    z_reg = np.minimum(1.-inftsml,np.maximum(-1.+inftsml,z))
    d2_dxz_dz2 = 2./9.*((1. + z_reg)**(-2./3.) + (1. - z_reg)**(-2./3.))

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
