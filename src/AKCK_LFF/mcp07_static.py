import numpy as np

from AKCK_LFF.alda import alda, lda_derivs

pi = np.pi

def mcp07_static(q,dv,param='PZ81'):

    rs = dv['rs']
    n = dv['n']
    kf = dv['kF']
    rsh = dv['rsh']
    cfac = 4*pi/kf**2
    q2 = q**2

    # bn according to the parametrization of Eq. (7) of
    # Massimiliano Corradini, Rodolfo Del Sole, Giovanni Onida, and Maurizia Palummo
    # Phys. Rev. B 57, 14569 (1998)
    # doi: 10.1103/PhysRevB.57.14569
    bn = 1.0 + 2.15*rsh + 0.435*rsh**3
    bn /= 3.0 + 1.57*rsh + 0.409*rsh**3

    f0 = alda(dv,param=param)
    akn = -f0/(4.0*pi*bn)

    ec,d_ec_d_rs = lda_derivs(dv,param=param)
    d_rs_ec_drs = ec + rs*d_ec_d_rs
    # The rs-dependent cn, multiplicative factor of d( r_s eps_c)/d(r_s)
    # eps_c is correlation energy per electron

    cn = -pi/(2.0*kf)*d_rs_ec_drs

    # The gradient term
    cxcn = 1.0 + 3.138*rs + 0.3*rs**2
    cxcd = 1.0 + 3.0*rs + 0.5334*rs**2
    cxc = -0.00238 + 0.00423*cxcn/cxcd
    dd = 2.0*cxc/(n**(4.0/3.0)*(4.0*pi*bn)) - 0.5*akn**2

    # The MCP07 kernel
    zp = akn*q2
    grad = 1.0 + dd*q2*q2

    qeps = 1.e-6
    fxcmcp07_taylor = f0 + q2 * ( 2*cxc/n**(4/3) + \
        (f0*(akn**2/6 + dd) - cfac*cn*akn**2)*q2 )

    if hasattr(q,'__len__'):

        fxcmcp07 = np.zeros_like(q)

        qm = q < qeps
        fxcmcp07[qm] = fxcmcp07_taylor[qm]

        qm = q >= qeps
        vc = 4.0*pi/q2[qm]
        cl = vc*bn
        cutdown = 1.0 + 1.0/(akn*q2[qm])**2
        fxcmcp07[qm] = cl*(np.exp(-zp[qm])*grad[qm] - 1.0) - cfac*cn/cutdown

    else:
        if q < qeps:
            fxcmcp07 = fxcmcp07_taylor
        else:
            vc = 4.0*pi/q2
            cl = vc*bn
            cutdown = 1.0 + 1.0/(akn*q2)**2
            fxcmcp07 = cl*(np.exp(-zp)*grad - 1.0) - cfac*cn/cutdown

    return fxcmcp07
