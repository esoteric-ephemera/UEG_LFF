import numpy as np

from AKCK_LFF.alda import alda,lda_derivs
from AKCK_LFF.mcp07_static import mcp07_static

pi = np.pi
gam = 1.311028777146059809410871821455657482147216796875
# NB: got this value from julia using the following script
# using SpecialFunctions
# BigFloat((gamma(0.25))^2/(32*pi)^(0.5))
cc = 4.81710873550434914847073741839267313480377197265625 # 23.0*pi/15.0

def exact_constraints(dv):

    n = dv['n']
    kf = dv['kF']
    rs = dv['rs']

    f0 = alda(dv,x_only=False,param='PW92')
    """
     from Iwamato and Gross, Phys. Rev. B 35, 3003 (1987),
     f(q,omega=infinity) = -4/5 n^(2/3)*d/dn[ eps_xc/n^(2/3)] + 6 n^(1/3) + d/dn[ eps_xc/n^(1/3)]
     eps_xc is XC energy per electron
    """

    # exchange contribution is -1/5 [3/(pi*n^2)]^(1/3)
    finf_x = -1.0/(5.0)*(3.0/(pi*n**2))**(1.0/3.0)

    # correlation contribution is -[22*eps_c + 26*rs*(d eps_c / d rs)]/(15*n)
    eps_c,d_eps_c_d_rs = lda_derivs(dv,param='PW92')
    finf_c = -(22.0*eps_c + 26.0*rs*d_eps_c_d_rs)/(15.0*n)
    finf = finf_x + finf_c

    bfac = (gam/cc)**(4.0/3.0)
    deltaf = finf - f0
    bn = bfac*deltaf**(4.0/3.0)

    return bn,finf

def GKI_im_freq(u,dv):

    bn,finf = exact_constraints(dv)
    y = bn**(0.5)*u
    y2 = y*y
    cp = (1.219946,0.973063,0.42106,1.301184,1.007578)

    inum = 1. - cp[0]*y + cp[1]*y2
    iden = 1. + y2*(cp[2] + y2*(cp[3] + y2*(cp[4] + y2*(cp[1]/gam)**(16./7.) )))
    interp = 1./gam*inum/iden**(7./16.)
    fxcu = -cc*bn**(3./4.)*interp + finf
    return fxcu

def g_rMCP07(q,u,dv):

    f0 = alda(dv,x_only=False,param='PW92')
    fxc_q = mcp07_static(q,dv,param='PW92')

    fp = {'a': 3.846991, 'b': 0.471351, 'c': 4.346063, 'd': 0.881313}
    kscr = dv['kF']*(fp['a'] + fp['b']*dv['kF']**(1.5))/(1. + dv['kF']**2)

    sclfun = (dv['rs']/fp['c'])**2

    pscl = sclfun + (1. - sclfun)*np.exp(-fp['d']*(q/kscr)**2)
    fxc_omega = GKI_im_freq(u*pscl,dv)
    fxc = (1.0 + np.exp(-(q/kscr)**2)*(fxc_omega/f0 - 1.0))*fxc_q

    return -q**2*fxc/(4.*pi)
