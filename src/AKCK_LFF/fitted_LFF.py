import numpy as np

from AKCK_LFF.asymptotics import get_g_minus_pars, get_g_plus_pars

pi = np.pi
rs_to_kf = (9*pi/4.)**(1./3.)

def smooth_step(x,a,b):
    f1 = np.exp(a*b)
    f2 = np.exp(-a*x)
    f = (f1 - 1.)*f2/(1. + (f1 - 2.)*f2)
    return f

def simple_LFF(q,rs,c,var):

    kf = rs_to_kf/rs
    q2 = (q/kf)**2
    q4 = q2*q2

    if var == '+':
        CA, CB, CC = get_g_plus_pars(rs)
    elif var == '-':
        CA, CB, CC = get_g_minus_pars(rs,0.)

    alpha = c[0] + c[1]*np.exp(-abs(c[2])*rs)

    interp1 = smooth_step(q4/16.,c[3],c[4])

    asymp1 = q2*(CA + alpha*q4)
    asymp2 = CB + CC*q2
    LFF = asymp1*interp1 + asymp2*(1. - interp1)

    return LFF

def g_plus_new(q,rs):
    cps = [-0.00451760, 0.0155766, 0.422624, 3.516054, 1.015830]
    return simple_LFF(q,rs,cps,'+')

def g_minus_new(q,rs):
    cms = [-0.00105483, 0.0157086, 0.345319, 2.850094, 0.935840]
    return simple_LFF(q,rs,cms,'-')

def g_plus_dlda(q,u,rs):
    from AKCK_LFF.alda import lda_derivs
    CA, _, _ = get_g_plus_pars(rs)

    kf = rs_to_kf/rs
    x2 = (q/kf)**2

    gp_inf_x = 3./20.
    dv = {'rs': rs, 'rsh': rs**(0.5), 'kF': kf, 'n': 3./(4.*pi*rs**3)}
    eps_c,d_eps_c_d_rs = lda_derivs(dv,param='PW92')
    gp_inf_c = pi*(22.0*eps_c + 26.0*rs*d_eps_c_d_rs)/(20.*kf)
    gp_inf = gp_inf_x + gp_inf_c

    wp = (3./rs**3)**(0.5)
    v = (u/(2.*wp))**2
    intp = smooth_step(v,3.,1.)

    enh = CA*intp + gp_inf*(1. - intp)

    return enh*x2

def gplus_dyn(q,u,rs):

    kf = rs_to_kf/rs
    x2 = (q/kf)**2
    gp_w = g_plus_dlda(q,u,rs)
    gp_q = g_plus_new(q,rs)
    CA, CB, _ = get_g_plus_pars(rs)
    itp = np.exp(-CA*x2/CB)
    gp = (1. + itp*(gp_w/(CA*x2) - 1.))*gp_q
    return gp

if __name__ == "__main__":

    g_plus_new(1.,1.)
