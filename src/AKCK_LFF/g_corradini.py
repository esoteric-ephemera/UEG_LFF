import numpy as np

from AKCK_LFF.alda import alda, lda_derivs

pi = np.pi

def g_corradini(q,dv):

    Q2 = (q/dv['kF'])**2

    ec,d_ec_d_rs = lda_derivs(dv,param='PW92')
    d_rs_ec_drs = ec + dv['rs']*d_ec_d_rs

    fxc_alda = alda(dv,x_only=False,param='PW92')
    A = -dv['kF']**2*fxc_alda/(4.*pi)

    a1 = 2.15
    a2 = 0.435
    b1 = 1.57
    b2 = 0.409
    B = (1. + dv['rsh']*(a1 + a2*dv['rs']))/(3. + dv['rsh']*(b1 + b2*dv['rs']))

    C = -pi*d_rs_ec_drs/(2.*dv['kF'])

    g = B/(A - C)
    alpha = 1.5*A/(dv['rsh']**(0.5)*B*g)
    beta = 1.2/(B*g)

    G = C*Q2 + B*Q2/(g + Q2) + alpha*Q2**2 *np.exp(-beta*Q2)

    return G

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    rs_l = [2,5,10]
    ql = np.linspace(1.,3.5,2000)

    for rs in rs_l:
        dv = {'rs': rs, 'rsh': rs**(0.5), 'n': 3./(4.*pi*rs**3),
            'kF': (9*pi/4.)**(1./3.)/rs }
        zl = ql*dv['kF']

        plt.plot(ql,g_corradini(zl,dv),label='$r_s={:}$'.format(rs))
    plt.xlim(1.,4.)
    plt.ylim(0.2,1.6)
    plt.legend()
    plt.show()
