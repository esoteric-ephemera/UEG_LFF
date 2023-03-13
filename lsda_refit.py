import numpy as np
from scipy.optimize import least_squares

from QMC_data import get_ck_chi_enh, get_CA_ec, get_HM_QMC_dat, get_AD_DMC_dat

pi = np.pi

c00 = (1. - np.log(2.))/pi**2
c01 = c00/2

c10 = 0.046644#0.093841/2
c11 = 0.025599#0.051475/2

kf_to_rs = (9.*pi/4.)**(1./3.)
c0_alpha = -1./(6.*pi**2)
PT_integral = 0.5315045266#0.531504
c1_alpha = (np.log(16.*pi*kf_to_rs) - 3. + PT_integral )/(6.*pi**2)

def spinf(z,pow):
    opz = np.minimum(2,np.maximum(0.0,1+z))
    omz = np.minimum(2,np.maximum(0.0,1-z))
    return (opz**pow + omz**pow)/2.0

def ec_hd(rs,A,B,C,D,E,F):
    lnrs = np.log(rs)
    return A*lnrs + B + rs*(C*lnrs + D) + rs**2*(E*lnrs + F)

def ec_ld(rs,gamma,beta1,beta2):
    return -gamma/(1. + beta1*rs**(0.5) + beta2*rs)

def get_cont_pars(A,B,C,gamma,beta1,beta2):

    tden = 1. + beta1 + beta2
    tden2 = tden*tden
    f1 = - gamma*(beta1 + beta2)/tden
    f2 = gamma*(beta1 + 2.*beta2)/(2.*tden2)
    f3 = -gamma*( (beta1 + 2.*beta2)**2/tden + beta1/2.  )/(2*tden2)

    D =  4*A - 4*B + 2*C + 4*f1 - 3*f2 + f3
    E =  3*A - 2*B +   C + 2*f1 - 2*f2 + f3
    F = -4*A + 3*B - 2*C - 3*f1 + 3*f2 - f3

    return D, E, F

def eps_c_interp(rs,A,B,C,gamma,beta1,beta2):

    D, E, F = get_cont_pars(A,B,C,gamma,beta1,beta2)

    if hasattr(rs,'__len__'):
        epsc = np.zeros(rs.shape)
        rmsk = rs <= 1.
        epsc[rmsk] = ec_hd(rs[rmsk],A,B,C,D,E,F)

        rmsk = rs > 1.
        epsc[rmsk] = ec_ld(rs[rmsk],gamma,beta1,beta2)
    else:

        if rs <= 1.:
            epsc = ec_hd(rs,A,B,C,D,E,F)
        elif rs > 1.:
            epsc = ec_ld(rs,gamma,beta1,beta2)

    return epsc

def eps_c(rs,z,fps):

    cc0 = {'A': c00, 'B': c10, 'C': fps[0], 'gamma': fps[1],
        'beta1': fps[2], 'beta2': fps[3] }

    cc1 = {'A': c01, 'B': c11, 'C': fps[4], 'gamma': fps[5],
        'beta1': fps[6], 'beta2': fps[7] }

    cca = {'A': abs(c0_alpha), 'B': c1_alpha, 'C': fps[8], 'gamma': fps[9],
        'beta1': fps[10], 'beta2': fps[11] }

    ec0 = eps_c_interp(rs,cc0['A'],cc0['B'],cc0['C'],cc0['gamma'],cc0['beta1'],\
        cc0['beta2'])

    ec1 = eps_c_interp(rs,cc1['A'],cc1['B'],cc1['C'],cc1['gamma'],cc1['beta1'],\
        cc1['beta2'])

    mac = eps_c_interp(rs,cca['A'],cca['B'],cca['C'],cca['gamma'],cca['beta1'],\
        cca['beta2'])

    fz_den = (2.**(1./3.)-1.)
    fdd0 = 4./9./fz_den
    dx_z = spinf(z,4./3.)
    fz = (dx_z - 1.)/fz_den

    z4 = z**4
    fzz4 = fz*z4

    ec = ec0 - mac/fdd0*(fz - fzz4) + (ec1 - ec0)*fzz4

    return ec

def chi_enh(rs,A,B,C,g,b1,b2):
    malpha_c = eps_c_interp(rs,A,B,C,g,b1,b2)
    chi_s_chi_p = 1. - rs/(pi*kf_to_rs) - 3.*(rs/kf_to_rs)**2*malpha_c
    return 1./chi_s_chi_p

def fit_LSDA():

    chi_rs, chi_QMC, chi_ucrt = get_ck_chi_enh()
    Nchi = chi_rs.shape[0]

    CA_d, NCA = get_CA_ec()
    HM_d, NHM = get_HM_QMC_dat()
    AD_d, NAD = get_AD_DMC_dat()

    Npts = Nchi + NCA + NHM + NAD

    def fobj(ps):

        res = np.zeros(Npts)
        chi = chi_enh(chi_rs,abs(c0_alpha),c1_alpha,*ps[8:])
        res[:Nchi] = (chi - chi_QMC)/chi_ucrt

        i = Nchi
        for az in CA_d:
            j = i + CA_d[az].shape[0]
            res[i:j] = (eps_c(CA_d[az][:,0],1.*az,ps) - CA_d[az][:,1])/CA_d[az][:,2]
            i = j

        i = Nchi + NCA
        for adict in [HM_d, AD_d]:
            for ars in adict:
                j = i + adict[ars].shape[0]
                res[i:j] = (eps_c(ars,adict[ars][:,0],ps) - adict[ars][:,1])/adict[ars][:,2]
                i = j

        return res

    PZ_81_d = {
        'C': [0.0020, 0.0007], 'gamma': [0.1423, 0.0843],
        'beta1': [1.0529, 1.3981], 'beta2': [0.3334, 0.2611]
    }

    fz_den = (2.**(1./3.)-1.)
    fdd0 = 4./9./fz_den
    for asymb in ['C','gamma']:
        PZ_81_d[asymb].append(PZ_81_d[asymb][0]*(1. - fdd0) + PZ_81_d[asymb][1]*fdd0)
    for asymb in ['beta1','beta2']:
        PZ_81_d[asymb].append(0.5*sum(PZ_81_d[asymb]))

    ips = []
    bdsu = []
    bdsl = []
    for i in range(3):
        bdsl += [-1.e2, 0., 0., 0.]
        bdsu += [1.e2, 1.e2, 1.e2, 1.e2]
        for asymb in ['C','gamma','beta1','beta2']:
            ips.append(PZ_81_d[asymb][i])

    #print(np.sum(fobj(ips)**2))
    lsqres = least_squares(fobj,ips,bounds = (bdsl,bdsu))
    print(lsqres)
    #print(np.sum(fobj(lsqres.x)**2))

    return

if __name__ == "__main__":

    fit_LSDA()
