import numpy as np

pi = np.pi
kf_to_rs = (9.*pi/4.)**(1./3.)


def spinf(z,pow):
    opz = np.minimum(2,np.maximum(0.0,1+z))
    omz = np.minimum(2,np.maximum(0.0,1-z))
    return (opz**pow + omz**pow)/2.0

def ts(rs,z):
    # kinetic energy per electron
    ts0 = 3./10.*(kf_to_rs/rs)**2
    ds = spinf(z,5./3.)
    return ts0*ds

def epsx(rs,z):
    # exchange energy per electron
    ex0 = -3./(4.*pi)*kf_to_rs/rs
    dx = spinf(z,4./3.)
    return ex0*dx

def ke_ex(rs,z):
    return ts(rs,z) + epsx(rs,z)

def get_ck_chi_enh():
    """
        Table I of C.A. Kukkonen and K. Chen,
        Phys. Rev. B 104, 195142 (2021),
        doi: 10.1103/PhysRevB.104.195142
    """
    rsl = np.array([1.,2.,3.,4.,5.])
    chi_chi0 = np.array([1.152,1.296,1.438,1.576,1.683])
    ucrt = np.array([2.e-3,6.e-3,9.e-3,9.e-3,1.5e-2])

    #kfl = kf_to_rs/rsl
    #ac = kfl**2/3.*(1./chi_chi0 - 1. + 1./(pi*kfl))
    return rsl, chi_chi0, ucrt
    #return rsl, ac, kfl**2/3./ucrt#

def get_CA_ec():

    etot_d = {
        0 : np.transpose((
            np.array([1.0,2.0,5.0,10.0,20.0,50.0,100.0]),
            np.array([1.174,0.0041,-0.1512,-0.10675,-0.06329,-0.02884,-0.015321]),
            np.array([1.e-3,4.e-4,1.e-4,5.e-5,3.e-5,1.e-5,5.e-6])
        )),
        1 : np.transpose((
            np.array([2.0,5.0,10.0,20.0,50.0,100.0]),
            np.array([0.2517,-0.1214,-0.1013,-0.06251,-0.02878,-0.015340]),
            np.array([6.e-4,2.e-4,1.e-4,3.e-5,2.e-5,5.e-6])
        ))
    }

    npts = 0
    for az in etot_d:
        etot_d[az][:,1] /= 2. # conversion from Rydberg
        etot_d[az][:,2] /= 2.

        etot_d[az][:,1]  -= ke_ex(etot_d[az][:,0],1.*az)
        npts += etot_d[az].shape[0]

    return etot_d, npts


def get_HM_QMC_dat():
    """
        Tables II-IV of Supplemental Material of
        M. Holzmann and S. Moroni, Phys. Rev. Lett. 124, 206404 (2020).
        doi: 10.1103/PhysRevLett.124.206404

        These are zero-variance extrapolated DMC values without SJ
    """

    etot_d = {
        70 : np.transpose(( \
            np.array([0., 0.42, 0.61, 0.79, 1.0]), \
            np.array([-21.37090, -21.36832, -21.36563,-21.36097, -21.34945]), \
            np.array([2.6e-4, 1.9e-4, 1.3e-4, 9.6e-4, 7.7e-4]) \
            )),
        100: np.transpose(( \
            np.array([0., 0.18, 0.42, 0.61, 0.79, 1.0]), \
            np.array([-15.38914, -15.38876, -15.38857, -15.38811, -15.38679, -15.38325]),\
            np.array([1.7e-4, 1.5e-4, 9.e-5, 2.1e-4, 9e-5, 3.e-5])
            )),
        120: np.transpose((\
            np.array([0., 0.18, 0.42, 0.61, 0.79, 1.0]), \
            np.array([-12.98786, -12.98758, -12.98760, -12.98748, -12.98663, -12.98470]), \
            np.array([2.7e-4, 1.2e-4, 5.e-5, 1.7e-4, 8.e-5, 7.e-5])
            ))
    }

    # conversion from mRy to Hartree
    conv_f = 5.e-4

    npts = 0
    ec_d = {}
    for rs in etot_d:
        etot_d[rs][:,1] *= conv_f
        etot_d[rs][:,2] *= conv_f
        npts += etot_d[rs].shape[0]

        ec_d[rs] = etot_d[rs].copy()
        etot_no_c = ke_ex(rs,ec_d[rs][:,0])
        ec_d[rs][:,1] -= etot_no_c

    return ec_d, npts

def get_AD_DMC_dat():

    """
        Data from Table IV of S. Azadi and N.D. Drummond,
        Phys. Rev. B 105, 245135 (2022),
        DOI: 10.1103/PhysRevB.105.245135
    """

    # first column is zeta, second is total energy in mHa, third is uncertainty in mHa
    etot_d = {
        30 : np.transpose((
            np.array([0.0, 0.5, 1.0]),
            -np.array([22.617,22.5862,22.4804]),
            np.array([8.e-3,5.e-4,6.e-4])
        )),
        40 : np.transpose((
            np.array([0.0, 0.5, 1.0]),
            -np.array([17.612,17.597,17.555]),
            np.array([4.e-3,2.e-3,2.e-3]),
        )),
        60 : np.transpose((
            np.array([0.0, 0.5, 1.0]),
            -np.array([12.254,12.2492,12.2413]),
            np.array([3.e-3,4.e-4,1.e-4])
        )),
        80 : np.transpose((
            np.array([0.0, 0.5, 1.0]),
            -np.array([9.4250,9.421,9.4242]),
            np.array([9.e-4,1.e-3,2.e-4])
        )),
        100 : np.transpose((
            np.array([0.0, 0.5, 1.0]),
            -np.array([7.6702,7.669976,7.6717]),
            np.array([4.e-4,7.e-6,9.e-4])
        ))
    }

    npts = 0
    for rs in etot_d:
        etot_d[rs][:,1] *= 1.e-3 # conversion from mHa to Hartree
        etot_d[rs][:,2] *= 1.e-3
        etot_d[rs][:,1] -= ke_ex(rs,etot_d[rs][:,0])
        npts += etot_d[rs].shape[0]

    return etot_d, npts

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    hm_dat, nhm = get_HM_QMC_dat()
    ad_dat, nad = get_AD_DMC_dat()

    zl = np.linspace(0.,1.,2000)
    fz = ((1. + zl)**(4./3.) + (1. - zl)**(4./3.) - 2.)/(2.**(4./3.) - 2.)
    plt.plot(zl,fz)
    for adict in [hm_dat,ad_dat]:
        for ars in adict:
            fz = (adict[ars][:,1] - adict[ars][0,1])/(adict[ars][-1,1] - adict[ars][0,1])
            plt.scatter(adict[ars][:,0],fz)
    plt.show(); exit()

    # sanity check plot, should look like Fig. 3 of Holzmann and Moroni
    etot_d,_ = get_HM_QMC_dat()
    colors = ['purple','darkgreen','darkblue']
    for irs, rs in enumerate(etot_d):
        etot_d[rs][:,1] += ke_ex(rs,etot_d[rs][:,0])
        epol = rs**(3./2.)*(etot_d[rs][:,1] - etot_d[rs][0,1])*2.e3
        sclucrt =  rs**(3./2.)*etot_d[rs][:,2]*2.e3
        plt.errorbar(etot_d[rs][:,0],epol,yerr=sclucrt,color=colors[irs],\
            markersize=3,marker='o',linewidth=0,elinewidth=1.5,\
            label='$r_\\mathrm{s}='+'{:}$'.format(rs))
    #plt.xlim(0.,1.)
    plt.ylim(0.,15.)
    plt.show()
    plt.close()

    colors = ['r','b','g','darkblue','k']
    etot_d, _ = get_AD_DMC_dat()
    for irs, rs in enumerate(etot_d):
        plt.errorbar(etot_d[rs][:,0],etot_d[rs][:,1],yerr=etot_d[rs][:,2],\
            color=colors[irs],\
            markersize=3,marker='o',linewidth=0,elinewidth=1.5,\
            label='$r_\\mathrm{s}='+'{:}$'.format(rs))
    plt.ylim(-9.e-3,-2.e-3)
    plt.show()
