from os import path
import numpy as np
import matplotlib.pyplot as plt

from asymptotics import get_g_plus_pars, gexc2

pi = np.pi
rs_to_kf = (9*pi/4.)**(1./3.)

rs_l = [1,2,5,10]

tdat_CK = {}
tdat_MCS = {}
npts = 0

for irs, rs in enumerate(rs_l):
    ckgpf = './data_files/CK_Gplus_rs_{:}.csv'.format(int(rs))
    mcsgpf = './data_files/MCS_Gplus_rs_{:}.csv'.format(int(rs))
    if path.isfile(ckgpf):
        tdat_CK[rs] = np.genfromtxt(ckgpf,delimiter=',',skip_header=1)

    if path.isfile(mcsgpf):
        tdat_MCS[rs] = np.genfromtxt(mcsgpf,delimiter=',',skip_header=1)

def get_mixing(rs,x,gp):

    # x = q/kf

    #kf = rs_to_kf/rs
    q2 = x**2#(q/kf)**2

    ap, bp, cp = get_g_plus_pars(rs)
    #dp = -(3.*pi**2)**(4./3.)*gexc2(rs)/(2.*pi)
    LQE = cp*q2 + bp
    q0 = 0.98*(bp/(ap - cp))**(0.5)
    dp = ((cp - ap)*q0**2 + bp)/q0**4
    print(rs,dp,q0)
    SQE = q2*ap + dp*q2*q2

    tdiff = SQE - LQE
    eps = 1.e-12
    tmsk = np.abs(tdiff) < eps
    tdiff[tmsk] = eps*np.sign(tdiff[tmsk])

    mix = (gp - LQE)/tdiff
    return mix

"""
rsl = np.exp(np.linspace(np.log(.1),np.log(100.),5000))
#ql = np.linspace(0.,3.,2000)
ap, bp, cp = get_g_plus_pars(rsl)
#plt.plot(ql,ql**2*ap)
#plt.plot(ql,ql**2*cp + bp)
plt.xscale('log')
plt.show()
exit()
#"""

mix_opt_CK = {}
mix_opt_MCS = {}
for rs in tdat_CK:
    tmix = get_mixing(rs,tdat_CK[rs][:,0],tdat_CK[rs][:,1])
    mix_opt_CK[rs] = np.transpose((tdat_CK[rs][:,0],tmix))
    plt.scatter(tdat_CK[rs][:,0],tmix)

for rs in tdat_MCS:
    tmix = get_mixing(rs,tdat_MCS[rs][:,0],tdat_MCS[rs][:,1])
    mix_opt_MCS[rs] = np.transpose((tdat_MCS[rs][:,0],tmix))
    plt.scatter(tdat_MCS[rs][:,0],tmix)

ql = np.linspace(0.,3.,2000)

alp = 10.
ff = (alp - 1.)/(alp - 2. + alp**((ql/2.)**4))
#plt.plot(ql,ff)
plt.ylim(-1.,2.)
plt.show()
