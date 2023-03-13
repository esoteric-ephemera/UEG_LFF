import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, bisect

from QMC_data import ke_ex, get_AD_DMC_dat

#def ec_LD(rs,gam,b1,b2):
#   return -gam/(1. + b1*rs**(0.5) + b2*rs)

def ec_LD(rs,gam,b1,b2,b3):
    rsh = rs**(0.5)
    return -gam/rs + b1/rs**(1.5) + b2/rs**2 + b3/rs**(2.5)

def ec_WC(rs,d0,d1,d2):
    return -d0/rs + d1/rs**(1.5) - d2/rs**2

def etot_wrap(rs,z,gam,b1,b2,b3):
    return ec_LD(rs,gam,b1,b2,b3) + ke_ex(rs,z)

# Table VII of Azadi and Drummond
unpol_crys = [.122,.923,.287]
pol_crys = [.052,.808,.166]

# Table VIII
ufp = [.13,1.0,.32,0.]
pfp = [.062,.97,.19,0.]

#ufp = [0.438855, 1.54858, -4.28415, 6.8502]
#pfp = [0.318591, 1.49998, -4.5807, 6.86035]

# Table V
afm_crys = np.transpose(np.array([
    [80.,90.,100.,125.],
    [9.4200,8.4585,7.6769,6.24472],
    [2.e-4,1.e-4,1.e-4,6.e-5]
]))

fm_crys = np.transpose(np.array([
    [80.,90.,100.,125.],
    [9.41999,8.4581,7.6774,6.24508],
    [9.e-5,1.e-4,1.e-4,6.e-5]
]))

# Table VI
unp_fluid = np.transpose(np.array([
    [30.,40.,60.,80.,100.],
    [22.6191,17.6143,12.2556,9.4259,7.6709],
    [7.e-4,3.e-4,3.e-4,4.e-4,3.e-4]
]))

pol_fluid = np.transpose(np.array([
    [30.,40.,60.,80.,100.],
    [22.4819,17.5558,12.2418,9.4246,7.6720],
    [7.e-4,7.e-4,5.e-4,3.e-4,4.e-4]
]))


#parnms = ['gamma','beta1','beta2','beta3']
parnms = ['d0','d1','d2','d3']

for i in range(2):
    afm_crys[:,1+i] *= (-1)**(i+1)*1.e-3
    fm_crys[:,1+i] *= (-1)**(i+1)*1.e-3
    unp_fluid[:,1+i] *= (-1)**(i+1)*1.e-3
    pol_fluid[:,1+i] *= (-1)**(i+1)*1.e-3
"""
print(unp_fluid[:,0])
print(unp_fluid[:,1]-ke_ex(unp_fluid[:,0],0.))
print(pol_fluid[:,0])
print(pol_fluid[:,1]-ke_ex(pol_fluid[:,0],1.))
exit()
#"""

uobj = lambda c : (ec_LD(unp_fluid[:,0],*c) - unp_fluid[:,1] \
    + ke_ex(unp_fluid[:,0],0.))/unp_fluid[:,2]
tmp = least_squares(uobj,ufp)
print('UNP obj=',np.sum(uobj(tmp.x)**2))
ufp = tmp.x
tstr = 'UNP fluid: \n'
for ipar, apar in enumerate(parnms):
    tstr += '{:} = {:.6f}\n'.format(apar,ufp[ipar])

pobj = lambda c : (ec_LD(pol_fluid[:,0],*c) - pol_fluid[:,1] \
    + ke_ex(pol_fluid[:,0],1.))/pol_fluid[:,2]
tmp = least_squares(pobj,pfp)
print('POL obj=',np.sum(pobj(tmp.x)**2))
pfp = tmp.x
tstr += '\nPOL fluid: \n'
for ipar, apar in enumerate(parnms):
    tstr += '{:} = {:.6f}\n'.format(apar,pfp[ipar])

dinit = [.89593,1.325,.365]

tobj = lambda c : (ec_WC(afm_crys[:,0],*c) - afm_crys[:,1])/afm_crys[:,2]
afm_pars = least_squares(tobj,dinit)
tstr += '\nAFM crystal: \n'

parnms = ['d0','d1','d2']
for ipar, apar in enumerate(parnms):
    tstr += '{:} = {:.6f}\n'.format(apar,afm_pars.x[ipar])

tobj = lambda c : (ec_WC(fm_crys[:,0],*c) - fm_crys[:,1])/fm_crys[:,2]
fm_pars = least_squares(tobj,dinit)
tstr += '\nFM crystal: \n'
for ipar, apar in enumerate(parnms):
    tstr += '{:} = {:.6f}\n'.format(apar,fm_pars.x[ipar])

fig, ax = plt.subplots(figsize=(6,4))
axins = ax.inset_axes((0.6,0.1,0.35,0.4) )

rsl = np.linspace(60.,125.,5000)
ax.scatter(afm_crys[:,0],afm_crys[:,1]*afm_crys[:,0]**(1.5) + dinit[0]*afm_crys[:,0]**(0.5),color='darkblue')

rslh = rsl**(0.5)
rslth = rslh*rsl
ec0 = dinit[0]*rslh
for anax in [ax,axins]:
    anax.plot(rsl,ec_WC(rsl,*afm_pars.x)*rslth + ec0,label='AFM crystal',color='darkblue')
    anax.plot(rsl,ec_WC(rsl,*fm_pars.x)*rslth + ec0,label='FM crystal',color='gray',linestyle='-.')

rsl = np.linspace(25.,150.,5000)
rslh = rsl**(0.5)
rslth = rslh*rsl
ec0 = dinit[0]*rslh

ax.scatter(unp_fluid[:,0], unp_fluid[:,1]*unp_fluid[:,0]**(1.5) \
    + dinit[0]*unp_fluid[:,0]**(0.5), color='darkorange')
ax.scatter(pol_fluid[:,0], pol_fluid[:,1]*pol_fluid[:,0]**(1.5) \
    + dinit[0]*pol_fluid[:,0]**(0.5), color='tab:green')

print(ec_LD(1.,*ufp))
print(ec_LD(1.,*pfp))

for anax in [ax,axins]:
    anax.plot(rsl,etot_wrap(rsl,0.,*ufp)*rslth + ec0,label='UNP fluid',color='darkorange')

    anax.plot(rsl,etot_wrap(rsl,1.,*pfp)*rslth + ec0,label='POL fluid',color='tab:green',linestyle='--')

ax.legend(fontsize=12)
#plt.scatter(fm_crys[:,0],fm_crys[:,1])
#plt.plot(rsl,ec_WC(rsl,*fm_pars.x))

ax.set_xlim(25.,150.)
axins.set_xlim(86.,93.)
axins.set_ylim(1.275,1.28)

ax.set_xlabel(r'$r_\mathrm{s}$',fontsize=14)
ax.set_ylabel(r'$\left[\varepsilon + ('\
    + '{:}'.format(dinit[0])\
    +r')/r_\mathrm{s}\right]r_\mathrm{s}^{3/2}$',fontsize=14)
ax.set_ylim(1.22,1.32)

#plt.show();exit()
plt.savefig('./AD_data.pdf',bbox_inches='tight',dpi=600)

rs0 = 87.

fnm = ['UNP fluid','POL fluid','AFM crystal','FM crystal']

def funs(rsc,ifun):
    if ifun == 0:
        tfun = etot_wrap(rsc,0.,*ufp)
    elif ifun == 1:
        tfun = etot_wrap(rsc,1.,*pfp)
    elif ifun == 2:
        tfun = ec_WC(rsc,*afm_pars.x)
    elif ifun == 3:
        tfun = ec_WC(rsc,*fm_pars.x)
    return tfun

nfun = len(fnm)

tstr += "\n\nTransition rs values:\n"
for i in range(nfun):
    for j in range(i+1,nfun):
        tobj = lambda rsc : funs(rsc,i) - funs(rsc,j)
        trs0 = bisect(tobj,rs0-10.,rs0+10.)
        tstr += '{:} to {:} rs = {:.2f}\n'.format(fnm[i],fnm[j],trs0)

with open('AD_data.txt','w+') as tfl:
    tfl.write(tstr)
