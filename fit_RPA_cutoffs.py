import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from corr import get_ec_GK, ec_rpa_unp

def gen_RPA_dat():
    rsl = [0.1]#[1.,2.,3.,4.,5.,10.,20.,40.,60.,80.,100.,120.]
    datl = np.zeros((len(rsl),5))
    uc = 4.0
    qc = 4.0
    maxinc = 100
    for irs, rs in enumerate(rsl):
        lpe = 1e20
        for iq in range(maxinc):
            for iu in range(maxinc):
                ec_rpa = get_ec_GK(rs,fxc='RPA',uc=uc,qc=qc)
                tpe = 100*(1. - ec_rpa/ec_rpa_unp(rs))
                if abs(tpe) < 1.:
                    break

                #rel_pe = 200*abs(lpe - tpe)/max(1.e-12,abs(lpe+tpe))
                adiff = abs(lpe - tpe)
                if adiff < .05:
                    break

                print(rs,qc,uc,tpe,adiff)
                lpe = tpe
                uc += .5

            if abs(tpe) < 1.:
                break

            qc += 0.5

        datl[irs,0] = rs
        datl[irs,1] = qc
        datl[irs,2] = uc
        datl[irs,3] = ec_rpa
        datl[irs,4] = tpe
        print(rs, qc,  uc, ec_rpa, tpe)
        np.savetxt('./ec_data/RPA_cutoffs.csv',datl[:irs+1,:],delimiter=',',\
            header='rs, q cutoff (1/kf) ,freq cutoff (1/kf**2), ec_rpa, PE (%)')
    return

def qcut(x,c):

    f = np.zeros(x.shape)
    tmsk = x <= 5.
    f[tmsk] = c[0] + c[1]*x[tmsk]

    tmsk = (5. < x) & (x <= 60.)
    f[tmsk] = c[0] + 5.*c[1] + c[2]*(x[tmsk] - 5.) + c[3]*(x[tmsk] - 5.)**2

    tmsk = (60. < x)
    f[tmsk] = c[0] + 5.*c[1] + 55.*(c[2] + 55.*c[3]) + c[4]*(x[tmsk] - 60.)

    return f

#gen_RPA_dat() ; exit()

tdat = np.genfromtxt('./ec_data/RPA_cutoffs.csv',delimiter=',',skip_header=1)

freq_cut = lambda x, c : c[0] + c[1]*x**(0.25) + c[2]*x**(0.5)

def freq_cut(x,c):

    bkpt = 40.
    ff = np.zeros(x.shape)
    tmsk = x <= bkpt

    ff[tmsk] = c[0] + c[1]*x[tmsk]**c[2]

    tmsk = x > bkpt
    ff[tmsk] = c[0] + c[1]*bkpt**c[2] + (x[tmsk] - bkpt)**c[3]

    return ff

fobj = lambda c : freq_cut(tdat[:,0],c) - tdat[:,2]
fres = least_squares(fobj,np.ones(4))
print(fres)
print(*fres.x)

qobj = lambda c : qcut(tdat[:,0],c) - tdat[:,1]
qres = least_squares(qobj,np.ones(5))
print(qres)
print(*qres.x)

rsl = np.linspace(.1,130.,4000)

plt.scatter(tdat[:,0],tdat[:,2],color='darkblue')
plt.plot(rsl,freq_cut(rsl,fres.x),color='darkblue')
plt.show()

plt.scatter(tdat[:,0],tdat[:,1],color='darkorange')
plt.plot(rsl,qcut(rsl,qres.x),color='darkorange')

plt.xscale('log')
plt.yscale('log')

plt.show()

tstr = 'FREQ cut pars:\n'
for ipar, apar in enumerate(fres.x):
    lchar = ', '
    if ipar == len(fres.x)-1:
        lchar = '\n\n'
    tstr += '{:.6f}{:}'.format(apar,lchar)
tstr += 'WVVCTR cut pars:\n'
for ipar, apar in enumerate(qres.x):
    lchar = ', '
    if ipar == len(qres.x)-1:
        lchar = '\n'
    tstr += '{:.6f}{:}'.format(apar,lchar)

print(tstr)
