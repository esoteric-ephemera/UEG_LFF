import numpy as np
import matplotlib.pyplot as plt

from asymptotics import get_g_plus_pars, gexc2

pi = np.pi

def aps(rs):
    Apos, Bpos, Cpos = get_g_plus_pars(rs)

    Dpos = -(3.*pi**2)**(4./3.)*gexc2(rs)/(2.*pi)
    return Apos, Bpos, Cpos, Dpos

#rsl = np.linspace(.1,20.,2000)
#ca, cb, cc, cd = aps(rsl)

#plt.plot(rsl,cd/(cc-ca))

rs = 1.
ca, cb, cc, cd = aps(rs)
ql = np.linspace(1.e-6,4.,5000)
q2 = ql*ql
q4 = q2*q2

ba = ca
bb = ca*cc/cb
bc = 4.*(cc/cb - cd/ca)
bd = (ca/cb)**4
gp = (ba + bb*q2)/(1. + bc*q2 + bd*q2**4)**(1./4.)

plt.plot(ql,ca*np.ones(ql.shape))
plt.plot(ql,cc + cb/q2)
plt.plot(ql,gp)
plt.ylim(-.01,1.)

plt.show()
