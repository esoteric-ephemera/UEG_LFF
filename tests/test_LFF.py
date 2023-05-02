import unittest

from AKCK_LFF.fitted_LFF import g_plus_new, g_minus_new

def test_LFF():
    tarr = []
    with open('./testdat.csv','r') as tfl:
        for irow, arow in enumerate(tfl):
            if irow == 0:
                continue
            tarr.append([float(x.strip()) for x in arow.split(',')])
    Ntest = len(tarr)

    thresh = 1.e-12

    tswt = unittest.TestSuite()

    for itest in range(Ntest):

        def test_g_plus():
            assert( abs(g_plus_new(tarr[itest][1],tarr[itest][0]) - tarr[itest][2]) \
                < thresh )
        tswt.addTest(unittest.FunctionTestCase(test_g_plus))

        def test_g_minus():
            assert( abs(g_minus_new(tarr[itest][1],tarr[itest][0]) - tarr[itest][3]) \
                < thresh )

        tswt.addTest(unittest.FunctionTestCase(test_g_minus))

    return tswt

if __name__ == "__main__":

    trun = unittest.TextTestRunner()
    trun.run(test_LFF())
