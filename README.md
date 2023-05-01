# Analytic parameterizations of the UEG local field factors

## Maintainer: Aaron Kaplan
***

This CC0-licensed repo contains Python3 libraries for computing the UEG local field factors (LFFs) developed by Aaron Kaplan and Carl Kukkonen in:

A.D. Kaplan and C.A. Kukkonen, "QMC-consistent static spin and density local field factors for the uniform electron gas", arXiv:2303.08626 (2023). *In review at Phys. Rev. B*.

Please cite the previously listed paper when using this code or the results of our paper.

## Basic use of the fitted LFFs

To use the fitted LFFs in Python code, type
*from fitted_LFF import g_plus_new, g_minus_new*
The arguments of both *g_plus_new* and *g_minus_new* are, in order, the wavevector (units: inverse bohr), and the Wigner-Seitz (WS) radius / *r*<sub>s</sub> (units: bohr).

Both *g_plus_new* and *g_minus_new* are wrappers around the parent structure *simple_LFF*.
This function takes the wavevector (units: inverse bohr), WS radius (units: bohr), coefficients (dimensionless), and variant (string, either *G<sub>+<\sub>* by setting *var == "+""*, or *G<sub>-<\sub>* by setting *var == "-"*).
*var* controls which set of asymptotic coefficients are used.

## Detailed breakdown of the repo contents

These are listed in order of their appearance on the Github repo.

# Directories

* **data_files**
  * Files of the form *CK_G\*\_rs\_\*.csv* contain Quantum Monte Carlo (QMC) data on *G<sub>+<\sub>* and *G<sub>-<\sub>* as a function of the wavevector (units: Fermi wavevector) from Ref. 1 (see **References** section below).
  Individual files are for a fixed value of the WS radius (units: bohr).
  When *plus* appears before *\_rs\_* in the file name, data for *G<sub>+<\sub>* is tabulated.
  Identically, when *minus* appears before *\_rs\_* in the file name, data for *G<sub>-<\sub>* is tabulated.
  The value of the WS radius appears after *\_rs\_* and before *.csv*.

  * Files of the form *MCS_Gplus_rs_\*.csv* present QMC data for *G<sub>+<\sub>* at a fixed value of the WS radius (same naming convention as before), but from Ref. 2.

* 

## References

1. C.A. Kukkonen and K. Chen, Phys. Rev. B **104**, 195142 (2021).
DOI: 10.1103/PhysRevB.104.195142

2. S. Moroni, D. M. Ceperley, and G. Senatore, Phys. Rev. Lett. **75**, 689 (1995).
DOI: 10.1103/PhysRevLett.75.689
