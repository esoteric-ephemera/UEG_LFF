# Analytic parameterizations of the UEG local field factors

## Maintainer: Aaron Kaplan
***

This CC0-licensed repo contains Python3 libraries for computing the uniform electron gas (UEG) local field factors (LFFs) developed by Aaron Kaplan and Carl Kukkonen in:

A.D. Kaplan and C.A. Kukkonen, "QMC-consistent static spin and density local field factors for the uniform electron gas", arXiv:2303.08626 (2023). *In review at Phys. Rev. B*.

Please cite the previously listed paper when using this code or the results of our paper.

## Basic use of the fitted LFFs

To use the fitted LFFs in Python code, type
*from AKCK_LFF.fitted_LFF import g_plus_new, g_minus_new*
The arguments of both *g_plus_new* and *g_minus_new* are, in order, the wavevector (units: inverse bohr), and the Wigner-Seitz (WS) radius / *r*<sub>s</sub> (units: bohr).

Both *g_plus_new* and *g_minus_new* are wrappers around the parent structure *simple_LFF*.
This function takes, in order, the wavevector (units: inverse bohr), WS radius (units: bohr), coefficients (dimensionless), and variant (string, either *G<sub>+* by setting *var == "+""*, or *G<sub>-* by setting *var == "-"*).
*var* controls which set of asymptotic coefficients are used.

## Detailed breakdown of the repo contents

These are listed in order of their appearance on the Github repo.
Unless specified otherwise, the units used in the following are hartree atomic units: lengths are measured in bohr radii, and energies in hartree (1 hartree = 2 Ry).

# Directories

* **data_files**
  * Files of the form *CK_G\*\_rs\_\*.csv* contain Quantum Monte Carlo (QMC) data on *G<sub>+* and *G<sub>-* as a function of the wavevector (units: Fermi wavevector) from Ref. 1 (see **References** section below).
  Individual files are for a fixed value of the WS radius (units: bohr).
  When *plus* appears before *\_rs\_* in the file name, data for *G<sub>+* is tabulated.
  Identically, when *minus* appears before *\_rs\_* in the file name, data for *G<sub>-* is tabulated.
  The value of the WS radius appears after *\_rs\_* and before *.csv*.

  * Files of the form *MCS_Gplus_rs_\*.csv* present QMC data for *G<sub>+* at a fixed value of the WS radius (same naming convention as before), but from Ref. 2.

* **ec_data**

  * Files of the form *eps_c_\*.csv** tabulate computed correlation energies (units: hartree, from the adiabatic connection fluctuation dissipation theorem) as a function of the WS radius.
  Some keywords are:
      * *COR*: the wavevector-dependent exchange-correlation (xc) kernel of Corradini *et al.* [3]
      * *NEW*: the present wavevector-dependent density LFF, or xc kernel
      * *RAD*: the wavevector- and frequency-dependent Richardson-Ashcroft xc kernel [4]
      * *RAS*: the static limit of the Richardson-Ashcroft kernel [4], which is still wavevector-dependent
      * *RPA*: the random phase approximation, whereby the xc kernel is set to zero
      * *rMCP07*: the wavevector- and frequency-dependent revised MCP07 kernel of Kaplan *et al.* [5]
      * The suffix *_GKQ* indicates a result from globally-adaptive Gauss-Kronrod quadrature, whereas no suffix indicates a result using the RPA cutoffs

  * *RPA_cutoffs.csv* contain the numeric RPA cutoffs described in the work
  * *eps_c_err.pdf* is Fig. 2 of the main text
  * *RPA_sanity.tex* is Table S2

* **figs** contains the figures of the main text and supplemental material

* **figs_from_fit** contains plots similar to those of the manuscript, but generated immediately after the fitting procedure.
These plots may use parameters in the LFFs that are not truncated at a fixed number of digits.

* **fitted_LFF_pars** contain the fitted parameters in the LFFs as both CSV- and LaTeX-formatted tables.
Initial guesses for the parameters are contained in *optpars_g\*.csv*.

* **quad_grids** is generated by *gauss_quad.py*, and contains grid points and weights for Gaussian quadrature
  * *GKQ* indicates Gauss-Kronrod quadrature
  * *GLQ* indicates Gauss-Legendre quadrature

* **stiffness_refit** contains the revised pars in the correlation spin stiffness (*alphac_pars_rev.tex*), and the susceptibility enhancement in tabular (*chi_enhance.tex*) and pictoral (*suscep_enhance.pdf*) form

# Python files

* *PW92.py* contains both the Perdew-Wang parameterization of the UEG correlation energy [6] in *ec_pw92*, and the Pade approximant of the UEG on-top pair distribution function [7] in *g0_unp_pw92_pade*

* *PZ81.py* contains the Perdew-Zunger parameterization of the UEG correlation energy [8] in *ec_pz81* and of the correlation spin-stiffness in *alpha_c_pz81*

* *QMC_data.py* contains QMC-computed UEG correlation energies from previous works.
Citations are given throughout the file.
This file is used in *stiffness_refit.py*.

* *alda.py* provides the adiabatic local density approximation to the xc kernel, which depends only on the WS radius.
This file is modified from a previous repo maintained by the author, https://github.com/esoteric-ephemera/tc21/tree/master/dft
Note that function *alda* thus requires a dictionary entry *dv* which takes the density *n*, Fermi wavevector *kF*, WS radius *rs*, and square-root of the WS radius *rsh* as keys.
Possible optional keywords are *x_only* for exchange only (default: False), and parameterization of the kernel (default: PZ81, PW92 is another possible option).

* *alpha_c_c1.py* recomputes the value of the next-to-leading order term in the high-density expansion of the correlation spin-stiffness, as in Eq. (19).

* *asymptotics.py* provides the asymptotic expansion coefficients of *G<sub>+* in *get_g_plus_pars*, which takes only the WS radius as input, and of *G<sub>-* in *get_g_minus_pars*, which takes the WS radius and relative spin polarization as input.

* Running *corr.py* computes the correlation energies of various density LFFs, as plotted in Fig. 2

* *fit_LFF.py* is the main fitting routine used here.
Running this file requires secondary options, specified as *key*=*value*
  * *routine*:
    * *init*: generates initial parameters for select values of the WS radius
    * *manip*: allows the user to manually manipulate the parameters in the LFF
    * *main*: main fitting routine which provides the parameters in the text
  * *var* is the variant of the LFF, either *+* for *G<sub>+* or *-* for *G<sub>-*.

* *fit_RPA_cutoffs.py* fits the numeric RPA cutoffs for the correlation energy, as described in the supplemental material

* *fitted_LFF.py* is described in the **Basic use of the fitted LFFs** section.

* *g_corradini.py* contains the density LFF *G<sub>+* of Corradini *et al.* [3] in *g_corradini*.
This function requires the wavevector (unit: inverse bohr) and a density dictionary similar to *alda* as input.

* *gauss_quad.py*, modified from https://github.com/esoteric-ephemera/tc21/tree/master/, is an all-purpose numeric integrator using adaptive mesh refinement.

* *mcp07_static.py* contains the static limit of the modified CP07 kernel of Ruzsinszky *et al.* [9], in *mcp07_static*.
This function takes the wavevector and density dictionary as inputs.
An optional keyword, *param* (default PZ81) can use either the Perdew-Zunger [8] (PZ81) or Perdew-Wang [6] (PW92) parameterization of the UEG correlation energy as input.

* *rMCP07.py* contains the wavevector- and frequency-dependent density LFF of Kaplan *et al.* [5] in *g_rMCP07*.
Note that this function ***assumes*** that the frequency is purely imaginary.
This function takes the wavevector, imaginary part of the frequency (entered as a real number), and density dictionary as input.

* *ra_lff.py* contains the various LFFs of Richardson and Ashcroft [4].
To use their density LFF *G<sub>+*, call *g_plus_ra*; to use their spin LFF *G<sub>-*, call *g_minus_ra*.
Both take the wavevector (units: bohr) and imaginary part of the frequency (units: inverse hartree energy), and WS radius (units: bohr) as inputs.
***Both assume that the frequency is purely imaginary.***

* *stiffness_refit.py* refits the correlation spin stiffness using the PW92 framework

* *surf_plots.py* produces the surface plots in the supplemental material.

## References

1. C.A. Kukkonen and K. Chen, Phys. Rev. B **104**, 195142 (2021).
DOI: 10.1103/PhysRevB.104.195142

2. S. Moroni, D. M. Ceperley, and G. Senatore, Phys. Rev. Lett. **75**, 689 (1995).
DOI: 10.1103/PhysRevLett.75.689

3. M. Corradini, R. Del Sole, G. Onida, and M. Palummo, Phys. Rev. B **57**, 14569 (1998).
DOI: 10.1103/PhysRevB.57.14569

4. C. F. Richardson and N. W. Ashcroft, Phys. Rev. B **50**, 8170 (1994).
DOI: 10.1103/PhysRevB.50.8170

5. A. D. Kaplan, N. K. Nepal, A. Ruzsinszky, P. Ballone, and J. P. Perdew, Phys. Rev. B **105**, 035123 (2022).
DOI: 10.1103/PhysRevB.105.035123

6. J.P. Perdew and Y. Wang, Phys. Rev. B **45**, 13244 (1992).
DOI: 10.1103/PhysRevB.45.13244

7. J. P. Perdew and Y. Wang, Phys. Rev. B **46**, 12947 (1992).
DOI: 10.1103/PhysRevB.46.12947,
and erratum Phys. Rev. B 56, 7018 (1997).
DOI: 10.1103/PhysRevB.56.7018

8. J. P. Perdew and A. Zunger, Phys. Rev. B **23**, 5048 (1981).
DOI: 10.1103/PhysRevB.23.5048

9. A. Ruzsinszky, N. K. Nepal, J. M. Pitarke, and J. P. Perdew, Phys. Rev. B **101**, 245135 (2020).
DOI: 10.1103/PhysRevB.101.245135
