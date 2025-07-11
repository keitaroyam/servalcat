# Servalcat
[![Build](https://github.com/keitaroyam/servalcat/workflows/CI/badge.svg)](https://github.com/keitaroyam/servalcat/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/servalcat?color=blue)](https://pypi.org/project/servalcat/)

**S**tructur**e** **r**efinement and **val**idation for **c**rystallography and single p**a**r**t**icle analysis

Servalcat implements pipelines that use Refmac5:
  * `servalcat refine_spa`: cryo-EM SPA refinement pipeline
  * `servalcat refine_cx`: small molecule crystallography

and a Refmac5 controller
  * `refmacat`: behaves as Refmac, but uses GEMMI for restraint generation instead of MAKECIF

Now “No Refmac5” refinement programs have been actively developed:
  * `servalcat refine_geom`: geometry optimization
  * `servalcat refine_spa_norefmac`: "No Refmac" version of refine\_spa
  * `servalcat refine_xtal_norefmac`: crystallographic refinement

Also, it has several utility commands: `servalcat util`.

## Installation

```
pip install servalcat
```
will install the stable version.

The required GEMMI version is now [v0.7.3](https://github.com/project-gemmi/gemmi/releases/tag/v0.7.3). It may not work with the latest gemmi code from the github. The policy is in the main branch I only push the code that works with the latest package of GEMMI.

To use the Refmac5 related commands, you also need to install [CCP4](https://www.ccp4.ac.uk/). For "No Refmac5" commands, you may just need [the monomer library](https://github.com/MonomerLibrary/monomers) if CCP4 is not installed.

**Notice:**
From ver. 0.4.6, Servalcat is no longer python-only package and has some C++ code. If you build Servalcat by yourself, probably you also need to build GEMMI using the same compiler.

## Usage
Please read the documentation: https://servalcat.readthedocs.io/en/latest/

## References
* [Yamashita, K., Wojdyr, M., Long, F., Nicholls, R. A., Murshudov, G. N. (2023) "GEMMI and Servalcat restrain REFMAC5" *Acta Cryst.* D**79**, 368-373](https://doi.org/10.1107/S2059798323002413)
* [Yamashita, K., Palmer, C. M., Burnley, T., Murshudov, G. N. (2021) "Cryo-EM single particle structure refinement and map calculation using Servalcat" *Acta Cryst. D***77**, 1282-1291](https://doi.org/10.1107/S2059798321009475)
