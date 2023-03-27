# Servalcat
[![Build](https://github.com/keitaroyam/servalcat/workflows/CI/badge.svg)](https://github.com/keitaroyam/servalcat/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/servalcat?color=blue)](https://pypi.org/project/servalcat/)

**S**tructur**e** **r**efinement and **val**idation for **c**rystallography and single p**a**r**t**icle analysis

## Installation

```
pip install servalcat
```
will install the stable version.

The required GEMMI version is now [v0.6.0](https://github.com/project-gemmi/gemmi/releases/tag/v0.6.0). It may not work with the latest gemmi code from the github. The policy is in the main branch I only push the code that works with the latest package of GEMMI.

**Notice:**
From ver. 0.4.6, Servalcat is no longer python-only package and has some C++ code. If you build Servalcat by yourself, probably you also need to build GEMMI using the same compiler.

## Usage
Please read the documentation: https://servalcat.readthedocs.io/en/latest/

## Reference
[Yamashita, K., Palmer, C. M., Burnley, T., Murshudov, G. N. (2021) "Cryo-EM single particle structure refinement and map calculation using Servalcat" *Acta Cryst. D***77**, 1282-1291](https://doi.org/10.1107/S2059798321009475)