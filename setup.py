from __future__ import division, absolute_import, print_function
import setuptools
from numpy.distutils.core import setup, Extension
from pybind11.setup_helpers import Pybind11Extension
import glob
import servalcat

ext_modules = [
    Pybind11Extension(
        "servalcat.ext",
        sorted(glob.glob("src/*.cpp") + ["gemmi/src/"+x for x in ("topo.cpp", "monlib.cpp", "polyheur.cpp", "resinfo.cpp", "riding_h.cpp")]), 
        include_dirs=["gemmi/include"],
    ),
]

setup(name='servalcat',
    version=servalcat.__version__,
    author='Keitaro Yamashita and Garib N. Murshudov',
    url='https://github.com/keitaroyam/servalcat',
    description= 'Structure refinement and validation for crystallography and single particle analysis',
    license='MPL-2.0',
    packages=setuptools.find_packages(),
    install_requires=['numpy>=1.15','scipy','pandas>=0.24.2', 'gemmi==0.6.0'],
    entry_points={
      'console_scripts': [
          'servalcat = servalcat.command_line:main',
          'refmacat  = servalcat.refmac.refmac_wrapper:command_line',
                          ],
      },
    ext_modules=ext_modules,
    zip_safe= False)
