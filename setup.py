from __future__ import division, absolute_import, print_function
import setuptools
from numpy.distutils.core import setup, Extension

version = {}
with open("./servalcat/config.py") as fp:
    exec(fp.read(), version)

setup(name='servalcat',
    version=version['__version__'],
    description= 'Structure refinement and validation for crystallography and single particle analysis',
    license='MPL-2.0',
    packages=setuptools.find_packages(),
    install_requires=['numpy','scipy','gemmi'],
    entry_points={
      'console_scripts': [
          'servalcat = servalcat.command_line:main',
                          ],
      },
    zip_safe= False)
