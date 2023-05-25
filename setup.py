from __future__ import division, absolute_import, print_function
import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools import distutils
try:
    from pybind11.setup_helpers import Pybind11Extension
except ImportError:
    from setuptools import Extension as Pybind11Extension
import glob
import sys
import os
import servalcat

ext_modules = [
    Pybind11Extension(
        "servalcat.ext",
        sorted(glob.glob("src/*.cpp") + ["gemmi/src/"+x for x in ("topo.cpp", "monlib.cpp", "polyheur.cpp", "resinfo.cpp", "riding_h.cpp", "eig3.cpp")]), 
        include_dirs=["gemmi/include", "eigen"],
    ),
]

# taken from https://github.com/project-gemmi/gemmi

# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
        # Don't trigger -Wunused-parameter.
        f.write('int main (int, char **) { return 0; }')
        fname = f.name
    try:
        compiler.compile([fname], extra_postargs=[flagname])
    except distutils.errors.CompileError:
        return False
    finally:
        try:
            os.remove(fname)
        except OSError:
            pass
    return True

def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.

    The newer version is prefered over c++11 (when it is available).
    """
    flags = ['-std=c++20', '-std=c++17', '-std=c++14', '-std=c++11']

    # C++17 on Mac requires higher -mmacosx-version-min, skip it for now
    if sys.platform == 'darwin':
        flags = flags[2:]

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support '
                       'is needed!')

class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc', '/D_CRT_SECURE_NO_WARNINGS'],
        'unix': [],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if sys.platform == 'win32':
        if sys.version_info[0] == 2:
            # without these variables distutils insist on using VS 2008
            os.environ['DISTUTILS_USE_SDK'] = '1'
            os.environ['MSSdk'] = '1'
        if sys.version_info[0] >= 3:
            c_opts['msvc'].append('/D_UNICODE')

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])

        if sys.platform == 'darwin':
            darwin_opts = []
            if 'MACOSX_DEPLOYMENT_TARGET' not in os.environ:
                import platform
                mac_ver = platform.mac_ver()
                current_macos = tuple(int(x) for x in mac_ver[0].split(".")[:2])
                if current_macos > (10, 9):
                    darwin_opts.append('-mmacosx-version-min=10.9')
            if has_flag(self.compiler, '-stdlib=libc++'):
                darwin_opts.append('-stdlib=libc++')
            opts += darwin_opts
            link_opts += darwin_opts

        if ct == 'unix':
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
            if 0:
                opts.append('-O0')
                opts.append('-g')
            elif has_flag(self.compiler, '-g0'):
                opts.append('-g0')
            if has_flag(self.compiler, '-Wl,-s'):
                link_opts.append('-Wl,-s')
        elif ct.startswith('mingw'):
            #opts.append('-std=c++14')
            opts.append(cpp_flag(self.compiler))
            opts.append('-fvisibility=hidden')
            opts.append('-g0')
            link_opts.append('-Wl,-s')
        for ext in self.extensions:
            ext.define_macros = [('VERSION_INFO',
                                  '"%s"' % self.distribution.get_version())]
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)

setup(name='servalcat',
    version=servalcat.__version__,
    author='Keitaro Yamashita and Garib N. Murshudov',
    url='https://github.com/keitaroyam/servalcat',
    description= 'Structure refinement and validation for crystallography and single particle analysis',
    long_description="Please see https://github.com/keitaroyam/servalcat",
    long_description_content_type='text/markdown',
    license='MPL-2.0',
    packages=setuptools.find_packages(),
    install_requires=['numpy>=1.15','scipy','pandas>=0.24.2', 'gemmi==0.6.2'],
    entry_points={
      'console_scripts': [
          'servalcat = servalcat.command_line:main',
          'refmacat  = servalcat.refmac.refmac_wrapper:command_line',
                          ],
      },
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe= False)
