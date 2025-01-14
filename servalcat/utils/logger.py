"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import sys
import datetime
import platform
import getpass
import traceback
import shlex
import servalcat

class Logger(object):
    def __init__(self, file_out=None, append=True):
        self.ofs = None
        self.stopped = False
        self.prefix = ""
        if file_out:
            self.set_file(file_out, append)
    # __init__()
    def stop_logging(self): self.stopped = True
    def start_logging(self): self.stopped = False
    def set_prefix(self, p): self.prefix = p
    def clear_prefix(self): self.prefix = ""
    
    def set_file(self, file_out, append=True):
        try:
            self.ofs = open(file_out, "a" if append else "w")
        except:
            print("Error: Cannot open log file to write")
    # set_file()

    def write(self, l, end="", flush=True, fs=None, print_fs=sys.stdout):
        if self.stopped: return
        if self.prefix:
            l = "".join(self.prefix + x for x in l.splitlines(keepends=True))
        print(l, end=end, file=print_fs, flush=flush)
        for f in (self.ofs, fs):
            if f is not None:
                f.write(l)
                f.write(end)
                if flush: f.flush()
    # write()

    def writeln(self, l, flush=True, fs=None, print_fs=sys.stdout):
        self.write(l, end="\n", flush=flush, fs=fs, print_fs=print_fs)
    # writeln()

    def error(self, l, end="\n", flush=True, fs=None):
        self.write(l, end, flush, fs, print_fs=sys.stderr)
    # error()

    def close(self):
        if self.ofs is not None:
            self.ofs.close()
            self.ofs = None
    # close()

    def flush(self): # to act as a file object
        if self.ofs:
            self.ofs.flush()
# class Logger

_logger = Logger() # singleton
set_file = _logger.set_file
write = _logger.write
writeln = _logger.writeln
error = _logger.error
close = _logger.close
flush = _logger.flush
stop = _logger.stop_logging
start = _logger.start_logging
set_prefix = _logger.set_prefix
clear_prefix = _logger.clear_prefix

def with_prefix(prefix):
    class WithPrefix(object): # should keep original prefix and restore?
        def __enter__(self):
            _logger.set_prefix(prefix)
            return _logger
        def __exit__(self, exc_type, exc_val, exc_tb):
            _logger.clear_prefix()
    return WithPrefix()

def silent():
    class Silent(object):
        def write(self, *args, **kwargs):
            pass
        def flush(self):
            pass
    return Silent()

def dependency_versions():
    import gemmi
    import scipy
    import numpy
    import pandas
    return dict(gemmi=gemmi.__version__,
                scipy=scipy.version.full_version,
                numpy=numpy.version.full_version,
                pandas=pandas.__version__)
# dependency_versions()

def versions_str():
    tmpl = "Servalcat {servalcat} with Python {python} ({deps})"
    return tmpl.format(servalcat=servalcat.__version__,
                       python=platform.python_version(),
                       deps=", ".join([x[0]+" "+x[1] for x in dependency_versions().items()]))
# versions_str()

def write_header(command="servalcat"):
    writeln("# Servalcat ver. {} (Python {})".format(servalcat.__version__, platform.python_version()))
    writeln("# Library vers. {}".format(", ".join([x[0]+" "+x[1] for x in dependency_versions().items()])))
    writeln("# Started on {}".format(datetime.datetime.now()))
    writeln("# Host: {} User: {}".format(platform.node(), getpass.getuser()))
    writeln("# Command-line:")
    writeln("# {} {}".format(command, " ".join(map(lambda x: shlex.quote(x), sys.argv[1:]))))
# write_header()

def exit_success():
    _logger.writeln("\n# Finished on {}\n".format(datetime.datetime.now()))

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    name = type(exc_value).__name__ if hasattr(type(exc_value), "__name__") else "(unknown)"
    #_logger.writeln("Uncaught exception: {}: {}".format(name, exc_value))
    _logger.error("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
    _logger.writeln("# Abnormally finished on {}\n".format(datetime.datetime.now()))
    _logger.close()

# handle_exception()

sys.excepthook = handle_exception
