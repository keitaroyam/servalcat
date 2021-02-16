"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import sys
import traceback

class Logger(object):
    def __init__(self, file_out=None, append=True):
        self.ofs = None
        if file_out:
            self.set_file(file_out, append)
    # __init__()
    
    def set_file(self, file_out, append=True):
        self.ofs = open(file_out, "a" if append else "w")
    # set_file()

    def write(self, l, end="\n", flush=True):
        print(l, end=end)
        if self.ofs is not None:
            self.ofs.write(l)
            self.ofs.write(end)
            if flush: self.ofs.flush()
    # write()

    def close(self):
        self.ofs.close()
        self.ofs = None
    # close()
# class Logger

_logger = Logger() # singleton
set_file = _logger.set_file
write = _logger.write
close = _logger.close


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    name = type(exc_value).__name__ if hasattr(type(exc_value), "__name__") else "(unknown)"
    #_logger.write("Uncaught exception: {}: {}".format(name, exc_value))
    _logger.write("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
# handle_exception()

sys.excepthook = handle_exception
