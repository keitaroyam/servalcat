"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import sys
import datetime
import traceback

class Logger(object):
    def __init__(self, file_out=None, append=True):
        self.ofs = None
        if file_out:
            self.set_file(file_out, append)
    # __init__()
    
    def set_file(self, file_out, append=True):
        try:
            self.ofs = open(file_out, "a" if append else "w")
        except:
            print("Error: Cannot open log file to write")
    # set_file()

    def write(self, l, end="\n", flush=True, fs=None):
        print(l, end=end)
        for f in (self.ofs, fs):
            if f is not None:
                f.write(l)
                f.write(end)
                if flush: f.flush()
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
    _logger.write("# Abnormally finished on {}\n".format(datetime.datetime.now()))

# handle_exception()

sys.excepthook = handle_exception
