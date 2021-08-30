"""
Created on November 04, 2020

@author: aukermaa@mskcc.org
"""
class EnsureByteContext(object):
    def __enter__(self):
        # Re-import modules 
        import builtins
        import io
        if not hasattr(builtins, "default_open"):
            # Backup open function as default_open
            print ("Reconfigurating [builtins] to have default_open() attribute")
            builtins.default_open = builtins.open
        def io_open(*args, **kwargs):
            # Define new IO function
            print ("__io_open__(): ", args, kwargs)
            if type(args[0])==io.BytesIO: return io.BufferedReader(args[0])
            else: return builtins.default_open(*args, **kwargs)
        if not hasattr(builtins, "io_open"):
            # Set open as new 
            print ("Reconfigurating [builtins] to use io_open() attribute")
            builtins.io_open = io_open
            builtins.open = builtins.io_open
        return self
    def __exit__(self, type, value, traceback):
        return self

