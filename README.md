# dsa110-meridian-fs
This repository contains routines to fringestop DSA-110 visibilities on the meridian and integrate visibilities.

## Installation Instructions
This install requires PSRDADA, which can be installed according to [these instructions](http://psrdada.sourceforge.net/download.shtml) and psrdada-python, which can be installed from the [github repository](https://github.com/AA-ALERT/psrdada-python).

For psrdada-python to work with PSRDADA, PSRDADA must be installed using the -fPIC flag.  If you are using gnu-autotools to install, set the environment variable C_FLAGS before installing.

If you have multiple installs of PSRDADA on your machine, you may need to set LD_LIBRARY_PATH in the setup.py file for psrdada-python.

dsa110-meridian-fs can then be installed with `python setup.py install`.
