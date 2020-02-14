from setuptools import setup

setup(name='dsa110-meridian-fs',
      version='0.1',
      url='http://github.com/dsa110/dsa110-calib/',
      author='Dana Simard',
      author_email='dana.simard@astro.caltech.edu',
      packages=['dsamfs'],
      package_data={'dsamfs':['data/*.all','data/templatekcal']},
      requirements=['dsacalib','casa-python','casa-data',
                    'astropy','scipy','psrdada-python','h5py','numba'],
      zip_safe=False)

