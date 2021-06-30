from setuptools import setup
from dsautils.version import get_git_version

version = get_git_version()
assert version is not None

setup(name='dsa110-meridian-fs',
      version=version,
      url='http://github.com/dsa110/dsa110-meridian-fs/',
      author='Dana Simard',
      author_email='dana.simard@astro.caltech.edu',
      packages=['dsamfs'],
      package_data={
                'dsamfs': ['data/*.txt', 'data/*.yaml'],
             },
      install_requires=['astropy',
                        'casatools',
                        'casadata',
                        'cython',
                        'h5py',
                        'matplotlib',
                        'numba',
                        'numpy',
                        'pandas',
                        'pytest',
                        'scipy',
                        'coverage',
                        'codecov',
                        'pyyaml',
                        'etcd3',
                        'structlog',
                        'dsa110-antpos',
                        'dsa110-pyutils',
                        'dsa110-calib',
                        'pyuvdata'
                        ],
      dependency_links=[
          "https://github.com/dsa110/dsa110-antpos/tarball/master#egg=dsa110-antpos",
          "https://github.com/dsa110/dsa110-pyutils/tarball/master#egg=dsa110-pyutils",
          "https://casa-pip.nrao.edu/repository/pypi-casa-release/simple",
          "https://github.com/dsa110/dsa110-calib/main#egg=dsa110-calib",
          ],
      zip_safe=False)

