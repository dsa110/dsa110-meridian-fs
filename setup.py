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
      zip_safe=False)

