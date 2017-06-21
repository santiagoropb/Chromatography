from setuptools import setup, find_packages


DISTNAME = 'pycadet'
VERSION = '0.0.0'
PACKAGES = find_packages()
EXTENSIONS = []
DESCRIPTION = 'Python Wrapper for Chromatography Analysis and Design Toolkit'
LONG_DESCRIPTION = open('README.md').read()
AUTHOR = 'DIPT/Jose-Santiago-Rodriguez '
MAINTAINER_EMAIL = 'TODO'
LICENSE = 'TODO'
URL = 'TODO'

setuptools_kwargs = {
    'zip_safe': False,
    'install_requires': ['pint', 'six','pyomo','pyyaml','h5py'],
    'scripts': [],
    'include_package_data': True
}


setup(name=DISTNAME,
      version=VERSION,
      packages=PACKAGES,
      ext_modules=EXTENSIONS,
      description='',
      long_description='',
      author=AUTHOR,
      maintainer_email=MAINTAINER_EMAIL,
      license=LICENSE,
      url=URL,
      **setuptools_kwargs)

