Installation
======================================

Pychrom can be installed as a standard Pypi package using setup.py

**Step 1**: Setup your Python environment

	Pychrom requires Python 3.5 along with several Python package dependencies.
	Python distributions like Anaconda, are recommended to manage 
	the Python environment.  Anaconda can be downloaded from https://www.continuum.io/downloads.  
	General information on Python can be found at https://www.python.org/.
	
	For more information on Python package dependencies, see :ref:`requirements`.  

**Step 2**: Install pychrom

        Users can install pychrom get the source code from github and install it
	using standard python setuptools.

	git clone https://github.com/santiagoropb/Chromatography.git pychrom
	cd pychrom
	
	**For users**: 	
	
	To install pychrom run::

		python setup.py install 
	
	**For developers**:

	To install pychrom run::

		python setup.py develop
	
	This will install the master branch from github

**Step 3**: Test installation

	To test that pychrom is install, open Python and run::
	
		import pychrom

	To verify the package was properly install run::

	        nosetests pychrom/test --with-coverage --cover-erase --cover-package=pychrom

.. _requirements:

Requirements
-------------
Pychrom runs on python 3.x along with several Python packages. 
The following Python packages are required:

* Numpy [vanderWalt2011]_: used to support large, multi-dimensional arrays and matrices, 
  http://www.numpy.org/
* Scipy [vanderWalt2011]_: used to support efficient routines for numerical integration, 
  http://www.scipy.org/
* Pandas [McKinney2013]_: used to analyze and store time series data, 
  http://pandas.pydata.org/
* Matplotlib [Hunter2007]_: used to produce figures, 
  http://matplotlib.org/
* Pyomo [Hart2012]_: used for formulating the optimization problems
  http://pyomo.org/

Solvers
-----------------

* CADET: used for simulating chromatography model
* IPOPT: used for solving the nonlinear optimization problems.

For the installation instructions please refer to

https://github.com/modsim/CADET
http://www.coin-or.org/Ipopt/documentation/node10.html

It is recommended to compile IPOPT with the HSL linear solvers. The examples and test problems have not been tested yet with other linear solvers besides HSL. The HSL software can be found at

http://hsl.rl.ac.uk/ipopt
