suspect
-------

============ =================
Travis CI    |build_status|
Coveralls    |coverage_status|
Code Climate |code_climate|
Waffle       |waffle|
============ =================

.. |build_status| image:: https://travis-ci.org/openmrslab/suspect.svg?branch=master
    :target: https://travis-ci.org/openmrslab/suspect

.. |coverage_status| image:: https://coveralls.io/repos/github/openmrslab/suspect/badge.svg?branch=master
    :target: https://coveralls.io/github/openmrslab/suspect?branch=master

.. |code_climate| image:: https://codeclimate.com/github/openmrslab/suspect/badges/gpa.svg
   :target: https://codeclimate.com/github/openmrslab/suspect

.. |waffle| image:: https://badge.waffle.io/openmrslab/suspect.svg?label=ready&title=Ready
 :target: https://waffle.io/openmrslab/suspect
 :alt: 'Stories in Ready'

Suspect is a Python package for processing MR spectroscopy data. It supports reading data from most common formats (with more on the way) and many different algorithms for core processing steps. Suspect allows researchers to build custom data processing scripts from reliable, modular building blocks and easily share their techniques with other labs around the world.

Installation
^^^^^^^^^^^^

Suspect itself is a pure Python package and is easy to install with `pip`_. However it does depend on various other packages, some of which are not so easy to install.

1. Obtain Python and the SciPy stack:

   Suspect requires Python 3 and makes heavy use of numpy and other parts of the Scientific Python stack. The easiest way to obtain this, along with a large number of other useful scientific packages, is to download the free Anaconda_ package. Alternatively check here_ for other ways to install these core packages.
2. Install pydicom:

   Suspect requires a version of pydicom >= v1.0. Unfortunately the latest release currently available from PyPI is v0.9.9 so pip cannot install it automatically. Instead, download the latest version of the code from https://github.com/darcymason/pydicom and use pip to install the code from the local folder with the command ``pip install path/to/download``
3. Install suspect

   ``pip install suspect`` will automatically download and install the latest version of suspect, along with all remaining other dependencies.

.. _pip: https://pip.pypa.io/en/stable/
.. _pydicom: https://pydicom.readthedocs.io/en/stable/index.html
.. _Anaconda: https://www.continuum.io/downloads
.. _here: http://www.scipy.org/install.html

Getting Started
^^^^^^^^^^^^^^^

Suspect is a very new package and we are working hard to get useful examples and documentation available to get people started. Please bear with us while we work to improve this aspect of the project. Offical documentation for the project is available at http://suspect.readthedocs.io/en/latest/

Contributing
^^^^^^^^^^^^

If you are interested in helping out with any part of suspect or the OpenMRSLab project, we would love to hear from you.

License
^^^^^^^

Suspect is released under the MIT license