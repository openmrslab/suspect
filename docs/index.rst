.. suspect documentation master file, created by
   sphinx-quickstart on Wed Feb 24 12:49:00 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Suspect documentation!
======================

Welcome! This is the documentation for Suspect |release|, last updated |today|

Parts of the documentation:

Getting started
===============

.. toctree::
   :maxdepth: 1
   :glob:

   notebooks/tut01_intro.ipynb
   notebooks/tut02_channels.ipynb
   notebooks/tut04_quant.ipynb
   notebooks/tut05_hsvd.ipynb
   notebooks/tut06_mpl.ipynb

Solving specific problems
=========================

.. toctree::
   :caption: Solving specific problems
   :hidden:

   notebooks/howto/coregister_images.ipynb

:doc:`notebooks/howto/coregister_images`
   Learn how to combine anatomical scans with your MRS voxels

API Reference
=============

.. toctree::
   :maxdepth: 1
   :glob:

   suspect_api.rst
   imagebase_api.rst
   mrs_data_api.rst
   frequency_correction_api.rst
   fitting_api.rst

Changelog
---------

Please see :doc:`the changelog </changelog>`.
