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

The Consensus Processing Pipeline
=================================

As part of the 2020 NMR in Biomed Special Edition on Spectroscopy, Near et al.
wrote a paper giving the consensus opinion on the post-acquisition processing
steps, at least for the single voxel case:

Near, J., Harris, A. D., Juchem, C., Kreis, R., Marjańska, M., Öz, G., et al. (2020). Preprocessing, analysis and quantification in single‐voxel magnetic resonance spectroscopy: experts' consensus recommendations. NMR in Biomedicine, 29, 323–23. http://doi.org/10.1002/nbm.4257

This is a highly worthwhile read for any spectroscopist, with excellent detail
on almost every component of the process, with the one notable exception of
channel combination.

Of course it is possible to perform all the processing steps from within
Suspect, here we provide a Jupyter notebook with the complete consensus
pipeline implemented.

.. toctree::
   :caption: Concensus
   :hidden:

   notebooks/consensus_playground.ipynb

:doc:`notebooks/consensus_playground`


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
