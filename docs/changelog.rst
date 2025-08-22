:orphan:

=========
Changelog
=========

* :release:`0.6.1 <22/08/25>`
* :support:`184` Handle both enhanced and interoperability tags Siemens DICOM XA
* :feature:`183` `suspect.io.load_twix()` to accept binary stream input
* :release:`0.6.0 <28/01/25>`
* :support:`181` Loader for XA RDA
* :support:`178` Loader for XA DICOM
* :support:`179` Use pip-review to install latest in new job
* :bug:`177` Updated regex for normal and pos
* :bug:`175` More general way to loop over dicom group
* :release:`0.5.2 <13/09/24>`
* :bug:`174` Use `pydicom.dcmread` instead of `pydicom.dicomio.readfile` for future proof
* :release:`0.5.1 <08/08/24>`
* :bug:`167` Compatibility issues in the latest version of Suspecs (v0.4.4)
* :bug:`166` Travis CI not working
* :support:`164` Added regex for extracting metadata on VIDA data
* :support:`162` Allow update on lmfit's version
* :bug:`160` pydicom >2.0.0 reads MRS with too many elements (thanks to @laurencejackson for this)
* :feature:`157` Add kwargs to lcmodel RAW file generation methods (thanks for @neurosutton for this)
* :bug:`155` Changed literal comparison of string to Python's comparison == (thanks for @joeranbosma for this)
* :support:`153` Added transformation matrix info for Philips (thanks for @oscarjalnefjord for this)
* :release:`0.4.9 <20/11/20>`
* :bug:`149` improved reliability of twix file header reading (thanks to @darrencl for this)
* :release:`0.4.3 <29/09/20>`
* :bug:`145` fixed issue with calculating transform for sagittally planned twix voxels
* :feature:`146` substantial overhaul of the singlet fitting module
* :release:`0.4.2 <20/08/20>`
* :bug:`143` changed .IMA loading to handle cases where data shape is calculated incorrectly
* :feature:`11` added anonymisation of twix VD/VE files
* :bug:`139` fixed an issue where axial/sagittal/coronal vectors could be calculated incorrectly
* :release:`0.4.1 <24/05/20>`
* :feature:`137` added functions to support absolute quantification
* :feature:`135` added TR parameter to MRSBase objects
* :feature:`133` improved interface to frequency correction methods and new method RATS
* :release:`0.4.0 <13/05/20>`
* :feature:`128` channel combination functions can now accept a channel axis
* :bug:`125` loading DICOM image volumes uses supplied extension to identify candidate slices
* :feature:`124` GE P-files are now supported
* :bug:`121` fixed missing return of MRSData.fid method
* :release:`0.3.9 <10/07/18>`
* :bug:`115` added ability to change encoding used for Philips SPAR files
* :release:`0.3.8 <22/05/18>`
* :bug:`107` LCModel processing works without a voxel transform
* :bug:`111` issues when processing with TARQUIN are reported to the user
* :release:`0.3.7 <30/04/18>`
* :feature:`109` TARQUIN processing now supports a water reference
* :feature:`85` new whiten function to decorrelate multi-channel data
* :feature:`87` singlet fitting returns MRSData for fit
* :bug:`102` fixed problem with loading anonymized twix files
* :bug:`100` improved loading of Philips sdat files (thanks to @jhamilx for help)
* :bug:`98` fixed an issue where 2D images could not be used to create a voxel mask
* :release:`0.3.6 <02/11/17>`
* :feature:`94` loading Siemens DICOM now includes a voxel transform
* :bug:`88` fixed an issue where certain Siemens DICOM files did not import
* :bug:`92` fixed a problem where row_vector and column_vector where swapped
* :bug:`90` fixed a problem with resampling to a single slice
* :release:`0.3.5 <25/09/17>`
* :bug:`82` fixed an issue with spectral registration over limited frequency ranges
* :release:`0.3.4 <05/08/17>`
* :feature:`80` Added resampling of 3D volumes to new coordinate systems
* :release:`0.3.3 <03/08/17>`
* :feature:`76` Added new auto-phasing algorithms
* :release:`0.3.2 <02/08/17>`
* :bug:`78` fixed an issue where SIFT denoising returns real values from complex input
* :release:`0.3.1 <01/08/17>`
* :bug:`74` changed image direction vectors to always be positive
* :bug:`72` fixed a problem where sometimes channel combination was done over the wrong axis
* :feature:`70` added direction vector accessors for spatial orientation
* :feature:`68` added support for save/load of Nifti format
* :bug:`67` created image mask is now an ImageBase object
* :feature:`63` coordinate transform functions now accept nd grids as input, not just single coordinates
* :feature:`62` can create a mask showing spectroscopy volume on structural volume
* :feature:`59` TARQUIN processing now includes plots of fits and data
* :feature:`56` added ability to get a slice to access a subset of spectrum
* :bug:`54` removed some additional PHI when anonymising twix data. Thanks to @josephmje for the fix
* :feature:`45` load_twix() now gets voxel positioning information
* :feature:`44` added ImageBase class to handle working with structural images
* :feature:`38` read TE from twix files
* :release:`0.3.0 <04/05/17>`
* :bug:`39` fixed issue with spline denoising receiving float instead of integer values
* :feature:`35` adjust_frequency() function for MRSData
* :bug:`33` negated initial guesses for spectral registration
* :bug:`31` all phase adjustments use common function
* :feature:`29` loading functions for Bruker data
* :support:`28` add documentation for water suppression methods
* :feature:`24` added MRSSpectrum object to match existing FID object. Thanks to @lasyasreepada for the feature
* :bug:`23` fixed denoising methods casting complex to real
* :feature:`21` added adjust_phase() function for MRSData
* :feature:`20` added support for MRS DICOM format
* :bug:`17` fixed bug where lcmodel files where created without quoted strings
* :support:`15` single location for current version information _version.py
* :support:`10` convert all docstrings to NumPy format, thanks to @lasyasreepada for a great job
