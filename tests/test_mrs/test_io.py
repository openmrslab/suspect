import suspect
import suspect.io.tarquin
import pytest
import unittest.mock
import builtins
from unittest.mock import patch
import os

from suspect.io._common import complex_array_from_iter

import numpy


def test_complex_from_iter():
    float_list = [1.0, 0.0, 0.0, 1.0]
    array = complex_array_from_iter(iter(float_list))
    assert array.shape == (2,)
    assert array[0] == 1
    assert array[1] == 1j


def test_shaped_complex_from_iter():
    float_list = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    array = complex_array_from_iter(iter(float_list), shape=[2, 2])
    assert array.shape == (2, 2)


def test_write_dpt():
    data = suspect.MRSData(numpy.zeros(1), 1e-3, 123.456)
    mock = unittest.mock.mock_open()
    with patch.object(builtins, 'open', mock):
        suspect.io.tarquin.save_dpt("/home/ben/test_dpt.dpt", data)
    #print(mock.mock_calls)
    #handle = mock()
    #print(handle.write.call_args())


def test_read_tarquin_results():
    fitting_result = suspect.io.tarquin.read_output("tests/test_data/tarquin/tarquin_results.txt")
    assert "metabolite_fits" in fitting_result
    assert "quality" in fitting_result
    assert fitting_result["quality"]["Metab FWHM (PPM)"] == 0.04754
    assert fitting_result["quality"]["Q"] == 4.048


def test_write_raw():
    # lcmodel needs to know the transform properties
    transform = suspect.transformation_matrix([1, 0, 0], [0, 1, 0], [0, 0, 0], [10, 10, 10])
    data = suspect.MRSData(numpy.zeros(1, 'complex'),
                           1e-3,
                           123.456,
                           transform=transform)
    mock = unittest.mock.mock_open()
    with patch.object(builtins, 'open', mock):
        suspect.io.lcmodel.save_raw("/home/ben/test_raw.raw", data)
    #print(mock().write.mock_calls)
    #handle = mock()
    #print(handle.write.call_args())


def test_lcmodel_all_files():
    # lcmodel needs to know the transform properties
    transform = suspect.transformation_matrix([1, 0, 0], [0, 1, 0], [0, 0, 0], [10, 10, 10])
    data = suspect.MRSData(numpy.zeros(1, 'complex'),
                           1e-3,
                           123.456,
                           transform=transform)
    mock = unittest.mock.mock_open()
    with patch.object(builtins, 'open', mock):
        suspect.io.lcmodel.write_all_files(os.path.join(os.getcwd(), "lcmodel"),
                                           data)
    #print(mock.call_args)
    #print(mock().write.mock_calls)


def test_lcmodel_read_coord():
    fitting_result = suspect.io.lcmodel.read_coord("tests/test_data/lcmodel/svs_97.COORD")
    assert len(fitting_result["metabolite_fits"]) == 41


def test_lcmodel_read_liver_coord():
    fitting_result = suspect.io.lcmodel.read_coord("tests/test_data/lcmodel/liver.COORD")


def test_lcmodel_read_basis():
    basis = suspect.io.lcmodel.read_basis("tests/test_data/lcmodel/press_30ms_3T.basis")
    #print(basis)
    #from matplotlib import pyplot
    #met = "NAA"
    #sw = 1.0 / basis["BASIS1"]["BADELT"]
    #fa = numpy.linspace(0, sw, len(basis[met]["data"]))
    #pyplot.plot(fa, numpy.abs(numpy.roll(basis[met]["data"], -basis[met]["ISHIFT"])))
    #pyplot.show()
    assert basis["BASIS1"]["BADELT"] == 0.000207357807
    assert basis["BASIS1"]["NDATAB"] == 4944
    assert "NAA" in basis["SPECTRA"]


def test_lcmodel_write_basis():
    basis = suspect.io.lcmodel.read_basis("tests/test_data/lcmodel/press_30ms_3T.basis")
    mock = unittest.mock.mock_open()
    with patch.object(builtins, 'open', mock):
        suspect.io.lcmodel.save_basis("/home/ben/test_raw.raw", basis)
        #print(mock().write.mock_calls)
        # handle = mock()
        # print(handle.write.call_args())


#def test_extract_csi_fid():
#    data = suspect.io.rda.load_rda("suspect/tests/test_data/CSITEST_20151028_97_1.rda")
#    single_voxel = data[0, 8, 8]
#    suspect.io.tarquin.save_dpt("/home/ben/test_dpt2.dpt", single_voxel)


#def test_load_sdat():
#    data = suspect.io.load_sdat("suspect/tests/test_data/SS0044_214-SS0044_214-WIP_SV_P40_LOC_R1-601_act.sdat")
#    assert data.te == 30
#    assert data.f0 == 127.769903
#    assert data.shape == (192, 2048)


def test_felix_save_mat():
    data = suspect.MRSData(numpy.zeros((16, 32), dtype='complex'), 1e-3, 123.456)
    mock = unittest.mock.mock_open()
    with patch.object(builtins, 'open', mock):
        suspect.io.felix.save_mat("test.mat", data)
        #print(mock.mock_calls)
        # handle = mock()
        # print(handle.write.call_args())