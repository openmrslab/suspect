import suspect
import suspect.io.tarquin
import pytest
import unittest.mock
import builtins
from unittest.mock import patch

import numpy


def test_write_dpt():
    data = suspect.MRSData(numpy.zeros(1), 1e-3, 123.456)
    mock = unittest.mock.mock_open()
    with patch.object(builtins, 'open', mock):
        suspect.io.tarquin.save_dpt("/home/ben/test_dpt.dpt", data)
    #print(mock.mock_calls)
    #handle = mock()
    #print(handle.write.call_args())


def test_write_raw():
    data = suspect.MRSData(numpy.zeros(1, 'complex'), 1e-3, 123.456)
    mock = unittest.mock.mock_open()
    with patch.object(builtins, 'open', mock):
        suspect.io.lcmodel.save_raw("/home/ben/test_raw.raw", data)
    #print(mock().write.mock_calls)
    #handle = mock()
    #print(handle.write.call_args())


def test_lcmodel_all_files():
    data = suspect.MRSData(numpy.zeros(1, 'complex'), 1e-3, 123.456)
    mock = unittest.mock.mock_open()
    with patch.object(builtins, 'open', mock):
        suspect.io.lcmodel.write_all_files("/home/ben/lcmodel",
                                           data)
    #print(mock.call_args)
    #print(mock().write.mock_calls)


def test_lcmodel_read_coord():
    fitting_result = suspect.io.lcmodel.read_coord("suspect/tests/test_data/lcmodel/svs_97.COORD")
    assert len(fitting_result["metabolite_fits"]) == 41


#def test_extract_csi_fid():
#    data = suspect.io.rda.load_rda("suspect/tests/test_data/CSITEST_20151028_97_1.rda")
#    single_voxel = data[0, 8, 8]
#    suspect.io.tarquin.save_dpt("/home/ben/test_dpt2.dpt", single_voxel)


#def test_load_sdat():
#    data = suspect.io.load_sdat("suspect/tests/test_data/SS0044_214-SS0044_214-WIP_SV_P40_LOC_R1-601_act.sdat")
#    assert data.te == 30
#    assert data.f0 == 127.769903
#    assert data.shape == (192, 2048)
