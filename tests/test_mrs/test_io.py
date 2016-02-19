import suspect
import suspect.io.tarquin

import numpy


def test_write_dpt():
    data = suspect.MRSData(numpy.zeros(1), 1e-3, 123.456)
    suspect.io.tarquin.save_dpt("/home/ben/test_dpt.dpt", data)


def test_extract_csi_fid():
    data = suspect.io.rda.load_rda("suspect/tests/test_data/CSITEST_20151028_97_1.rda")
    single_voxel = data[0, 8, 8]
    suspect.io.tarquin.save_dpt("/home/ben/test_dpt2.dpt", single_voxel)