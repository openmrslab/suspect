import suspect
import numpy


def test_load_dicom_volume():
    data = suspect.image.load_dicom_volume("tests/test_data/siemens/mri/T1.0001.IMA")

    numpy.testing.assert_allclose(data.position, [-106.11758473, -99.7457628, -60.63559341])
    numpy.testing.assert_allclose(data.voxel_size, [0.224609375, 0.224609375, 5])


def test_load_dicom():
    test_dcm = "tests/test_data/dicom/No_Name/Mrs_Dti_Qa/MRSshortTELN_401/IM-0001-0002.dcm"
    data = suspect.io.load_dicom(test_dcm)
    assert data.shape == (2, 1024)

