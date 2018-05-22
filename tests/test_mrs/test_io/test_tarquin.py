import suspect.io.tarquin
import unittest.mock as mock
import pytest


def test_load_fit_file():
    metabolites, fit_data = suspect.io.tarquin.read_fit_file("tests/test_data/tarquin/tarquin_megapress_fit.txt")
    assert fit_data.shape == (1, 1, 1, 2048, 12)


def test_tarquin_error():
    with mock.patch("subprocess.run") as mock_run, mock.patch("suspect.io.tarquin.save_dpt") as mock_save:
        mock_run.return_value.returncode = 255
        mock_run.return_value.stderr = "This is an error"
        with pytest.raises(Exception):
            suspect.io.tarquin.process("test_data")
        mock_run.return_value.returncode = 0
        with pytest.raises(FileNotFoundError):
            suspect.io.tarquin.process("test_data")
