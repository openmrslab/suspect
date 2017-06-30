import suspect.io.tarquin


def test_load_fit_file():
    metabolites, fit_data = suspect.io.tarquin.read_fit_file("tests/test_data/tarquin/tarquin_megapress_fit.txt")
    assert fit_data.shape == (1, 1, 1, 2048, 12)
