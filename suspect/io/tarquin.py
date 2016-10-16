import subprocess


def save_dpt(filename, data):
    with open(filename, 'wb') as fout:
        fout.write("Dangerplot_version\t1.0\n".encode())
        fout.write("Number_of_points\t{}\n".format(data.np).encode())
        fout.write("Sampling_frequency\t{0:8.8e}\n".format(1.0 / data.dt).encode())
        fout.write("Transmitter_frequency\t{0:8.8e}\n".format(data.f0 * 1e6).encode())
        fout.write("Phi0\t{0:8.8e}\n".format(0).encode())
        fout.write("Phi1\t{0:8.8e}\n".format(0).encode())
        fout.write("PPM_reference\t{0:8.8e}\n".format(data.ppm0).encode())
        fout.write("Echo_time\t{0:8.8e}\n".format(data.te * 1e-3).encode())
        fout.write("Real_FID\tImag_FID\t\n".encode())
        for x in data:
            fout.write("{0.real:8.8e} {0.imag:8.8e}\n".format(x).encode())


def read_output(filename):
    """Reads in a Tarquin txt results file and returns a dict of the information

    Parameters
    ----------
    filename : txt file
        The filename to read from

    Returns
    -------
    result : dict

    """
    with open(filename) as fin:
        data = fin.read()

        metabolite_fits = {}
        fit_quality = {}

        sections = data.split("\n\n")

        # first section is the metabolite concentrations
        metabolite_lines = sections[0].splitlines()[2:]
        for line in metabolite_lines:
            name, concentration, pc_sd, sd = line.split()
            metabolite_fits[name] = {
                "concentration": concentration,
                "sd": pc_sd,
            }

        # second section is the fit quality
        lines = sections[1].splitlines()[2:]
        for line in lines:
            key, value = line.split(":")
            fit_quality[key.strip()] = float(value.strip())

        return {
            "metabolite_fits": metabolite_fits,
            "quality": fit_quality
        }


def process(data, options={}):
    save_dpt("/tmp/temp.dpt", data)
    option_string = ""
    for key, value in options.items():
        option_string += " --{} {}".format(key, value)
    subprocess.run("tarquin --input {} --format dpt --output_txt {}{}".format(
        "/tmp/temp.dpt", "/tmp/output.txt", option_string
    ), shell=True)
    # with open("/tmp/output.txt") as fin:
    #    result = fin.read()
    result = read_output("/tmp/output.txt")
    return result
