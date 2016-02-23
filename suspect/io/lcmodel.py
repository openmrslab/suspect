import numpy


def save_raw(filename, data):
    with open(filename, 'w') as fout:
        fout.write(" $SEQPAR\n")
        fout.write(" ECHOT = {}\n".format(data.te))
        fout.write(" HZPPPM = {}\n".format(data.f0))
        fout.write(" SEQ = 'PRESS'\n")
        fout.write(" $END\n")
        fout.write(" $NMID\n")
        fout.write(" FMTDAT = '(2E15.6)'\n")
        # convert the volume from mm^3 to cc
        fout.write(" VOLUME = %f\n".format(data.voxel_size() * 1e-3))
        fout.write(" $END\n")
        for point in numpy.nditer(data, order='C'):
            fout.write("  { 4.6e}  { 4.6e}\n".format(point.real, point.imag))
