import struct


def save_mat(filename, data):
    with open(filename, "wb") as fout:
        header_bytes = struct.pack("<BBBB6I89I27III10I2f4I2f16I4f10If19I64s49I",
                                   1, 2, 3, 4,
                                   256,
                                   1, 0, 1, 32, 210,
                                   *range(6, 95),
                                   *range(1, 28),
                                   data.shape[1],
                                   data.shape[0],
                                   *range(124, 134),
                                   data.f0,
                                   data.f0,
                                   *range(136, 140),
                                   data.sw,
                                   1e3 / 0.8,
                                   *range(142, 158),
                                   0,
                                   0,
                                   0,
                                   0,
                                   *range(162, 172),
                                   300,
                                   *range(173, 192),
                                   b"THIS IS A TTTTEST",
                                   *range(208, 257))

        # byte_code = struct.pack("<BBBB", 1, 2, 3, 4)
        # frame_size = struct.pack("<I", 256)
        # overall_header = struct.pack("<IIIII", 1, 0, 1, 32, 210)
        # words_to_frame_header = struct.pack("<89I", *range(6, 95))
        # frame_header = struct.pack("32I", *range(32))
        # rest_of_header = struct.pack("130I", *range(127, 257))
        #
        # header_bytes = b"".join([byte_code,
        #                          frame_size,
        #                          overall_header,
        #                          words_to_frame_header,
        #                          frame_header,
        #                          rest_of_header])

        fout.write(header_bytes)

        # write each fid of the COSY line by line
        for fid in data:
            # start with the number of data words in the FID
            fout.write(struct.pack("<I", len(fid) * 2))
            # write out each point, real then imaginary
            for point in fid:
                fout.write(struct.pack("<ff", point.real, point.imag))
