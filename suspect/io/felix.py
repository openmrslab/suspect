import struct

# this is what we have been able to deduce about the Felix file header
# word -1: byte key
# word 0: 256 (size of header)
# word 1: 1 (number of frames)
# word 2: 0 (not sure, possibly data type)
# word 3: 1 (not sure, possibly an unused parameter)
# word 4: 32 (size of frame header)
# word 5: 210 (Felix version number)
# word 6-94: digits 6 - 94 (not used for anything)
# word 95-121: digits 1-27 (not used for anything)
# word 122: FID length
# word 123: number of increments
# word 124-133: digits 124-133 (not used for anything)
# word 134: FID f0
# word 135: increment f0
# word 136-139: digits 136-139 (not used for anything)
# word 140: FID sw
# word 141: increment sw
# word 142-157: digits 142-157 (not used for anything)
# word 146: d1
# word 158: zero order phase
# word 159: first order phase
# word 160: zero order phase d2
# word 161: first order phase d2
# word 162-171: digits 162-171 (not used for anything)
# word 172: temperature
# word 173-191: digits 173-191 (not used for anything)
# word 192-207: description string
# word 208-256: digits 208-256 (not used for anything)


def save_mat(filename, data):
    """
    Parameters
    ----------
    filename :
    data :

    """
    with open(filename, "wb") as fout:
        header_bytes = struct.pack("<BBBB6I89I27III10I2f4I2f16I4f10If19I64s49I",
                                   1, 2, 3, 4,
                                   256,
                                   1, 0, 1, 32, 210,
                                   *range(6, 95),
                                   1, 1,
                                   *range(1, 26),
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
