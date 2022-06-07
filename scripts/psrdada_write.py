"""
A quick script to write to a psrdada buffer in order to test a psrdada reader.
"""

import os
import subprocess
from time import sleep
import numpy as np
from psrdada import Writer

KEY_STRING = 'adad'
KEY = 0xadad
NANT = 64
NCHAN = 384
NPOL = 2
NBLS = NANT * (NANT + 1) // 2


def main():
    """Writes a psrdada buffer for test"""
    vis_temp = np.arange(NBLS * NCHAN * NPOL * 2, dtype=np.float32)

    # Define the data rate, including the buffer size
    # and the header size
    samples_per_frame = 1
    header_size = 4096
    buffer_size = int(4 * NBLS * NPOL * NCHAN * samples_per_frame * 2)
    assert buffer_size == vis_temp.nbytes, (
        "Sample data size and buffer size do not match.")

    # Create the buffer
    os.system(f"dada_db -a {header_size} -b {buffer_size} -k {KEY_STRING}")
    print("Buffer created")

    # Start the reader
    read = (
        "python ./meridian_fringestop.py /home/ubuntu/data/ "
        "/home/ubuntu/proj/dsa110-shell/dsa110-meridian-fs/dsamfs/data/test_parameters.yaml "
        "/home/ubuntu/proj/dsa110-shell/dsa110-meridian-fs/dsamfs/data/test_header.txt")

    with open("/home/ubuntu/data/tmp/write.log", 'w', encoding='utf-8') as read_log:
        with subprocess.Popen(read, shell=True, stdout=read_log, stderr=read_log) as _read_proc:
            print("Reader started")
            sleep(0.1)

            # Write to the buffer
            writer = Writer(KEY)
            print('Writer created')
            for i in range(48):
                page = writer.getNextPage()
                data = np.asarray(page)
                data[...] = vis_temp.view(np.int8)
                if i < 9:
                    writer.markFilled()
                else:
                    writer.markEndOfData()
                    vis_temp += 1
                    # Wait to allow reader to clear pages
                sleep(1)

            writer.disconnect()

    os.system(f"dada_db -d -k {KEY_STRING}")


if __name__ == "__main__":
    main()
