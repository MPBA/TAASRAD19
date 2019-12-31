import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def unpack_and_downscale_hdf_wrapper(run_id_and_hdf_archive_path):
    # unpack arguments
    if len(run_id_and_hdf_archive_path) != 2 and len(run_id_and_hdf_archive_path) != 3:
        raise ValueError("Invalid arguments passed to wrapper function")

    return unpack_and_downscale_hdf(*run_id_and_hdf_archive_path)


def unpack_and_downscale_hdf(run_id, hdf_archive_path, outsize=(64, 64)):
    with h5py.File(hdf_archive_path, "r", libver="latest") as hdf_archive:
        run = hdf_archive[str(run_id)]
        run = np.array(run)

        # first axis is number of sequences, i.e. num of channels
        # move it to the last position because we are interested in resizing
        # the radar scan
        run = np.moveaxis(run, 0, -1)

        run = run.astype(np.float32)
        run = cv2.resize(
            run, dsize=outsize, interpolation=cv2.INTER_LINEAR
        )
        run = run.astype(np.float16)
        # put the axis where where it was
        run = np.moveaxis(run, -1, 0)

    return run, run.shape[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Convert the radar dataset from TAR/ZIP format to HDF5."
    )

    parser.add_argument("radar_directory", type=str)
    args = parser.parse_args()

    metadata = pd.read_csv(
        os.path.join(args.radar_directory, "hdf_metadata.csv"), index_col="id"
    )
    sort_meta = metadata.sort_values(by="start_datetime", ascending=True)

    hdf_archive_path = os.path.join(
        args.radar_directory, "hdf_archives", "all_data.hdf5"
    )

    runs = []
    count = 0

    with ProcessPoolExecutor() as executor:
        worker_args = []
        for run_id in sort_meta.index:
            # generate argument list
            worker_args.append((run_id, hdf_archive_path))

        # call with executor
        with tqdm(total=len(worker_args)) as pbar:
            for run, c in executor.map(unpack_and_downscale_hdf_wrapper, worker_args):
                runs.append(run)
                count += c
                pbar.update(1)

    runs = np.concatenate(runs)

    print(f"Processed {count} radar scans")
    print("runs.shape:", runs.shape)

    outfile_path = os.path.join(args.radar_directory, "runs_64x64.npz")
    with open(outfile_path, "wb") as outfile:
        np.savez_compressed(outfile, runs)


if __name__ == "__main__":
    main()
