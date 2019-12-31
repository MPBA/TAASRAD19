import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from tqdm import tqdm

from imageio import imwrite
from settings import CROP_SIZE, MASK


def get_run_counts(run_id_and_hdf_archive_path):
    # unpack arguments
    run_id, hdf_archive_path = run_id_and_hdf_archive_path

    with h5py.File(hdf_archive_path, "r", libver="latest") as hdf_archive:
        counts = np.zeros((CROP_SIZE, CROP_SIZE, 526), np.uint64)
        run = hdf_archive[str(run_id)]
        for scan in run:
            # for i in range(CROP_SIZE):
            #     for j in range(CROP_SIZE):
            #         if MASK[i][j]:
            #             counts[i][j][int(scan[i][j] * 10)] += 1

            scan_scaled = (scan * 10).astype(np.int)
            values = np.unique(scan_scaled)

            for value in values:
                value_mask = scan_scaled == value
                value_mask[~MASK] = 0
                counts[:, :, value] += value_mask

    count = run.shape[0]
    return counts, count


def get_mask(counts, count, output_dir):
    normalized_data = np.divide(counts, count, dtype=np.float64)

    m = normalized_data[MASK].mean(axis=0, dtype=np.float64)
    S = np.cov(normalized_data[MASK], rowvar=False)
    Sinv = np.linalg.pinv(S)

    distances = np.full((CROP_SIZE, CROP_SIZE), np.inf, dtype=np.float64)
    for i in range(CROP_SIZE):
        for j in range(CROP_SIZE):
            if MASK[i][j]:
                distances[i][j] = mahalanobis(normalized_data[i][j], m, Sinv)

    dist_mean = distances[MASK].mean()
    dist_stddev = distances[MASK].std()

    mask = distances <= dist_mean + 3 * dist_stddev
    mask = mask.astype(np.uint8)

    output_path = os.path.join(output_dir, "mask.png")
    imwrite(output_path, mask)


def main():
    parser = argparse.ArgumentParser(
        description="Convert the radar dataset from TAR/ZIP format to HDF5."
    )

    parser.add_argument("radar_directory", type=str)
    args = parser.parse_args()

    metadata = pd.read_csv(
        os.path.join(args.radar_directory, "hdf_metadata.csv"), index_col="id"
    )
    metadata = metadata.sample(frac=.2, random_state=42)
    sort_meta = metadata.sort_values(by="run_length", ascending=True)

    # For each pixel, initialize counts for each possible value to 0.
    counts = np.zeros((CROP_SIZE, CROP_SIZE, 526), np.uint64)
    count = 0

    hdf_archive_path = os.path.join(
        args.radar_directory, "hdf_archives", "all_data.hdf5"
    )
    with ProcessPoolExecutor(max_workers=32) as executor:
        worker_args = []
        for run_id in sort_meta.index:
            # generate argument list
            worker_args.append((run_id, hdf_archive_path))

        # call with executor
        with tqdm(total=len(worker_args)) as pbar:
            for image_counts, image_count in executor.map(get_run_counts, worker_args):
                counts += image_counts
                count += image_count
                pbar.update(1)

    print(f"Processed {count} radar scans")
    get_mask(counts, count, args.radar_directory)


if __name__ == "__main__":
    main()
