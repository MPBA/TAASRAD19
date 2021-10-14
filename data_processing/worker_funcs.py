import gzip
import os
import tarfile

import h5py
import numpy as np
import pandas as pd
from scipy.signal import convolve2d

from settings import (
    CROP_SIZE,
    GAP_LENGTH,
    IMG_SIZE,
    MIN_RUN_SIZE,
    SKIP_ROWS,
    PARSE_OFFSET,
    MASK,
    PARSED_TYPE
)


class BadAsciiException(Exception):
    """Raised when the shape size is unexpected or the ascii file is malformed"""


def clean(
    arr: np.ndarray,
    lowpass_threshold=7,  # 7 dbZ = > 0.1 mm/h
    highpass_threshold=40,
    highpass_neighbors=30,
    midpass_threshold=15,
    midpass_neighbors=15,
) -> np.ndarray:
    # lowpass filter, remove all signal where dbz < 6
    arr[arr <= lowpass_threshold] = 0

    # remove high values isolated noise (dbz>40) using a 7x7 conv
    # +-------------+
    # |1 1 1 1 1 1 1|
    # |1 1 1 1 1 1 1|
    # |1 1 1 1 1 1 1|
    # |1 1 1 0 1 1 1|
    # |1 1 1 1 1 1 1|
    # |1 1 1 1 1 1 1|
    # |1 1 1 1 1 1 1|
    # +-------------+
    k_7x7 = np.ones((7, 7))
    k_7x7[3, 3] = 0
    neighbors_large = convolve2d(arr.astype(np.bool), k_7x7, mode="same")
    arr[(arr > highpass_threshold) & (neighbors_large < highpass_neighbors)] = 0

    # remove mid values isolated noise (dbz>15) using a 5x5 conv
    # +---------+
    # |1 1 1 1 1|
    # |1 1 1 1 1|
    # |1 1 0 1 1|
    # |1 1 1 1 1|
    # |1 1 1 1 1|
    # +---------+
    k_5x5 = np.ones((5, 5))
    k_5x5[2, 2] = 0
    neighbors_mid = convolve2d(arr.astype(np.bool), k_5x5, mode="same")
    arr[(arr > midpass_threshold) & (neighbors_mid < midpass_neighbors)] = 0

    # remove remaining isolated noise (less than 4 neighbors with dbz>0) using a 3x3 conv
    # +-----+
    # |1 1 1|
    # |1 0 1|
    # |1 1 1|
    # +-----+
    k_3x3 = np.ones((3, 3))
    k_3x3[1, 1] = 0
    neighbors_small = convolve2d(arr.astype(np.bool), k_3x3, mode="same")
    arr[neighbors_small < 4] = 0

    # smooth noise inside rainy patches:
    # if the pixel is dbz>20 and the average difference
    # between the pixel and the surrounding is >10dbz (80)
    # then replace the pixel value with the average of the surrounding pixels
    # we apply this correction only in the central part 340 x 340 of the image
    # to avoid touching the radar circle borders
    # +--------+
    # |-1 -1 -1|
    # |-1  8 -1|
    # |-1 -1 -1|
    # +--------+
    k_deriv = -1 * np.ones((3, 3))
    k_deriv[1, 1] = 8
    convolved_deriv = convolve2d(arr, k_deriv, mode="same")
    mask_deriv = np.zeros(arr.shape)
    mask_deriv[(arr > 20) & (convolved_deriv > 80)] = 1

    # apply only to the central 340 x 340 area to avoid touching the radar circle borders
    mask_deriv[:70] = 0
    mask_deriv[-70:] = 0
    mask_deriv[:, :70] = 0
    mask_deriv[:, -70:] = 0

    # compute surronding pixel average
    avg_3x3 = convolve2d(arr, k_3x3, mode="same") / 8

    # replace pixels with the surrounding average
    condlist = [mask_deriv == 1, arr > 0]
    choicelist = [avg_3x3, arr]
    arr = np.select(condlist, choicelist)

    return arr


def parse_ascii(
    fileobj,
    offset=PARSE_OFFSET,
    scan_size=IMG_SIZE,
    skip_header=SKIP_ROWS,
    output_type=PARSED_TYPE,
    do_clean=True
):
    try:
        arr = np.genfromtxt(
            fileobj,
            skip_footer=offset,
            skip_header=skip_header + offset,
            delimiter="\t",
            autostrip=True,
            usecols=range(offset, scan_size - offset),
            dtype=np.float32,
        )
        arr[np.where(arr == (-99.0))] = 0.0
        arr[np.where(arr == (55.0))] = 0.0
        if arr.shape != (scan_size - offset * 2, scan_size - offset * 2):
            raise BadAsciiException("{} has wrong shape".format(fileobj.name))
        
        if output_type:
            arr = arr.astype(output_type)

        if do_clean:
            arr = clean(arr)

        return np.around(arr, 1)
    except:
        raise BadAsciiException("{} is broken".format(fileobj.name))
    finally:
        fileobj.close()


def recursively_check_run_consistency(f, run, last):
    runs = []
    if len(run) >= MIN_RUN_SIZE:
        try:
            well_formed_scans = []
            for run_scan in run:
                with f.extractfile(run_scan) as extr, gzip.GzipFile(
                    fileobj=extr
                ) as gfile:
                    well_formed_scans.append(parse_ascii(gfile))
        # Note: the parse_ascii function raises a BadAsciiException when the radar scan is malformed.
        # When this happens: the scans to the left are a run and are well-formed. They are saved if they are
        # long enough. We know the scans to the right are a run, but not if they are well-formed.
        except BadAsciiException:
            if len(well_formed_scans) >= MIN_RUN_SIZE:
                runs.append(
                    (
                        np.stack(well_formed_scans),
                        (
                            last - GAP_LENGTH * (len(run) - 1),
                            last - GAP_LENGTH * (len(run) - len(well_formed_scans)),
                        ),
                    )
                )
            runs += recursively_check_run_consistency(
                f, run[len(well_formed_scans) + 1 :], last
            )
        else:
            runs.append(
                (
                    np.stack(well_formed_scans),
                    (last - GAP_LENGTH * (len(run) - 1), last),
                )
            )
    return runs


def identify_runs(f, scans, tags):
    # select only the MAX Z product
    scans = [scan for scan in scans if "cmaZ" in scan.name]
    if scans:
        # A run is sequence of consecutive radar scans within GAP_LENGTH time.
        # This list will contain tuples whose first element is the run data
        # and the second element is a tuple of  (start_datetime, end_datetime)
        # for that run.
        runs = []

        # Run length counts the number of consecutive scans within GAP_LENGTH time.
        run_length = 0

        # First date in scans
        last = pd.to_datetime(scans[0].name[13:-9])

        for idx, scan in enumerate(scans):
            if ".ascii.gz" in scan.name:
                scan_time = pd.to_datetime(scan.name[13:-9])
                #print(scan_time)

                # If scan is within GAP_LENGTH of last scan, increase run_length window
                if scan_time - last <= GAP_LENGTH:
                    run_length += 1
                else:
                    # If run is wide enough, save run, then start a new run
                    if run_length >= MIN_RUN_SIZE:
                        runs += recursively_check_run_consistency(
                            f, scans[idx - run_length : idx], last
                        )
                    run_length = 1
                last = scan_time
        if run_length >= MIN_RUN_SIZE:
            runs += recursively_check_run_consistency(f, scans[-run_length:], last)

        # set to 0 all the points outside the radar circle
        for run, _ in runs:
            run[:, ~MASK] = 0

        # iterate through all the runs and remove those without enough rainfall
        # discard runs with mean < 0.5 dbz
        # discard runs with mean < 1 dbz and without any weak label

        def check_run(run):
            mean = np.mean(run)
            if mean >= 1:
                return True
            elif mean >= 0.5 and tags != "":
                return True
            else:
                return False

        runs = [(run, periods) for run, periods in runs if check_run(run)]

        return runs if runs else None


def worker(args):

    day, radar_directory, tags = args
    path = os.path.join(radar_directory, str(day.year), day.strftime("%Y%m%d.tar"))
    if os.path.exists(path):
        with tarfile.open(path) as tar_archive:
            day_runs = identify_runs(
                tar_archive,
                scans=[
                    file_obj
                    for file_obj in sorted(
                        tar_archive.getmembers()[1:], key=lambda m: m.name
                    )
                ],
                tags=tags,
            )
        if day_runs is not None:
            metadata = []
            i = 0
            with h5py.File(
                os.path.join(
                    radar_directory, "hdf_archives", day.strftime("%Y%m%d.hdf5")
                ),
                "w",
                libver="latest",
            ) as hdf_archive:
                for run, periods in day_runs:
                    avg_value = np.mean(run)
                    metadata.append(
                        {
                            "start_datetime": periods[0],
                            "end_datetime": periods[1],
                            "run_length": run.shape[0],
                            "avg_cell_value": avg_value,
                            "tags": tags,
                        }
                    )
                    dset = hdf_archive.create_dataset(
                        "{}".format(i),
                        chunks=(1, CROP_SIZE, CROP_SIZE),
                        shuffle=True,
                        fletcher32=True,
                        compression="gzip",
                        compression_opts=9,
                        dtype=PARSED_TYPE,
                        data=run,
                    )
                    dset.attrs["start_datetime"] = str(periods[0])
                    dset.attrs["end_datetime"] = str(periods[1])
                    dset.attrs["run_length"] = run.shape[0]
                    dset.attrs["avg_cell_value"] = avg_value
                    dset.attrs["tags"] = str(tags)
                    i += 1
                hdf_archive.flush()
            return metadata
