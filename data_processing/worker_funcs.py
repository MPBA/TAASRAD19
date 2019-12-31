import gzip
import os
import tarfile

import h5py
import numpy as np
import pandas as pd

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


def parse_ascii(
    fileobj,
    offset=PARSE_OFFSET,
    scan_size=IMG_SIZE,
    skip_header=SKIP_ROWS,
    output_type=PARSED_TYPE,
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

        return arr
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
                    hdf_archive.create_dataset(
                        "{}".format(i),
                        chunks=(1, CROP_SIZE, CROP_SIZE),
                        shuffle=True,
                        fletcher32=True,
                        compression="gzip",
                        compression_opts=9,
                        data=run,
                    )
                    i += 1
                hdf_archive.flush()
            return metadata
