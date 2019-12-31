import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import h5py
import pandas as pd
from tqdm import tqdm

from settings import END_DATE, START_DATE
from worker_funcs import worker


"""
This script generates `run` of scans. Each `run` is a consecutive series of radar scans
such the difference in minute between every pair of consecutive scans is not greater 
than GAP_LENGTH (default to 5 minutes).
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert the radar dataset from TAR/ZIP format to HDF5."
    )

    parser.add_argument("radar_directory", type=str)
    args = parser.parse_args()

    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
    csv_path = os.path.join(args.radar_directory, "hdf_metadata.csv")
    ouput_dir = os.path.join(args.radar_directory, "hdf_archives")
    date_descriptions_path = os.path.join(
        args.radar_directory, "daily_weather_report_tags.csv"
    )

    if os.path.exists(csv_path):
        metadata = pd.read_csv(csv_path, index_col="id")
    else:
        metadata = pd.DataFrame(
            columns=[
                "start_datetime",
                "end_datetime",
                "run_length",
                "avg_cell_value",
                "tags",
            ]
        )
    run_n = len(metadata)
    os.makedirs(ouput_dir, exist_ok=True)
    date_descriptions = pd.read_csv(date_descriptions_path)
    date_descriptions = date_descriptions.fillna("")

    date_tags = {}

    for _, row in date_descriptions.iterrows():
        date_tags[row.date_iso] = row.tags

    with h5py.File(
        os.path.join(ouput_dir, "all_data.hdf5"), "w", libver="latest"
    ) as hdf_archive:
        with ProcessPoolExecutor() as executor:
            worker_args = [
                (day, args.radar_directory, date_tags[str(day.date())],)
                for day in date_range
            ]
            with tqdm(total=len(worker_args)) as pbar:
                for metadata_list in executor.map(worker, worker_args):
                    if metadata_list:
                        for idx, metadata_dict in enumerate(metadata_list):
                            metadata.loc[run_n] = metadata_dict
                            hdf_archive[str(run_n)] = h5py.ExternalLink(
                                os.path.join(
                                    ouput_dir,
                                    metadata_dict["start_datetime"].strftime(
                                        "%Y%m%d.hdf5"
                                    ),
                                ),
                                str(idx),
                            )
                            run_n += 1
                            hdf_archive.flush()
                    pbar.update(1)

    metadata.to_csv(csv_path, index_label="id")
    print(metadata)
