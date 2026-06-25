"""
LAUSCHER – Flexible Auditory Spike Conversion Chain

Reference: https://arxiv.org/abs/1910.07407
"""

import argparse
import logging
import os
from os.path import isfile, isdir
import numpy as np

from lauscher.audiowaves import FileMonoAudioWave
from lauscher.helpers import CommandLineArguments
from lauscher.transformations.wave2spike import Wave2Spike

def main(input_name: str,
         output_name: str,
         num_channels: int,
         rng_seed: int | None):
    if rng_seed == None:
        global_rng = np.random.default_rng()
    else:
        global_rng = np.random.default_rng(rng_seed)
    if isfile(input_name):
        print(f'Processing {input_name} into {output_name}')
        process_single_file(input_name, output_name, num_channels, global_rng)
    elif isdir(input_name):
        list_of_files = os.listdir(input_name)
        list_of_files.sort()
        for file in list_of_files:
            if file[-5:] == ".flac":
                i_file_path = os.path.join(input_name, file)
                o_file_path = os.path.join(output_name, f"{file[:-5]}.npz")
                print(f'Processing {i_file_path} into {o_file_path}')
                process_single_file(i_file_path, o_file_path, num_channels, global_rng)
            else:
                print(f'Skipping {input_name}/{file}')
    else:
        raise IOError(f"Input file/folder '{input_name}' not found.")

def process_single_file(input_file: str, output_file: str, num_channels: int, rng: np.random.Generator):
    trafo = Wave2Spike(num_channels=num_channels, rng=rng)
    spikes = FileMonoAudioWave(input_file).transform(trafo)
    spikes.export(output_file)

def main_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_name", type=str,
                        help="""Path to the input sound file (.flac), or folder
                                containing sound files, to be converted to
                                spike train files.""")
    parser.add_argument("output_name", type=str,
                        help="""Path to the output file, or folder, in case
                                of an input folder. Spike train files will
                                be written into it in .npz. Folder processing will
                                use the same name as the input files scanned""")
    parser.add_argument("--num_channels", type=int, default=700,
                        help="Number of frequency selective channels.")
    parser.add_argument("--rng_seed", type=int, default=None,
                        help="""Seed for the global random number generator. 
                                Uses random seed if unspecified. Outputs 
                                equivalent even if the jobs parameter is different
                                between runs.""")
    parser.add_argument("-j", "--jobs", type=int, default=None,
                        help="""Number of concurrent jobs used for data 
                             processing.""")
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    global_args = CommandLineArguments()
    global_args.num_concurrent_jobs = args.jobs
    main(args.input_name, args.output_name, args.num_channels, args.rng_seed)


if __name__ == "__main__":
    main_cli()
