import argparse
import time
from pathlib import Path
from typing import Any, Dict, Generator, Tuple, Union, Optional, List
from handler import Trainer
from tqdm import tqdm
import multiprocessing
from omegaconf import OmegaConf
import re

import src.sampler as smp
from tokenizer_cfg import define_argparser
from util import file_util

path = Path(__file__).absolute().parent

conf = OmegaConf.load('config.yaml')
file_path = Path('.') / conf['corpora']


if __name__ == "__main__":
    start_time = time.time()
    config = define_argparser()

    file_list = [
        str(files.absolute()) for files in Path(config.files).glob("*.txt")
    ]

    if config.sample_rate:
        procs = []

        for file in file_list:
            proc = multiprocessing.Process(
                target=smp.shuf_sampling,
                args=(
                    file,
                    file_util.path_manager(file, conf.output_path),
                    config.sample_rate,
                    config.use_morph,
                ),
            )
            procs.append(proc)
            proc.start()

        for proc in tqdm(procs):
            proc.join()

        source_list = [
            input_file.replace("corpus", "sample") for input_file in file_list
        ]
    else:
        source_list = [
            str(files.absolute()) for files in Path(config.files).glob("*.txt")
        ]

    print(f"--- {time.time() - start_time} seconds ---")

    model = Trainer(config, source_list)
    model.tokenizer_handler()

