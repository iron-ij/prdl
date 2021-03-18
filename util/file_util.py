from typing import Any, Callable, Dict, Generator, Tuple, Union, Optional, List
from pathlib import Path
import multiprocessing
from tqdm import tqdm
import re

def path_manager(input, output):
    if isinstance(input, Path):  
        input = str(input)
    return input.replace(input, output)

def get_input_list(file_path: Union[str, Path], out_path) -> List[Path]:
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    file_list = []
    for dirs in file_path.iterdir():
        if dirs.is_dir():
            file_list.extend(list(dirs.glob('*.txt')))
        elif dirs.suffix == '.txt':
            file_list.append(dirs)

    file_list_set = [
        [input_file, 
         input_file.parent.parent / out_path / input_file.name
        ]
        for input_file
        in file_list
    ]
    return file_list_set

def multi_proc_executor(file_params, func):
    procs = []
    for i, o, r in file_params:
        proc = multiprocessing.Process(
            target=func,
            args=(i, o, r),
        )
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()