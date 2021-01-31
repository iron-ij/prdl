import argparse
import time
from pathlib import Path
from typing import Any, Dict, Tuple, Union, Optional, List
from handler import Trainer
from tqdm import tqdm
import multiprocessing

import sampler

path = Path(__file__).absolute().parent


def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True, type=str)

    # Define sample ratio
    p.add_argument("--sample_rate", type=float)

    # Common args

    # Common init args
    p.add_argument("--vocab", type=Optional[Union[str, Dict[str, int]]], default=None)
    p.add_argument("--lowercase", type=bool, default=True)

    # Common training args
    p.add_argument(
        "--files", required=True, type=str, help="input corpus parent directory path"
    )
    p.add_argument("--vocab_size", type=int, default=30000)
    p.add_argument("--min_frequency", type=int, default=2)
    p.add_argument("--special_tokens", default=["<unk>"])

    # CBPE init args
    p.add_argument(
        "--merges",
        type=Optional[Union[str, Dict[Tuple[int, int], Tuple[int, int]]]],
        default=None,
    )
    p.add_argument("--dropout", type=Optional[float], default=None)
    p.add_argument("--unicode_normalizer", type=Optional[str], default=None)
    p.add_argument("--cbpe_unk_token", default="<unk>")  # check
    p.add_argument("--suffix", type=str, default="</w>")
    p.add_argument("--bert_normalizer", type=bool, default=True)
    p.add_argument("--split_on_whitespace_only", type=bool, default=False)

    # CBPE training args
    p.add_argument("--limit_alphabet", type=int, default=1000)
    p.add_argument("--initial_alphabet", type=List[str], default=[])
    p.add_argument("--cpbe_train_shuffix", default="</w>")  # check

    # BBPE init args
    p.add_argument("--add_prefix_space", type=bool, default=False)
    p.add_argument("--continuing_subword_prefix", type=Optional[str], default=None)
    p.add_argument("--end_of_word_suffix", type=Optional[str], default=None)
    p.add_argument("--trim_offsets", type=bool, default=False)

    # BBPE training args
    p.add_argument("--bbpe_special_tokens", default=[])

    # Wordpiece init args
    p.add_argument("--unk_token", default="[UNK]")
    p.add_argument("--sep_token", default="[SEP]")
    p.add_argument("--cls_token", default="[CLS]")
    p.add_argument("--pad_token", default="[PAD]")
    p.add_argument("--mask_token", default="[MASK]")
    p.add_argument("--clean_text", type=bool, default=True)
    p.add_argument("--handle_chinese_chars", type=bool, default=True)
    p.add_argument(
        "--strip_accents", type=Optional[bool], default=None
    )  # Must be False if lowercase is False
    p.add_argument("--wordpieces_prefix", type=str, default="##")

    # Wordpiece train args
    p.add_argument(
        "--word_piece_special_tokens",
        default=[
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
        ],
    )

    config = p.parse_args()

    return config


if __name__ == "__main__":
    start_time = time.time()
    config = define_argparser()

    if config.sample_rate:
        file_list = [
            str(files.absolute()) for files in Path(config.files).glob("*.txt")
        ]
        procs = []

        for file in file_list:
            proc = multiprocessing.Process(
                target=sampler.execute_script,
                args=(
                    file,
                    config.sample_rate,
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
    print(source_list)

    model = Trainer(config, source_list)
    model.tokenizer_handler()
