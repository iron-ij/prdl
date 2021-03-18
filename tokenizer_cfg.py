import argparse
from typing import Any, Dict, Tuple, Union, Optional, List

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True, type=str)

    # Define sample ratio
    p.add_argument("--sample_rate", type=float)
    p.add_argument("--use_morph", type=bool, default=False)

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