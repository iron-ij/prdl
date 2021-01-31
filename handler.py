import argparse
from os import error
from pathlib import Path
from tokenizers import CharBPETokenizer, ByteLevelBPETokenizer, BertWordPieceTokenizer
from typing import Any, Dict, Tuple, Union, Optional, List

path = Path(__file__).absolute().parent


class Trainer:
    def __init__(self, conf, files) -> None:
        self.conf = conf
        self.files = files

    def _bbpe(self):
        tokenizer = ByteLevelBPETokenizer(
            vocab=self.conf.vocab,
            merges=self.conf.merges,
            add_prefix_space=self.conf.add_prefix_space,
            lowercase=self.conf.lowercase,
            dropout=self.conf.dropout,
            unicode_normalizer=self.conf.unicode_normalizer,
            continuing_subword_prefix=self.conf.continuing_subword_prefix,
            end_of_word_suffix=self.conf.end_of_word_suffix,
            trim_offsets=self.conf.trim_offsets,
        )

        tokenizer.train(
            files=self.files,
            vocab_size=self.conf.vocab_size,
            min_frequency=self.conf.min_frequency,
            special_tokens=self.conf.bbpe_special_tokens,
        )

        return tokenizer

    def _cbpe(self):
        tokenizer = CharBPETokenizer(
            vocab=self.conf.vocab,
            merges=self.conf.merges,
            unk_token=self.conf.cbpe_unk_token,
            suffix=self.conf.suffix,
            dropout=self.conf.dropout,
            lowercase=self.conf.lowercase,
            unicode_normalizer=self.conf.unicode_normalizer,
            bert_normalizer=self.conf.bert_normalizer,
            split_on_whitespace_only=self.conf.split_on_whitespace_only,
        )

        tokenizer.train(
            files=self.files,
            vocab_size=self.conf.vocab_size,
            min_frequency=self.conf.min_frequency,
            special_tokens=self.conf.special_tokens,
            limit_alphabet=self.conf.limit_alphabet,
            initial_alphabet=self.conf.initial_alphabet,
            suffix=self.conf.cpbe_train_shuffix,
        )

        return tokenizer

    def _wordpiece(self):
        tokenizer = BertWordPieceTokenizer(
            vocab=self.conf.vocab,
            unk_token=self.conf.unk_token,
            sep_token=self.conf.sep_token,
            cls_token=self.conf.cls_token,
            pad_token=self.conf.pad_token,
            mask_token=self.conf.mask_token,
            clean_text=self.conf.clean_text,
            handle_chinese_chars=self.conf.handle_chinese_chars,
            strip_accents=self.conf.strip_accents,
            lowercase=self.conf.lowercase,
            wordpieces_prefix=self.conf.wordpieces_prefix,
        )

        tokenizer.train(
            files=self.files,
            vocab_size=self.conf.vocab_size,
            min_frequency=self.conf.min_frequency,
            limit_alphabet=self.conf.limit_alphabet,
            initial_alphabet=self.conf.initial_alphabet,
            special_tokens=self.conf.word_piece_special_tokens,
            wordpieces_prefix=self.conf.wordpieces_prefix,
        )

        return tokenizer

    def tokenizer_handler(self):
        tokenizer_dict = {
            "bbpe": self._bbpe,
            "cbpe": self._cbpe,
            "wordpiece": self._wordpiece,
        }

        used_tokenizer = self.conf.tokenizer

        if self.conf.tokenizer not in tokenizer_dict.keys():
            raise Exception("Check tokenizer!")
        else:
            trained_tokenizer = tokenizer_dict[self.conf.tokenizer]()

        trained_tokenizer.save(
            f"./{str(used_tokenizer)}_{str(self.conf.vocab_size)}.json"
        )
