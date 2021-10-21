# Copyright 2021 The neiss authors. All Rights Reserved.
#
# This file is part of tf_neiss_nlp.
#
# tf_neiss_nlp is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# tf_neiss_nlp is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# tf_neiss_nlp. If not, see http://www.gnu.org/licenses/.
# ==============================================================================
from collections import Counter
from paiargparse import pai_dataclass, PAIArgumentParser
from dataclasses import dataclass

from tfaip_addons.util.file.pai_file import File
from tfaip_scenario.nlp.text_class.data.pipelines.map_text_2_id import compute_ngrams_py
from tfaip_scenario.nlp.text_class.util.simpletokenizer import split_text

DC = "dc"
PC = "pc"
NER = "ner"


def save(filename, counts):
    with open(filename, "w+") as f:
        for word, freq in counts:
            f.write(f"{word},{freq}\n")


@pai_dataclass
@dataclass
class VocabCreatorParams:
    corpus: str
    token_vocab_name: str = "tokens.vocab"
    char_vocab_name: str = "char.vocab"
    ngram_vocab_name: str = "ngram.vocab"
    label_vocab_name: str = "labels.vocab"
    remove_punct: bool = True
    ignore_chars: str = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
    tokenizer: str = "simpleTokenizer"
    mode: str = DC
    min_n: int = 3
    max_n: int = 3


def add_ner_labels(entities, cntr_labels):
    if (len(entities.keys())) != 1:
        raise RuntimeError("several modul ids found. Unsure which own to take")

    for modul_id in entities.keys():
        labels = [result.label for result in entities.get(modul_id).results]
        cntr_labels.update(labels)


def add_class_labels(class_dict, cntr_labels: Counter):
    if (len(class_dict.keys())) != 1:
        raise RuntimeError("several modul ids found. Unsure which own to take")

    for modul_id in class_dict.keys():
        if len(class_dict.get(modul_id).results) != 1:
            raise RuntimeError("several classifications. Which one is correct?")
        label = [class_dict.get(modul_id).results[0].label]
        cntr_labels.update(label)


def run(args: VocabCreatorParams):
    if args.tokenizer == "simpleTokenizer":
        splitter = split_text
    else:
        raise RuntimeError("unsupported tokenization")

    with open(args.corpus, "r") as f:
        files = [line.strip(" \n") for line in f if line.strip(" \n")]

    cntr_words = Counter()
    cntr_ngrams = Counter()
    cntr_labels = Counter()
    i = 0
    for file in files:
        if i % 1 == 0:
            # print(f"processed {i} of {len(files)} text files")
            print(f"processed {i} of {len(files)} text files")
        i += 1

        if file.endswith("txt"):
            with open(file, "r") as f:
                text = f.read().replace("\n", " ").replace(" +", " ").strip()
            if args.mode == DC:
                with open(file + ".info", "r") as f:
                    label = [f.read().replace("\n", " ").replace(" +", " ").strip()]
                cntr_labels.update(label)
            else:
                raise RuntimeError("Unsupported mode")
        elif file.endswith("json"):

            paifile: File = File.load(file)
            text = " ".join([line.text.content for line in paifile.get_lines(allow_multiple_pages=True)])
            if args.mode == DC:
                add_class_labels(paifile.get_classifications(), cntr_labels)
            elif args.mode == PC:
                for page in paifile.get_pages():
                    add_class_labels(page.get_classifications(), cntr_labels)
            elif args.mode == NER:
                add_ner_labels(paifile.get_entities(), cntr_labels)
            class_dict = paifile.get_classifications()

        else:
            raise RuntimeError("unsuported file type")

        words = splitter(text, args.remove_punct, args.ignore_chars)
        ngrams = [ngram for word in words for ngram in compute_ngrams_py(word, args.min_n, args.max_n)]
        cntr_words.update(words)
        cntr_ngrams.update(ngrams)

    cntr_chars = Counter()

    for word, freq in cntr_words.most_common(len(cntr_words)):
        cntr_chars.update(word)

    save(args.char_vocab_name, cntr_chars.most_common(len(cntr_chars)))
    save(args.token_vocab_name, cntr_words.most_common(int(len(cntr_words) * 0.2)))
    save(args.ngram_vocab_name, cntr_ngrams.most_common(int(len(cntr_ngrams))))
    save(args.label_vocab_name, cntr_labels.most_common(len(cntr_labels)))


if __name__ == "__main__":
    parser = PAIArgumentParser()
    parser.add_root_argument("myArgs", VocabCreatorParams, flat=True)
    args = parser.parse_args()
    run(args)
