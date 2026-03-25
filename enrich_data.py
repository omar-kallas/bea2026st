#!/usr/bin/env python3
import csv
from pathlib import Path
from typing import Dict
from tqdm import tqdm
from phonemizer import phonemize
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from phonemizer.backend import EspeakBackend
from dragonmapper.transcriptions import pinyin_to_ipa
from pypinyin import pinyin, Style
from wordfreq import zipf_frequency
import pyphen
import Levenshtein


def corpus_frequency(word):
    return zipf_frequency(word, "en")  # Zipf scale (0–8), higher = more frequent


PREFIXES = ["un", "re", "in", "im", "dis", "pre", "mis"]
SUFFIXES = ["ing", "ed", "ly", "ness", "ment", "tion", "able", "ible", "er", "est"]

def count_morphemes(word):
    count = 1  # root
    for p in PREFIXES:
        if word.startswith(p):
            count += 1
            word = word[len(p):]
    for s in SUFFIXES:
        if word.endswith(s):
            count += 1
            word = word[:-len(s)]
    return count


dic = pyphen.Pyphen(lang='en')

def syllable_count(word):
    hyphenated = dic.inserted(word)
    return len(hyphenated.split("-"))


def edit_distance(w1, w2):
    return Levenshtein.distance(w1, w2)


EspeakWrapper.set_library("/home/omar.kallas/.local/lib/libespeak-ng.so")
be_en = EspeakBackend("en-us")
be_de = EspeakBackend("de")
be_es = EspeakBackend("es")

def get_ipa(en_word, l1_word, l1):
    ipa_en, ipa_l1 = None, None
    try:
        ipa_en = be_en.phonemize([en_word], strip=True)[0]
    except Exception as e:
        print(f"An error has occurec processing {en_word}")
        print(f"Error: {e.with_traceback()}")
    
    try:
        if l1 == 'de':
            ipa_l1 = be_de.phonemize([l1_word], strip=True)[0]
        elif l1 == 'es':
            ipa_l1 = be_es.phonemize([l1_word], strip=True)[0]
        elif l1 == 'cn':
            py = pinyin(l1_word, style=Style.TONE3, v_to_u=True)
            py = " ".join([s[0] for s in py])
            ipa_l1 = pinyin_to_ipa(py)
    except Exception as e:
        print(f"An error has occurec processing {l1_word}, {l1}, {en_word}")
        print(f"Error: {e}")

    return ipa_en, ipa_l1


def augment_row(row: Dict[str, str], lang: str) -> Dict[str, str]:
    """
    Placeholder row-level feature augmentation.
    Replace this implementation with real feature engineering logic.
    """
    augmented = dict(row)
            
    augmented["ipa_en"], augmented["ipa_l1"] = get_ipa(row["en_target_word"], row["L1_source_word"], row["L1"])

    augmented.update({
        "freq": corpus_frequency(row["en_target_word"]),
        "morpheme_count": count_morphemes(row["en_target_word"]),
        "char_length": len(row["en_target_word"]),
        "syllables": syllable_count(row["en_target_word"]),
        "edit_distance": edit_distance(row["en_target_word"], row["L1_source_word"]) if row["L1"] in ['es', 'de'] else '-',
        "phon_edit_distance": edit_distance(augmented["ipa_en"], augmented["ipa_l1"])
    })

    return augmented


def enrich_csv_file(input_path: Path, output_path: Path, lang: str) -> None:
    with input_path.open("r", encoding="utf-8", newline="") as in_f:
        reader = csv.DictReader(in_f)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header row in: {input_path}")

        original_fieldnames = list(reader.fieldnames)
        new_fieldnames = [
            "ipa_en",
            "ipa_l1",
            "freq",
            "morpheme_count",
            "char_length",
            "syllables",
            "edit_distance",
            "phon_edit_distance",
        ]
        output_fieldnames = original_fieldnames + [
            f for f in new_fieldnames if f not in original_fieldnames
        ]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=output_fieldnames)
            writer.writeheader()
            for row in tqdm(reader):
                writer.writerow(augment_row(row, lang=lang))


def enrich_directory(input_root: Path, output_root: Path) -> None:
    csv_files = sorted(input_root.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {input_root}")

    for input_file in csv_files:
        relative_path = input_file.relative_to(input_root)
        if len(relative_path.parts) < 2:
            raise ValueError(
                f"Expected path format '<split>/<lang>/file.csv', got: {relative_path}"
            )
        lang = relative_path.parts[1]
        if lang not in {"cn", "es", "de"}:
            raise ValueError(f"Unexpected language '{lang}' in path: {relative_path}")
        output_file = output_root / relative_path
        print(f"Enriching: {relative_path}")
        enrich_csv_file(input_file, output_file, lang=lang)
        print(f"Enriched: {input_file} -> {output_file}")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    input_root = project_root / "data"
    output_root = project_root / "data_enriched"

    if not input_root.exists():
        raise FileNotFoundError(f"Input directory not found: {input_root}")
    
    enrich_directory(input_root, output_root)
    print(f"Done. Enriched files written to: {output_root}")


if __name__ == "__main__":
    main()
