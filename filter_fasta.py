#!/usr/bin/env python3
"""Mask low-probability predictions and bases already known in model_input."""

import argparse
import csv
import gzip
import math
from pathlib import Path
import re
import sys


REFERENCE_IDS = {"normal", "ref", "reference"}


def open_fasta(path, mode):
    """Read or write text FASTA, using gzip for filenames ending in .gz."""
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    encoding = "utf-8-sig" if mode == "rt" else "utf-8"
    return opener(path, mode, encoding=encoding)


def read_fasta(path):
    records = []
    header = None
    chunks = []
    with open_fasta(path, "rt") as handle:
        for number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header = line[1:]
                if not header.strip():
                    raise ValueError(f"{path}:{number}: empty FASTA header")
                chunks = []
            else:
                if header is None:
                    raise ValueError(f"{path}:{number}: sequence before FASTA header")
                chunks.append("".join(line.split()))
    if header is not None:
        records.append((header, "".join(chunks)))
    if not records:
        raise ValueError(f"{path}: no FASTA records")
    ids = set()
    for header, sequence in records:
        identifier = header.split()[0]
        if identifier in ids:
            raise ValueError(f"{path}: duplicate FASTA ID {identifier!r}")
        if not sequence:
            raise ValueError(f"{path}: empty sequence {identifier!r}")
        ids.add(identifier)
    return records


def read_probabilities(path, lengths, site_base):
    """Accept sample-by-site, long or wide TSV and validate all coordinates."""
    probabilities = {identifier: {} for identifier in lengths}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)):
            raise ValueError(f"{path}: missing or duplicate TSV column names")
        normalized = {field.strip().lower(): field for field in fields}

        def column(*aliases):
            return next((normalized[a] for a in aliases if a in normalized), None)

        site_col = column("site", "position", "pos", "site_index")
        id_col = column("sample_id", "sequence_id", "sequence", "seq_id", "id", "name")
        prob_col = column("probability", "top_probability", "top_probability_per_site", "prob")
        site_columns = {
            field: int(match.group(1)) - site_base
            for field in fields
            if (match := re.fullmatch(r"site_(\d+)", field.strip(), re.IGNORECASE))
        }
        matrix_format = id_col is not None and bool(site_columns) and site_col is None
        if site_col is None and not matrix_format:
            raise ValueError(f"{path}: expected a 'site' column (1-based by default)")
        long_format = id_col is not None and prob_col is not None
        if not long_format and not matrix_format:
            missing = set(lengths) - set(fields)
            if missing:
                raise ValueError(
                    f"{path}: expected sequence_id/site/probability columns or "
                    f"site plus sequence-ID columns; missing {sorted(missing)}"
                )

        def add(identifier, index, raw, row_number):
            if identifier not in lengths:
                return  # Allows reference rows and unrelated sequences in the table.
            if not 0 <= index < lengths[identifier]:
                raise ValueError(f"{path}:{row_number}: site outside sequence {identifier!r}")
            if index in probabilities[identifier]:
                raise ValueError(f"{path}:{row_number}: duplicate site for {identifier!r}")
            try:
                value = float(raw)
            except (TypeError, ValueError):
                raise ValueError(f"{path}:{row_number}: invalid probability {raw!r}") from None
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{path}:{row_number}: probability must be between 0 and 1")
            probabilities[identifier][index] = value

        for row in reader:
            row_number = reader.line_num
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}:{row_number}: wrong number of TSV fields")
            if matrix_format:
                for field, index in site_columns.items():
                    add(row[id_col].strip(), index, row[field], row_number)
                continue
            try:
                index = int(row[site_col]) - site_base
            except (ValueError, TypeError):
                raise ValueError(f"{path}:{row_number}: site must be an integer") from None
            if long_format:
                add(row[id_col].strip(), index, row[prob_col], row_number)
            else:
                for identifier in lengths:
                    add(identifier, index, row[identifier], row_number)
    for identifier, length in lengths.items():
        missing = length - len(probabilities[identifier])
        if missing:
            raise ValueError(f"{path}: {missing} site probabilities missing for {identifier!r}")
    return probabilities


def run(args):
    if not math.isfinite(args.cutoff) or not 0 <= args.cutoff <= 1:
        raise ValueError("cutoff must be between 0 and 1")
    probability_path = args.fasta.parent / "top_probability_per_site.tsv"
    for source in (args.fasta, args.model_input, probability_path):
        if args.output.resolve() == source.resolve() or (
            args.output.exists() and source.exists() and args.output.samefile(source)
        ):
            raise ValueError("output must differ from all input files")
    records = read_fasta(args.fasta)
    models = read_fasta(args.model_input)
    model_by_id = {header.split()[0]: sequence for header, sequence in models}
    lengths = {
        header.split()[0]: len(sequence)
        for header, sequence in records
        if header.split()[0].lower() not in REFERENCE_IDS
    }
    masks = {}
    for identifier, length in lengths.items():
        mask = models[0][1] if len(models) == 1 else model_by_id.get(identifier)
        if mask is None:
            raise ValueError(f"model_input has no record matching {identifier!r}")
        if len(mask) != length:
            raise ValueError(f"model_input length differs from {identifier!r} ({len(mask)} vs {length})")
        masks[identifier] = mask
    probabilities = read_probabilities(probability_path, lengths, args.site_base) if lengths else {}
    output = []
    for header, sequence in records:
        identifier = header.split()[0]
        if identifier in lengths:
            sequence = "".join(
                base if masks[identifier][index] in "-?" and probabilities[identifier][index] >= args.cutoff
                else "-"
                for index, base in enumerate(sequence)
            )
        output.append(f">{header}\n{sequence}\n")
    # Validate everything before opening the output file.
    with open_fasta(args.output, "wt") as handle:
        handle.write("".join(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fasta", type=Path, help="Prediction FASTA (plain or .gz); TSV must be in the same directory")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Output FASTA; use .gz for gzip compression")
    parser.add_argument("--cutoff", required=True, type=float, help="Keep probabilities >= this value (0 to 1)")
    parser.add_argument("--model-input", "--model_input", dest="model_input", required=True, type=Path,
                        help="Aligned FASTA (plain or .gz): any character other than '-' or '?' marks a known site")
    parser.add_argument("--site-base", type=int, choices=(0, 1), default=1,
                        help="TSV site numbering: 0 or 1 (default: 1); counts gaps too")
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError, EOFError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
