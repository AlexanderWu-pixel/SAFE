"""DemoDiff NPE boundaries -> SAFE custom slicer. No neural embedding is used."""
from __future__ import annotations
from typing import Any, TextIO
from graphbpe_slicer import BPESlicer

import argparse
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / 'data' / 'tokenizer'
from rdkit import Chem
from converter import SAFEConverter
from tqdm import tqdm


def identity(mol: Chem.Mol) -> str:
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


# Observe the cuts SAFE actually accepts, after its stereo protection.
# This subclass collects diagnostics; the main converter performs SAFE encoding.
class AuditedConverter(SAFEConverter):
    def _fragment(self, mol: Chem.Mol, allow_empty: bool=False) -> list[tuple[int, int]]:
        self.callback_smiles = Chem.MolToSmiles(mol, canonical=False)
        # This map-labelled string is only an audit reference, NOT encoder input.
        indexed = Chem.Mol(mol)
        for atom in indexed.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
        self.indexed_source = Chem.MolToSmiles(indexed, canonical=False)
        result = super()._fragment(mol, allow_empty=allow_empty)
        self.used: list[tuple[int, int]] = [tuple(pair) for pair in result]
        copy = Chem.RWMol(mol)
        for u, v in self.used:
            copy.RemoveBond(u, v)
        self.actual_groups: list[list[int]] = [list(g) for g in Chem.GetMolFrags(copy.GetMol())]
        return result


# Per-worker conversion pipeline. Reuses the loaded vocabulary across rows.
# Strict default: preserve both the BPE boundaries AND the isomeric molecular graph.
class Pipeline:
    def __init__(self, prefix: str | Path, allow_coarsening: bool=False) -> None:
        if not importlib.metadata.version('safe-mol').startswith('0.2.0'):
            raise RuntimeError('Install the tested safe-mol==0.2.0 version')
        self.converter: AuditedConverter = AuditedConverter(slicer='bpe', bpe_vocab_path=str(prefix), ignore_stereo=False)
        self.slicer: BPESlicer = self.converter.slicer
        self.allow_coarsening = allow_coarsening

    def convert(self, smiles: str) -> dict[str, Any]:
        source = Chem.MolFromSmiles(smiles)
        if source is None or source.GetNumAtoms() == 0:
            raise ValueError('Invalid or empty SMILES')
        if any(a.GetAtomMapNum() for a in source.GetAtoms()):
            raise ValueError('Atom-mapped input is not supported; remove maps explicitly')
        if any(a.GetAtomicNum() == 0 for a in source.GetAtoms()):
            raise ValueError('Wildcard/dummy atoms require open-fragment SAFE semantics; '
                             'not supported by this closed-molecule pipeline')
        normalized = identity(source)
        # SAFE invokes the slicer on its own working molecule. This avoids
        # reusing indices from a differently ordered input molecule.
        encoded = self.converter.encoder(
            normalized, canonical=True, randomize=False, allow_empty=True,
        )
        # Compare undirected edge sets; endpoint order does not affect a cut.
        requested = {tuple(sorted(p)) for p in self.slicer.requested}
        used = {tuple(sorted(p)) for p in self.converter.used}
        exact = requested == used
        if not exact and not self.allow_coarsening:
            raise ValueError('SAFE stereo protection changed BPE cuts; use '
                             '--allow-stereo-coarsening to explicitly allow this')
        if len(encoded.split('.')) != len(self.converter.actual_groups):
            raise ValueError('SAFE serialization fallback changed the fragment partition')
        # Strict RDKit parse: never repair dangling attachment labels.
        reconstructed = Chem.MolFromSmiles(encoded)
        if reconstructed is None or identity(reconstructed) != normalized:
            raise ValueError('SAFE round-trip failed (including stereo/charge/isotope)')
        return {
            'normalized_smiles': normalized,
            'safe': encoded,
            'bpe_node_ids': self.slicer.node_ids,
            'bpe_groups': self.slicer.groups,
            'requested_cuts': self.slicer.requested,
            'used_cuts': self.converter.used,
            'safe_groups': self.converter.actual_groups,
            'indexed_source_smiles': self.converter.indexed_source,
            'bpe_boundary_exact': exact,
            'num_bpe_fragments': len(self.slicer.groups),
            'num_safe_fragments': len(self.converter.actual_groups),
            'roundtrip_ok': True,
        }


PIPELINE = None


# ProcessPool initializer: each process gets its own mutable slicer state.
def initialize(prefix: str | Path, allow_coarsening: bool) -> None:
    global PIPELINE
    PIPELINE = Pipeline(prefix, allow_coarsening)


# Preserve source identifiers/representations and isolate per-row failures.
# A failed molecule goes to failed.jsonl instead of silently disappearing.
def process(record: tuple[int, dict[str, str | None]]) -> tuple[bool, dict[str, Any]]:
    index, row = record
    base = {'record_index': index, 'chembl_id': row.get('chembl_id'),
            'canonical_smiles': row.get('canonical_smiles'),
            'standard_inchi': row.get('standard_inchi'),
            'standard_inchi_key': row.get('standard_inchi_key')}
    try:
        if None in row or any(v is None for v in row.values()):
            raise ValueError('Malformed CSV record')
        smiles = (row['canonical_smiles'] or '').strip()
        if not smiles:
            raise ValueError('Empty canonical_smiles')
        return True, {**base, **PIPELINE.convert(smiles)}
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return False, {
            **base,
            'error_type': type(exc).__name__,
            'error': str(exc),
    }


def input_reader(fin: TextIO, delimiter: str='auto') -> tuple[csv.DictReader, str]:
    """Detect CSV/TSV from the header, not the filename or chemical contents."""
    csv.field_size_limit(10_000_000)
    header = fin.readline()
    fin.seek(0)
    separator = ('\t' if '\t' in header else ',') if delimiter == 'auto' else {'tab': '\t', 'comma': ','}[delimiter]
    reader = csv.DictReader(fin, delimiter=separator)
    if not {'chembl_id', 'canonical_smiles'} <= set(reader.fieldnames or []):
        raise ValueError(f'Missing required columns; actual header: {reader.fieldnames}')
    return reader, separator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--vocab-prefix', type=Path,
                        default=VENDOR / 'vocab3000ring300' / 'pretrain-token')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--delimiter', choices=['auto', 'tab', 'comma'], default='auto')
    parser.add_argument('--allow-stereo-coarsening', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1 or args.limit < 0:
        parser.error('workers/batch-size must be positive; limit must be nonnegative')
    if not args.input.is_file():
        parser.error(f'Input not found: {args.input}')
    for suffix in ('.node', '.edge', '.ring'):
        if not Path(str(args.vocab_prefix) + suffix).is_file():
            parser.error(f'Missing vocabulary: {args.vocab_prefix}{suffix}')
    initialize(str(args.vocab_prefix), args.allow_stereo_coarsening)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    errors = Counter()
    started = time.monotonic()
    pool = None
    try:
        if args.workers > 1:
            pool = ProcessPoolExecutor(
                max_workers=args.workers, mp_context=mp.get_context('spawn'),
                initializer=initialize,
                initargs=(str(args.vocab_prefix), args.allow_stereo_coarsening),
            )
        with args.input.open(encoding='utf-8-sig', newline='') as fin, \
                (args.output_dir / 'records.jsonl').open('w', encoding='utf-8') as fout, \
                (args.output_dir / 'safe.txt').open('w', encoding='utf-8') as corpus, \
                (args.output_dir / 'failed.jsonl').open('w', encoding='utf-8') as ferr, \
                tqdm(desc='BPE -> SAFE', unit='mol') as progress:
            reader, separator = input_reader(fin, args.delimiter)
            records = enumerate(reader, 1)
            if args.limit:
                records = islice(records, args.limit)
            # Bounded batches avoid queuing the entire ChEMBL file in memory.
            # executor.map preserves input order within each batch.
            while True:
                batch = list(islice(records, args.batch_size))
                if not batch:
                    break
                results = pool.map(process, batch, chunksize=8) if pool else map(process, batch)
                for ok, result in results:
                    counts['processed'] += 1
                    counts['success' if ok else 'failed'] += 1
                    if ok:
                        counts['boundary_exact'] += int(result['bpe_boundary_exact'])
                        # Training corpus line n matches success JSON record n,
                        # not necessarily original input row n (failures omitted).
                        corpus.write(result['safe'] + '\n')
                    else:
                        errors[result['error_type']] += 1
                    (fout if ok else ferr).write(json.dumps(result, ensure_ascii=False) + '\n')
                    progress.update(1)
                fout.flush()
                corpus.flush()
                ferr.flush()
                progress.set_postfix(ok=counts['success'], failed=counts['failed'])
    finally:
        if pool:
            pool.shutdown(wait=True)
    summary = {
        'status': 'complete', 'counts': dict(counts), 'errors': dict(errors),
        'delimiter': 'tab' if separator == '\t' else 'comma',
        'limit': args.limit, 'elapsed_seconds': round(time.monotonic() - started, 3),
        'input': str(args.input.resolve()),
        'input_size_bytes': args.input.stat().st_size,
        'allow_stereo_coarsening': args.allow_stereo_coarsening,
        'versions': {p: importlib.metadata.version(p) for p in ['safe-mol', 'rdkit', 'datamol']},
        'vocab_sha256': {s: hashlib.sha256(Path(str(args.vocab_prefix) + s).read_bytes()).hexdigest()
                         for s in ['.node', '.edge', '.ring']},

    }
    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary['counts'], ensure_ascii=False))


if __name__ == '__main__':
    main()
