"""
Disk storage and serialization routines for nanoreactor candidate reactions.

Handles persistence, reading, extxyz formatting, and cryptographic manifest
integrity verification for frozen candidate sets and their molecular species.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import ase.io
import numpy as np
from ase import Atoms

from nrnet.engines import EngineSettings
from nrnet.nanoreactor.candidates import (
    Candidate,
    CandidateSet,
    EndpointPair,
    Species,
    extract_candidate_set,
)

__all__ = [
    "freeze",
    "load_candidates",
    "write_candidate_files",
]

CANDIDATE_FILES: tuple[str, ...] = (
    "candidates.json",
    "pairs.extxyz",
    "species.extxyz",
)


def pair_frame_metadata(
    reaction_id: str,
    pair_idx: int,
    pair: EndpointPair,
) -> dict[str, Any]:
    """
    Format metadata dictionary for endpoint frame serialization.

    :param reaction_id: Identifier of the parent candidate reaction.
    :type reaction_id: str
    :param pair_idx: Index of pair within candidate.
    :type pair_idx: int
    :param pair: Relaxed endpoint pair.
    :type pair: EndpointPair
    :return: Frame metadata dictionary for extxyz output.
    :rtype: dict[str, Any]
    """
    return {
        "reaction_id": reaction_id,
        "pair_index": pair_idx,
        "replicate": pair.replicate,
        "time_fs": pair.time_fs,
        "atom_ids": np.array(pair.atom_ids),
    }


def pair_to_frames(
    reaction_id: str,
    pair_idx: int,
    pair: EndpointPair,
) -> tuple[Atoms, Atoms]:
    """
    Format endpoint pair before and after frames with extxyz metadata.

    :param reaction_id: Identifier of the parent candidate reaction.
    :type reaction_id: str
    :param pair_idx: Index of pair within candidate.
    :type pair_idx: int
    :param pair: Relaxed endpoint pair.
    :type pair: EndpointPair
    :return: Pair of before and after Atoms structures with info dictionaries.
    :rtype: tuple[Atoms, Atoms]
    """
    meta = pair_frame_metadata(reaction_id, pair_idx, pair)
    f_b, f_a = pair.before.copy(), pair.after.copy()
    f_b.info.update({**meta, "side": "before"})
    f_a.info.update({**meta, "side": "after"})
    return f_b, f_a


def candidate_pair_frames(
    candidate: Candidate,
    offset: int,
) -> tuple[list[Atoms], list[list[int]]]:
    """
    Convert candidate pairs to frames with tracked frame indices.

    :param candidate: Candidate reaction instance.
    :type candidate: Candidate
    :param offset: Current start index in the cumulative frame collection.
    :type offset: int
    :return: List of Atoms frames and list of index pairs.
    :rtype: tuple[list[Atoms], list[list[int]]]
    """
    frames: list[Atoms] = []
    indices: list[list[int]] = []
    for pair_idx, pair in enumerate(candidate.pairs):
        f_b, f_a = pair_to_frames(candidate.reaction_id, pair_idx, pair)
        indices.append([offset + len(frames), offset + len(frames) + 1])
        frames.extend([f_b, f_a])
    return frames, indices


def write_pairs_extxyz(
    candidates: Sequence[Candidate],
    path: Path,
) -> list[list[list[int]]]:
    """
    Write all candidate endpoint pairs to an extxyz file and track indices.

    :param candidates: Sequence of Candidate reactions.
    :type candidates: Sequence[Candidate]
    :param path: Destination file path for pairs.extxyz.
    :type path: Path
    :return: Nested frame indices per candidate and pair.
    :rtype: list[list[list[int]]]
    """
    all_frames: list[Atoms] = []
    indices: list[list[list[int]]] = []
    for cand in candidates:
        frames, cand_indices = candidate_pair_frames(cand, len(all_frames))
        all_frames.extend(frames)
        indices.append(cand_indices)
    ase.io.write(path, all_frames, format="extxyz")
    return indices


def serialize_candidate(
    candidate: Candidate,
    pair_indices: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """
    Serialize a Candidate reaction into a JSON-compatible dictionary.

    :param candidate: Candidate reaction instance.
    :type candidate: Candidate
    :param pair_indices: Frame indices in pairs.extxyz for this candidate.
    :type pair_indices: Sequence[Sequence[int]]
    :return: Serialized candidate mapping.
    :rtype: dict[str, Any]
    """
    return {
        "reaction_id": candidate.reaction_id,
        "state_a": list(candidate.state_a),
        "state_b": list(candidate.state_b),
        "reactant_keys": list(candidate.reactant_keys),
        "product_keys": list(candidate.product_keys),
        "mapped_smiles": candidate.mapped_smiles,
        "n_events": candidate.n_events,
        "first_seen_fs": candidate.first_seen_fs,
        "pair_frame_indices": [list(pair) for pair in pair_indices],
    }


def serialized_candidates(
    candidates: Sequence[Candidate],
    frame_indices: Sequence[Sequence[Sequence[int]]],
) -> list[dict[str, Any]]:
    """
    Serialize candidate reactions with their extxyz frame indices.

    :param candidates: Sequence of candidate reactions.
    :type candidates: Sequence[Candidate]
    :param frame_indices: Nested frame indices for each candidate pair.
    :type frame_indices: Sequence[Sequence[Sequence[int]]]
    :return: List of serialized candidate dictionaries.
    :rtype: list[dict[str, Any]]
    """
    return [
        serialize_candidate(cand, indices)
        for cand, indices in zip(candidates, frame_indices, strict=True)
    ]


def write_candidates_json(
    candidate_set: CandidateSet,
    frame_indices: Sequence[Sequence[Sequence[int]]],
    path: Path,
) -> None:
    """
    Write candidates.json containing reactant state and candidate metadata.

    :param candidate_set: Complete candidate set instance.
    :type candidate_set: CandidateSet
    :param frame_indices: Nested frame indices for each candidate pair.
    :type frame_indices: Sequence[Sequence[Sequence[int]]]
    :param path: Destination file path for candidates.json.
    :type path: Path
    :return: None.
    :rtype: None
    """
    data = {
        "reactant_state": list(candidate_set.reactant_state),
        "stats": candidate_set.stats,
        "candidates": serialized_candidates(
            candidate_set.candidates, frame_indices
        ),
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def write_species_extxyz(
    species_map: Mapping[str, Species],
    path: Path,
) -> None:
    """
    Write all relaxed species structures and identifiers to species.extxyz.

    :param species_map: Dictionary mapping species keys to Species dataclasses.
    :type species_map: Mapping[str, Species]
    :param path: Destination file path for species.extxyz.
    :type path: Path
    :return: None.
    :rtype: None
    """
    frames: list[Atoms] = []
    for spec in species_map.values():
        frame = spec.atoms.copy()
        frame.info["key"] = spec.key
        frame.info["smiles"] = spec.smiles if spec.smiles else "None"
        frame.info["inchikey"] = spec.inchikey if spec.inchikey else "None"
        frame.info["multiplicity"] = spec.multiplicity
        frames.append(frame)
    ase.io.write(path, frames, format="extxyz")


def write_manifest_json(
    out_dir: Path,
    filenames: Sequence[str],
) -> None:
    """
    Write manifest.json containing sha256 checksums of candidate files.

    :param out_dir: Target candidate set directory.
    :type out_dir: Path
    :param filenames: Sequence of relative file names to hash.
    :type filenames: Sequence[str]
    :return: None.
    :rtype: None
    """
    manifest: dict[str, str] = {}
    for name in sorted(filenames):
        content = (out_dir / name).read_bytes()
        manifest[name] = hashlib.sha256(content).hexdigest()
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)


def write_candidate_files(
    candidate_set: CandidateSet,
    out_dir: Path,
) -> None:
    """
    Serialize all candidate set components to disk in the output directory.

    :param candidate_set: Frozen CandidateSet to persist.
    :type candidate_set: CandidateSet
    :param out_dir: Output directory path.
    :type out_dir: Path
    :return: None.
    :rtype: None
    """
    write_species_extxyz(candidate_set.species, out_dir / "species.extxyz")
    frame_indices = write_pairs_extxyz(
        candidate_set.candidates, out_dir / "pairs.extxyz"
    )
    write_candidates_json(
        candidate_set, frame_indices, out_dir / "candidates.json"
    )
    write_manifest_json(out_dir, CANDIDATE_FILES)


def verify_manifest(out_dir: Path) -> dict[str, str]:
    """
    Verify sha256 checksums of candidate files against manifest.json.

    :param out_dir: Candidate directory path.
    :type out_dir: Path
    :return: Parsed manifest dictionary mapping file names to hashes.
    :rtype: dict[str, str]
    :raises ValueError: If manifest is missing or a file hash does not match.
    """
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Missing manifest.json in candidate directory.")
    with manifest_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    for name, expected_hash in data.items():
        target = out_dir / name
        if not target.is_file():
            raise ValueError(f"Missing file named in manifest: {name}")
        actual_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Hash mismatch for file: {name}")
    return data


def species_from_frame(frame: Atoms) -> Species:
    """
    Construct a Species dataclass from an extxyz frame.

    :param frame: Atoms frame loaded from species.extxyz.
    :type frame: Atoms
    :return: Reconstructed Species dataclass instance.
    :rtype: Species
    """
    raw_s = frame.info.get("smiles")
    smiles = None if raw_s in (None, "", "None") else str(raw_s)
    raw_i = frame.info.get("inchikey")
    inchikey = None if raw_i in (None, "", "None") else str(raw_i)
    return Species(
        key=str(frame.info["key"]),
        atoms=frame.copy(),
        smiles=smiles,
        inchikey=inchikey,
        multiplicity=int(frame.info["multiplicity"]),
    )


def load_species_extxyz(path: Path) -> dict[str, Species]:
    """
    Load species dictionary from species.extxyz.

    :param path: Path to species.extxyz file.
    :type path: Path
    :return: Dictionary mapping species keys to Species instances.
    :rtype: dict[str, Species]
    """
    frames = ase.io.read(path, index=":", format="extxyz")
    return {spec.key: spec for spec in (species_from_frame(f) for f in frames)}


def pair_from_frames(f_b: Atoms, f_a: Atoms) -> EndpointPair:
    """
    Reconstruct an EndpointPair from before and after extxyz Atoms frames.

    :param f_b: Before endpoint frame.
    :type f_b: Atoms
    :param f_a: After endpoint frame.
    :type f_a: Atoms
    :return: Reconstructed EndpointPair instance.
    :rtype: EndpointPair
    """
    raw_ids = f_b.info["atom_ids"]
    atom_ids = tuple(int(x) for x in raw_ids)
    return EndpointPair(
        replicate=int(f_b.info["replicate"]),
        time_fs=float(f_b.info["time_fs"]),
        atom_ids=atom_ids,
        before=f_b.copy(),
        after=f_a.copy(),
    )


def parse_pair_frames(
    indices: Sequence[Sequence[int]],
    pair_frames: Sequence[Atoms],
) -> tuple[EndpointPair, ...]:
    """
    Reconstruct endpoint pairs from indexed frame pairs.

    :param indices: List of [before_index, after_index] pairs.
    :type indices: Sequence[Sequence[int]]
    :param pair_frames: Sequence of all Atoms frames from pairs.extxyz.
    :type pair_frames: Sequence[Atoms]
    :return: Reconstructed EndpointPair tuple.
    :rtype: tuple[EndpointPair, ...]
    """
    return tuple(
        pair_from_frames(pair_frames[b], pair_frames[a]) for b, a in indices
    )


def candidate_from_record(
    data: Mapping[str, Any],
    pairs: tuple[EndpointPair, ...],
) -> Candidate:
    """
    Instantiate Candidate dataclass from JSON mapping and parsed pairs.

    :param data: Candidate JSON dictionary.
    :type data: Mapping[str, Any]
    :param pairs: Parsed endpoint pairs.
    :type pairs: tuple[EndpointPair, ...]
    :return: Instantiated Candidate dataclass instance.
    :rtype: Candidate
    """
    return Candidate(
        reaction_id=str(data["reaction_id"]),
        state_a=tuple(data["state_a"]),
        state_b=tuple(data["state_b"]),
        reactant_keys=tuple(data["reactant_keys"]),
        product_keys=tuple(data["product_keys"]),
        pairs=pairs,
        mapped_smiles=str(data["mapped_smiles"]),
        n_events=int(data["n_events"]),
        first_seen_fs=float(data["first_seen_fs"]),
    )


def build_candidate_from_json(
    data: Mapping[str, Any],
    pair_frames: Sequence[Atoms],
) -> Candidate:
    """
    Reconstruct a Candidate instance from JSON data and pair frames.

    :param data: Candidate JSON dictionary.
    :type data: Mapping[str, Any]
    :param pair_frames: Sequence of all Atoms frames from pairs.extxyz.
    :type pair_frames: Sequence[Atoms]
    :return: Reconstructed Candidate dataclass instance.
    :rtype: Candidate
    """
    pairs = parse_pair_frames(data["pair_frame_indices"], pair_frames)
    return candidate_from_record(data, pairs)


def read_candidates_json(path: Path) -> dict[str, Any]:
    """
    Read and parse candidates.json file.

    :param path: Path to candidates.json.
    :type path: Path
    :return: Parsed JSON content dictionary.
    :rtype: dict[str, Any]
    """
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_candidate_reactions(
    target_dir: Path,
) -> tuple[tuple[Candidate, ...], tuple[str, ...], dict[str, int]]:
    """
    Load candidate reactions, reactant state, and stats from directory files.

    :param target_dir: Candidate directory path.
    :type target_dir: Path
    :return: Tuple of candidate reactions, reactant state keys, and stats.
    :rtype: tuple[tuple[Candidate, ...], tuple[str, ...], dict[str, int]]
    """
    pairs_frames = ase.io.read(
        target_dir / "pairs.extxyz", index=":", format="extxyz"
    )
    cand_data = read_candidates_json(target_dir / "candidates.json")
    candidates = tuple(
        build_candidate_from_json(item, pairs_frames)
        for item in cand_data["candidates"]
    )
    raw_stats = cand_data.get("stats", {})
    stats = {str(k): int(v) for k, v in raw_stats.items()}
    return candidates, tuple(cand_data["reactant_state"]), stats


def load_candidates(out_dir: str | Path) -> CandidateSet:
    """
    Load frozen candidate set from directory and verify file integrity.

    :param out_dir: Candidate directory path.
    :type out_dir: str or Path
    :return: Reconstructed frozen CandidateSet instance.
    :rtype: CandidateSet
    :raises ValueError: If manifest integrity verification fails.
    """
    target_dir = Path(out_dir)
    verify_manifest(target_dir)
    species_map = load_species_extxyz(target_dir / "species.extxyz")
    candidates, reactant_state, stats = load_candidate_reactions(target_dir)
    return CandidateSet(
        reactant_state=reactant_state,
        species=species_map,
        candidates=candidates,
        stats=stats,
    )


def freeze(
    reactant: Atoms,
    trajectories: list[list[Atoms]],
    min_frames: int,
    dump_fs: float,
    engine_settings: EngineSettings | str | Path,
    out_dir: str | Path,
) -> CandidateSet:
    """
    Freeze nanoreactor trajectories into a validated candidate set on disk.

    :param reactant: Initial unreacted system configuration.
    :type reactant: Atoms
    :param trajectories: List of replicate trajectory frame lists.
    :type trajectories: list[list[Atoms]]
    :param min_frames: Minimum frame duration for stable states.
    :type min_frames: int
    :param dump_fs: Trajectory frame dump interval in femtoseconds.
    :type dump_fs: float
    :param engine_settings: Settings, or the engines.yaml path, for xTB.
    :type engine_settings: EngineSettings or str or Path
    :param out_dir: Destination directory for the frozen files.
    :type out_dir: str or Path
    :return: The candidate set that was written.
    :rtype: CandidateSet
    """
    candidate_set = extract_candidate_set(
        reactant, trajectories, min_frames, dump_fs, engine_settings
    )
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    write_candidate_files(candidate_set, out_path)
    return candidate_set
