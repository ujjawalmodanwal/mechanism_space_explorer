"""
Literature reaction product specifications and candidate matching.

Parses literature reference definitions and evaluates whether nanoreactor
candidate reactions generated specific expected multi-component products.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import ase.io
import yaml
from ase import Atoms
from rdkit import Chem

from nrnet.chem import state_key
from nrnet.engines import EngineSettings, attach
from nrnet.nanoreactor.candidates import (
    FREEZE_FMAX_EV_PER_ANG,
    FREEZE_MAX_STEPS,
    Candidate,
    CandidateSet,
    EndpointPair,
    Species,
    find_fragment_in_atoms,
    pair_mapped_smiles,
    reacting_atom_ids,
    relax_species,
    resolve_engine_settings,
)
from nrnet.optimize import relax

__all__ = [
    "LiteratureProduct",
    "LiteratureReaction",
    "add_literature_to_candidates",
    "append_literature_candidate",
    "apply_literature_reaction",
    "build_literature_pair",
    "build_literature_reaction",
    "candidate_has_product",
    "candidate_matches_states",
    "candidate_set_holds_product",
    "candidate_state_holds_product",
    "ensure_candidate_species",
    "find_matching_candidate_index",
    "label_matched_candidate",
    "load_literature_products",
    "load_literature_reactions",
    "make_literature_candidate",
    "next_reaction_id",
    "parse_literature_products",
    "parse_literature_reactions",
    "process_literature_reaction",
    "record_literature_candidate",
    "relax_endpoint",
    "smiles_to_inchikey_prefix",
    "state_inchikey_prefixes",
    "update_matched_candidate",
    "validate_literature_states",
    "validate_reaction_symbols",
]


@dataclass(frozen=True, slots=True)
class LiteratureProduct:
    """
    Literature reference reaction product description.

    :param name: Product identifier name.
    :type name: str
    :param smiles: Tuple of canonical product component SMILES.
    :type smiles: tuple[str, ...]
    """

    name: str
    smiles: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LiteratureReaction:
    """
    Literature reference reaction specification with atomic endpoints.

    :param name: Unique name of the literature reaction.
    :type name: str
    :param reactant: Atomic structure of the initial reactant.
    :type reactant: Atoms
    :param product: Atomic structure of the product.
    :type product: Atoms
    :param note: Descriptive literature notes or commentary.
    :type note: str
    """

    name: str
    reactant: Atoms = field(compare=False)
    product: Atoms = field(compare=False)
    note: str = ""


def validate_reaction_symbols(
    reactant: Atoms,
    product: Atoms,
    name: str,
) -> None:
    """
    Validate that reactant and product have identical chemical symbols.

    :param reactant: Reactant atomic configuration.
    :type reactant: Atoms
    :param product: Product atomic configuration.
    :type product: Atoms
    :param name: Reaction identifier name.
    :type name: str
    :return: None.
    :rtype: None
    :raises ValueError: If atom symbols do not match identically in order.
    """
    r_syms = reactant.get_chemical_symbols()
    p_syms = product.get_chemical_symbols()
    if r_syms != p_syms:
        raise ValueError(
            f"Reaction '{name}' chemical symbols mismatch: {r_syms} != {p_syms}"
        )


def build_literature_reaction(
    item: Mapping[str, Any],
    system_dir: Path,
) -> LiteratureReaction:
    """
    Construct and validate a LiteratureReaction from YAML entry.

    :param item: Dictionary defining a literature reaction entry.
    :type item: Mapping[str, Any]
    :param system_dir: Directory containing reactant and product xyz files.
    :type system_dir: Path
    :return: Validated literature reaction instance.
    :rtype: LiteratureReaction
    """
    name = str(item["name"])
    reactant = ase.io.read(system_dir / item["reactant"], format="xyz")
    product = ase.io.read(system_dir / item["product"], format="xyz")
    validate_reaction_symbols(reactant, product, name)
    note = str(item.get("note", ""))
    return LiteratureReaction(
        name=name,
        reactant=reactant,
        product=product,
        note=note,
    )


def parse_literature_reactions(
    data: Mapping[str, Any],
    system_dir: Path,
) -> list[LiteratureReaction]:
    """
    Parse LiteratureReaction instances from raw loaded YAML mapping.

    :param data: Parsed YAML dictionary holding reactions list.
    :type data: Mapping[str, Any]
    :param system_dir: Directory containing system xyz coordinate files.
    :type system_dir: Path
    :return: List of validated literature reactions.
    :rtype: list[LiteratureReaction]
    """
    raw_reactions = data.get("reactions", []) if isinstance(data, dict) else []
    reactions: list[LiteratureReaction] = []
    for item in raw_reactions:
        reactions.append(build_literature_reaction(item, system_dir))
    return reactions


def load_literature_reactions(
    path: str | Path,
) -> list[LiteratureReaction]:
    """
    Load literature reference reactions from a YAML file or directory.

    :param path: Path to literature.yaml or the enclosing system directory.
    :type path: str or Path
    :return: List of validated literature reactions.
    :rtype: list[LiteratureReaction]
    """
    p = Path(path)
    yaml_path = p if p.is_file() else p / "literature.yaml"
    if not yaml_path.is_file():
        return []
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return parse_literature_reactions(data, yaml_path.parent)


def relax_endpoint(
    atoms: Atoms,
    settings: EngineSettings,
) -> Atoms:
    """
    Relax atomic geometry with xTB using standard freeze parameters.

    :param atoms: Input atomic structure to optimize.
    :type atoms: Atoms
    :param settings: Engine configuration settings for xTB.
    :type settings: EngineSettings
    :return: Relaxed atomic structure.
    :rtype: Atoms
    """
    attached = attach(atoms.copy(), settings)
    res = relax(
        attached,
        fmax_ev_per_ang=FREEZE_FMAX_EV_PER_ANG,
        max_steps=FREEZE_MAX_STEPS,
    )
    return res.atoms


def validate_literature_states(
    rxn: LiteratureReaction,
    relaxed_r: Atoms,
    relaxed_p: Atoms,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """
    Validate stereo state keys for relaxed literature endpoints.

    :param rxn: Input literature reaction specification.
    :type rxn: LiteratureReaction
    :param relaxed_r: Relaxed reactant atomic geometry.
    :type relaxed_r: Atoms
    :param relaxed_p: Relaxed product atomic geometry.
    :type relaxed_p: Atoms
    :return: Tuple of reactant and product stereo-aware state keys.
    :rtype: tuple[tuple[str, ...], tuple[str, ...]]
    :raises ValueError: If both ends share state or product state changed.
    """
    orig_p_key = state_key(rxn.product, stereo=True)
    post_r_key = state_key(relaxed_r, stereo=True)
    post_p_key = state_key(relaxed_p, stereo=True)
    if post_p_key != orig_p_key:
        raise ValueError(
            f"Relaxing product for '{rxn.name}' changed state from "
            f"{orig_p_key} to {post_p_key}"
        )
    if post_r_key == post_p_key:
        raise ValueError(
            f"Both ends of reaction '{rxn.name}' give identical state: "
            f"{post_r_key}"
        )
    return post_r_key, post_p_key


def build_literature_pair(
    relaxed_r: Atoms,
    relaxed_p: Atoms,
) -> EndpointPair:
    """
    Construct EndpointPair for reacting atoms between relaxed endpoints.

    :param relaxed_r: Relaxed reactant structure.
    :type relaxed_r: Atoms
    :param relaxed_p: Relaxed product structure.
    :type relaxed_p: Atoms
    :return: EndpointPair holding reacting atoms subset.
    :rtype: EndpointPair
    """
    atom_ids = tuple(reacting_atom_ids(relaxed_r, relaxed_p))
    ids = atom_ids if atom_ids else tuple(range(len(relaxed_r)))
    return EndpointPair(
        replicate=0,
        time_fs=0.0,
        atom_ids=ids,
        before=relaxed_r[list(ids)],
        after=relaxed_p[list(ids)],
    )


def candidate_matches_states(
    candidate: Candidate,
    state_r: tuple[str, ...],
    state_p: tuple[str, ...],
) -> bool:
    """
    Check if a candidate has the same reacting molecules, either direction.

    Whole states are not compared: a water that only stands by gives the
    same states as a water that relays a proton (discovery log, F12).

    :param candidate: Candidate reaction to inspect.
    :type candidate: Candidate
    :param state_r: Literature reacting molecules, reactant side.
    :type state_r: tuple[str, ...]
    :param state_p: Literature reacting molecules, product side.
    :type state_p: tuple[str, ...]
    :return: True if candidate has the same reacting molecules.
    :rtype: bool
    """
    target = {state_r, state_p}
    return {candidate.reactant_keys, candidate.product_keys} == target


def update_matched_candidate(candidate: Candidate, name: str) -> Candidate:
    """
    Label a reaction the MD already found with its literature name.

    No hand-built pair is added: the MD pairs are the only geometries.

    :param candidate: Existing matched candidate reaction.
    :type candidate: Candidate
    :param name: Literature reaction name.
    :type name: str
    :return: The candidate with literature_name set.
    :rtype: Candidate
    """
    return replace(candidate, literature_name=name)


def make_literature_candidate(
    reaction_id: str,
    rxn_name: str,
    state_r: tuple[str, ...],
    state_p: tuple[str, ...],
    pair: EndpointPair,
) -> Candidate:
    """
    Create a new candidate reaction flagged with source='literature'.

    :param reaction_id: Formatted reaction identifier (e.g. 'R005').
    :type reaction_id: str
    :param rxn_name: Literature reaction name identifier.
    :type rxn_name: str
    :param state_r: Reactant stereo-aware state key.
    :type state_r: tuple[str, ...]
    :param state_p: Product stereo-aware state key.
    :type state_p: tuple[str, ...]
    :param pair: Validated literature endpoint pair.
    :type pair: EndpointPair
    :return: Newly created Candidate reaction.
    :rtype: Candidate
    """
    return Candidate(
        reaction_id=reaction_id,
        state_a=state_r,
        state_b=state_p,
        reactant_keys=state_key(pair.before, stereo=True),
        product_keys=state_key(pair.after, stereo=True),
        pairs=(pair,),
        mapped_smiles=pair_mapped_smiles(pair),
        n_events=0,
        first_seen_fs=0.0,
        source="literature",
        literature_name=rxn_name,
    )


def next_reaction_id(candidates: Sequence[Candidate]) -> str:
    """
    Determine next candidate reaction identifier in sequence.

    :param candidates: Sequence of existing candidate reactions.
    :type candidates: Sequence[Candidate]
    :return: Next formatted reaction identifier (e.g. 'R001').
    :rtype: str
    """
    max_idx = 0
    for cand in candidates:
        if cand.reaction_id.startswith("R") and cand.reaction_id[1:].isdigit():
            max_idx = max(max_idx, int(cand.reaction_id[1:]))
    return f"R{max_idx + 1:03d}"


def ensure_candidate_species(
    candidate: Candidate,
    species_map: dict[str, Species],
    settings: EngineSettings,
) -> None:
    """
    Ensure all species for candidate endpoints exist in species map.

    :param candidate: Candidate reaction to inspect.
    :type candidate: Candidate
    :param species_map: Mutable species map to populate.
    :type species_map: dict[str, Species]
    :param settings: Engine configuration settings.
    :type settings: EngineSettings
    :return: None.
    :rtype: None
    """
    pair = candidate.pairs[0]
    keys = set(candidate.reactant_keys) | set(candidate.product_keys)
    for key in keys:
        if key not in species_map:
            frag = find_fragment_in_atoms(pair.before, key)
            if frag is None:
                frag = find_fragment_in_atoms(pair.after, key)
            if frag is not None:
                spec = relax_species(frag, key, settings)
                if spec is not None:
                    species_map[key] = spec


def process_literature_reaction(
    rxn: LiteratureReaction,
    settings: EngineSettings,
) -> tuple[EndpointPair, tuple[str, ...], tuple[str, ...]]:
    """
    Relax endpoints, validate states, and build literature pair.

    :param rxn: Input literature reaction specification.
    :type rxn: LiteratureReaction
    :param settings: Engine configuration settings.
    :type settings: EngineSettings
    :return: Triplet of pair, reactant state key, and product state key.
    :rtype: tuple[EndpointPair, tuple[str, ...], tuple[str, ...]]
    """
    res_r = relax_endpoint(rxn.reactant, settings)
    res_p = relax_endpoint(rxn.product, settings)
    st_r, st_p = validate_literature_states(rxn, res_r, res_p)
    return build_literature_pair(res_r, res_p), st_r, st_p


def find_matching_candidate_index(
    candidates: Sequence[Candidate],
    state_r: tuple[str, ...],
    state_p: tuple[str, ...],
) -> int | None:
    """
    Find index of first candidate with the same reacting molecules.

    :param candidates: Sequence of candidate reactions.
    :type candidates: Sequence[Candidate]
    :param state_r: Reacting molecules, reactant side.
    :type state_r: tuple[str, ...]
    :param state_p: Reacting molecules, product side.
    :type state_p: tuple[str, ...]
    :return: Index of matching candidate or None.
    :rtype: int or None
    """
    for idx, cand in enumerate(candidates):
        if candidate_matches_states(cand, state_r, state_p):
            return idx
    return None


def label_matched_candidate(
    idx: int,
    name: str,
    candidates: list[Candidate],
) -> Candidate:
    """
    Label the MD candidate that matches a literature reaction.

    :param idx: Index of matching candidate in candidate list.
    :type idx: int
    :param name: Literature reaction identifier name.
    :type name: str
    :param candidates: Mutable list of Candidate reactions.
    :type candidates: list[Candidate]
    :return: Labelled Candidate reaction.
    :rtype: Candidate
    """
    cand = update_matched_candidate(candidates[idx], name)
    candidates[idx] = cand
    print(
        f"Literature reaction '{name}' was found by the MD as "
        f"{cand.reaction_id}; labelled, nothing added"
    )
    return cand


def append_literature_candidate(
    name: str,
    pair: EndpointPair,
    st_r: tuple[str, ...],
    st_p: tuple[str, ...],
    candidates: list[Candidate],
) -> Candidate:
    """
    Construct and append new candidate with source='literature'.

    :param name: Literature reaction identifier name.
    :type name: str
    :param pair: Relaxed literature endpoint pair.
    :type pair: EndpointPair
    :param st_r: Reactant state key.
    :type st_r: tuple[str, ...]
    :param st_p: Product state key.
    :type st_p: tuple[str, ...]
    :param candidates: Mutable list of Candidate reactions.
    :type candidates: list[Candidate]
    :return: Appended Candidate reaction.
    :rtype: Candidate
    """
    cand = make_literature_candidate(
        next_reaction_id(candidates), name, st_r, st_p, pair
    )
    candidates.append(cand)
    print(
        f"Added literature reaction '{name}' as candidate "
        f"{cand.reaction_id} (source=literature)"
    )
    return cand


def record_literature_candidate(
    name: str,
    pair: EndpointPair,
    st_r: tuple[str, ...],
    st_p: tuple[str, ...],
    candidates: list[Candidate],
) -> Candidate:
    """
    Update or append candidate for a literature reaction.

    :param name: Literature reaction identifier name.
    :type name: str
    :param pair: Relaxed endpoint pair.
    :type pair: EndpointPair
    :param st_r: Reactant state key.
    :type st_r: tuple[str, ...]
    :param st_p: Product state key.
    :type st_p: tuple[str, ...]
    :param candidates: Mutable list of Candidate reactions.
    :type candidates: list[Candidate]
    :return: Added or updated Candidate reaction.
    :rtype: Candidate
    """
    idx = find_matching_candidate_index(
        candidates,
        state_key(pair.before, stereo=True),
        state_key(pair.after, stereo=True),
    )
    if idx is not None:
        return label_matched_candidate(idx, name, candidates)
    return append_literature_candidate(name, pair, st_r, st_p, candidates)


def apply_literature_reaction(
    rxn: LiteratureReaction,
    candidates: list[Candidate],
    species_map: dict[str, Species],
    settings: EngineSettings,
) -> None:
    """
    Incorporate a literature reaction into candidate list and species map.

    :param rxn: Literature reaction to incorporate.
    :type rxn: LiteratureReaction
    :param candidates: Mutable list of Candidate reactions.
    :type candidates: list[Candidate]
    :param species_map: Mutable dictionary of species.
    :type species_map: dict[str, Species]
    :param settings: Engine configuration settings.
    :type settings: EngineSettings
    :return: None.
    :rtype: None
    """
    pair, st_r, st_p = process_literature_reaction(rxn, settings)
    cand = record_literature_candidate(rxn.name, pair, st_r, st_p, candidates)
    ensure_candidate_species(cand, species_map, settings)


def add_literature_to_candidates(
    candidate_set: CandidateSet,
    reactions: Sequence[LiteratureReaction],
    engine_settings: EngineSettings | str | Path,
) -> CandidateSet:
    """
    Incorporate literature reactions into a CandidateSet.

    :param candidate_set: Frozen CandidateSet instance.
    :type candidate_set: CandidateSet
    :param reactions: Sequence of literature reactions to add.
    :type reactions: Sequence[LiteratureReaction]
    :param engine_settings: Configuration settings for relaxation engine.
    :type engine_settings: EngineSettings or str or Path
    :return: New CandidateSet including literature reactions and species.
    :rtype: CandidateSet
    """
    settings = resolve_engine_settings(engine_settings)
    cands = list(candidate_set.candidates)
    species_map = dict(candidate_set.species)
    for rxn in reactions:
        apply_literature_reaction(rxn, cands, species_map, settings)
    stats = dict(candidate_set.stats)
    stats["candidates"] = len(cands)
    stats["species"] = len(species_map)
    return CandidateSet(
        reactant_state=candidate_set.reactant_state,
        species=species_map,
        candidates=tuple(cands),
        stats=stats,
    )


def smiles_to_inchikey_prefix(smiles: str) -> str | None:
    """
    Compute 14-character InChIKey skeleton prefix for a SMILES string.

    :param smiles: Molecular SMILES representation.
    :type smiles: str
    :return: 14-character skeleton prefix or None if parsing fails.
    :rtype: str or None
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    key = Chem.MolToInchiKey(mol)
    return key[:14] if key else None


def parse_literature_products(
    data: Mapping[str, Any],
) -> list[LiteratureProduct]:
    """
    Parse LiteratureProduct instances from raw loaded YAML mapping.

    :param data: Parsed YAML dictionary holding products list.
    :type data: Mapping[str, Any]
    :return: List of validated literature products.
    :rtype: list[LiteratureProduct]
    """
    raw_products = data.get("products", []) if isinstance(data, dict) else []
    products: list[LiteratureProduct] = []
    for item in raw_products:
        smiles = tuple(str(s) for s in item.get("smiles", []))
        products.append(
            LiteratureProduct(name=str(item["name"]), smiles=smiles)
        )
    return products


def load_literature_products(
    path: str | Path,
) -> list[LiteratureProduct]:
    """
    Load literature reference reaction products from a YAML file.

    :param path: Filesystem path to literature.yaml.
    :type path: str or Path
    :return: List of literature products with names and SMILES components.
    :rtype: list[LiteratureProduct]
    """
    yaml_path = Path(path)
    if not yaml_path.is_file():
        return []
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return parse_literature_products(data)


def state_inchikey_prefixes(
    species_map: Mapping[str, Species],
    state_keys: Sequence[str],
) -> list[str]:
    """
    Extract 14-character InChIKey prefixes for species keys in a state.

    :param species_map: Mapping of species keys to Species dataclasses.
    :type species_map: Mapping[str, Species]
    :param state_keys: Sequence of species keys present in a chemical state.
    :type state_keys: Sequence[str]
    :return: List of 14-character InChIKey prefixes.
    :rtype: list[str]
    """
    prefixes: list[str] = []
    for key in state_keys:
        species = species_map.get(key)
        if species is not None and species.inchikey:
            prefixes.append(species.inchikey[:14])
        elif species is not None and species.smiles:
            pfx = smiles_to_inchikey_prefix(species.smiles)
            if pfx:
                prefixes.append(pfx)
    return prefixes


def candidate_state_holds_product(
    state_prefixes: Sequence[str],
    product: LiteratureProduct,
) -> bool:
    """
    Determine if a candidate state contains all components of a product.

    :param state_prefixes: 14-char InChIKey prefixes of species in state.
    :type state_prefixes: Sequence[str]
    :param product: Target literature product with expected SMILES.
    :type product: LiteratureProduct
    :return: True if state holds every required product component.
    :rtype: bool
    """
    if not product.smiles:
        return False
    req_prefixes = [smiles_to_inchikey_prefix(s) for s in product.smiles]
    if any(pfx is None for pfx in req_prefixes):
        return False
    state_counts = Counter(state_prefixes)
    req_counts = Counter(req_prefixes)
    return all(state_counts[pfx] >= req_counts[pfx] for pfx in req_counts)


def candidate_has_product(
    candidate: Candidate,
    species_map: Mapping[str, Species],
    product: LiteratureProduct,
) -> bool:
    """
    Check if any chemical state of a candidate reaction holds the product.

    :param candidate: Candidate reaction to inspect.
    :type candidate: Candidate
    :param species_map: Mapping of species keys to Species objects.
    :type species_map: Mapping[str, Species]
    :param product: Literature product to search for.
    :type product: LiteratureProduct
    :return: True if any candidate endpoint state holds every product SMILES.
    :rtype: bool
    """
    states = (
        candidate.state_a,
        candidate.state_b,
        candidate.reactant_keys,
        candidate.product_keys,
    )
    for state in states:
        prefixes = state_inchikey_prefixes(species_map, state)
        if candidate_state_holds_product(prefixes, product):
            return True
    return False


def candidate_set_holds_product(
    candidate_set: CandidateSet,
    product: LiteratureProduct,
) -> bool:
    """
    Determine whether a candidate set contains a literature product.

    :param candidate_set: Frozen CandidateSet instance.
    :type candidate_set: CandidateSet
    :param product: Literature product to match.
    :type product: LiteratureProduct
    :return: True if any candidate in the set holds the product.
    :rtype: bool
    """
    return any(
        candidate_has_product(cand, candidate_set.species, product)
        for cand in candidate_set.candidates
    )
