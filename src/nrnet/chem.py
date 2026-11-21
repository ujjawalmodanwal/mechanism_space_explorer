"""
Chemical graph construction, hashing, bond perception, and reaction mapping.

Provides pure functions operating on ASE Atoms structures and NetworkX graphs
to support state representation and reaction extraction in nanoreactor networks.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx
from ase import Atoms
from ase.formula import Formula
from ase.neighborlist import NeighborList, natural_cutoffs
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

__all__ = [
    "DOUBLET",
    "HYDROGEN_RADIUS_ANG",
    "SINGLET",
    "SpeciesLabels",
    "connectivity_graph",
    "fragments",
    "mapped_reaction_smiles",
    "multiplicity",
    "radical_count",
    "rdkit_mol",
    "reacting_atoms",
    "single_bond_stereo_smiles",
    "species_key",
    "species_labels",
    "state_key",
    "stereo_descriptor",
]

SINGLET = 1
DOUBLET = 2
HYDROGEN_RADIUS_ANG = 0.37


@dataclass(frozen=True, slots=True)
class SpeciesLabels:
    """
    Chemical identifiers for a molecular species.

    :param smiles: Canonical SMILES string without hydrogens, or None.
    :type smiles: str or None
    :param inchikey: Standard InChIKey string, or None.
    :type inchikey: str or None
    """

    smiles: str | None
    inchikey: str | None


def connectivity_graph(atoms: Atoms, scale: float = 1.2) -> nx.Graph:
    """
    Build atomic connectivity graph from natural covalent cutoffs.

    :param atoms: Atomic configuration.
    :type atoms: Atoms
    :param scale: Multiplicative scale factor for covalent cutoffs.
    :type scale: float
    :return: Connectivity graph with element attributes on each node.
    :rtype: nx.Graph
    """
    # H radius 0.37 A (half of H2's bond): ASE's 0.31 A gives an H-H cutoff
    # of 0.744 A, shorter than H2 itself (0.777 A in GFN2-xTB), so H2 was
    # read as two H atoms.
    # ASE uses an override as given, without mult, so scale it here.
    h_cutoff = HYDROGEN_RADIUS_ANG * scale
    cutoffs = natural_cutoffs(atoms, mult=scale, H=h_cutoff)
    # skin=0: ASE's default skin widens every cutoff by 0.6 A in total.
    neighbor_list = NeighborList(
        cutoffs,
        skin=0.0,
        self_interaction=False,
        bothways=True,
    )
    neighbor_list.update(atoms)
    graph = nx.Graph()
    for idx, symbol in enumerate(atoms.get_chemical_symbols()):
        graph.add_node(idx, element=symbol)
    for idx in range(len(atoms)):
        indices, _ = neighbor_list.get_neighbors(idx)
        for neighbor in indices:
            graph.add_edge(idx, int(neighbor))
    return graph


def fragments(graph: nx.Graph) -> list[list[int]]:
    """
    Extract sorted connected components sorted by their smallest atom index.

    :param graph: Molecular connectivity graph.
    :type graph: nx.Graph
    :return: List of sorted component atom index lists.
    :rtype: list[list[int]]
    """
    components = [sorted(comp) for comp in nx.connected_components(graph)]
    components.sort(key=lambda comp: comp[0])
    return components


def atoms_to_xyz(
    atoms: Atoms,
    atom_ids: Sequence[int] | None = None,
) -> str:
    """
    Format selected atoms from an Atoms object as an XYZ coordinate block.

    :param atoms: Atomic structure.
    :type atoms: Atoms
    :param atom_ids: Specific atom indices to include, or None for all.
    :type atom_ids: Sequence[int] or None
    :return: XYZ format string.
    :rtype: str
    """
    indices = range(len(atoms)) if atom_ids is None else atom_ids
    lines = [str(len(indices)), ""]
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    for idx in indices:
        pos = positions[idx]
        lines.append(f"{symbols[idx]} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}")
    return "\n".join(lines)


def rdkit_mol(
    atoms: Atoms,
    atom_ids: Sequence[int] | None = None,
) -> Chem.RWMol:
    """
    Build an RDKit molecule whose bonds are this module's connectivity.

    RDKit's own bond perception uses different distance rules; building
    from connectivity_graph keeps labels consistent with species keys.

    :param atoms: Atomic structure.
    :type atoms: Atoms
    :param atom_ids: Atom indices to include, or None for all atoms.
    :type atom_ids: Sequence[int] or None
    :return: Molecule with single bonds only, in subset order.
    :rtype: Chem.RWMol
    """
    indices = range(len(atoms)) if atom_ids is None else atom_ids
    subset = atoms[list(indices)]
    mol = Chem.RWMol(Chem.MolFromXYZBlock(atoms_to_xyz(subset)))
    for begin, end in connectivity_graph(subset).edges():
        mol.AddBond(int(begin), int(end), Chem.BondType.SINGLE)
    for atom in mol.GetAtoms():
        # Every hydrogen is explicit; RDKit must not add hidden ones.
        atom.SetNoImplicit(True)
    return mol


def single_bond_stereo_smiles(atoms: Atoms) -> str:
    """
    Generate canonical isomeric SMILES from 3D single-bond connectivity.

    :param atoms: Atomic structure.
    :type atoms: Atoms
    :return: Canonical isomeric SMILES with tetrahedral stereo perception.
    :rtype: str
    """
    mol = rdkit_mol(atoms)
    Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
    Chem.AssignStereochemistryFrom3D(mol)
    return Chem.MolToSmiles(mol)


def stereo_descriptor(
    atoms: Atoms,
    atom_ids: Sequence[int] | None = None,
) -> str:
    """
    Compute canonical stereo descriptor for tetrahedral stereocenters.

    Builds an RDKit molecule from connectivity with single bonds only,
    explicit hydrogens, and 3D coordinates. Assigns tetrahedral
    stereochemistry from the 3D conformer and compares with its mirror
    image (negated x coordinates) to keep the lexicographically smaller
    isomeric SMILES, ensuring enantiomers share the same descriptor while
    diastereomers differ. E/Z stereochemistry around double bonds is out
    of scope because all bonds are represented as single bonds.

    :param atoms: Atomic configuration.
    :type atoms: Atoms
    :param atom_ids: Subset of atom indices belonging to species, or None.
    :type atom_ids: Sequence[int] or None
    :return: Descriptor 's' + 8 hex digits, or '' if no tetrahedral stereo.
    :rtype: str
    """
    indices = range(len(atoms)) if atom_ids is None else atom_ids
    subset = atoms[list(indices)]
    smi = single_bond_stereo_smiles(subset)
    mirror = subset.copy()
    mirror.positions[:, 0] = -mirror.positions[:, 0]
    mirror_smi = single_bond_stereo_smiles(mirror)
    chosen = min(smi, mirror_smi)
    if "@" not in chosen:
        return ""
    digest = hashlib.sha256(chosen.encode("utf-8")).hexdigest()[:8]
    return f"s{digest}"


def species_key(
    graph: nx.Graph,
    atom_ids: Sequence[int],
    atoms: Atoms | None = None,
    stereo: bool = False,
) -> str:
    """
    Compute canonical species key from Hill formula and WL graph hash.

    When stereo is True and atoms is provided, appends the stereo
    descriptor to the formula-hash key if stereocenters are present.

    :param graph: Molecular connectivity graph.
    :type graph: nx.Graph
    :param atom_ids: Atom indices belonging to this species fragment.
    :type atom_ids: Sequence[int]
    :param atoms: Atomic structure containing coordinates, or None.
    :type atoms: Atoms or None
    :param stereo: Whether to include 3D tetrahedral stereo descriptor.
    :type stereo: bool
    :return: Canonical species key in formula-hash[-descriptor] format.
    :rtype: str
    """
    symbols = "".join(graph.nodes[idx]["element"] for idx in atom_ids)
    formula = Formula(symbols).format("hill")
    subgraph = graph.subgraph(atom_ids)
    graph_hash = nx.weisfeiler_lehman_graph_hash(
        subgraph,
        node_attr="element",
        iterations=3,
        digest_size=4,
    )
    base_key = f"{formula}-{graph_hash}"
    if not stereo or atoms is None:
        return base_key
    desc = stereo_descriptor(atoms, atom_ids)
    return f"{base_key}-{desc}" if desc else base_key


def state_key(
    atoms: Atoms,
    scale: float = 1.2,
    stereo: bool = False,
) -> tuple[str, ...]:
    """
    Compute canonical state key as a sorted tuple of fragment species keys.

    :param atoms: Full atomic configuration containing all fragments.
    :type atoms: Atoms
    :param scale: Multiplicative scale factor for covalent cutoffs.
    :type scale: float
    :param stereo: Whether to include 3D tetrahedral stereo descriptors.
    :type stereo: bool
    :return: Sorted tuple of fragment species keys.
    :rtype: tuple[str, ...]
    """
    graph = connectivity_graph(atoms, scale=scale)
    components = fragments(graph)
    keys = [
        species_key(graph, frag, atoms=atoms, stereo=stereo)
        for frag in components
    ]
    return tuple(sorted(keys))


def reacting_atoms(
    graph_before: nx.Graph,
    graph_after: nx.Graph,
) -> list[int]:
    """
    Identify reacting atoms grown to full molecules across both frames.

    :param graph_before: Connectivity graph of initial frame.
    :type graph_before: nx.Graph
    :param graph_after: Connectivity graph of final frame.
    :type graph_after: nx.Graph
    :return: Sorted unique indices of atoms participating in reaction.
    :rtype: list[int]
    """
    edges_before = {tuple(sorted(edge)) for edge in graph_before.edges()}
    edges_after = {tuple(sorted(edge)) for edge in graph_after.edges()}
    diff_nodes = {node for edge in edges_before ^ edges_after for node in edge}
    reacting: set[int] = set()
    for node in diff_nodes:
        reacting.update(nx.node_connected_component(graph_before, node))
        reacting.update(nx.node_connected_component(graph_after, node))
    return sorted(reacting)


def species_labels(atoms: Atoms) -> SpeciesLabels:
    """
    Assign bond orders on our connectivity and derive SMILES and InChIKey.

    :param atoms: Atomic configuration of a single species.
    :type atoms: Atoms
    :return: SpeciesLabels with identifiers or None if perception fails.
    :rtype: SpeciesLabels
    """
    mol = rdkit_mol(atoms)
    try:
        rdDetermineBonds.DetermineBondOrders(mol, charge=0)
    except ValueError:
        return SpeciesLabels(smiles=None, inchikey=None)
    smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
    return SpeciesLabels(smiles=smiles, inchikey=Chem.MolToInchiKey(mol))


def frame_to_mapped_smiles(
    atoms: Atoms,
    atom_ids: Sequence[int],
) -> str:
    """
    Generate mapped SMILES for an atom subset of a single frame.

    :param atoms: Atomic configuration.
    :type atoms: Atoms
    :param atom_ids: Indices of atoms included in the subset.
    :type atom_ids: Sequence[int]
    :return: Canonical mapped SMILES string with explicit hydrogens.
    :rtype: str
    """
    mol = rdkit_mol(atoms, atom_ids)
    try:
        rdDetermineBonds.DetermineBondOrders(mol, charge=0)
    except ValueError:
        # A failed bond-order pass can leave orders behind; start clean.
        mol = rdkit_mol(atoms, atom_ids)
    for idx, orig_idx in enumerate(atom_ids):
        mol.GetAtomWithIdx(idx).SetAtomMapNum(orig_idx + 1)
    return Chem.MolToSmiles(mol)


def mapped_reaction_smiles(
    before: Atoms,
    after: Atoms,
    atom_ids: Sequence[int] | None = None,
) -> str:
    """
    Generate atom-mapped reaction SMILES between two endpoint frames.

    :param before: Initial atomic configuration.
    :type before: Atoms
    :param after: Final atomic configuration.
    :type after: Atoms
    :param atom_ids: Subset of reacting atom indices, or None for all atoms.
    :type atom_ids: Sequence[int] or None
    :return: Mapped reaction SMILES in reactants>>products format.
    :rtype: str
    """
    selected = list(range(len(before))) if atom_ids is None else atom_ids
    reactants_smiles = frame_to_mapped_smiles(before, selected)
    products_smiles = frame_to_mapped_smiles(after, selected)
    return f"{reactants_smiles}>>{products_smiles}"


def multiplicity(atoms: Atoms, charge: int = 0) -> int:
    """
    Determine spin multiplicity from total electron count parity.

    :param atoms: Atomic structure.
    :type atoms: Atoms
    :param charge: Total net charge of the system, defaults to 0.
    :type charge: int
    :return: 1 for even electron count (singlet), 2 for odd (doublet).
    :rtype: int
    """
    electrons = sum(atoms.get_atomic_numbers()) - charge
    return SINGLET if electrons % 2 == 0 else DOUBLET


def radical_count(atoms: Atoms, atom_ids: Sequence[int]) -> int:
    """
    Count the open-shell molecules among the given atoms.

    A side with two radicals is a diradical that closed-shell engines
    cannot describe, so callers reject such reactions.

    :param atoms: Atomic structure.
    :type atoms: Atoms
    :param atom_ids: Atom indices to inspect.
    :type atom_ids: Sequence[int]
    :return: Number of fragments with an odd electron count.
    :rtype: int
    """
    subset = atoms[list(atom_ids)]
    graph = connectivity_graph(subset)
    return sum(
        multiplicity(subset[frag]) == DOUBLET for frag in fragments(graph)
    )
