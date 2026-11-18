"""
Unit tests for chemical graph and reaction mapping routines.

Tests cover connectivity graph generation, species keys, state keys,
reacting atom identification, RDKit labels, reaction SMILES, and spin.
"""

import re

import networkx as nx
from ase import Atoms
from ase.build import molecule
from ase.formula import Formula
from rdkit import Chem
from rdkit.Chem import AllChem

from nrnet.chem import (
    connectivity_graph,
    fragments,
    mapped_reaction_smiles,
    multiplicity,
    radical_count,
    reacting_atoms,
    species_key,
    species_labels,
    state_key,
)


def mol_to_atoms(mol: Chem.Mol) -> Atoms:
    """
    Convert an RDKit Mol with 3D coordinates to an ASE Atoms object.

    :param mol: Embedded RDKit molecule.
    :type mol: Chem.Mol
    :return: ASE Atoms representation.
    :rtype: Atoms
    """
    conformer = mol.GetConformer()
    positions = conformer.GetPositions()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return Atoms(symbols=symbols, positions=positions)


def build_molecule_from_smiles(
    smiles: str,
    random_seed: int = 42,
) -> Atoms:
    """
    Generate an ASE Atoms object from SMILES via 3D embedding.

    :param smiles: Chemical SMILES string.
    :type smiles: str
    :param random_seed: Seed for reproducible conformer generation.
    :type random_seed: int
    :return: 3D embedded ASE Atoms instance.
    :rtype: Atoms
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=random_seed)
    AllChem.UFFOptimizeMolecule(mol)
    return mol_to_atoms(mol)


def reorder_atoms_to_symbols(
    atoms: Atoms,
    target_symbols: list[str],
) -> Atoms:
    """
    Permute atoms to match a specified sequence of elemental symbols.

    :param atoms: Source atomic structure.
    :type atoms: Atoms
    :param target_symbols: Target elemental symbol sequence.
    :type target_symbols: list[str]
    :return: Reordered Atoms instance.
    :rtype: Atoms
    """
    cur_symbols = atoms.get_chemical_symbols()
    used_indices: set[int] = set()
    permutation: list[int] = []
    for symbol in target_symbols:
        for idx, cur_sym in enumerate(cur_symbols):
            if cur_sym == symbol and idx not in used_indices:
                used_indices.add(idx)
                permutation.append(idx)
                break
    return atoms[permutation]


def extract_map_numbers(smiles: str) -> set[int]:
    """
    Extract set of integer atom map numbers from a mapped SMILES string.

    :param smiles: Mapped SMILES string.
    :type smiles: str
    :return: Set of unique integer map numbers.
    :rtype: set[int]
    """
    map_pattern = re.compile(r":(\d+)]")
    return {int(num) for num in map_pattern.findall(smiles)}


def test_separated_molecules_species() -> None:
    """
    Verify disconnected molecules 6 A apart yield separate species.

    :return: None.
    :rtype: None
    """
    hcho = molecule("H2CO")
    nh3 = molecule("NH3")
    nh3.translate([6.0, 0.0, 0.0])
    graph = connectivity_graph(hcho + nh3)
    components = fragments(graph)
    assert len(components) == 2
    formulas = [
        Formula("".join(graph.nodes[idx]["element"] for idx in comp)).format(
            "hill"
        )
        for comp in components
    ]
    assert formulas == ["CH2O", "H3N"]


def test_species_key_invariance() -> None:
    """
    Verify species_key is invariant to rotation, translation, and permutation.

    :return: None.
    :rtype: None
    """
    initial = molecule("H2CO")
    graph_initial = connectivity_graph(initial)
    key_initial = species_key(graph_initial, list(range(len(initial))))

    transformed = initial.copy()
    transformed.rotate(45, "z")
    transformed.translate([2.5, -1.0, 3.0])
    transformed = transformed[[2, 0, 3, 1]]
    graph_transformed = connectivity_graph(transformed)
    key_transformed = species_key(
        graph_transformed, list(range(len(transformed)))
    )
    assert key_initial == key_transformed


def test_ethanol_vs_dimethyl_ether() -> None:
    """
    Verify structural isomers have identical Hill formula but distinct keys.

    :return: None.
    :rtype: None
    """
    ethanol = molecule("CH3CH2OH")
    dme = molecule("CH3OCH3")
    graph_ethanol = connectivity_graph(ethanol)
    graph_dme = connectivity_graph(dme)
    key_ethanol = species_key(graph_ethanol, list(range(len(ethanol))))
    key_dme = species_key(graph_dme, list(range(len(dme))))

    formula_ethanol = key_ethanol.split("-")[0]
    formula_dme = key_dme.split("-")[0]
    assert formula_ethanol == "C2H6O"
    assert formula_dme == "C2H6O"
    assert key_ethanol != key_dme


def test_aminomethanol_vs_separated_reactants() -> None:
    """
    Verify aminomethanol differs from separated reactant state.

    :return: None.
    :rtype: None
    """
    hcho = molecule("H2CO")
    nh3 = molecule("NH3")
    nh3.translate([6.0, 0.0, 0.0])
    reactants_state = state_key(hcho + nh3)

    aminomethanol = build_molecule_from_smiles("NCO", random_seed=42)
    product_state = state_key(aminomethanol)

    assert len(reactants_state) == 2
    assert len(product_state) == 1
    assert product_state[0].startswith("CH5NO-")
    assert reactants_state != product_state


def build_test_graph(
    edges: list[tuple[int, int]],
    node_count: int = 11,
) -> nx.Graph:
    """
    Construct a networkx Graph with designated node count and edges.

    :param edges: List of node index edge pairs.
    :type edges: list[tuple[int, int]]
    :param node_count: Total number of nodes, defaults to 11.
    :type node_count: int
    :return: Constructed graph instance.
    :rtype: nx.Graph
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    graph.add_edges_from(edges)
    return graph


def test_reacting_atoms_with_spectator() -> None:
    """
    Verify reacting_atoms isolates reacting molecules from spectator solvent.

    :return: None.
    :rtype: None
    """
    edges_hcho = [(0, 1), (0, 2), (0, 3)]
    edges_nh3 = [(4, 5), (4, 6), (4, 7)]
    edges_h2o = [(8, 9), (8, 10)]
    before = build_test_graph(edges_hcho + edges_nh3 + edges_h2o)

    edges_amino = [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (1, 7)]
    after = build_test_graph(edges_amino + edges_h2o)

    active = reacting_atoms(before, after)
    assert active == list(range(8))


def test_species_labels_hcho_and_radical() -> None:
    """
    Verify species_labels on closed-shell HCHO and open-shell radical CH3.

    :return: None.
    :rtype: None
    """
    hcho = molecule("H2CO")
    labels_hcho = species_labels(hcho)
    assert labels_hcho.smiles == "C=O"
    assert labels_hcho.inchikey == "WSFSSNUMVMOOMR-UHFFFAOYSA-N"

    ch3 = molecule("CH3")
    labels_ch3 = species_labels(ch3)
    assert labels_ch3.smiles is None
    assert labels_ch3.inchikey is None


def test_mapped_reaction_smiles() -> None:
    """
    Verify mapped_reaction_smiles maps all atoms consistently across reaction.

    :return: None.
    :rtype: None
    """
    hcho = molecule("H2CO")
    nh3 = molecule("NH3")
    nh3.translate([6.0, 0.0, 0.0])
    reactants = hcho + nh3
    amino = build_molecule_from_smiles("NCO", random_seed=42)
    products = reorder_atoms_to_symbols(amino, reactants.get_chemical_symbols())
    rxn = mapped_reaction_smiles(reactants, products)
    assert rxn.count(">>") == 1
    reactants_part, products_part = rxn.split(">>")
    expected = set(range(1, len(reactants) + 1))
    assert extract_map_numbers(reactants_part) == expected
    assert extract_map_numbers(products_part) == expected


def test_multiplicity() -> None:
    """
    Verify spin multiplicity logic on singlet HCHO and doublet CH3.

    :return: None.
    :rtype: None
    """
    hcho = molecule("H2CO")
    ch3 = molecule("CH3")
    assert multiplicity(hcho) == 1
    assert multiplicity(ch3) == 2
    assert multiplicity(ch3, charge=1) == 1


def test_connectivity_has_no_neighbor_list_skin() -> None:
    """
    Check that C-N at 2.17 A is not a bond while 1.47 A is.

    ASE's default 0.3 A skin per atom once made the 2.17 A contact bond.

    :return: None.
    :rtype: None
    """
    far = Atoms("CN", positions=[[0, 0, 0], [2.17, 0, 0]])
    near = Atoms("CN", positions=[[0, 0, 0], [1.47, 0, 0]])
    assert connectivity_graph(far).number_of_edges() == 0
    assert connectivity_graph(near).number_of_edges() == 1


def test_radical_mapped_smiles_keeps_three_hydrogens() -> None:
    """
    Check the fallback SMILES of CH3 adds no hidden hydrogen.

    :return: None.
    :rtype: None
    """
    methyl = molecule("CH3")
    reactants = mapped_reaction_smiles(methyl, methyl).split(">>")[0]
    assert reactants == "[C:1]([H:2])([H:3])[H:4]"


def test_radical_count_counts_open_shell_fragments() -> None:
    """
    Check two far-apart H atoms count as two radicals and H2 as none.

    :return: None.
    :rtype: None
    """
    apart = Atoms("HH", positions=[[0, 0, 0], [5.0, 0, 0]])
    bonded = Atoms("HH", positions=[[0, 0, 0], [0.74, 0, 0]])
    assert radical_count(apart, [0, 1]) == 2
    assert radical_count(bonded, [0, 1]) == 0
