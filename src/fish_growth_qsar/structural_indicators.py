"""Calculate comparator indicators from the reported structure, including counterions."""

import argparse
from functools import lru_cache
from pathlib import Path

import pandas as pd
from rdkit import Chem


SMARTS = {
    "carbonyl": "[#6]=[O]",
    "aromatic_hydroxyl": "[c]-[O;H1]",
    "carboxylic_acid_or_carboxylate": "[CX3](=[OX1])[OX2H1,OX1-]",
    "carboxylic_ester": "[#6][CX3](=[OX1])[OX2H0][#6]",
    "amide": "[CX3](=[OX1])[NX3]",
    "aldehyde": "[CX3H1](=[OX1])",
    "ketone": "[#6][CX3](=[OX1])[#6]",
    "carbamate": "[NX3][CX3](=[OX1])[OX2][#6]",
    "carbon_halogen": "[#6]~[F,Cl,Br,I]",
    "phosphorus": "[#15]",
    "sulfur": "[#16]",
    "aryl_nitrogen": "[c]-[N;X3;!$(N=*)]",
}
PHENOL_TYPES = (
    "nitro_on_phenolic_ring",
    "halogen_on_phenolic_ring",
    "multiple_aromatic_hydroxyls",
    "hydrocarbon_monophenol",
    "other_aromatic_hydroxyl",
)
INDICATOR_ORDER = tuple(SMARTS) + PHENOL_TYPES
QUERIES = {name: Chem.MolFromSmarts(pattern) for name, pattern in SMARTS.items()}
NITRO = Chem.MolFromSmarts("[c]-[N+](=O)[O-]")
HALOGEN = Chem.MolFromSmarts("[c]-[F,Cl,Br,I]")


@lru_cache(maxsize=None)
def structural_indicators(smiles):
    """Return the archived binary features in their original column order."""
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError("Invalid reported chemical structure")
    flags = {name: bool(molecule.HasSubstructMatch(query)) for name, query in QUERIES.items()}
    hydroxyls = molecule.GetSubstructMatches(QUERIES["aromatic_hydroxyl"])
    hydroxyl_carbons = {match[0] for match in hydroxyls}
    aromatic_rings = [
        set(ring)
        for ring in molecule.GetRingInfo().AtomRings()
        if all(molecule.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
    ]

    def same_ring(query):
        matches = molecule.GetSubstructMatches(query)
        return any(
            hydroxyl_carbons & ring and any(match[0] in ring for match in matches)
            for ring in aromatic_rings
        )

    family = None
    if hydroxyls:
        if same_ring(NITRO):
            family = PHENOL_TYPES[0]
        elif same_ring(HALOGEN):
            family = PHENOL_TYPES[1]
        elif len(hydroxyls) > 1:
            family = PHENOL_TYPES[2]
        elif (
            all(atom.GetAtomicNum() in (1, 6, 8) for atom in molecule.GetAtoms())
            and sum(atom.GetAtomicNum() == 8 for atom in molecule.GetAtoms()) == 1
        ):
            family = PHENOL_TYPES[3]
        else:
            family = PHENOL_TYPES[4]
    flags.update({name: family == name for name in PHENOL_TYPES})
    return tuple(int(flags[name]) for name in INDICATOR_ORDER)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    table = pd.read_csv(args.input)
    if args.smiles_column not in table:
        raise ValueError("SMILES column is absent")
    values = [structural_indicators(value) for value in table[args.smiles_column]]
    result = pd.concat([table, pd.DataFrame(values, columns=INDICATOR_ORDER)], axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
