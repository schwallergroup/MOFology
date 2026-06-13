"""Default concept generation rules for MOF concept-vector probing."""

NUMERIC_CONCEPT_COLUMNS = [
    "Density",
    "Largest cavity diameter",
    "Pore limiting diameter",
    "Unit cell volume",
    "Band gap (PBE)",
    "CO2 uptake",
    "H2O uptake",
]

CATEGORICAL_CONCEPT_COLUMNS = [
    "crystal_system",
    "topology",
]

MULTIVALUE_TOKEN_COLUMNS = [
    "metal_cluster_elements",
    "linker_smiles",
]

