# MOFology: A Knowledge Graph for Engineering Direct Air Capture Materials

MOFology is an ontology-grounded Knowledge Graph (KG) framework specifically designed to accelerate the engineering and discovery of Metal-Organic Frameworks (MOFs) for Direct Air Capture (DAC) of CO₂. By integrating ~250,000 MOFs from six disparate databases, MOFology provides a unified, semantically-rich graph containing roughly 8.4 million RDF triples. 

This repository contains the codebase used to generate the Knowledge Graph, train graph embeddings, and perform the downstream experiments described in the paper.

## Repository Structure

The codebase is organized into several key modules:

*   **`kg_construction/`**: Code for extracting data from the raw databases (OpenDAC25, DigiMOF, QMOF, MOF-ChemUnity, MOF-FreeEnergy, and SynMOF), normalizing the data, and constructing the RDF triples using OWL reasoning (via HermiT).
*   **`ontology/`**: Contains the MOFology ontology file (`MOF_EMMO_ontology.ttl`), which extends the Elementary Multiperspective Material Ontology (EMMO) with MOF-specific classes (e.g., `Linker`, `MetalCluster`, `FunctionalizedMOF`).
*   **`embeddings/`**: Implementations of the graph embedding models used in the study, including CompGCN, Node2Vec, and TransE.
*   **`studies/`**: The downstream machine learning studies and evaluations, including:
    *   Link prediction and chemical property prediction (`run_full_study.py`).
    *   Concept vector probing and extraction.
    *   Multi-criteria DAC screening (`run_dac_screen.py`).
*   **`scripts/`**: Scripts for generating the publication-quality figures and t-SNE visualizations.

## Setup and Installation

1.  Clone this repository.
2.  Install the required dependencies via pip:

```bash
pip install -r requirements.txt
```

*Note: You may need to install PyTorch separately depending on your specific CUDA/CPU setup.*

## Usage

### 1. Building the Knowledge Graph
To build the Knowledge Graph from the raw data sources, ensure the raw datasets are placed in a `data/raw/` directory at the root level, then run the build script:

```bash
python kg_construction/build_kg.py
```
This will extract, normalize, and construct the base `.ttl` graph, followed by enrichment using the OWL-RL reasoner. The final KG will be saved to `data/kg/`.

### 2. Running the Core ML Studies
To run the primary evaluations comparing CompGCN, Node2Vec, and TransE embeddings on link prediction and chemical property prediction:

```bash
python studies/run_full_study.py
```
This script handles the mapping of embeddings, regression evaluation against standard properties, and generation of comparative t-SNE visualizations.

### 3. Multi-Criteria DAC Screening
To rank novel MOFs for their DAC suitability using the predictive pipelines:

```bash
python studies/run_dac_screen.py
```

## Data Availability
The raw datasets required for KG construction (e.g., OpenDAC25, QMOF, DigiMOF, etc.) should be downloaded from their respective sources and placed in the corresponding folders under `data/raw/`.

## License
MIT License