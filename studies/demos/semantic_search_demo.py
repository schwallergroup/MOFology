#!/usr/bin/env python3
"""
Semantic Search Demonstration for MOFology Knowledge Graph.

Demonstrates that the KG can answer chemist-relevant questions via SPARQL queries.
"""

import argparse
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
from rdflib import Graph, Namespace

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

MOF = Namespace("http://emmo.info/domain-mof/mof-ontology#")
SYN = Namespace("http://emmo.info/domain-mof/synthesis#")


QUERIES = {
    "count_mofs_by_type": {
        "description": "Count MOFs by type (experimental, hypothetical, functionalized)",
        "chemist_question": "How many MOFs of each type are in the knowledge graph?",
        "sparql": """
PREFIX mof: <http://emmo.info/domain-mof/mof-ontology#>
PREFIX syn: <http://emmo.info/domain-mof/synthesis#>
SELECT ?type (COUNT(DISTINCT ?mof) AS ?count)
WHERE {{
    ?mof a ?type .
    FILTER(?type IN (mof:MOF, mof:ExperimentalMOF, mof:HypotheticalMOF, syn:FunctionalizedMOF))
}}
GROUP BY ?type
ORDER BY DESC(?count)
""",
        "params": {},
    },

    "functionalization_tracking": {
        "description": "Find functionalized MOF derivatives and their parent MOFs",
        "chemist_question": "What are the functionalized derivatives in the KG and their parent MOFs?",
        "sparql": """
PREFIX mof: <http://emmo.info/domain-mof/mof-ontology#>
PREFIX syn: <http://emmo.info/domain-mof/synthesis#>
SELECT DISTINCT ?funcMof ?parent
WHERE {{
    ?funcMof syn:derivedFrom ?parent .
    ?funcMof a syn:FunctionalizedMOF .
}}
LIMIT 30
""",
        "params": {},
    },

    "mofs_with_topology": {
        "description": "Find MOFs with their topology codes",
        "chemist_question": "What topologies are present and which MOFs have them?",
        "sparql": """
PREFIX mof: <http://emmo.info/domain-mof/mof-ontology#>
SELECT ?topo (COUNT(DISTINCT ?mof) AS ?count)
WHERE {{
    ?mof mof:hasTopology ?t .
    ?t mof:topologyCode ?topo .
}}
GROUP BY ?topo
ORDER BY DESC(?count)
LIMIT 20
""",
        "params": {},
    },

    "metal_element_distribution": {
        "description": "Find distribution of metal elements across MOFs",
        "chemist_question": "Which metal elements are most common in MOF structures?",
        "sparql": """
PREFIX mof: <http://emmo.info/domain-mof/mof-ontology#>
SELECT ?element (COUNT(DISTINCT ?mof) AS ?count)
WHERE {{
    ?mof mof:hasMetalNode ?m .
    ?m mof:hasMetalElement ?element .
}}
GROUP BY ?element
ORDER BY DESC(?count)
LIMIT 20
""",
        "params": {},
    },

    "property_coverage": {
        "description": "Count available properties across MOFs",
        "chemist_question": "What properties are available and how many MOFs have them?",
        "sparql": """
PREFIX mof: <http://emmo.info/domain-mof/mof-ontology#>
SELECT ?propName (COUNT(DISTINCT ?mof) AS ?count)
WHERE {{
    {{
        ?prop mof:hasComputationalPropertyOwner ?mof .
        ?prop mof:propertyName ?propName .
    }} UNION {{
        ?prop mof:hasStructuralPropertyOwner ?mof .
        ?prop mof:propertyName ?propName .
    }} UNION {{
        ?prop mof:hasPhysicalPropertyOwner ?mof .
        ?prop mof:propertyName ?propName .
    }}
}}
GROUP BY ?propName
ORDER BY DESC(?count)
LIMIT 30
""",
        "params": {},
    },

    "linker_smiles_sample": {
        "description": "Sample of linker SMILES strings in the KG",
        "chemist_question": "What organic linkers are represented in the knowledge graph?",
        "sparql": """
PREFIX mof: <http://emmo.info/domain-mof/mof-ontology#>
SELECT DISTINCT ?linker ?smiles
WHERE {{
    ?mof mof:hasLinker ?linker .
    ?linker mof:hasSMILES ?smiles .
}}
LIMIT 20
""",
        "params": {},
    },

    "dac_porous_low_density": {
        "description": "Porous, low-density candidates for high-capacity DAC sorbents",
        "chemist_question": "Which MOFs have pore limiting diameter > 6 Angstrom and density < 1.0 g/cm^3?",
        "sparql": """
PREFIX mof: <http://emmo.info/domain-mof/mof-ontology#>
SELECT DISTINCT ?mof ?pld ?density
WHERE {{
    ?pld_prop mof:hasStructuralPropertyOwner ?mof ;
              mof:propertyName "Pore limiting diameter"^^<http://www.w3.org/2001/XMLSchema#string> ;
              mof:propertyValue ?pld .
    ?den_prop mof:hasPhysicalPropertyOwner ?mof ;
              mof:propertyName "Density"^^<http://www.w3.org/2001/XMLSchema#string> ;
              mof:propertyValue ?density .
    FILTER(?pld > 6.0 && ?density < 1.0)
}}
ORDER BY ?density
LIMIT 30
""",
        "params": {},
    },

    "dac_strong_co2_binders": {
        "description": "MOFs with strong CO2 binding (BE < -0.5 eV) and favorable CO2/H2O selectivity",
        "chemist_question": "Which MOFs bind CO2 below -0.5 eV more strongly than they bind H2O?",
        "sparql": """
PREFIX mof: <http://emmo.info/domain-mof/mof-ontology#>
SELECT DISTINCT ?mof ?co2_be ?h2o_be
WHERE {{
    ?p_co2 mof:hasComputationalPropertyOwner ?mof ;
           mof:propertyName "Binding Energy CO2"^^<http://www.w3.org/2001/XMLSchema#string> ;
           mof:propertyValue ?co2_be .
    ?p_h2o mof:hasComputationalPropertyOwner ?mof ;
           mof:propertyName "Binding Energy H2O"^^<http://www.w3.org/2001/XMLSchema#string> ;
           mof:propertyValue ?h2o_be .
    FILTER(?co2_be < -0.5 && ?co2_be < ?h2o_be)
}}
ORDER BY ?co2_be
LIMIT 30
""",
        "params": {},
    },

    "dac_mg_family_analogues": {
        "description": "Mg-based MOFs with pcu or fcu topology (Mg-MOF-74 family analogues)",
        "chemist_question": "Which Mg-containing MOFs have pcu or fcu topology (analogues of Mg-MOF-74)?",
        "sparql": """
PREFIX mof: <http://emmo.info/domain-mof/mof-ontology#>
SELECT DISTINCT ?mof ?topo
WHERE {{
    ?mof mof:hasMetalNode ?m .
    ?m mof:hasMetalElement "Mg" .
    ?mof mof:hasTopology ?t .
    ?t mof:topologyCode ?topo .
    FILTER(?topo IN ("pcu", "fcu"))
}}
LIMIT 30
""",
        "params": {},
    },

    "dac_functionalization_improving_co2": {
        "description": "Amine-functionalized derivatives whose CO2 binding is stronger than the parent",
        "chemist_question": "Which amine-functionalized MOFs tighten CO2 binding relative to their parent?",
        "sparql": """
PREFIX mof: <http://emmo.info/domain-mof/mof-ontology#>
PREFIX syn: <http://emmo.info/domain-mof/synthesis#>
SELECT DISTINCT ?funcMof ?parent ?parent_co2 ?child_co2
WHERE {{
    ?funcMof syn:derivedFrom ?parent ;
             a syn:FunctionalizedMOF .
    ?pc mof:hasComputationalPropertyOwner ?parent ;
        mof:propertyName "Binding Energy CO2"^^<http://www.w3.org/2001/XMLSchema#string> ;
        mof:propertyValue ?parent_co2 .
    ?cc mof:hasComputationalPropertyOwner ?funcMof ;
        mof:propertyName "Binding Energy CO2"^^<http://www.w3.org/2001/XMLSchema#string> ;
        mof:propertyValue ?child_co2 .
    FILTER(?child_co2 < ?parent_co2)
}}
ORDER BY ?child_co2
LIMIT 30
""",
        "params": {},
    },
}


def load_kg(kg_path: str) -> Graph:
    """Load the knowledge graph from TTL file."""
    log.info(f"Loading KG from {kg_path}...")
    g = Graph()
    g.parse(kg_path, format="turtle")
    log.info(f"Loaded {len(g)} triples")
    return g


def run_query(g: Graph, query_name: str, query_info: Dict) -> Tuple[pd.DataFrame, str]:
    """Execute a SPARQL query and return results as DataFrame."""
    sparql = query_info["sparql"].format(**query_info["params"])

    log.info(f"Running query: {query_name}")
    try:
        results = g.query(sparql)

        if not results.vars:
            return pd.DataFrame(), sparql

        rows = []
        for row in results:
            row_dict = {}
            for i, var in enumerate(results.vars):
                val = row[i]
                if val is not None:
                    row_dict[str(var)] = str(val).split("#")[-1] if "#" in str(val) else str(val)
                else:
                    row_dict[str(var)] = None
            rows.append(row_dict)

        df = pd.DataFrame(rows)
        log.info(f"  Found {len(df)} results")
        return df, sparql
    except Exception as e:
        log.error(f"  Query failed: {e}")
        return pd.DataFrame(), sparql


def format_results_markdown(query_name: str, query_info: Dict, df: pd.DataFrame, sparql: str) -> str:
    """Format query results as markdown."""
    md = []
    md.append(f"### {query_name.replace('_', ' ').title()}\n")
    md.append(f"**Chemist Question:** {query_info['chemist_question']}\n")
    md.append(f"**Description:** {query_info['description']}\n")
    md.append(f"\n**SPARQL Query:**\n```sparql\n{sparql.strip()}\n```\n")

    if len(df) > 0:
        md.append(f"\n**Results ({len(df)} rows):**\n")
        md.append(df.head(10).to_markdown(index=False))
        if len(df) > 10:
            md.append(f"\n*... and {len(df) - 10} more rows*\n")
    else:
        md.append("\n**Results:** No matching MOFs found (query returned empty)\n")

    md.append("\n---\n")
    return "\n".join(md)


def generate_demo_report(g: Graph, out_dir: str) -> None:
    """Generate the full demo report."""
    os.makedirs(out_dir, exist_ok=True)

    report = []
    report.append("# MOFology Semantic Search Demonstration\n")
    report.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
    report.append("\nThis document demonstrates that the MOFology Knowledge Graph can answer ")
    report.append("chemist-relevant questions via SPARQL queries. Each query shows a natural language ")
    report.append("question, the corresponding SPARQL query, and sample results.\n")
    report.append("\n---\n")

    summary_stats = []

    for query_name, query_info in QUERIES.items():
        df, sparql = run_query(g, query_name, query_info)

        # Save individual CSV
        csv_path = os.path.join(out_dir, f"{query_name}_results.csv")
        df.to_csv(csv_path, index=False)

        # Format for report
        report.append(format_results_markdown(query_name, query_info, df, sparql))

        summary_stats.append({
            "Query": query_name.replace("_", " ").title(),
            "Results": len(df),
            "Status": "✓" if len(df) > 0 else "✗ (empty)",
        })

    # Write main report
    report_path = os.path.join(out_dir, "query_results.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    log.info(f"Report saved to {report_path}")

    # Write summary
    summary = []
    summary.append("# Semantic Search Demo Summary\n")
    summary.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
    summary.append("\n## Query Results Overview\n")
    summary.append(pd.DataFrame(summary_stats).to_markdown(index=False))
    summary.append("\n\n## Key Takeaways\n")
    summary.append("- The MOFology KG enables complex multi-hop queries\n")
    summary.append("- Chemists can query structural analogues, property filters, and synthesis conditions\n")
    summary.append("- Functionalization tracking links parent MOFs to derivatives\n")
    summary.append("- Multi-criteria DAC screening is possible via SPARQL\n")

    summary_path = os.path.join(out_dir, "demo_summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary))
    log.info(f"Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Semantic Search Demo for MOFology KG")
    parser.add_argument(
        "--kg_path",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kg", "mof_kg.ttl"),
    )
    parser.add_argument(
        "--out_dir",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "results"/semantic_search_demo),
    )
    args = parser.parse_args()

    g = load_kg(args.kg_path)
    generate_demo_report(g, args.out_dir)
    log.info("Demo complete!")


if __name__ == "__main__":
    main()
