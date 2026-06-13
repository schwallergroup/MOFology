"""
SynMOF Data Extractor

Extracts synthesis entities from SynMOF literature-mined CSV files:
- SynMOF_A.csv: Synthesis conditions, solvents, additives, counterions
- Synmof_M_210618.csv: Synthesis conditions, solvents, additives (no counterions)
- Synmof_Me_210618.csv: Synthesis conditions, solvents (with mol ratios), additives, counterions

Data from all three files is merged by CSD code (filename minus '_clean' suffix).
Solvents and additives are identified by PubChem CIDs.
"""

import pandas as pd
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set

from ..datamodels.entitymodels import (
    SynthesisProcessEntity,
    SynthesisConditionEntity,
    SolventEntity,
    ChemicalEntity,
)


class SynMOFExtractor:
    """Extractor for SynMOF literature-mined synthesis data."""

    def __init__(self, data_dir: Path):
        """
        Initialize the SynMOF extractor.

        Args:
            data_dir: Path to the directory containing SynMOF CSV files.
        """
        self.data_dir = Path(data_dir)

        # Data file paths
        self.synmof_a_file = self.data_dir / "SynMOF_A.csv"
        self.synmof_m_file = self.data_dir / "Synmof_M_210618.csv"
        self.synmof_me_file = self.data_dir / "Synmof_Me_210618.csv"

        # Merged data store keyed by CSD code
        self._merged: Dict[str, dict] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_clean(filename: str) -> str:
        """Normalize SynMOF filename into canonical CSD code."""
        s = str(filename).strip()
        # Normalize known SynMOF suffixes that are not part of canonical CSD code.
        # Apply repeatedly in case of chained suffixes.
        changed = True
        while changed:
            changed = False
            low = s.lower()
            for suffix in ("_clean", "_charged"):
                if low.endswith(suffix):
                    s = s[: -len(suffix)]
                    changed = True
                    break
        return s

    @staticmethod
    def _parse_float(val) -> Optional[float]:
        """Safely parse a float from a value that may contain unicode tilde etc."""
        if val is None:
            return None
        s = str(val).strip()
        if not s or s.lower() == "nan":
            return None
        # Remove common non-numeric prefixes like '∼', '~', '>', '<', '≈'
        s = re.sub(r"^[∼~><≈≤≥]+", "", s).strip()
        if not s:
            return None
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _collect_list(row: pd.Series, prefix: str, count: int = 5) -> list:
        """Collect non-null values from numbered columns (e.g. solvent1..solvent5)."""
        items = []
        for i in range(1, count + 1):
            col = f"{prefix}{i}"
            if col in row.index:
                val = row[col]
                if pd.notna(val):
                    s = str(val).strip()
                    if s and s.lower() != "nan":
                        items.append(s)
        return items

    # ------------------------------------------------------------------
    # Loading & merging
    # ------------------------------------------------------------------

    def _load_and_merge(self):
        """Load all three CSVs and merge rows by CSD code."""
        if self._loaded:
            return

        merged: Dict[str, dict] = {}

        def _init_record(csd_code: str) -> dict:
            return {
                "csd_code": csd_code,
                "temperature_c": None,
                "time_hours": None,
                "yield_percent": None,
                "counterions": [],
                "solvents": [],      # PubChem CID strings
                "additives": [],     # PubChem CID strings
            }

        def _merge_row(record: dict, row: pd.Series, has_counterions: bool):
            # Temperature
            if record["temperature_c"] is None:
                record["temperature_c"] = self._parse_float(row.get("temperature_Celsius"))

            # Time
            if record["time_hours"] is None:
                record["time_hours"] = self._parse_float(row.get("time_h"))

            # Yield
            if record["yield_percent"] is None:
                record["yield_percent"] = self._parse_float(row.get("Yield_Percent"))

            # Counterions (text strings like Cl, NO3, F)
            if has_counterions:
                for ci in self._collect_list(row, "counterions", 5):
                    if ci not in record["counterions"]:
                        record["counterions"].append(ci)

            # Solvents (PubChem CIDs as strings, e.g. '962.0' → '962')
            for sv in self._collect_list(row, "solvent", 5):
                cid = self._normalize_cid(sv)
                if cid and cid not in record["solvents"]:
                    record["solvents"].append(cid)

            # Additives
            for ad in self._collect_list(row, "additive", 5):
                cid = self._normalize_cid(ad)
                if cid and cid not in record["additives"]:
                    record["additives"].append(cid)

        def _process_df(df: pd.DataFrame, has_counterions: bool):
            for _, row in df.iterrows():
                fn = str(row.get("filename", "")).strip()
                if not fn or fn.lower() == "nan":
                    continue
                csd_code = self._strip_clean(fn)
                if not csd_code:
                    continue
                if csd_code not in merged:
                    merged[csd_code] = _init_record(csd_code)
                _merge_row(merged[csd_code], row, has_counterions)

        # --- SynMOF_A (has counterions columns, though often empty) ---
        if self.synmof_a_file.exists():
            df_a = pd.read_csv(self.synmof_a_file)
            _process_df(df_a, has_counterions=True)
            print(f"  SynMOF_A: loaded {len(df_a)} rows")
        else:
            print(f"  Warning: {self.synmof_a_file} not found")

        # --- Synmof_M (no counterions) ---
        if self.synmof_m_file.exists():
            df_m = pd.read_csv(self.synmof_m_file)
            _process_df(df_m, has_counterions=False)
            print(f"  Synmof_M: loaded {len(df_m)} rows")
        else:
            print(f"  Warning: {self.synmof_m_file} not found")

        # --- Synmof_Me (has counterions with data) ---
        if self.synmof_me_file.exists():
            df_me = pd.read_csv(self.synmof_me_file)
            _process_df(df_me, has_counterions=True)
            print(f"  Synmof_Me: loaded {len(df_me)} rows")
        else:
            print(f"  Warning: {self.synmof_me_file} not found")

        self._merged = merged
        self._loaded = True
        print(f"  Merged {len(merged)} unique MOFs from SynMOF data")

    @staticmethod
    def _normalize_cid(val: str) -> Optional[str]:
        """Normalize a PubChem CID value (e.g. '962.0' → '962')."""
        try:
            return str(int(float(val)))
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Public extraction methods
    # ------------------------------------------------------------------

    def extract_synthesis_processes(self) -> List[SynthesisProcessEntity]:
        """
        Extract synthesis process entities from merged SynMOF data.

        Returns:
            List of SynthesisProcessEntity instances (one per MOF).
        """
        self._load_and_merge()
        processes = []

        for csd_code, rec in self._merged.items():
            syn = SynthesisProcessEntity(
                synthesis_id=f"SYN_{csd_code}",
                mof_id=csd_code,
                yield_percent=rec["yield_percent"],
                counterions=rec["counterions"],
                data_source="SynMOF",
            )
            processes.append(syn)

        print(f"Extracted {len(processes)} synthesis processes from SynMOF")
        return processes

    def extract_synthesis_conditions(self) -> List[SynthesisConditionEntity]:
        """
        Extract synthesis condition entities from merged SynMOF data.

        Returns:
            List of SynthesisConditionEntity instances (one per MOF with data).
        """
        self._load_and_merge()
        conditions = []

        for csd_code, rec in self._merged.items():
            temp = rec["temperature_c"]
            time_h = rec["time_hours"]

            if temp is not None or time_h is not None:
                cond = SynthesisConditionEntity(
                    condition_id=f"COND_{csd_code}",
                    synthesis_id=f"SYN_{csd_code}",
                    temperature_c=temp,
                    time_hours=time_h,
                )
                conditions.append(cond)

        print(f"Extracted {len(conditions)} synthesis conditions from SynMOF")
        return conditions

    def extract_solvents(self) -> Tuple[List[SolventEntity], List[Dict]]:
        """
        Extract unique solvent entities and their relationships to synthesis processes.

        Returns:
            Tuple of (solvent entities, relationship dicts).
            Each relationship dict has keys: synthesis_id, solvent_id.
        """
        self._load_and_merge()
        solvent_dict: Dict[str, SolventEntity] = {}
        relationships: List[Dict] = []

        for csd_code, rec in self._merged.items():
            synthesis_id = f"SYN_{csd_code}"
            for cid in rec["solvents"]:
                solvent_id = f"SOLVENT_CID_{cid}"

                # Create unique solvent entity
                if solvent_id not in solvent_dict:
                    solvent_dict[solvent_id] = SolventEntity(
                        solvent_id=solvent_id,
                        canonical_name=f"PubChem CID {cid}",
                        smiles="",
                        canonical_smiles="",
                        all_names=[f"PubChem CID {cid}"],
                        data_sources=["SynMOF"],
                    )

                # Create relationship
                relationships.append({
                    "synthesis_id": synthesis_id,
                    "solvent_id": solvent_id,
                })

        solvents = list(solvent_dict.values())
        print(f"Extracted {len(solvents)} unique solvents and {len(relationships)} solvent relationships from SynMOF")
        return solvents, relationships

    def extract_additives(self) -> Tuple[List[ChemicalEntity], List[Dict]]:
        """
        Extract unique additive entities and their relationships to synthesis processes.

        Additives are stored as ChemicalEntity objects and will be typed as
        syn:Additive in the KG by the entity converter.

        Returns:
            Tuple of (additive chemical entities, relationship dicts).
            Each relationship dict has keys: synthesis_id, chemical_id.
        """
        self._load_and_merge()
        additive_dict: Dict[str, ChemicalEntity] = {}
        relationships: List[Dict] = []

        for csd_code, rec in self._merged.items():
            synthesis_id = f"SYN_{csd_code}"
            for cid in rec["additives"]:
                chemical_id = f"ADDITIVE_CID_{cid}"

                # Create unique additive entity
                if chemical_id not in additive_dict:
                    additive_dict[chemical_id] = ChemicalEntity(
                        chemical_id=chemical_id,
                        canonical_name=f"PubChem CID {cid}",
                        all_names=[f"PubChem CID {cid}"],
                        data_sources=["SynMOF"],
                    )

                # Create relationship
                relationships.append({
                    "synthesis_id": synthesis_id,
                    "chemical_id": chemical_id,
                })

        additives = list(additive_dict.values())
        print(f"Extracted {len(additives)} unique additives and {len(relationships)} additive relationships from SynMOF")
        return additives, relationships
