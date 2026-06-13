"""
OpenDAC25 Data Extractor

Extracts entities from mof_analysis_final.json:
- Parent MOFs (base experimental MOFs from CSD)
- Functionalized MOFs (parent MOF + functionalization with degree)
- Functional Group Chemicals (unique functional groups used)
- Properties (CO2 and H2O binding energies)

Incorporates logic from incorporate_energies.py
"""

import json
import re
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..datamodels.entitymodels import (
    MOFEntity,
    FunctionalizedMOFEntity,
    FunctionalizationEntity,
    PropertyEntity,
    ChemicalEntity,
)


class OpenDAC25Extractor:
    """Extractor for OpenDAC25 data sources."""
    
    # Amine Definitions from incorporate_energies.py
    AMINES = {
        "en": {"name": "Ethylenediamine", "type": "1°/1°"},
        "nmen": {"name": "N-Methylethylenediamine", "type": "1°/2°"},
        "een": {"name": "N-Ethylethylenediamine", "type": "1°/2°"},
        "ipen": {"name": "N-Isopropylethylenediamine", "type": "1°/2°"},
        "dmen": {"name": "N,N-Dimethylethylenediamine", "type": "1°/3°"},
        "deen": {"name": "N,N-Diethylethylenediamine", "type": "1°/3°"},
        "mmen": {"name": "Dimethylethylenediamine", "type": "2°/2°"}, 
        "eeen": {"name": "N,N’-Diethylethylenediamine", "type": "2°/2°"},
        "mden": {"name": "N,N,N’-Trimethylethylenediamine", "type": "2°/3°"},
        "tmen": {"name": "N,N,N’,N’-Tetramethylethylenediamine", "type": "3°/3°"}
    }
    
    def __init__(self, data_file: Optional[Path] = None):
        """
        Initialize the OpenDAC25 extractor.
        
        Args:
            data_file: Path to mof_analysis_final.json. If None, uses default path.
        """
        if data_file is None:
            # Default to data/raw/OpenDAC25/mof_analysis_final.json
            project_root = Path(__file__).parent.parent.parent
            data_file = project_root / "data" / "raw" / "OpenDAC25" / "mof_analysis_final.json"
        
        self.data_file = Path(data_file)
        self._data = None

    def _load_data(self):
        if self._data is None:
            if not self.data_file.exists():
                print(f"Warning: {self.data_file} not found")
                self._data = {}
            else:
                with open(self.data_file, 'r') as f:
                    self._data = json.load(f)
        return self._data
    
    def extract_functionalized_mofs(self) -> List[FunctionalizedMOFEntity]:
        """
        Extract functionalized MOF entities from mof_analysis_final.json.
        """
        data = self._load_data()
        func_mofs = []
        
        for parent_csd, func_groups in data.items():
            if not isinstance(func_groups, dict):
                continue
            
            for amine_code, func_data in func_groups.items():
                if not isinstance(func_data, dict):
                    continue
                
                mof_name_base = func_data.get('mof_name_base', '').strip()
                if not mof_name_base:
                    continue
                
                # Create functionalized MOF ID
                func_mof_id = mof_name_base
                
                # Get functional group name from AMINES dict if possible
                func_group_name = self.AMINES.get(amine_code, {}).get("name", amine_code)
                
                # Parse functionalization degree from mof_name_base if possible
                func_degree = None
                # Try to find decimal pattern (e.g., 0.16)
                degree_match = re.search(r'_(\d+\.\d+)_', mof_name_base)
                if degree_match:
                    try:
                        func_degree = float(degree_match.group(1))
                    except ValueError:
                        pass
                if func_degree is None:
                    degree_match = re.search(r'_(\d+)$', mof_name_base)
                    if degree_match:
                        try:
                            func_degree = float(degree_match.group(1))
                        except ValueError:
                            pass
                
                func_mof = FunctionalizedMOFEntity(
                    func_mof_id=func_mof_id,
                    parent_csd_code=parent_csd,
                    canonical_name=mof_name_base,
                    functionalization_type="AmineFunctionalization",
                    functional_group_name=func_group_name,
                    functionalization_degree=func_degree,
                    data_sources=["OpenDAC25"],
                )
                func_mofs.append(func_mof)
        
        print(f"Extracted {len(func_mofs)} functionalized MOFs from OpenDAC25")
        return func_mofs
    
    def extract_functionalizations(self) -> List[FunctionalizationEntity]:
        """
        Extract functionalization entities from mof_analysis_final.json.
        """
        data = self._load_data()
        functionalizations = []
        
        for parent_csd, func_groups in data.items():
            if not isinstance(func_groups, dict):
                continue
            
            for amine_code, func_data in func_groups.items():
                if not isinstance(func_data, dict):
                    continue
                
                mof_name_base = func_data.get('mof_name_base', '').strip()
                if not mof_name_base:
                    continue

                func_mof_id = mof_name_base
                func_group_name = self.AMINES.get(amine_code, {}).get("name", amine_code)
                
                # Parse degree
                func_degree = None
                degree_match = re.search(r'_(\d+\.\d+)_', mof_name_base)
                if degree_match:
                    try: func_degree = float(degree_match.group(1))
                    except: pass
                if func_degree is None:
                    degree_match = re.search(r'_(\d+)$', mof_name_base)
                    if degree_match:
                        try: func_degree = float(degree_match.group(1))
                        except: pass
                
                functionalization = FunctionalizationEntity(
                    functionalization_id=f"FUNC_{func_mof_id}",
                    func_mof_id=func_mof_id,
                    functionalization_type="AmineFunctionalization",
                    functional_group_name=func_group_name,
                    functional_group_id=f"CHEM_{amine_code}" if amine_code in self.AMINES else f"CHEM_{func_group_name}", 
                    functionalization_degree=func_degree,
                    data_sources=["OpenDAC25"],
                )
                functionalizations.append(functionalization)
        
        print(f"Extracted {len(functionalizations)} functionalizations from OpenDAC25")
        return functionalizations
    
    def extract_properties(self) -> List[PropertyEntity]:
        """
        Extract property entities (binding energies) from mof_analysis_final.json.
        """
        data = self._load_data()
        properties = []
        
        for parent_csd, func_groups in data.items():
            if not isinstance(func_groups, dict):
                continue
            
            for amine_code, func_data in func_groups.items():
                if not isinstance(func_data, dict):
                    continue
                
                mof_name_base = func_data.get('mof_name_base', '').strip()
                if not mof_name_base:
                    continue
                    
                func_mof_id = mof_name_base
                energies = func_data.get('energies', {})
                
                if not isinstance(energies, dict):
                    continue
                
                for adsorbate, energy_value in energies.items():
                    try:
                        value = float(energy_value)
                        prop = PropertyEntity(
                            property_id=f"PROP_{func_mof_id}_binding_{adsorbate}",
                            mof_id=func_mof_id,
                            property_name=f"Binding Energy {adsorbate}",
                            property_type="ComputationalProperty",
                            value=value,
                            units="eV",
                            data_source="OpenDAC25",
                        )
                        properties.append(prop)
                    except (ValueError, TypeError):
                        pass
        
        print(f"Extracted {len(properties)} properties from OpenDAC25")
        return properties
    
    def extract_parent_mofs(self) -> List[MOFEntity]:
        """
        Extract parent MOF entities from mof_analysis_final.json.
        """
        data = self._load_data()
        parent_mofs = {}
        
        for parent_csd, func_groups in data.items():
            if not isinstance(func_groups, dict):
                continue
            
            if parent_csd not in parent_mofs:
                mof = MOFEntity(
                    mof_id=parent_csd,
                    canonical_name=parent_csd,
                    all_names=[parent_csd],
                    csd_code=parent_csd,
                    is_experimental=True,
                    data_sources=["OpenDAC25_Parent"],
                )
                parent_mofs[parent_csd] = mof
        
        result = list(parent_mofs.values())
        print(f"Extracted {len(result)} parent MOFs from OpenDAC25")
        return result
    
    def extract_functional_groups(self) -> List[ChemicalEntity]:
        """
        Extract functional group chemical entities.
        """
        # We extract primarily from the AMINES dict as those are the canonical definitions used
        functional_groups = []
        
        for code, info in self.AMINES.items():
            chemical = ChemicalEntity(
                chemical_id=f"CHEM_{code}",
                canonical_name=info['name'],
                all_names=[info['name'], code],
                data_sources=["OpenDAC25"],
            )
            functional_groups.append(chemical)
            
        print(f"Extracted {len(functional_groups)} unique functional groups (amines) from OpenDAC25")
        return functional_groups
