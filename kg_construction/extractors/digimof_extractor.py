"""
DigiMOF Data Extractor

Extracts entities from DigiMOF data sources:
- Organized/Abstracts_with_Synthesis_Method.csv: MOFs, Abstracts, Synthesis (method, yield, conditions)
- Organized/LinkerData.csv: Linkers
- Organized/LinkersandProperties.csv: MOFs (with topology), Linkers, Metal Clusters, Properties 
  (LCD, PLD, Density, ASA, NASA, OMS, Void fractions), Topologies
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import json
import hashlib

from ..datamodels.entitymodels import (
    MOFEntity,
    LinkerEntity,
    MetalClusterEntity,
    SynthesisProcessEntity,
    SynthesisConditionEntity,
    SynthesisProcedureEntity,
    AbstractEntity,
    PropertyEntity,
    TopologyEntity,
)


class DigiMOFExtractor:
    """Extractor for DigiMOF data sources."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the DigiMOF extractor.
        
        Args:
            data_dir: Path to DigiMOF data directory. If None, uses default path.
        """
        if data_dir is None:
            # Default to data/raw/DigiMOF relative to project root
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data" / "raw" / "DigiMOF"
        
        self.data_dir = Path(data_dir)
        
        # Data file paths
        self.abstracts_file = self.data_dir / "Organized" / "Abstracts_with_Synthesis_Method.csv"
        self.linker_data_file = self.data_dir / "Organized" / "LinkerData.csv"
        self.linkers_props_file = self.data_dir / "Organized" / "LinkersandProperties.csv"
        self.excel_file_1 = self.data_dir / "cm3c00788_si_001.xlsx"
        self.excel_file_2 = self.data_dir / "cm3c00788_si_002.xlsx"
    
    def _generate_linker_id(self, smiles: str) -> str:
        """Generate linker ID from SMILES using hash."""
        if not smiles:
            return ""
        # Use first 12 chars of MD5 hash
        return hashlib.md5(smiles.encode()).hexdigest()[:12]
    
    def extract_mofs(self) -> List[MOFEntity]:
        """
        Extract MOF entities from Excel files and abstracts.
        
        Returns:
            List of MOFEntity instances
        """
        mofs = []
        mof_dict = {}  # Track by CSD code to merge data
        
        # Extract from abstracts (has CSD codes and names)
        if self.abstracts_file.exists():
            df = pd.read_csv(self.abstracts_file)
            for _, row in df.iterrows():
                csd_code = str(row.get('Ref Code', '')).strip()
                if not csd_code or csd_code == 'nan':
                    continue
                
                mof_name = str(row.get('MOF Name', '')).strip()
                if not mof_name or mof_name == 'nan':
                    mof_name = csd_code
                
                # Parse names
                all_names = [name.strip() for name in mof_name.split('<|>') if name.strip()]
                canonical_name = all_names[0] if all_names else csd_code
                
                # Get reference DOI
                reference = str(row.get('Reference', '')).strip()
                if reference == 'nan':
                    reference = None
                
                if csd_code not in mof_dict:
                    mof = MOFEntity(
                        mof_id=csd_code,
                        canonical_name=canonical_name,
                        all_names=all_names,
                        csd_code=csd_code,
                        is_experimental=True,
                        data_sources=["DigiMOF"],
                    )
                    if reference:
                        mof.other_ids['doi'] = reference
                    mof_dict[csd_code] = mof
                else:
                    # Merge names
                    existing = mof_dict[csd_code]
                    existing.all_names.extend([n for n in all_names if n not in existing.all_names])
                    if reference and 'doi' not in existing.other_ids:
                        existing.other_ids['doi'] = reference
        
        # Extract from LinkersandProperties (has topology info)
        if self.linkers_props_file.exists():
            df = pd.read_csv(self.linkers_props_file)
            # Skip header row if it's a count row
            df = df[df['Refcode'] != 'Counts']
            
            for _, row in df.iterrows():
                csd_code = str(row.get('Refcode', '')).strip()
                if not csd_code or csd_code == 'nan':
                    continue
                
                # Get topology
                topology = str(row.get('CN_Topology', '')).strip()
                if topology == 'nan' or not topology:
                    topology = None
                
                if csd_code not in mof_dict:
                    mof = MOFEntity(
                        mof_id=csd_code,
                        canonical_name=csd_code,
                        all_names=[csd_code],
                        csd_code=csd_code,
                        is_experimental=True,
                        topology=topology,
                        data_sources=["DigiMOF"],
                    )
                    mof_dict[csd_code] = mof
                else:
                    # Update topology if not set
                    if not mof_dict[csd_code].topology and topology:
                        mof_dict[csd_code].topology = topology
        
        mofs = list(mof_dict.values())
        print(f"Extracted {len(mofs)} MOFs from DigiMOF")
        return mofs
    
    def extract_linkers(self) -> List[LinkerEntity]:
        """
        Extract linker entities from LinkerData.csv.
        
        Returns:
            List of LinkerEntity instances
        """
        if not self.linker_data_file.exists():
            print(f"Warning: {self.linker_data_file} not found")
            return []
        
        df = pd.read_csv(self.linker_data_file)
        linkers = []
        linker_dict = {}  # Track by name to deduplicate
        
        for _, row in df.iterrows():
            # Extract linker names from Linker 1-9 columns
            for i in range(1, 10):
                linker_col = f'Linker {i}'
                if linker_col not in df.columns:
                    continue
                
                linker_name = str(row.get(linker_col, '')).strip()
                if not linker_name or linker_name == 'nan' or linker_name == '':
                    continue
                
                # Use name as canonical name (will be converted to SMILES later)
                if linker_name not in linker_dict:
                    linker = LinkerEntity(
                        linker_id=f"LINKER_{self._generate_linker_id(linker_name)}",
                        canonical_name=linker_name,
                        smiles="",  # Will be filled during standardization
                        canonical_smiles="",  # Will be filled during standardization
                        all_names=[linker_name],
                        data_sources=["DigiMOF"],
                    )
                    linker_dict[linker_name] = linker
                else:
                    # Add to all_names if not already there
                    existing = linker_dict[linker_name]
                    if linker_name not in existing.all_names:
                        existing.all_names.append(linker_name)
        
        linkers = list(linker_dict.values())
        print(f"Extracted {len(linkers)} linkers from DigiMOF")
        return linkers
    
    def extract_metal_clusters(self) -> Tuple[List[MetalClusterEntity], List[Dict]]:
        """
        Extract metal cluster entities from Excel files.
        
        Returns:
            Tuple[List[MetalClusterEntity], List[Dict]]: (Entities, Relationships)
        """
        clusters = []
        cluster_dict = {}  # Track by metal elements
        relationships = []
        
        # Extract from LinkersandProperties (has metal info)
        if self.linkers_props_file.exists():
            df = pd.read_csv(self.linkers_props_file)
            df = df[df['Refcode'] != 'Counts']
            
            for _, row in df.iterrows():
                csd_code = str(row.get('Refcode', '')).strip()
                if not csd_code or csd_code == 'nan':
                    continue
                    
                metals_str = str(row.get('Metal', '')).strip()
                if not metals_str or metals_str == 'nan' or metals_str == '0':
                    continue
                
                # Parse metals (could be comma-separated or single)
                metal_elements = [m.strip() for m in metals_str.split(',') if m.strip()]
                if not metal_elements:
                    continue
                
                # Create cluster ID from metals
                cluster_id = f"CLUSTER_{'_'.join(sorted(metal_elements))}"
                
                # Create relationship
                relationships.append({
                    'mof_id': csd_code,
                    'metal_node_id': cluster_id
                })
                
                if cluster_id not in cluster_dict:
                    cluster = MetalClusterEntity(
                        cluster_id=cluster_id,
                        metal_elements=metal_elements,
                        data_sources=["DigiMOF"],
                    )
                    cluster_dict[cluster_id] = cluster
        
        clusters = list(cluster_dict.values())
        print(f"Extracted {len(clusters)} metal clusters and {len(relationships)} relationships from DigiMOF")
        return clusters, relationships
    
    def extract_topologies(self) -> Tuple[List[TopologyEntity], List[Dict]]:
        """
        Extract topology entities from LinkersandProperties.csv.
        
        Returns:
            Tuple[List[TopologyEntity], List[Dict]]: (Entities, Relationships)
        """
        if not self.linkers_props_file.exists():
            print(f"Warning: {self.linkers_props_file} not found")
            return [], []
        
        df = pd.read_csv(self.linkers_props_file)
        df = df[df['Refcode'] != 'Counts']
        
        topologies = {}
        relationships = []
        
        for _, row in df.iterrows():
            csd_code = str(row.get('Refcode', '')).strip()
            topology = str(row.get('CN_Topology', '')).strip()
            
            if not topology or topology == 'nan' or topology == '0':
                continue
            
            # Skip UNKNOWN topologies
            if topology == 'UNKNOWN':
                continue
            
            topology_id = topology.lower()
            
            # Create relationship if valid MOF ID
            if csd_code and csd_code != 'nan':
                relationships.append({
                    'mof_id': csd_code,
                    'topology_id': topology_id
                })
            
            if topology_id not in topologies:
                topologies[topology_id] = TopologyEntity(
                    topology_id=topology_id,
                    topology_name=topology,
                    data_sources=["DigiMOF"]
                )
        
        result = list(topologies.values())
        print(f"Extracted {len(result)} unique valid topologies and {len(relationships)} relationships from DigiMOF")
        return result, relationships
    
    def extract_properties(self) -> List[PropertyEntity]:
        """
        Extract property entities from LinkersandProperties.csv.
        
        Returns:
            List of PropertyEntity instances
        """
        if not self.linkers_props_file.exists():
            print(f"Warning: {self.linkers_props_file} not found")
            return []
        
        df = pd.read_csv(self.linkers_props_file)
        df = df[df['Refcode'] != 'Counts']
        
        properties = []
        
        # Property columns to extract
        property_columns = {
            'LCD (Å)': ('Largest cavity diameter', 'Å', 'StructuralProperty'),
            'PLD (Å)': ('Pore limiting diameter', 'Å', 'StructuralProperty'),
            'Density (kg/m3)': ('Density', 'kg/m³', 'PhysicalProperty'),
            'ASA (m2/g)': ('Accessible surface area', 'm²/g', 'ComputationalProperty'),
            'NASA (m2/g)': ('Non-accessible surface area', 'm²/g', 'ComputationalProperty'),
            'No. OMS': ('Number of open metal sites', 'count', 'StructuralProperty'),
            'Void fraction (POAV)': ('Void fraction (accessible)', 'dimensionless', 'ComputationalProperty'),
            'Void fraction (PONAV)': ('Void fraction (non-accessible)', 'dimensionless', 'ComputationalProperty'),
        }
        
        for _, row in df.iterrows():
            csd_code = str(row.get('Refcode', '')).strip()
            if not csd_code or csd_code == 'nan':
                continue
            
            # Extract each property
            for col_name, (prop_name, units, prop_type) in property_columns.items():
                if col_name not in df.columns:
                    continue
                
                value = row.get(col_name)
                if pd.isna(value) or value == '' or value == 0:
                    continue
                
                try:
                    value_float = float(value)
                    # Skip zero values (often means not available)
                    if value_float == 0:
                        continue
                except (ValueError, TypeError):
                    continue
                
                # Create property entity
                prop_id = f"PROP_{csd_code}_{col_name.replace(' ', '_').replace('(', '').replace(')', '').replace('.', '_')}"
                prop = PropertyEntity(
                    property_id=prop_id,
                    mof_id=csd_code,
                    property_name=prop_name,
                    property_type=prop_type,
                    value=value_float,
                    units=units,
                    data_source="DigiMOF",
                )
                properties.append(prop)
        
        print(f"Extracted {len(properties)} properties from DigiMOF")
        return properties
    
    def extract_synthesis(self) -> List[SynthesisProcessEntity]:
        """
        Extract synthesis process entities from Abstracts_with_Synthesis_Method.csv.
        
        Returns:
            List of SynthesisProcessEntity instances
        """
        if not self.abstracts_file.exists():
            print(f"Warning: {self.abstracts_file} not found")
            return []
        
        df = pd.read_csv(self.abstracts_file)
        synthesis_list = []
        
        for _, row in df.iterrows():
            csd_code = str(row.get('Ref Code', '')).strip()
            if not csd_code or csd_code == 'nan':
                continue
            
            # Get synthesis method
            method = str(row.get('mentioned_Synthesis_Method', '')).strip()
            if method == 'nan':
                method = None
            
            # Get yield
            yield_str = str(row.get('synthesis_yield_percent', '')).strip()
            yield_pct = None
            if yield_str and yield_str != 'nan':
                try:
                    yield_pct = float(yield_str)
                except (ValueError, TypeError):
                    pass
            
            # Create synthesis process
            synthesis = SynthesisProcessEntity(
                synthesis_id=f"SYN_{csd_code}",
                mof_id=csd_code,
                method=method,
                yield_percent=yield_pct,
                data_source="DigiMOF",
            )
            synthesis_list.append(synthesis)
        
        print(f"Extracted {len(synthesis_list)} synthesis processes from DigiMOF")
        return synthesis_list
    
    def extract_synthesis_conditions(self) -> List[SynthesisConditionEntity]:
        """
        Extract synthesis condition entities from Abstracts_with_Synthesis_Method.csv.
        
        Returns:
            List of SynthesisConditionEntity instances
        """
        if not self.abstracts_file.exists():
            print(f"Warning: {self.abstracts_file} not found")
            return []
        
        df = pd.read_csv(self.abstracts_file)
        conditions = []
        
        for _, row in df.iterrows():
            csd_code = str(row.get('Ref Code', '')).strip()
            if not csd_code or csd_code == 'nan':
                continue
            
            # Extract temperature, time, and pressure
            temp_str = str(row.get('synthesis_temperature_c', '')).strip()
            time_str = str(row.get('synthesis_time_hours', '')).strip()
            pressure_str = str(row.get('synthesis_pressure_bar', '')).strip()
            
            temp = None
            time_hours = None
            pressure = None
            
            if temp_str and temp_str != 'nan':
                try:
                    temp = float(temp_str)
                except (ValueError, TypeError):
                    pass
            
            if time_str and time_str != 'nan':
                try:
                    time_hours = float(time_str)
                except (ValueError, TypeError):
                    pass
            
            if pressure_str and pressure_str != 'nan':
                try:
                    pressure = float(pressure_str)
                except (ValueError, TypeError):
                    pass
            
            # Only create condition if at least one value is present
            if temp is not None or time_hours is not None or pressure is not None:
                condition = SynthesisConditionEntity(
                    condition_id=f"COND_{csd_code}",
                    synthesis_id=f"SYN_{csd_code}",
                    temperature_c=temp,
                    time_hours=time_hours,
                    pressure_bar=pressure,
                )
                conditions.append(condition)
        
        print(f"Extracted {len(conditions)} synthesis conditions from DigiMOF")
        return conditions
    
    def extract_abstracts(self) -> List[AbstractEntity]:
        """
        Extract abstract entities from Abstracts_with_Synthesis_Method.csv.
        
        Returns:
            List of AbstractEntity instances
        """
        if not self.abstracts_file.exists():
            print(f"Warning: {self.abstracts_file} not found")
            return []
        
        df = pd.read_csv(self.abstracts_file)
        abstracts = []
        
        for _, row in df.iterrows():
            csd_code = str(row.get('Ref Code', '')).strip()
            if not csd_code or csd_code == 'nan':
                continue
            
            abstract_text = str(row.get('Abstract', '')).strip()
            if not abstract_text or abstract_text == 'nan':
                continue
            
            # Get title (from MOF Name)
            title = str(row.get('MOF Name', '')).strip()
            if title == 'nan':
                title = None
            
            # Get authors
            authors = str(row.get('authors', '')).strip()
            if authors == 'nan':
                authors = None
            
            # Get reference DOI
            reference = str(row.get('Reference', '')).strip()
            if reference == 'nan':
                reference = None
            
            abstract = AbstractEntity(
                abstract_id=f"ABSTRACT_{csd_code}",
                mof_id=csd_code,
                title=title,
                abstract_text=abstract_text,
                authors=authors,
                doi=reference,
                data_source="DigiMOF",
            )
            abstracts.append(abstract)
        
        print(f"Extracted {len(abstracts)} abstracts from DigiMOF")
        return abstracts
