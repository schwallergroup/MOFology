"""
MaterialsProject Data Extractor

Extracts entities from MaterialsProject_cleaned.json:
- MOFs (hypothetical and experimental, with topology, formula, identifiers)
- Properties (band gaps, PLD, LCD, density, volume, magnetic properties, charges, spins)
- Linkers (from smilesLinkers)
- Metal Clusters (from smilesNodes)
- Topologies (from topology field or MOFid)
- Space Groups (from spacegroup field)
- Crystal Systems (inferred from space groups)
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import hashlib
import re
from ..datamodels.entitymodels import (
    MOFEntity,
    PropertyEntity,
    LinkerEntity,
    MetalClusterEntity,
    TopologyEntity,
    SpaceGroupEntity,
    CrystalSystemEntity,
)


class MaterialsProjectExtractor:
    """Extractor for MaterialsProject data sources."""
    
    def __init__(self, data_file: Optional[Path] = None):
        """
        Initialize the MaterialsProject extractor.
        
        Args:
            data_file: Path to MaterialsProject_cleaned.json. If None, uses default path.
        """
        if data_file is None:
            # Default to data/raw/MaterialsProjQMOF/MaterialsProject_cleaned.json
            project_root = Path(__file__).parent.parent.parent
            data_file = project_root / "data" / "raw" / "MaterialsProjQMOF" / "MaterialsProject_cleaned.json"
        
        self.data_file = Path(data_file)
    
    def _generate_linker_id(self, smiles: str) -> str:
        """Generate linker ID from SMILES using hash."""
        if not smiles:
            return ""
        return hashlib.md5(smiles.encode()).hexdigest()[:12]
    
    def extract_mofs(self) -> List[MOFEntity]:
        """
        Extract MOF entities from MaterialsProject_cleaned.json.
        
        Returns:
            List of MOFEntity instances
        """
        if not self.data_file.exists():
            print(f"Warning: {self.data_file} not found")
            return []
        
        with open(self.data_file, 'r') as f:
            data = json.load(f)
        
        mofs = []
        
        for entry in data:
            identifiers = entry.get('identifiers', {})
            mp_id = identifiers.get('mp_id', '').strip()
            if not mp_id:
                continue
            
            # Get name
            name = identifiers.get('name', '').strip()
            if not name:
                name = mp_id
            
            # Get raw data
            raw_data = entry.get('raw_data', {})
            data_obj = raw_data.get('data', {})
            
            if not data_obj:
                continue
            
            # Check if it has CSD code (some entries are from CSD)
            csd_code = data_obj.get('csdRefcode')
            if csd_code:
                csd_code = str(csd_code).strip()
                if not csd_code or csd_code == 'nan':
                    csd_code = None
            else:
                csd_code = None
            
            # Get formula
            formula = data_obj.get('reducedFormula')
            if formula:
                formula = str(formula).strip()
                if not formula or formula == 'nan':
                    formula = None
            else:
                formula = None
            
            # Get MOFid if available (lowercase 'mofid' in data)
            mofid = data_obj.get('mofid')
            if mofid:
                mofid = str(mofid).strip()
                if not mofid or mofid == 'nan':
                    mofid = None
            else:
                mofid = None
            
            # Get topology from topology field
            topology = data_obj.get('topology')
            if topology:
                topology = str(topology).strip()
                if not topology or topology == 'nan':
                    topology = None
            else:
                # Try to extract from mofid if topology field not available
                if mofid:
                    match = re.search(r'MOFid-v1\.([^.]+)\.', mofid)
                    if match:
                        topology_code = match.group(1)
                        if topology_code not in ['UNKNOWN', 'ERROR']:
                            topology = topology_code
            
            # Determine if experimental (has CSD code) or hypothetical
            is_experimental = csd_code is not None
            
            # Use CSD code as mof_id if available, otherwise MP ID
            mof_id = csd_code if csd_code else mp_id
            
            mof = MOFEntity(
                mof_id=mof_id,
                canonical_name=name,
                all_names=[name],
                formula=formula,
                csd_code=csd_code,
                mp_id=mp_id,
                mofid=mofid,
                is_experimental=is_experimental,
                topology=topology,
                data_sources=["MaterialsProject"],
            )
            
            # Add DOI if available
            doi = data_obj.get('doi')
            if doi:
                doi = str(doi).strip()
                if doi and doi != 'nan':
                    mof.other_ids['doi'] = doi
            
            # Add MOFkey if available
            mofkey = data_obj.get('mofkey')
            if mofkey:
                mofkey = str(mofkey).strip()
                if mofkey and mofkey != 'nan':
                    mof.other_ids['mofkey'] = mofkey
            
            # Add source if available
            source = data_obj.get('source')
            if source:
                source = str(source).strip()
                if source and source != 'nan':
                    mof.other_ids['source'] = source
            
            # Add synthesized flag if available
            synthesized = data_obj.get('synthesized')
            if synthesized:
                synthesized_str = str(synthesized).strip()
                if synthesized_str and synthesized_str != 'nan':
                    mof.other_ids['synthesized'] = synthesized_str
            
            mofs.append(mof)
        
        print(f"Extracted {len(mofs)} MOFs from MaterialsProject")
        return mofs
    
    def extract_properties(self) -> List[PropertyEntity]:
        """
        Extract property entities from MaterialsProject_cleaned.json.
        
        Returns:
            List of PropertyEntity instances
        """
        if not self.data_file.exists():
            print(f"Warning: {self.data_file} not found")
            return []
        
        with open(self.data_file, 'r') as f:
            data = json.load(f)
        
        properties = []
        
        # Property mappings from raw_data.data
        property_mappings = {
            'EgPBE': ('Band gap (PBE)', 'eV', 'ComputationalProperty'),
            'EgHLE17': ('Band gap (HLE17)', 'eV', 'ComputationalProperty'),
            'EgHSE06star': ('Band gap (HSE06*)', 'eV', 'ComputationalProperty'),
            'EgHSE06': ('Band gap (HSE06)', 'eV', 'ComputationalProperty'),
            'pld': ('Pore limiting diameter', 'Å', 'ComputationalProperty'),
            'lcd': ('Largest cavity diameter', 'Å', 'ComputationalProperty'),
            'density': ('Density', 'g/cm³', 'PhysicalProperty'),
            'volume': ('Unit cell volume', 'Å³', 'StructuralProperty'),
            'ASA': ('Accessible surface area', 'm²/g', 'ComputationalProperty'),
            'NASA': ('Non-accessible surface area', 'm²/g', 'ComputationalProperty'),
            'natoms': ('Number of atoms', 'count', 'StructuralProperty'),
            'netPBEMagmom': ('Net magnetic moment (PBE)', 'μB', 'PhysicalProperty'),
            'maxPBEChargeDDEC': ('Max charge (DDEC)', 'e', 'ComputationalProperty'),
            'maxPBEChargeCM5': ('Max charge (CM5)', 'e', 'ComputationalProperty'),
            'maxPBEMagmom': ('Max magnetic moment (PBE)', 'μB', 'PhysicalProperty'),
            'maxPBESpinDDEC': ('Max spin (DDEC)', 'dimensionless', 'ComputationalProperty'),
            'spacegroupNumber': ('Space group number', 'number', 'StructuralProperty'),
        }
        
        for entry in data:
            identifiers = entry.get('identifiers', {})
            mp_id = identifiers.get('mp_id', '').strip()
            if not mp_id:
                continue
            
            raw_data = entry.get('raw_data', {})
            data_obj = raw_data.get('data', {})
            
            if not data_obj:
                continue
            
            # Get CSD code if available, otherwise use MP ID
            csd_code = data_obj.get('csdRefcode')
            if csd_code:
                csd_code = str(csd_code).strip()
                if not csd_code or csd_code == 'nan':
                    csd_code = None
            else:
                csd_code = None
            mof_id = csd_code if csd_code else mp_id
            
            # Extract properties
            for key, (prop_name, units, prop_type) in property_mappings.items():
                prop_data = data_obj.get(key)
                if prop_data is None:
                    continue
                
                # Handle nested structure (value, display, error, unit)
                if isinstance(prop_data, dict):
                    value = prop_data.get('value')
                else:
                    value = prop_data
                
                if value is None:
                    continue
                
                try:
                    value_float = float(value)
                except (ValueError, TypeError):
                    continue
                
                prop = PropertyEntity(
                    property_id=f"PROP_{mof_id}_{key}",
                    mof_id=mof_id,
                    property_name=prop_name,
                    property_type=prop_type,
                    value=value_float,
                    units=units,
                    data_source="MaterialsProject",
                )
                properties.append(prop)
        
        print(f"Extracted {len(properties)} properties from MaterialsProject")
        return properties
    
    def extract_linkers(self) -> List[LinkerEntity]:
        """
        Extract linker entities from MaterialsProject_cleaned.json.
        
        Returns:
            List of LinkerEntity instances
        """
        if not self.data_file.exists():
            print(f"Warning: {self.data_file} not found")
            return []
        
        with open(self.data_file, 'r') as f:
            data = json.load(f)
        
        linkers = []
        linker_dict = {}  # Track by SMILES to deduplicate
        
        for entry in data:
            raw_data = entry.get('raw_data', {})
            data_obj = raw_data.get('data', {})
            
            if not data_obj:
                continue
            
            smiles_linkers = data_obj.get('smilesLinkers')
            if not smiles_linkers:
                continue
            smiles_linkers = str(smiles_linkers).strip()
            if not smiles_linkers or smiles_linkers == 'nan':
                continue
            
            # Split by comma if multiple linkers
            linker_smiles_list = [s.strip() for s in smiles_linkers.split(',') if s.strip()]
            
            for smiles in linker_smiles_list:
                if not smiles or smiles in linker_dict:
                    continue
                
                linker_id = f"LINKER_{self._generate_linker_id(smiles)}"
                
                linker = LinkerEntity(
                    linker_id=linker_id,
                    canonical_name=smiles,  # Use SMILES as name for now
                    smiles=smiles,
                    canonical_smiles=smiles,  # Assume already canonical
                    all_names=[smiles],
                    data_sources=["MaterialsProject"],
                )
                linker_dict[smiles] = linker
        
        linkers = list(linker_dict.values())
        print(f"Extracted {len(linkers)} linkers from MaterialsProject")
        return linkers
    
    def extract_metal_clusters(self) -> Tuple[List[MetalClusterEntity], List[Dict]]:
        """
        Extract metal cluster entities from MaterialsProject_cleaned.json.
        
        Returns:
            Tuple[List[MetalClusterEntity], List[Dict]]: (Entities, Relationships)
        """
        if not self.data_file.exists():
            print(f"Warning: {self.data_file} not found")
            return [], []
        
        with open(self.data_file, 'r') as f:
            data = json.load(f)
        
        clusters = []
        cluster_dict = {}  # Track by SMILES to deduplicate
        relationships = []
        
        for entry in data:
            identifiers = entry.get('identifiers', {})
            mp_id = identifiers.get('mp_id', '').strip()
            if not mp_id:
                continue
                
            # Determine MOF ID (CSD or MP ID)
            raw_data = entry.get('raw_data', {})
            data_obj = raw_data.get('data', {})
            csd_code = data_obj.get('csdRefcode') if data_obj else None
            if csd_code:
                csd_code = str(csd_code).strip()
                if not csd_code or csd_code == 'nan':
                    csd_code = None
            mof_id = csd_code if csd_code else mp_id
            
            if not data_obj:
                continue
            
            smiles_nodes = data_obj.get('smilesNodes')
            if not smiles_nodes:
                continue
            smiles_nodes = str(smiles_nodes).strip()
            if not smiles_nodes or smiles_nodes == 'nan':
                continue
            
            # Parse metal elements from SMILES (e.g., "[La]", "[Cu]", "O,[Ba],[Cu]")
            # Extract elements in brackets
            elements = re.findall(r'\[([A-Z][a-z]?)\]', smiles_nodes)
            
            if not elements:
                continue
            
            # Create cluster ID from sorted elements
            cluster_id = f"CLUSTER_{'_'.join(sorted(set(elements)))}"
            
            # Create relationship
            relationships.append({
                'mof_id': mof_id,
                'metal_node_id': cluster_id
            })
            
            if cluster_id not in cluster_dict:
                cluster = MetalClusterEntity(
                    cluster_id=cluster_id,
                    metal_elements=list(set(elements)),
                    data_sources=["MaterialsProject"],
                )
                cluster_dict[cluster_id] = cluster
        
        clusters = list(cluster_dict.values())
        print(f"Extracted {len(clusters)} metal clusters and {len(relationships)} relationships from MaterialsProject")
        return clusters, relationships
    
    def extract_topologies(self) -> List[TopologyEntity]:
        """
        Extract topology entities from MaterialsProject_cleaned.json.
        Note: Topology relationships are currently handled via MOFEntity.topology field in Normalizer.
        
        Returns:
            List of TopologyEntity instances
        """
        if not self.data_file.exists():
            print(f"Warning: {self.data_file} not found")
            return []
        
        with open(self.data_file, 'r') as f:
            data = json.load(f)
        
        topologies = {}
        
        for entry in data:
            raw_data = entry.get('raw_data', {})
            data_obj = raw_data.get('data', {})
            
            if not data_obj:
                continue
            
            # Get topology from topology field
            topology = data_obj.get('topology')
            if topology:
                topology = str(topology).strip()
                if topology and topology != 'nan':
                    topology_id = topology.lower()
                    if topology_id not in topologies:
                        topologies[topology_id] = TopologyEntity(
                            topology_id=topology_id,
                            topology_name=topology,
                            data_sources=["MaterialsProject"]
                        )
                    continue
            
            # Try to extract from mofid (lowercase in data)
            mofid = data_obj.get('mofid')
            if mofid:
                mofid = str(mofid).strip()
                if mofid and mofid != 'nan':
                    match = re.search(r'MOFid-v1\.([^.]+)\.', mofid)
                    if match:
                        topology_code = match.group(1)
                        if topology_code not in ['UNKNOWN', 'ERROR']:
                            topology_id = topology_code.lower()
                            if topology_id not in topologies:
                                topologies[topology_id] = TopologyEntity(
                                    topology_id=topology_id,
                                    topology_name=topology_code,
                                    data_sources=["MaterialsProject"]
                                )
        
        result = list(topologies.values())
        print(f"Extracted {len(result)} unique topologies from MaterialsProject")
        return result
    
    def extract_space_groups(self) -> Tuple[List[SpaceGroupEntity], List[Dict]]:
        """
        Extract space group entities from MaterialsProject_cleaned.json.
        
        Returns:
            Tuple[List[SpaceGroupEntity], List[Dict]]: (Entities, Relationships)
        """
        if not self.data_file.exists():
            print(f"Warning: {self.data_file} not found")
            return [], []
        
        with open(self.data_file, 'r') as f:
            data = json.load(f)
        
        space_groups = {}
        relationships = []
        
        for entry in data:
            identifiers = entry.get('identifiers', {})
            mp_id = identifiers.get('mp_id', '').strip()
            if not mp_id:
                continue
                
            # Determine MOF ID
            raw_data = entry.get('raw_data', {})
            data_obj = raw_data.get('data', {})
            csd_code = data_obj.get('csdRefcode') if data_obj else None
            if csd_code:
                csd_code = str(csd_code).strip()
                if not csd_code or csd_code == 'nan':
                    csd_code = None
            mof_id = csd_code if csd_code else mp_id
            
            if not data_obj:
                continue
            
            spacegroup = data_obj.get('spacegroup')
            if not spacegroup:
                continue
            
            spacegroup = str(spacegroup).strip()
            if not spacegroup or spacegroup == 'nan':
                continue
            
            space_group_id = spacegroup
            
            # Create relationship
            relationships.append({
                'mof_id': mof_id,
                'space_group_id': space_group_id
            })
            
            if space_group_id not in space_groups:
                sg = SpaceGroupEntity(
                    space_group_id=space_group_id,
                    space_group_name=spacegroup,
                    data_sources=["MaterialsProject"]
                )
                space_groups[space_group_id] = sg
        
        result = list(space_groups.values())
        print(f"Extracted {len(result)} unique space groups and {len(relationships)} relationships from MaterialsProject")
        return result, relationships
    
    def extract_crystal_systems(self) -> Tuple[List[CrystalSystemEntity], List[Dict]]:
        """
        Extract crystal system entities from MaterialsProject_cleaned.json.
        Uses spacegroupCrystal field from data, falls back to inference if needed.
        
        Returns:
            Tuple[List[CrystalSystemEntity], List[Dict]]: (Entities, Relationships)
        """
        if not self.data_file.exists():
            print(f"Warning: {self.data_file} not found")
            return [], []
        
        with open(self.data_file, 'r') as f:
            data = json.load(f)
        
        crystal_systems = {}
        relationships = []
        
        for entry in data:
            identifiers = entry.get('identifiers', {})
            mp_id = identifiers.get('mp_id', '').strip()
            if not mp_id:
                continue
                
            # Determine MOF ID
            raw_data = entry.get('raw_data', {})
            data_obj = raw_data.get('data', {})
            csd_code = data_obj.get('csdRefcode') if data_obj else None
            if csd_code:
                csd_code = str(csd_code).strip()
                if not csd_code or csd_code == 'nan':
                    csd_code = None
            mof_id = csd_code if csd_code else mp_id
            
            if not data_obj:
                continue
            
            crystal_system_id = None
            crystal_system_name = None
            
            # First try to get from spacegroupCrystal field (most reliable)
            crystal_system = data_obj.get('spacegroupCrystal')
            if crystal_system:
                crystal_system = str(crystal_system).strip().lower()
                if crystal_system and crystal_system != 'nan':
                    crystal_system_id = crystal_system
                    crystal_system_name = crystal_system.capitalize()
            
            # Fall back to inferring from spacegroup if spacegroupCrystal not available
            if not crystal_system_id:
                spacegroup = data_obj.get('spacegroup')
                if spacegroup:
                    spacegroup = str(spacegroup).strip()
                    if spacegroup and spacegroup != 'nan':
                        # Infer crystal system from spacegroup
                        sys_name = None
                        if spacegroup in ['P-1', 'P1']:
                            sys_name = 'triclinic'
                        elif spacegroup.startswith('C2') or spacegroup.startswith('P2'):
                            sys_name = 'monoclinic'
                        elif spacegroup.startswith('P3') or spacegroup.startswith('R3'):
                            sys_name = 'trigonal'
                        elif spacegroup.startswith('P4') or spacegroup.startswith('I4'):
                            sys_name = 'tetragonal'
                        elif spacegroup.startswith('P6'):
                            sys_name = 'hexagonal'
                        elif spacegroup in ['Fm-3m', 'Im-3m', 'Pm-3m', 'Fd-3m', 'Ia-3d']:
                            sys_name = 'cubic'
                        elif 'orthorhombic' in spacegroup.lower():
                            sys_name = 'orthorhombic'
                        
                        if sys_name:
                            crystal_system_id = sys_name.lower()
                            crystal_system_name = sys_name.capitalize()
            
            if crystal_system_id:
                # Create relationship
                relationships.append({
                    'mof_id': mof_id,
                    'crystal_system_id': crystal_system_id
                })
                
                if crystal_system_id not in crystal_systems:
                    cs = CrystalSystemEntity(
                        crystal_system_id=crystal_system_id,
                        crystal_system_name=crystal_system_name,
                        data_sources=["MaterialsProject"]
                    )
                    crystal_systems[crystal_system_id] = cs
        
        result = list(crystal_systems.values())
        print(f"Extracted {len(result)} crystal systems and {len(relationships)} relationships from MaterialsProject")
        return result, relationships
