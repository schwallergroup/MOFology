"""
ChemUnity Data Extractor

Extracts entities from ChemUnity data sources:
- MOF_names_and_CSD_codes.csv: MOF entities (with formula and topology from computational_properties.csv)
- computational_properties.csv: Computational properties, space groups, crystal systems, lattice parameters, 
  topologies, linkers (from SMILES), metal clusters
- all_experimental_properties.csv: Experimental properties
- applications.csv: MOF applications/capabilities
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import re
import hashlib

from ..datamodels.entitymodels import (
    MOFEntity,
    PropertyEntity,
    SpaceGroupEntity,
    CrystalSystemEntity,
    LatticeParameterEntity,
    CapabilityEntity,
    TopologyEntity,
    LinkerEntity,
    MetalClusterEntity,
)


class ChemUnityExtractor:
    """Extractor for ChemUnity data sources."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the ChemUnity extractor.
        
        Args:
            data_dir: Path to ChemUnity data directory. If None, uses default path.
        """
        if data_dir is None:
            # Default to data/raw/ChemUnity relative to project root
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data" / "raw" / "ChemUnity"
        
        self.data_dir = Path(data_dir)
        
        # Data file paths
        self.mof_names_file = self.data_dir / "MOF_names_and_CSD_codes.csv"
        self.computational_props_file = self.data_dir / "computational_properties.csv"
        self.experimental_props_file = self.data_dir / "all_experimental_properties.csv"
        self.applications_file = self.data_dir / "applications.csv"
    
    def extract_mofs(self) -> List[MOFEntity]:
        """
        Extract MOF entities from MOF_names_and_CSD_codes.csv.
        Joins with computational_properties.csv to get formula and topology.
        
        Returns:
            List of MOFEntity instances
        """
        if not self.mof_names_file.exists():
            print(f"Warning: {self.mof_names_file} not found")
            return []
        
        df_names = pd.read_csv(self.mof_names_file)
        
        # Load computational properties for formula and topology
        comp_props = {}
        if self.computational_props_file.exists():
            df_comp = pd.read_csv(self.computational_props_file)
            for _, row in df_comp.iterrows():
                csd_code = str(row.get('CSD code', '')).strip()
                if csd_code and csd_code != 'nan':
                    comp_props[csd_code] = row
        
        mofs = []
        
        for _, row in df_names.iterrows():
            # Get CSD code (primary identifier)
            csd_code = str(row.get('Ref Code', '')).strip()
            if not csd_code or csd_code == 'nan':
                continue
            
            # Get MOF name
            mof_name = str(row.get('MOF Name', '')).strip()
            if not mof_name or mof_name == 'nan':
                mof_name = csd_code  # Fallback to CSD code
            
            # Parse names - MOF Name can contain multiple names separated by <|>
            all_names = [name.strip() for name in mof_name.split('<|>') if name.strip()]
            canonical_name = all_names[0] if all_names else csd_code
            
            # Get reference DOI
            reference = str(row.get('Reference', '')).strip()
            if reference == 'nan':
                reference = None
            
            # Get formula and topology from computational properties
            formula = None
            topology = None
            if csd_code in comp_props:
                comp_row = comp_props[csd_code]
                
                # Extract formula
                formula_val = str(comp_row.get('Molecular formula', '')).strip()
                if formula_val and formula_val != 'nan':
                    formula = formula_val
                
                # Extract topology from MOFid
                mofid = str(comp_row.get('MOFid', '')).strip()
                if mofid and mofid != 'nan':
                    # Parse topology from MOFid (format: MOFid-v1.{topology}.cat{X})
                    match = re.search(r'MOFid-v1\.([^.]+)\.', mofid)
                    if match:
                        topology_code = match.group(1)
                        if topology_code not in ['UNKNOWN', 'ERROR']:
                            topology = topology_code
            
            # Create MOF entity
            mof = MOFEntity(
                mof_id=csd_code,  # Primary key = CSD code
                canonical_name=canonical_name,
                all_names=all_names,
                csd_code=csd_code,
                formula=formula,
                topology=topology,
                is_experimental=True,  # ChemUnity contains experimental MOFs
                data_sources=["ChemUnity"],
            )
            
            # Store reference in other_ids if available
            if reference:
                mof.other_ids['doi'] = reference
            
            mofs.append(mof)
        
        print(f"Extracted {len(mofs)} MOFs from ChemUnity")
        return mofs
    
    def extract_properties(self) -> List[PropertyEntity]:
        """
        Extract computational properties from computational_properties.csv.
        
        Returns:
            List of PropertyEntity instances
        """
        if not self.computational_props_file.exists():
            print(f"Warning: {self.computational_props_file} not found")
            return []
        
        df = pd.read_csv(self.computational_props_file)
        properties = []
        
        # Property columns to extract (excluding structural properties)
        property_columns = {
            'Band gap (eV)': ('Band gap', 'eV', 'ComputationalProperty'),
            'CO2 uptake at LP (mol/kg)': ('CO2 uptake at low pressure', 'mol/kg', 'ComputationalProperty'),
            'CO2 uptake at HP (mol/kg)': ('CO2 uptake at high pressure', 'mol/kg', 'ComputationalProperty'),
            'CH4 uptake at HP (mol/kg)': ('CH4 uptake at high pressure', 'mol/kg', 'ComputationalProperty'),
            'logKH_CO2': ('log(KH) CO2', 'dimensionless', 'ComputationalProperty'),
            'logKH_CH4': ('log(KH) CH4', 'dimensionless', 'ComputationalProperty'),
            'Largest included sphere diameter (A)': ('Largest included sphere diameter', 'Å', 'StructuralProperty'),
            'Total volumetric surface area (m^2/m^3)': ('Total volumetric surface area', 'm²/m³', 'ComputationalProperty'),
            'ASA [m^2/cm^3]': ('Accessible surface area', 'm²/cm³', 'ComputationalProperty'),
            'NASA [m^2/cm^3]': ('Non-accessible surface area', 'm²/cm³', 'ComputationalProperty'),
            'POAV [cm^3/g]': ('Pore volume (accessible)', 'cm³/g', 'ComputationalProperty'),
            'PONAV [cm^3/g]': ('Pore volume (non-accessible)', 'cm³/g', 'ComputationalProperty'),
            'POAVF': ('Pore volume fraction (accessible)', 'dimensionless', 'ComputationalProperty'),
            'PONAVF': ('Pore volume fraction (non-accessible)', 'dimensionless', 'ComputationalProperty'),
            'density [g/cm^3]': ('Density', 'g/cm³', 'PhysicalProperty'),
            'total_SA_gravimetric': ('Total surface area (gravimetric)', 'm²/g', 'ComputationalProperty'),
            'total_POV_volumetric': ('Total pore volume (volumetric)', 'cm³/cm³', 'ComputationalProperty'),
            'total_POV_gravimetric': ('Total pore volume (gravimetric)', 'cm³/g', 'ComputationalProperty'),
            'pure_CO2_kH': ('CO2 Henry constant', 'mol/(kg·Pa)', 'ComputationalProperty'),
            'pure_CO2_widomHOA': ('CO2 Widom insertion energy', 'kJ/mol', 'ComputationalProperty'),
            'pure_methane_kH': ('CH4 Henry constant', 'mol/(kg·Pa)', 'ComputationalProperty'),
            'pure_methane_widomHOA': ('CH4 Widom insertion energy', 'kJ/mol', 'ComputationalProperty'),
            'pure_uptake_methane_298.00_580000': ('CH4 uptake at 298K, 58 bar', 'mol/kg', 'ComputationalProperty'),
            'CH4DC': ('CH4 delivery capacity', 'cm³(STP)/cm³', 'ComputationalProperty'),
            'CH4HPSTP': ('CH4 high pressure storage', 'cm³(STP)/cm³', 'ComputationalProperty'),
        }
        
        for _, row in df.iterrows():
            csd_code = str(row.get('CSD code', '')).strip()
            if not csd_code or csd_code == 'nan':
                continue
            
            # Extract each property
            for col_name, (prop_name, units, prop_type) in property_columns.items():
                if col_name not in df.columns:
                    continue
                
                value = row.get(col_name)
                if pd.isna(value) or value == '':
                    continue
                
                try:
                    value_float = float(value)
                except (ValueError, TypeError):
                    continue
                
                # Create property entity
                prop = PropertyEntity(
                    property_id=f"PROP_{csd_code}_{col_name.replace(' ', '_').replace('(', '').replace(')', '').replace('.', '_')}",
                    mof_id=csd_code,  # Link to MOF via CSD code
                    property_name=prop_name,
                    property_type=prop_type,
                    value=value_float,
                    units=units,
                    data_source="ChemUnity_computational",
                )
                properties.append(prop)
        
        print(f"Extracted {len(properties)} computational properties from ChemUnity")
        return properties
    
    def extract_experimental_properties(self) -> List[PropertyEntity]:
        """
        Extract experimental properties from all_experimental_properties.csv.
        
        Returns:
            List of PropertyEntity instances
        """
        if not self.experimental_props_file.exists():
            print(f"Warning: {self.experimental_props_file} not found")
            return []
        
        df = pd.read_csv(self.experimental_props_file)
        properties = []
        
        for _, row in df.iterrows():
            csd_code = str(row.get('Ref Code', '')).strip()
            if not csd_code or csd_code == 'nan':
                continue
            
            property_name = str(row.get('Property', '')).strip()
            if not property_name or property_name == 'nan':
                continue
            
            # Get value
            value_str = str(row.get('Value', '')).strip()
            if value_str == 'nan' or not value_str:
                continue
            
            # Try to parse as float
            try:
                value = float(value_str)
            except (ValueError, TypeError):
                # Skip non-numeric values for now
                continue
            
            # Get units
            units = str(row.get('Units', '')).strip()
            if units == 'nan':
                units = None
            
            # Get conditions
            conditions = str(row.get('Condition', '')).strip()
            if conditions == 'nan':
                conditions = None
            
            # Determine property type based on property name
            prop_type = "PhysicalProperty"  # Default
            if any(keyword in property_name.lower() for keyword in ['crystal system', 'space group', 'unit cell']):
                prop_type = "StructuralProperty"
            
            # Create property entity
            prop = PropertyEntity(
                property_id=f"PROP_{csd_code}_exp_{property_name.replace(' ', '_').replace('(', '').replace(')', '')}",
                mof_id=csd_code,  # Link to MOF via CSD code
                property_name=property_name,
                property_type=prop_type,
                value=value,
                units=units,
                conditions=conditions,
                data_source="ChemUnity_experimental",
            )
            properties.append(prop)
        
        print(f"Extracted {len(properties)} experimental properties from ChemUnity")
        return properties
    
    def extract_space_groups(self) -> Tuple[List[SpaceGroupEntity], List[Dict]]:
        """
        Extract space group entities from computational_properties.csv.
        
        Returns:
            Tuple[List[SpaceGroupEntity], List[Dict]]: (Entities, Relationships)
        """
        if not self.computational_props_file.exists():
            print(f"Warning: {self.computational_props_file} not found")
            return [], []
        
        df = pd.read_csv(self.computational_props_file)
        space_groups = {}
        relationships = []
        
        for _, row in df.iterrows():
            csd_code = str(row.get('CSD code', '')).strip()
            space_group = str(row.get('Space group', '')).strip()
            
            if not space_group or space_group == 'nan':
                continue
            
            # Use space group as ID
            space_group_id = space_group
            
            # Create relationship if we have a valid MOF ID
            if csd_code and csd_code != 'nan':
                relationships.append({
                    'mof_id': csd_code,
                    'space_group_id': space_group_id
                })
            
            # Skip if already processed entity
            if space_group_id in space_groups:
                continue
            
            # Get crystal system (for additional info)
            crystal_system = str(row.get('Crystal system', '')).strip()
            if crystal_system == 'nan':
                crystal_system = None
            
            # Create space group entity
            sg = SpaceGroupEntity(
                space_group_id=space_group_id,
                space_group_name=space_group,
                crystal_system=crystal_system,
                data_sources=["ChemUnity"],
            )
            space_groups[space_group_id] = sg
        
        result = list(space_groups.values())
        print(f"Extracted {len(result)} unique space groups and {len(relationships)} relationships from ChemUnity")
        return result, relationships
    
    def extract_crystal_systems(self) -> Tuple[List[CrystalSystemEntity], List[Dict]]:
        """
        Extract crystal system entities from computational_properties.csv.
        
        Returns:
            Tuple[List[CrystalSystemEntity], List[Dict]]: (Entities, Relationships)
        """
        if not self.computational_props_file.exists():
            print(f"Warning: {self.computational_props_file} not found")
            return [], []
        
        df = pd.read_csv(self.computational_props_file)
        crystal_systems = {}
        relationships = []
        
        for _, row in df.iterrows():
            csd_code = str(row.get('CSD code', '')).strip()
            crystal_system = str(row.get('Crystal system', '')).strip()
            
            if not crystal_system or crystal_system == 'nan':
                continue
            
            # Normalize to lowercase for ID
            crystal_system_id = crystal_system.lower()
            
            # Create relationship if we have a valid MOF ID
            if csd_code and csd_code != 'nan':
                relationships.append({
                    'mof_id': csd_code,
                    'crystal_system_id': crystal_system_id
                })
            
            # Skip if already processed entity
            if crystal_system_id in crystal_systems:
                continue
            
            # Create crystal system entity
            cs = CrystalSystemEntity(
                crystal_system_id=crystal_system_id,
                crystal_system_name=crystal_system,
                data_sources=["ChemUnity"],
            )
            crystal_systems[crystal_system_id] = cs
        
        result = list(crystal_systems.values())
        print(f"Extracted {len(result)} unique crystal systems and {len(relationships)} relationships from ChemUnity")
        return result, relationships
    
    def extract_lattice_parameters(self) -> List[LatticeParameterEntity]:
        """
        Extract lattice parameter entities from computational_properties.csv.
        
        Returns:
            List of LatticeParameterEntity instances
        """
        if not self.computational_props_file.exists():
            print(f"Warning: {self.computational_props_file} not found")
            return []
        
        df = pd.read_csv(self.computational_props_file)
        lattice_params = []
        
        for _, row in df.iterrows():
            csd_code = str(row.get('CSD code', '')).strip()
            if not csd_code or csd_code == 'nan':
                continue
            
            # Extract unit cell parameters
            try:
                a = float(row.get('a', 0)) if pd.notna(row.get('a')) else None
                b = float(row.get('b', 0)) if pd.notna(row.get('b')) else None
                c = float(row.get('c', 0)) if pd.notna(row.get('c')) else None
                alpha = float(row.get('alpha', 0)) if pd.notna(row.get('alpha')) else None
                beta = float(row.get('beta', 0)) if pd.notna(row.get('beta')) else None
                gamma = float(row.get('gamma', 0)) if pd.notna(row.get('gamma')) else None
                volume = float(row.get('CellV [A^3]', 0)) if pd.notna(row.get('CellV [A^3]')) else None
            except (ValueError, TypeError):
                continue
            
            # Skip if no valid parameters
            if not any([a, b, c]):
                continue
            
            # Create lattice parameter entity
            lattice = LatticeParameterEntity(
                lattice_param_id=f"LATTICE_{csd_code}",
                mof_id=csd_code,  # Link to MOF via CSD code
                a=a,
                b=b,
                c=c,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                volume=volume,
                data_source="ChemUnity",
            )
            lattice_params.append(lattice)
        
        print(f"Extracted {len(lattice_params)} lattice parameter sets from ChemUnity")
        return lattice_params
    
    def extract_topologies(self) -> List[TopologyEntity]:
        """
        Extract topology entities from MOFid column in computational_properties.csv.
        
        Returns:
            List of TopologyEntity instances
        """
        if not self.computational_props_file.exists():
            print(f"Warning: {self.computational_props_file} not found")
            return []
        
        df = pd.read_csv(self.computational_props_file)
        topologies = {}
        
        for _, row in df.iterrows():
            mofid = str(row.get('MOFid', '')).strip()
            if not mofid or mofid == 'nan':
                continue
            
            # Parse topology from MOFid (format: MOFid-v1.{topology}.cat{X})
            match = re.search(r'MOFid-v1\.([^.]+)\.', mofid)
            if match:
                topology_code = match.group(1)
                # Skip invalid topologies
                if topology_code in ['UNKNOWN', 'ERROR']:
                    continue
                # Skip topologies with ERROR or UNKNOWN prefix/suffix
                if 'ERROR' in topology_code or 'UNKNOWN' in topology_code:
                    continue
                
                topology_id = topology_code.lower()
                if topology_id not in topologies:
                    topologies[topology_id] = TopologyEntity(
                        topology_id=topology_id,
                        topology_name=topology_code,
                        data_sources=["ChemUnity"]
                    )
        
        result = list(topologies.values())
        print(f"Extracted {len(result)} unique valid topologies from ChemUnity")
        return result
    
    def extract_linkers(self) -> List[LinkerEntity]:
        """
        Extract linker entities from smiles_linker column in computational_properties.csv.
        
        Returns:
            List of LinkerEntity instances
        """
        if not self.computational_props_file.exists():
            print(f"Warning: {self.computational_props_file} not found")
            return []
        
        df = pd.read_csv(self.computational_props_file)
        linkers = {}
        
        for _, row in df.iterrows():
            smiles_str = str(row.get('smiles_linker', '')).strip()
            if not smiles_str or smiles_str == 'nan':
                continue
            
            # Parse list of SMILES (stored as string representation of list)
            try:
                # Handle string representation of list: "['smiles1', 'smiles2']"
                smiles_list = eval(smiles_str) if smiles_str.startswith('[') else [smiles_str]
                for smiles in smiles_list:
                    smiles = str(smiles).strip()
                    if smiles and smiles != 'nan':
                        # Generate linker ID from SMILES hash
                        linker_id = f"LINKER_{hashlib.md5(smiles.encode()).hexdigest()[:16]}"
                        if linker_id not in linkers:
                            linkers[linker_id] = LinkerEntity(
                                linker_id=linker_id,
                                canonical_name=f"Linker_{linker_id[-8:]}",
                                smiles=smiles,
                                canonical_smiles=smiles,
                                data_sources=["ChemUnity"]
                            )
            except (SyntaxError, ValueError, TypeError):
                # If eval fails, skip this entry
                continue
        
        result = list(linkers.values())
        print(f"Extracted {len(result)} unique linkers from ChemUnity")
        return result
    
    def extract_metal_clusters(self) -> Tuple[List[MetalClusterEntity], List[Dict]]:
        """
        Extract metal cluster entities from Metal types column in computational_properties.csv.
        
        Returns:
            Tuple[List[MetalClusterEntity], List[Dict]]: (Entities, Relationships)
        """
        if not self.computational_props_file.exists():
            print(f"Warning: {self.computational_props_file} not found")
            return [], []
        
        df = pd.read_csv(self.computational_props_file)
        clusters = {}
        relationships = []
        
        for _, row in df.iterrows():
            csd_code = str(row.get('CSD code', '')).strip()
            metal_types = str(row.get('Metal types', '')).strip()
            
            if not metal_types or metal_types == 'nan':
                continue
            
            # Split comma-separated metals
            metals = [m.strip() for m in metal_types.split(',') if m.strip()]
            if metals:
                # Create cluster ID from sorted metal list
                cluster_id = f"CLUSTER_{'_'.join(sorted(metals))}"
                
                # Create relationship if we have a valid MOF ID
                if csd_code and csd_code != 'nan':
                    relationships.append({
                        'mof_id': csd_code,
                        'metal_node_id': cluster_id
                    })
                
                if cluster_id not in clusters:
                    clusters[cluster_id] = MetalClusterEntity(
                        cluster_id=cluster_id,
                        metal_elements=metals,
                        data_sources=["ChemUnity"]
                    )
        
        result = list(clusters.values())
        print(f"Extracted {len(result)} unique metal clusters and {len(relationships)} relationships from ChemUnity")
        return result, relationships
    
    def extract_capabilities(self) -> List[CapabilityEntity]:
        """
        Extract capability entities from applications.csv.
        
        Returns:
            List of CapabilityEntity instances
        """
        if not self.applications_file.exists():
            print(f"Warning: {self.applications_file} not found")
            return []
        
        df = pd.read_csv(self.applications_file)
        capabilities = []
        
        # Map application names to capability types
        capability_type_map = {
            'Hydrogen storage': 'HydrogenStorageCapability',
            'CO2 capture': 'CO2CaptureCapability',
            'CO2Capture': 'CO2CaptureCapability',
            'Methane storage': 'MethaneStorageCapability',
            'CH4 storage': 'MethaneStorageCapability',
            'Luminescent sensing': 'LuminescentSensingCapability',
            'Photocatalytic': 'PhotocatalyticCapability',
            'Catalysis': 'CatalysisCapability',
            'DAC': 'DACCapability',
        }
        
        for _, row in df.iterrows():
            csd_code = str(row.get('Ref Code', '')).strip()
            if not csd_code or csd_code == 'nan':
                continue
            
            application = str(row.get('Application', '')).strip()
            if not application or application == 'nan' or application == 'Not Provided':
                continue
            
            # Map application to capability type
            capability_type = None
            for key, cap_type in capability_type_map.items():
                if key.lower() in application.lower():
                    capability_type = cap_type
                    break
            
            # Default to generic capability if no match
            if not capability_type:
                capability_type = 'Capability'
            
            # Create capability entity
            capability = CapabilityEntity(
                capability_id=f"CAP_{csd_code}_{capability_type}",
                mof_id=csd_code,  # Link to MOF via CSD code
                capability_type=capability_type,
                data_source="ChemUnity",
            )
            capabilities.append(capability)
        
        print(f"Extracted {len(capabilities)} capabilities from ChemUnity")
        return capabilities
