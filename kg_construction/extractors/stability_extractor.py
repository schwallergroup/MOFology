import csv
import re
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
import uuid
import hashlib

from ..datamodels.entitymodels import MOFEntity, PropertyEntity, LinkerEntity

class StabilityExtractor:
    """
    Extracts stability data (Free Energy, Strain Energy) from CSV files
    and links them to existing MOFs in the Knowledge Graph.
    Also ingests new MOFs with provenance flags.
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.fe_dir = data_dir / "fe_atom"
        self.se_dir = data_dir / "se_atom"
        
        # Structural Index: (topology, frozenset(linker_smiles)) -> [mof_ids]
        self.structural_index: Dict[Tuple[str, frozenset], List[str]] = {}
        
        # Maps Linker SMILES -> Linker ID (for linking new MOFs to existing linkers)
        self.linker_smiles_map: Dict[str, str] = {}

    def extract_properties(self, existing_mofs: List[MOFEntity], linkers: List[LinkerEntity], linker_rels: List[Dict]) -> Tuple[List[MOFEntity], List[PropertyEntity]]:
        """
        Extract properties by matching against existing MOFs.
        Creates NEW MOFs for unmatched entries, marked as hypothetical.
        
        Args:
            existing_mofs: List of MOFEntity objects
            linkers: List of LinkerEntity objects
            linker_rels: List of dictionaries [{'mof_id': ..., 'linker_id': ...}, ...]
            
        Returns:
            Tuple[List[MOFEntity], List[PropertyEntity]]: (New MOFs, All Properties)
        """
        print(f"StabilityExtractor: processing against {len(existing_mofs)} existing MOFs")
        
        properties = []
        new_mofs = []
        
        # Track seen mofids/keys to avoid creating duplicates within this run
        seen_new_mofids = set() 
        
        # Build lookup maps
        mofid_lookup = {}
        mofkey_lookup = {}
        name_lookup = {}
        
        # Build Linker Lookup (ID -> SMILES) & Reverse Lookup (SMILES -> ID)
        for linker in linkers:
            # Prefer canonical, fallback to original
            s = linker.canonical_smiles if linker.canonical_smiles else linker.smiles
            if s:
                clean_s = self._clean_smiles(s)
                self.linker_smiles_map[clean_s] = linker.linker_id
        
        # Map MOF IDs to Linker IDs using linker_rels
        mof_to_linkers = {}
        for rel in linker_rels:
            mid = rel.get('mof_id')
            lid = rel.get('linker_id')
            if mid and lid:
                if mid not in mof_to_linkers:
                    mof_to_linkers[mid] = set()
                mof_to_linkers[mid].add(lid)
        
        for mof in existing_mofs:
            # MOFid lookup
            if mof.mofid:
                mofid_lookup[mof.mofid] = mof.mof_id
            
            # MOFkey lookup (check other_ids)
            if mof.other_ids and 'mofkey' in mof.other_ids:
                mofkey_lookup[mof.other_ids['mofkey']] = mof.mof_id
            
            # Name lookup - canonical name
            if mof.canonical_name:
                name_lookup[mof.canonical_name] = mof.mof_id
            
            # Name lookup - CSD code
            if mof.csd_code:
                name_lookup[mof.csd_code] = mof.mof_id
                
            # Name lookup - alternative names
            for name in mof.all_names:
                name_lookup[name] = mof.mof_id
                
            # Populate Structural Index
            linked_linker_ids = set(mof.linker_ids) if mof.linker_ids else set()
            if mof.mof_id in mof_to_linkers:
                linked_linker_ids.update(mof_to_linkers[mof.mof_id])
            
            if mof.topology and linked_linker_ids:
                mof_linker_smiles = set()
                for lid in linked_linker_ids:
                    # Invert linker_smiles_map to get SMILES from ID? No, we didn't store that.
                    # We need ID -> SMILES map. 
                    # Re-iterate linkers to find it? Or store it earlier.
                    # Optimization: create ID->SMILES map above.
                    pass
                
                # Re-do this properly:
                # We need ID -> SMILES map
                id_to_smiles = {v: k for k, v in self.linker_smiles_map.items()}
                
                for lid in linked_linker_ids:
                    if lid in id_to_smiles:
                        mof_linker_smiles.add(id_to_smiles[lid])
                
                if mof_linker_smiles:
                    key = (mof.topology.lower(), frozenset(mof_linker_smiles))
                    if key not in self.structural_index:
                        self.structural_index[key] = []
                    self.structural_index[key].append(mof.mof_id)

        print(f"  Built structural index with {len(self.structural_index)} unique topology+linker combinations")

        # Process FE data (Free Energy)
        print("  Processing Free Energy data...")
        nm1, props1 = self._process_directory(
            self.fe_dir, "Free Energy (atom)", "FreeEnergy", "FE_atom", "eV/atom", 
            mofid_lookup, mofkey_lookup, name_lookup, seen_new_mofids
        )
        new_mofs.extend(nm1)
        properties.extend(props1)

        # Process SE data (Strain Energy)
        print("  Processing Strain Energy data...")
        nm2, props2 = self._process_directory(
            self.se_dir, "Strain Energy (atom)", "StrainEnergy", "SE_atom", "eV/atom", 
            mofid_lookup, mofkey_lookup, name_lookup, seen_new_mofids
        )
        new_mofs.extend(nm2)
        properties.extend(props2)
        
        print(f"StabilityExtractor: Extracted {len(properties)} properties and created {len(new_mofs)} new MOFs")
        return new_mofs, properties

    def _process_directory(self, directory: Path, prop_name: str, prop_type: str, 
                           value_col: str, units: str,
                           mofid_lookup: Dict[str, str], mofkey_lookup: Dict[str, str], 
                           name_lookup: Dict[str, str], seen_new_mofids: Set[str]) -> Tuple[List[MOFEntity], List[PropertyEntity]]:
        properties = []
        new_mofs = []
        
        if not directory.exists():
            print(f"    Warning: Directory {directory} does not exist")
            return [], []
            
        files = list(directory.glob("**/*.csv"))
        print(f"    Found {len(files)} CSV files in {directory.name}")
        
        matches_strict = 0
        matches_fuzzy = 0
        new_entries = 0
        total_rows = 0
        
        for file_path in files:
            # Skip hidden files
            if file_path.name.startswith('.'):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        total_rows += 1
                        
                        # 1. Try to find match
                        mof_id, method = self._find_match(row, mofid_lookup, mofkey_lookup, name_lookup)
                        
                        # 2. If no match, create new MOF
                        if not mof_id:
                            # Check if we've already created this new MOF in this run
                            # Identity based on mofid_v1 or mofkey
                            unique_key = row.get('mofid_v1') or row.get('mofkey') or row.get('mof_name')
                            if not unique_key:
                                continue
                                
                            # Hash it for ID
                            new_id_hash = hashlib.md5(unique_key.encode()).hexdigest()[:8]
                            generated_id = f"STAB_{new_id_hash}"
                            
                            # If we haven't seen this generated ID yet, create the entity
                            if generated_id not in seen_new_mofids:
                                new_mof = self._construct_mof_entity(row, generated_id)
                                if new_mof:
                                    new_mofs.append(new_mof)
                                    seen_new_mofids.add(generated_id)
                                    # Add to lookups so next row finds it? 
                                    # Ideally yes, but seen_new_mofids handles the uniqueness
                                    new_entries += 1
                                    
                            mof_id = generated_id
                            is_new_mof = True
                        else:
                            if method == 'fuzzy':
                                matches_fuzzy += 1
                            else:
                                matches_strict += 1

                        # 3. Add Property
                        if mof_id:
                            try:
                                if value_col not in row or not row[value_col]:
                                    continue
                                    
                                value = float(row[value_col])
                                
                                # Create unique property ID
                                prop_id = f"{prop_type}_{mof_id}_{uuid.uuid4().hex[:8]}"
                                
                                prop = PropertyEntity(
                                    property_id=prop_id,
                                    mof_id=mof_id,
                                    property_name=prop_name,
                                    property_type=prop_type,
                                    value=value,
                                    units=units,
                                    data_source="MOF-FreeEnergy",
                                    conditions=f"Source File: {file_path.name}; Match Method: {method if not is_new_mof else 'new_entry'}"
                                )
                                properties.append(prop)
                            except (ValueError, KeyError):
                                continue 
            except Exception as e:
                print(f"    Error reading {file_path}: {e}")
                            
        print(f"    Matched {matches_strict} (strict) + {matches_fuzzy} (fuzzy). Created {new_entries} new MOFs. Total rows: {total_rows}")
        return new_mofs, properties

    def _construct_mof_entity(self, row: Dict, mof_id: str) -> Optional[MOFEntity]:
        """Construct a new MOFEntity from a CSV row."""
        try:
            name = row.get('mof_name', f"Unknown_MOF_{mof_id}")
            mofid = row.get('mofid_v1')
            mofkey = row.get('mofkey')
            topology = row.get('Topology')
            
            # Extract Linkers to link to existing ones
            linker_ids = []
            if mofid:
                components = self._parse_mofid_components(mofid)
                for comp_smiles in components:
                    if comp_smiles in self.linker_smiles_map:
                        linker_ids.append(self.linker_smiles_map[comp_smiles])
            
            # Create Entity
            mof = MOFEntity(
                mof_id=mof_id,
                canonical_name=name,
                all_names=[name],
                mofid=mofid,
                other_ids={'mofkey': mofkey} if mofkey else {},
                topology=topology if topology else None,
                is_experimental=False, # PROVENANCE FLAG: Hypothetical
                data_sources=["MOF-FreeEnergy"],
                linker_ids=linker_ids
            )
            return mof
        except Exception as e:
            print(f"Error constructing MOF entity: {e}")
            return None

    def _find_match(self, row: Dict, mofid_lookup: Dict, mofkey_lookup: Dict, name_lookup: Dict) -> Tuple[Optional[str], str]:
        """
        Find matching MOF ID using available identifiers in row.
        Returns (mof_id, match_method)
        """
        
        # 1. Try MOFid (highest precision)
        if 'mofid_v1' in row and row['mofid_v1']:
            val = row['mofid_v1'].strip()
            if val in mofid_lookup:
                return mofid_lookup[val], 'mofid'
                
        # 2. Try MOFkey
        if 'mofkey' in row and row['mofkey']:
            val = row['mofkey'].strip()
            if val in mofkey_lookup:
                return mofkey_lookup[val], 'mofkey'
        
        # 3. Try Name
        if 'mof_name' in row and row['mof_name']:
            val = row['mof_name'].strip()
            if val in name_lookup:
                return name_lookup[val], 'name'
        
        # 4. Try Fuzzy Component Match
        # Extract topology and component SMILES from row
        if 'mofid_v1' in row and 'Topology' in row:
            topo = row['Topology'].strip().lower()
            mofid_str = row['mofid_v1']
            
            # Parse components from MOFid
            components = self._parse_mofid_components(mofid_str)
            if not components:
                return None, 'none'
                
            # Iterate through structural index
            # We are looking for a KG MOF where:
            # 1. Topology matches
            # 2. KG MOF Linkers are a SUBSET of the Stability MOF components
            
            candidates = []
            
            # Optimization: only check entries with matching topology
            for (idx_topo, idx_linkers), mof_ids in self.structural_index.items():
                if idx_topo == topo:
                    # Check if KG linkers are present in row components
                    if idx_linkers.issubset(components):
                        candidates.extend(mof_ids)
            
            if candidates:
                # Actually, let's just take the first unique one.
                return candidates[0], 'fuzzy'

        return None, 'none'

    def _parse_mofid_components(self, mofid_str: str) -> Set[str]:
        """
        Parses MOFid string to extract unique component SMILES.
        Format: [SMILES].[SMILES] MOFid-v1...
        """
        if not mofid_str:
            return set()
            
        try:
            # Split SMILES part from metadata part (separated by space)
            parts = mofid_str.split(' ')
            smiles_part = parts[0]
            
            # Split individual components by '.'
            raw_components = smiles_part.split('.')
            
            clean_components = set()
            for c in raw_components:
                clean = self._clean_smiles(c)
                if clean:
                    clean_components.add(clean)
            
            return clean_components
        except Exception:
            return set()

    def _clean_smiles(self, smiles: str) -> str:
        """
        Normalize SMILES string (simple string cleaning).
        """
        if not smiles:
            return ""
        # Remove whitespace
        s = smiles.strip()
        # Should we uppercase? SMILES is case sensitive (C vs c for aromatic).
        # So typically NO uppercase.
        return s
