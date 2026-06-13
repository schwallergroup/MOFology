"""
Linker Extractor

Extracts and standardizes linker information from:
1. ChemUnity (computational_properties.csv)
2. MaterialsProject (MaterialsProject_raw.json)
3. DigiMOF (Extracted_paper_info.csv)

Consolidates logic from extract_linkers.py and update_linker_links.py
"""

import json
import re
import hashlib
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any, Set, Tuple
from datetime import datetime

from ..datamodels.entitymodels import LinkerEntity

class LinkerExtractor:
    """Extractor for Organic Linkers from multiple sources."""
    
    def __init__(self, raw_data_dir: Optional[Path] = None):
        if raw_data_dir is None:
            self.raw_data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
        else:
            self.raw_data_dir = Path(raw_data_dir)
            
    def normalize_smiles(self, smiles_str: Any) -> Optional[str]:
        """Normalize SMILES string."""
        if pd.isna(smiles_str) or not smiles_str:
            return None
        # Handle list format like "['SMILES1', 'SMILES2']"
        if isinstance(smiles_str, str) and smiles_str.startswith('['):
            try:
                import ast
                smiles_list = ast.literal_eval(smiles_str)
                if isinstance(smiles_list, list) and len(smiles_list) > 0:
                    return smiles_list[0]  # Take first SMILES
            except:
                pass
        return str(smiles_str).strip()

    def generate_linker_id(self, smiles: Optional[str] = None, name: Optional[str] = None) -> Optional[str]:
        """Generate stable linker_id from SMILES (preferred) or name."""
        if smiles and pd.notna(smiles):
            # Use hash of normalized SMILES
            clean_smiles = str(smiles).strip().upper()
            return f"LINKER_{hashlib.md5(clean_smiles.encode()).hexdigest()[:8].upper()}"
        elif name and pd.notna(name):
            # Use hash of normalized name
            clean_name = re.sub(r'[^a-zA-Z0-9]', '', str(name).upper())
            return f"LINKER_{hashlib.md5(clean_name.encode()).hexdigest()[:8].upper()}"
        return None

    def extract_chemunity_linkers(self) -> List[Dict[str, Any]]:
        """Extract linkers from ChemUnity."""
        print("  Extracting linkers from ChemUnity...")
        filepath = self.raw_data_dir / 'ChemUnity' / 'computational_properties.csv'
        if not filepath.exists():
            print(f"    Warning: {filepath} not found")
            return []

        df = pd.read_csv(filepath, low_memory=False)
        linkers = []
        
        for _, row in df.iterrows():
            csd_code = str(row.get('CSD code', '')).strip()
            smiles_linker = row.get('smiles_linker', '')
            chemical_name = row.get('Chemical Name', '')
            synonyms = row.get('Synonyms', '')
            
            smiles = self.normalize_smiles(smiles_linker)
            
            if smiles or chemical_name:
                linker_id = self.generate_linker_id(smiles=smiles, name=chemical_name)
                if linker_id:
                    all_names = []
                    if pd.notna(chemical_name):
                        all_names.append(str(chemical_name))
                    if pd.notna(synonyms):
                        try:
                            syn_list = json.loads(synonyms) if isinstance(synonyms, str) and synonyms.startswith('[') else [synonyms]
                            all_names.extend([str(s) for s in syn_list if pd.notna(s)])
                        except:
                            if pd.notna(synonyms):
                                all_names.append(str(synonyms))
                    
                    linkers.append({
                        'linker_id': linker_id,
                        'canonical_name': chemical_name if pd.notna(chemical_name) else f"Linker_{linker_id}",
                        'smiles': smiles,
                        'all_names': list(set(all_names)),
                        'data_source': 'ChemUnity',
                        'source_mof_id': csd_code if csd_code and csd_code != 'nan' else None
                    })
        return linkers

    def extract_digimof_linkers(self) -> List[Dict[str, Any]]:
        """Extract linker names from DigiMOF abstracts/titles."""
        print("  Extracting linkers from DigiMOF...")
        filepath = self.raw_data_dir / 'DigiMOF' / 'Extracted_paper_info.csv'
        if not filepath.exists():
            print(f"    Warning: {filepath} not found")
            return []

        df = pd.read_csv(filepath, low_memory=False)
        linkers = []
        
        # Common linker abbreviations/patterns
        patterns = [
            r'\b(BDC|H2BDC|1,4-benzenedicarboxylate|terephthalate)\b',
            r'\b(BTC|H3BTC|1,3,5-benzenetricarboxylate|trimesate)\b',
            r'\b(NDC|H2NDC|2,6-naphthalenedicarboxylate)\b',
            r'\b(INA|isonicotinate|isonicotinic acid)\b',
            r'\b(pyridine|pyridyl)\b',
            r'\b(imidazolate|IM)\b',
            r'\b(azolate|tetrazolate)\b',
        ]
        
        for _, row in df.iterrows():
            csd_code = str(row.get('Ref Code', '')).strip() # Was 'CSD' in extract_linkers.py but 'Ref Code' in digimof_extractor.py
            if not csd_code: csd_code = str(row.get('CSD', '')).strip()
            
            text = f"{row.get('Title', '')} {row.get('Abstract', '')}".lower()
            found_names = set()
            for p in patterns:
                matches = re.findall(p, text, re.IGNORECASE)
                found_names.update([m[0] if isinstance(m, tuple) else m for m in matches])
            
            for name in found_names:
                if name:
                    linker_id = self.generate_linker_id(name=name)
                    if linker_id:
                        linkers.append({
                            'linker_id': linker_id,
                            'canonical_name': name,
                            'smiles': None,
                            'all_names': [name],
                            'data_source': 'DigiMOF',
                            'source_mof_id': csd_code if csd_code and csd_code != 'nan' else None
                        })
        return linkers

    def extract_materialsproject_linkers(self) -> List[Dict[str, Any]]:
        """Placeholder for MP linkers."""
        return []

    def extract_all_linkers(self) -> Tuple[List[LinkerEntity], List[Dict[str, str]]]:
        """
        Extract, merge, and deduplicate linkers from all sources.
        Returns:
            Tuple of (List[LinkerEntity], List[RelationshipDict])
        """
        raw_linkers = []
        raw_linkers.extend(self.extract_chemunity_linkers())
        raw_linkers.extend(self.extract_digimof_linkers())
        
        # Merge by linker_id
        merged = {}
        relationships = []
        
        for l in raw_linkers:
            lid = l['linker_id']
            
            # Capture relationship if we have a source MOF
            if l.get('source_mof_id'):
                relationships.append({
                    'mof_id': l['source_mof_id'],
                    'linker_id': lid,
                    'source': l['data_source']
                })
            
            if lid not in merged:
                merged[lid] = {
                    'linker_id': lid,
                    'canonical_name': l['canonical_name'],
                    'smiles': l['smiles'],
                    'all_names': set(l['all_names']),
                    'data_sources': {l['data_source']}
                }
            else:
                existing = merged[lid]
                # Prefer smiles if available
                if not existing['smiles'] and l['smiles']:
                    existing['smiles'] = l['smiles']
                # Merge names
                existing['all_names'].update(l['all_names'])
                # Update canonical name if better (not starting with Linker_)
                if existing['canonical_name'].startswith('Linker_') and not l['canonical_name'].startswith('Linker_'):
                    existing['canonical_name'] = l['canonical_name']
                # Merge sources
                existing['data_sources'].add(l['data_source'])
        
        # Convert to Entities
        entities = []
        for data in merged.values():
            entities.append(LinkerEntity(
                linker_id=data['linker_id'],
                canonical_name=data['canonical_name'],
                smiles=data['smiles'] if data['smiles'] else "",
                canonical_smiles=data['smiles'] if data['smiles'] else "", # Assuming input smiles are somewhat canonical
                all_names=list(data['all_names']),
                data_sources=list(data['data_sources'])
            ))
            
        print(f"Extracted {len(entities)} unique linkers total.")
        print(f"Extracted {len(relationships)} MOF-Linker relationships.")
        return entities, relationships
