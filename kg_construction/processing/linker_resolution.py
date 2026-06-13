"""
Linker Resolution Module

Handles matching MOFs to Linkers across different data sources using SMILES and Name matching.
Based on logic from update_linker_links.py.
"""

import pandas as pd
import json
import re
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from ..datamodels.entitymodels import LinkerEntity, MOFEntity

class LinkerResolver:
    """Resolves relationships between MOFs and Linkers."""
    
    def __init__(self, linkers: List[LinkerEntity]):
        self.linkers = linkers
        self.smiles_lookup = {}
        self.name_lookup = {}
        self._build_lookups()
        
    def _build_lookups(self):
        """Build lookup dictionaries for fast matching."""
        for linker in self.linkers:
            # Index by canonical SMILES
            if linker.canonical_smiles:
                s = self._normalize_smiles(linker.canonical_smiles)
                if s:
                    self.smiles_lookup[s.upper()] = linker.linker_id
            
            # Index by original SMILES
            if linker.smiles:
                s = self._normalize_smiles(linker.smiles)
                if s:
                    self.smiles_lookup[s.upper()] = linker.linker_id
            
            # Index by all names
            names = [linker.canonical_name] + linker.all_names
            for name in names:
                if name:
                    clean_name = self._clean_name(name)
                    if clean_name:
                        self.name_lookup[clean_name] = linker.linker_id

    def _normalize_smiles(self, smiles_str: str) -> Optional[str]:
        """Normalize SMILES string."""
        if not smiles_str:
            return None
        # Handle list format
        if isinstance(smiles_str, str) and smiles_str.startswith('['):
            try:
                import ast
                l = ast.literal_eval(smiles_str)
                if l: return str(l[0]).strip()
            except: pass
        return str(smiles_str).strip()

    def _clean_name(self, name: str) -> str:
        """Normalize chemical name for matching."""
        return re.sub(r'[^a-zA-Z0-9]', '', str(name).upper())

    def resolve_linker(self, smiles: Optional[str] = None, name: Optional[str] = None) -> Optional[str]:
        """
        Find a matching linker ID given a SMILES string or name.
        """
        # Try SMILES match first
        if smiles:
            norm_smiles = self._normalize_smiles(smiles)
            if norm_smiles:
                lid = self.smiles_lookup.get(norm_smiles.upper())
                if lid: return lid
        
        # Try Name match
        if name:
            clean_name = self._clean_name(name)
            if clean_name:
                lid = self.name_lookup.get(clean_name)
                if lid: return lid
                
        return None
