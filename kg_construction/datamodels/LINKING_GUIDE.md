# Entity Linking Guide

## How Properties (and other entities) Link to MOFs

### Key Principle: Use CSD Code as the Linking Key

During entity extraction, **all entities that reference MOFs should use the CSD code** (or the MOF's primary `mof_id`) to establish relationships.

### MOF Entity Primary Key

The `MOFEntity.mof_id` field is the primary key used for linking:
- **For experimental MOFs**: `mof_id` = CSD code (e.g., "ABAVIJ")
- **For hypothetical MOFs**: `mof_id` = MP ID (e.g., "MP_12345") or generated ID
- **This is the identifier that all other entities reference**

### Property-to-MOF Linking

When extracting properties from data sources:

```python
# Example: Extracting properties from computational_properties.csv
def extract_properties(self) -> List[PropertyEntity]:
    df = pd.read_csv("computational_properties.csv")
    properties = []
    
    for _, row in df.iterrows():
        csd_code = row['CSD code']  # Get CSD code from data
        
        # Create property entity
        prop = PropertyEntity(
            property_id=f"PROP_{csd_code}_bandgap",
            mof_id=csd_code,  # ← Use CSD code to link to MOF
            property_name="Band gap (eV)",
            property_type="ComputationalProperty",
            value=row['Band gap (eV)'],
            units="eV"
        )
        properties.append(prop)
    
    return properties
```

**Important**: The `mof_id` in `PropertyEntity` must match the `mof_id` in `MOFEntity`.

### Other Entity-to-MOF Linking

The same principle applies to all entities that reference MOFs:

#### Synthesis Processes
```python
synthesis = SynthesisProcessEntity(
    synthesis_id=f"SYN_{csd_code}",
    mof_id=csd_code,  # ← Link to MOF using CSD code
    method="Solvothermal"
)
```

#### Abstracts
```python
abstract = AbstractEntity(
    abstract_id=f"ABSTRACT_{csd_code}",
    mof_id=csd_code,  # ← Link to MOF using CSD code
    title=row['Title'],
    abstract_text=row['Abstract']
)
```

#### Lattice Parameters
```python
lattice = LatticeParameterEntity(
    lattice_param_id=f"LATTICE_{csd_code}",
    mof_id=csd_code,  # ← Link to MOF using CSD code
    a=row['a'],
    b=row['b'],
    c=row['c']
)
```

#### Capabilities
```python
capability = CapabilityEntity(
    capability_id=f"CAP_{csd_code}_co2",
    mof_id=csd_code,  # ← Link to MOF using CSD code
    capability_type="CO2CaptureCapability"
)
```

### Extraction Workflow

1. **Extract MOFs first** (establish the primary keys):
   ```python
   mofs = extractor.extract_mofs()
   # Each MOF has mof_id = CSD code (or MP ID)
   ```

2. **Extract properties** (reference MOFs by CSD code):
   ```python
   properties = extractor.extract_properties()
   # Each property has mof_id = CSD code from data source
   ```

3. **Link during normalization/validation**:
   ```python
   # Verify all property.mof_id values exist in MOF entities
   mof_ids = {mof.mof_id for mof in mofs}
   for prop in properties:
       if prop.mof_id not in mof_ids:
           print(f"Warning: Property {prop.property_id} references unknown MOF {prop.mof_id}")
   ```

### Handling Missing CSD Codes

If a property comes from a data source without a CSD code:

1. **Try to match by other identifiers** (MP ID, name, formula)
2. **Or create a placeholder MOF** with a generated ID
3. **Or skip the property** if no MOF can be matched

### Example: Complete Extraction Pattern

```python
class ChemUnityExtractor:
    def extract_mofs(self) -> List[MOFEntity]:
        """Extract MOFs - establishes primary keys"""
        df = pd.read_csv("MOF_names_and_CSD_codes.csv")
        mofs = []
        
        for _, row in df.iterrows():
            csd_code = row['Ref Code']  # This becomes mof_id
            mof = MOFEntity(
                mof_id=csd_code,  # Primary key
                csd_code=csd_code,
                canonical_name=row['MOF Name'],
                # ...
            )
            mofs.append(mof)
        
        return mofs
    
    def extract_properties(self) -> List[PropertyEntity]:
        """Extract properties - links to MOFs via CSD code"""
        df = pd.read_csv("computational_properties.csv")
        properties = []
        
        for _, row in df.iterrows():
            csd_code = row['CSD code']  # Use this to link
            
            # Extract band gap property
            if pd.notna(row.get('Band gap (eV)')):
                prop = PropertyEntity(
                    property_id=f"PROP_{csd_code}_bandgap",
                    mof_id=csd_code,  # ← Links to MOF with mof_id=csd_code
                    property_name="Band gap (eV)",
                    property_type="ComputationalProperty",
                    value=float(row['Band gap (eV)']),
                    units="eV"
                )
                properties.append(prop)
        
        return properties
```

### Summary

- **MOFEntity.mof_id** = Primary key (CSD code for experimental MOFs)
- **PropertyEntity.mof_id** = Foreign key (must match MOFEntity.mof_id)
- **All other entities** that reference MOFs use the same pattern
- **During extraction**: Use CSD code from data source as the linking key
- **During validation**: Verify all foreign keys reference existing MOFs





