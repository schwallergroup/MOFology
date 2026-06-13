"""
Data models for all MOF Knowledge Graph entities.

Each dataclass corresponds to an ontology class and includes:
- Fields matching ontology datatype properties
- Proper typing and defaults
- Ontology class information in docstrings

Ontology Namespaces:
- : (default) = http://emmo.info/domain-mof/mof-ontology#
- syn: = http://emmo.info/domain-mof/synthesis#
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

# ============================================================================
# Core MOF Entities
# ============================================================================

@dataclass
class MOFEntity:
    """
    MOF Entity Model
    
    Ontology Class: :MOF (base class)
    Subclasses: :ExperimentalMOF, :HypotheticalMOF, syn:FunctionalizedMOF
    
    Ontology Properties:
    - :hasCSDCode (xsd:string) - CSD code identifier (primary ID for experimental MOFs)
    - :hasMPID (xsd:string) - Materials Project ID (for hypothetical/computational MOFs)
    - :hasFormula (xsd:string) - Chemical formula
    - :hasCanonicalName (xsd:string) - Primary name
    - :hasAlternativeName (xsd:string) - Alternative names (repeatable property)
    - syn:isHypothetical (xsd:boolean) - Experimental vs hypothetical
    
    Note: The is_experimental field is inverted from syn:isHypothetical in the ontology:
    - is_experimental=True → syn:isHypothetical=false (ExperimentalMOF)
    - is_experimental=False → syn:isHypothetical=true (HypotheticalMOF)
    
    MOF Identifiers:
    - CSD code (csd_code): Primary identifier for experimentally synthesized MOFs
    - MP ID (mp_id): Materials Project identifier for computational/hypothetical MOFs
    - MOFid (mofid): Structural identifier (like SMILES) - encodes MOF structure
      Format: "MOFid-v1.{topology}.{category};{csd_code}_clean"
      Example: "MOFid-v1.fsc.cat1;BUKYAJ_clean"
    - Other IDs (other_ids): Dictionary for additional identifiers from other databases
    - mof_id: Primary key used in KG (preferentially CSD code, fallback to MP ID or generated)
      This is the identifier used to link properties, synthesis, etc. to MOFs
    
    Relationships:
    - :hasLinker → :OrganicLinker (required, min 1)
    - :hasMetalNode → :MetalCluster
    - :hasProperty → :MaterialProperty
    - :hasStructuralProperty → :StructuralProperty (for space group, crystal system, lattice parameters)
    - :hasSynthesisProcess → syn:SynthesisProcess
    - :hasAbstract → :Abstract
    - :hasPublication → :Publication
    
    Note: Topology (e.g., "pcu", "fcu", "dia") describes the overall framework
    topology/net topology of the MOF structure. This is important structural
    information that characterizes the connectivity pattern of the framework.
    
    Structural Information:
    - Space groups, crystal systems, and lattice parameters are stored as separate
      entities (SpaceGroupEntity, CrystalSystemEntity, LatticeParameterEntity) and
      linked to MOFs via :hasStructuralProperty relationships. This allows querying
      and grouping MOFs by structural characteristics.
    """
    mof_id: str  # Primary key for KG linking (CSD code if available, otherwise MP_id or generated)
    # This is the identifier used to link properties, synthesis, abstracts, etc. to MOFs
    canonical_name: str  # Maps to :hasCanonicalName
    all_names: List[str] = field(default_factory=list)  # Maps to :hasAlternativeName (repeatable property)
    formula: Optional[str] = None  # Maps to :hasFormula
    csd_code: Optional[str] = None  # Maps to :hasCSDCode (primary identifier for experimental MOFs)
    mp_id: Optional[str] = None  # Maps to :hasMPID (Materials Project identifier)
    mofid: Optional[str] = None  # Structural identifier (like SMILES) - encodes MOF structure
    # Format: "MOFid-v1.{topology}.{category};{csd_code}_clean"
    # Example: "MOFid-v1.fsc.cat1;BUKYAJ_clean" or "MOFid-v1.UNKNOWN.cat0;QONLAI_clean"
    # Additional identifiers can be stored in source_ids dict if needed
    other_ids: Dict[str, str] = field(default_factory=dict)  # Other MOF identifiers (e.g., from other databases)
    is_experimental: bool = True  # Determines :ExperimentalMOF vs :HypotheticalMOF
    topology: Optional[str] = None  # Framework topology (e.g., "pcu", "fcu", "dia", "CN_Topology" from DigiMOF)
    # Note: Space group, crystal system, and lattice parameters are separate entities
    # linked via relationships (not stored directly here)
    data_sources: List[str] = field(default_factory=list)  # For provenance
    linker_ids: List[str] = field(default_factory=list) # List of linker IDs used in this MOF
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())




@dataclass
class FunctionalizedMOFEntity:
    """
    Functionalized MOF Entity Model
    
    Ontology Class: syn:FunctionalizedMOF (subclass of :MOF)
    
    Ontology Properties:
    - :hasCanonicalName (xsd:string) - Name of functionalized MOF
    - :hasFormula (xsd:string) - Chemical formula
    
    Relationships:
    - syn:derivedFrom → :MOF (parent MOF)
    - syn:hasFunctionalization → syn:Functionalization
    - :hasProperty → :MaterialProperty (binding energies, etc.)
    """
    func_mof_id: str  # Primary key
    parent_csd_code: str  # Foreign key to parent MOF (CSD code)
    canonical_name: str  # Maps to :hasCanonicalName
    functionalization_type: str  # e.g., "AmineFunctionalization"
    formula: Optional[str] = None  # Maps to :hasFormula
    functional_group_name: Optional[str] = None  # e.g., "N-Methylethylenediamine"
    functionalization_degree: Optional[float] = None  # Maps to syn:functionalizationDegree
    data_sources: List[str] = field(default_factory=list)
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# Component Entities
# ============================================================================

@dataclass
class LinkerEntity:
    """
    Organic Linker Entity Model
    
    Ontology Class: :OrganicLinker (subclass of :Chemical)
    
    Ontology Properties:
    - :hasSMILES (xsd:string) - SMILES representation
    - :hasCanonicalSMILES (xsd:string) - Canonical SMILES
    - :hasChemicalName (xsd:string) - Primary name
    - :hasAlternativeChemicalName (xsd:string) - Alternative names (repeatable property)
    
    Relationships:
    - :usedInMOF (inverse of :hasLinker) → :MOF
    """
    linker_id: str  # Primary key (generated from canonical SMILES hash)
    canonical_name: str  # Maps to :hasChemicalName
    smiles: str  # Maps to :hasSMILES
    canonical_smiles: str  # Maps to :hasCanonicalSMILES
    all_names: List[str] = field(default_factory=list)  # Maps to :hasAlternativeChemicalName (repeatable property)
    data_sources: List[str] = field(default_factory=list)
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class MetalClusterEntity:
    """
    Metal Cluster Entity Model
    
    Ontology Class: :MetalCluster
    
    Ontology Properties:
    - :hasMetalElement (xsd:string) - Metal elements (repeatable property)
    - :hasClusterFormula (xsd:string) - Formula of cluster
    - :hasClusterDescription (xsd:string) - Description
    - :coordinationNumber (xsd:integer) - Coordination number
    
    Relationships:
    - :isComponentOf (inverse of :hasMetalNode) → :MOF
    
    Note: Topology for metal clusters describes the coordination geometry and
    connectivity pattern of the metal nodes (e.g., octahedral, tetrahedral,
    paddlewheel, etc.). This is distinct from the MOF framework topology and
    describes the local structure of the metal cluster itself.
    """
    cluster_id: str  # Primary key
    metal_elements: List[str] = field(default_factory=list)  # Maps to :hasMetalElement (repeatable property)
    formula: Optional[str] = None  # Maps to :hasClusterFormula
    description: Optional[str] = None  # Maps to :hasClusterDescription
    coordination_number: Optional[int] = None  # Maps to :coordinationNumber
    #topology: Optional[str] = None  # Metal cluster topology/geometry (e.g., "octahedral", "paddlewheel")
    data_sources: List[str] = field(default_factory=list)
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TopologyEntity:
    """
    Topology Entity Model
    
    Ontology Class: :Topology
    
    Ontology Properties:
    - :hasTopology (xsd:string) - Topology of the MOF
    """
    topology_id: str  # Primary key
    topology_name: str  # Maps to :hasTopology
    data_sources: List[str] = field(default_factory=list)
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SolventEntity:
    """
    Solvent Entity Model
    
    Ontology Class: syn:Solvent (subclass of :Chemical)
    
    Ontology Properties:
    - :hasSMILES (xsd:string) - SMILES representation
    - :hasCanonicalSMILES (xsd:string) - Canonical SMILES
    - :hasChemicalName (xsd:string) - Primary name
    - :hasAlternativeChemicalName (xsd:string) - Alternative names (repeatable property)
    
    Relationships:
    - :usedInSynthesis (inverse of syn:usesSolvent) → syn:SynthesisProcess
    """
    solvent_id: str  # Primary key (generated from canonical SMILES hash)
    canonical_name: str  # Maps to :hasChemicalName
    smiles: str  # Maps to :hasSMILES
    canonical_smiles: str  # Maps to :hasCanonicalSMILES
    all_names: List[str] = field(default_factory=list)  # Maps to :hasAlternativeChemicalName (repeatable property)
    data_sources: List[str] = field(default_factory=list)
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ChemicalEntity:
    """
    Generic Chemical Entity Model
    
    Ontology Class: :Chemical (base class for chemicals)
    Used for: Functional groups, precursors, adsorbates, etc.
    
    Ontology Properties:
    - :hasSMILES (xsd:string) - SMILES representation
    - :hasCanonicalSMILES (xsd:string) - Canonical SMILES
    - :hasChemicalName (xsd:string) - Primary name
    - :hasAlternativeChemicalName (xsd:string) - Alternative names (repeatable property)
    
    Relationships:
    - syn:usesFunctionalGroup (from syn:Functionalization) → syn:Functionalization
    """
    chemical_id: str  # Primary key
    canonical_name: str  # Maps to :hasChemicalName
    smiles: Optional[str] = None  # Maps to :hasSMILES
    canonical_smiles: Optional[str] = None  # Maps to :hasCanonicalSMILES
    all_names: List[str] = field(default_factory=list)  # Maps to :hasAlternativeChemicalName (repeatable property)
    data_sources: List[str] = field(default_factory=list)
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class MetalPrecursorEntity(ChemicalEntity):
    """
    Metal Precursor Entity Model
    
    Ontology Class: syn:MetalPrecursor (subclass of :Chemical)
    
    Used in MOF synthesis as metal source.
    
    Relationships:
    - :usedAsMetalPrecursorIn (inverse of syn:usesMetalPrecursor) → syn:SynthesisProcess
    """
    metal_element: Optional[str] = None  # Primary metal element provided


@dataclass
class LinkerPrecursorEntity(ChemicalEntity):
    """
    Linker Precursor Entity Model
    
    Ontology Class: syn:LinkerPrecursor (subclass of :Chemical)
    
    Used in MOF synthesis as linker source.
    
    Relationships:
    - :usedAsLinkerPrecursorIn (inverse of syn:usesLinkerPrecursor) → syn:SynthesisProcess
    """
    target_linker_id: Optional[str] = None  # Linker produced by this precursor


# ============================================================================
# Structural Classification Entities
# ============================================================================

@dataclass
class SpaceGroupEntity:
    """
    Space Group Entity Model
    
    Ontology Class: Could be :StructuralProperty or a new class if needed
    
    Purpose: Represents crystallographic space groups (e.g., "P1", "P3¯c1", "P21/c")
    Many MOFs share the same space group, so this allows querying/grouping by space group.
    
    Relationships:
    - :hasSpaceGroup (from :MOF) → :MOF (many-to-many relationship)
    - Could be linked via :hasStructuralProperty → :StructuralProperty
    """
    space_group_id: str  # Primary key (e.g., "P1", "P3¯c1")
    space_group_name: str  # Full name (e.g., "P 1", "P 3¯ c 1")
    space_group_number: Optional[int] = None  # International space group number (e.g., 165)
    crystal_system: Optional[str] = None  # e.g., "triclinic", "monoclinic", "orthorhombic"
    point_group: Optional[str] = None  # Point group symbol
    data_sources: List[str] = field(default_factory=list)
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CrystalSystemEntity:
    """
    Crystal System Entity Model
    
    Ontology Class: Could be :StructuralProperty or a new class if needed
    
    Purpose: Represents crystal systems (e.g., "Triclinic", "Monoclinic", "Orthorhombic", 
    "Tetragonal", "Trigonal", "Hexagonal", "Cubic")
    Many MOFs share the same crystal system, so this allows querying/grouping by crystal system.
    
    Relationships:
    - :hasCrystalSystem (from :MOF) → :MOF (many-to-many relationship)
    - Could be linked via :hasStructuralProperty → :StructuralProperty
    """
    crystal_system_id: str  # Primary key (e.g., "triclinic", "monoclinic")
    crystal_system_name: str  # Full name (e.g., "Triclinic", "Monoclinic")
    description: Optional[str] = None  # Description of the crystal system
    data_sources: List[str] = field(default_factory=list)
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class LatticeParameterEntity:
    """
    Lattice Parameter Entity Model
    
    Ontology Class: :StructuralProperty
    
    Purpose: Represents unit cell lattice parameters for a specific MOF.
    Contains all unit cell dimensions (a, b, c, α, β, γ) and volume.
    These are specific to each MOF structure.
    
    Ontology Properties:
    - :propertyName (xsd:string) - e.g., "Unit cell parameters"
    - :propertyValue (xsd:decimal) - Could store volume or individual parameters
    - :propertyUnits (xsd:string) - Units (Å, degrees, Å³)
    - :propertyConditions (xsd:string) - Measurement conditions
    
    Relationships:
    - :hasStructuralProperty (from :MOF) → :MOF
    
    Linking to MOFs:
    - mof_id MUST match MOFEntity.mof_id (typically CSD code)
    - Use CSD code from data source to link lattice parameters to MOFs
    """
    lattice_param_id: str  # Primary key
    mof_id: str  # Foreign key to MOF - MUST match MOFEntity.mof_id (typically CSD code)
    # Unit cell lengths
    a: Optional[float] = None  # Unit cell parameter a (Å)
    b: Optional[float] = None  # Unit cell parameter b (Å)
    c: Optional[float] = None  # Unit cell parameter c (Å)
    # Unit cell angles
    alpha: Optional[float] = None  # Angle α (degrees)
    beta: Optional[float] = None  # Angle β (degrees)
    gamma: Optional[float] = None  # Angle γ (degrees)
    # Derived properties
    volume: Optional[float] = None  # Unit cell volume (Å³)
    z_value: Optional[int] = None  # Z (number of formula units per unit cell)
    # Metadata
    property_name: str = "Unit cell parameters"  # Maps to :propertyName
    units: str = "Å, degrees, Å³"  # Maps to :propertyUnits
    conditions: Optional[str] = None  # Maps to :propertyConditions
    data_source: str = ""  # For provenance
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# Property Entities
# ============================================================================

@dataclass
class PropertyEntity:
    """
    Material Property Entity Model
    
    Ontology Class: :MaterialProperty (base class)
    Subclasses: :StructuralProperty, :ComputationalProperty, :PhysicalProperty
    
    Ontology Properties:
    - :propertyName (xsd:string) - Name of property
    - :propertyValue (xsd:decimal) - Numeric value
    - :propertyUnits (xsd:string) - Units
    - :propertyConditions (xsd:string) - Measurement conditions
    
    Relationships:
    - :hasPropertyOwner (inverse of :hasProperty) → :MOF
    - :hasStructuralPropertyOwner (inverse of :hasStructuralProperty) → :MOF
    - :hasComputationalPropertyOwner (inverse of :hasComputationalProperty) → :MOF
    - :hasPhysicalPropertyOwner (inverse of :hasPhysicalProperty) → :MOF
    
    Property-to-MOF Linking:
    - Properties are linked to MOFs via mof_id (foreign key)
    - mof_id should match the MOFEntity.mof_id (typically CSD code)
    - During extraction, use CSD code from the data source to link properties to MOFs
    - Example: If property data has CSD code "ABAVIJ", set mof_id="ABAVIJ" to link to MOF with mof_id="ABAVIJ"
    """
    property_id: str  # Primary key
    mof_id: str  # Foreign key to MOF - MUST match MOFEntity.mof_id (typically CSD code)
    # Use CSD code from data source to link properties to MOFs during extraction
    # For functionalized MOFs, use func_mof_id instead
    property_name: str  # Maps to :propertyName
    property_type: str  # "StructuralProperty", "ComputationalProperty", or "PhysicalProperty"
    value: Optional[float] = None  # Maps to :propertyValue
    units: Optional[str] = None  # Maps to :propertyUnits
    conditions: Optional[str] = None  # Maps to :propertyConditions
    data_source: str = ""  # For provenance
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# Synthesis Entities
# ============================================================================

@dataclass
class SynthesisProcessEntity:
    """
    Synthesis Process Entity Model
    
    Ontology Class: syn:SynthesisProcess
    
    Ontology Properties:
    - syn:hasSynthesisMethod (xsd:string) - Method name (e.g., "Solvothermal")
    - syn:hasYield (xsd:decimal) - Yield percentage
    
    Relationships:
    - :hasMOF (inverse of syn:hasSynthesisProcess) → :MOF
    - syn:hasCondition → syn:SynthesisCondition
    - syn:hasProcedure → syn:SynthesisProcedure
    - syn:usesSolvent → syn:Solvent
    - syn:usesMetalPrecursor → syn:MetalPrecursor
    - syn:usesLinkerPrecursor → syn:LinkerPrecursor
    
    Linking to MOFs:
    - mof_id MUST match MOFEntity.mof_id (typically CSD code)
    - Use CSD code from data source to link synthesis to MOFs
    """
    synthesis_id: str  # Primary key
    mof_id: str  # Foreign key to MOF - MUST match MOFEntity.mof_id (typically CSD code)
    method: Optional[str] = None  # Maps to syn:hasSynthesisMethod
    yield_percent: Optional[float] = None  # Maps to syn:hasYield
    counterions: List[str] = field(default_factory=list)  # Maps to syn:hasCounterion (repeatable property)
    data_source: str = ""  # For provenance
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SynthesisConditionEntity:
    """
    Synthesis Condition Entity Model
    
    Ontology Class: syn:SynthesisCondition
    
    Ontology Properties:
    - syn:hasTemperature (xsd:decimal) - Temperature in Celsius
    - syn:hasPressure (xsd:decimal) - Pressure in bar
    - syn:hasReactionTime (xsd:decimal) - Time in hours
    
    Relationships:
    - syn:hasCondition (from syn:SynthesisProcess) → syn:SynthesisProcess
    """
    condition_id: str  # Primary key
    synthesis_id: str  # Foreign key to SynthesisProcess
    temperature_c: Optional[float] = None  # Maps to syn:hasTemperature
    pressure_bar: Optional[float] = None  # Maps to syn:hasPressure
    time_hours: Optional[float] = None  # Maps to syn:hasReactionTime
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SynthesisProcedureEntity:
    """
    Synthesis Procedure Entity Model
    
    Ontology Class: syn:SynthesisProcedure
    
    Ontology Properties:
    - syn:synthesisText (xsd:string) - Textual description of procedure
    
    Relationships:
    - syn:hasProcedure (from syn:SynthesisProcess) → syn:SynthesisProcess
    """
    procedure_id: str  # Primary key
    synthesis_id: str  # Foreign key to SynthesisProcess
    procedure_text: Optional[str] = None  # Maps to syn:synthesisText
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# Literature Entities
# ============================================================================

@dataclass
class AbstractEntity:
    """
    Abstract Entity Model
    
    Ontology Class: :Abstract (subclass of :Publication)
    
    Ontology Properties:
    - :publicationTitle (xsd:string) - Title of publication
    - :publicationAbstract (xsd:string) - Abstract text
    - :publicationAuthors (xsd:string) - Authors
    - :publicationJournal (xsd:string) - Journal name
    - :publicationDOI (xsd:anyURI) - DOI
    
    Relationships:
    - :describedInAbstract (inverse of :hasAbstract) → :MOF
    
    Linking to MOFs:
    - mof_id MUST match MOFEntity.mof_id (typically CSD code)
    - Use CSD code from data source to link abstracts to MOFs
    """
    abstract_id: str  # Primary key
    mof_id: str  # Foreign key to MOF - MUST match MOFEntity.mof_id (typically CSD code)
    title: Optional[str] = None  # Maps to :publicationTitle
    abstract_text: Optional[str] = None  # Maps to :publicationAbstract
    authors: Optional[str] = None  # Maps to :publicationAuthors
    journal: Optional[str] = None  # Maps to :publicationJournal
    doi: Optional[str] = None  # Maps to :publicationDOI
    data_source: str = ""  # For provenance
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class PublicationEntity:
    """
    Publication Entity Model
    
    Ontology Class: :Publication
    
    Ontology Properties:
    - :publicationTitle (xsd:string) - Title
    - :publicationAbstract (xsd:string) - Abstract text
    - :publicationAuthors (xsd:string) - Authors
    - :publicationJournal (xsd:string) - Journal
    - :publicationDOI (xsd:anyURI) - DOI
    
    Relationships:
    - :describedIn (inverse of :hasPublication) → :MOF
    """
    publication_id: str  # Primary key
    title: str  # Maps to :publicationTitle
    abstract_text: Optional[str] = None  # Maps to :publicationAbstract
    authors: Optional[str] = None  # Maps to :publicationAuthors
    journal: Optional[str] = None  # Maps to :publicationJournal
    doi: Optional[str] = None  # Maps to :publicationDOI
    data_source: str = ""  # For provenance
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# Functionalization Entities
# ============================================================================

@dataclass
class FunctionalizationEntity:
    """
    Functionalization Process Entity Model
    
    Ontology Class: syn:Functionalization
    
    Ontology Properties:
    - syn:functionalizationMethod (xsd:string) - Method description
    - syn:functionalizationDegree (xsd:decimal) - Degree of functionalization
    - syn:functionalGroupName (xsd:string) - Name of functional group
    - syn:functionalGroupSMILES (xsd:string) - SMILES of functional group
    
    Relationships:
    - syn:hasFunctionalization (from :MOF) → :MOF or syn:FunctionalizedMOF
    - syn:hasFunctionalizationType → syn:FunctionalizationType (e.g., syn:AmineFunctionalization)
    - syn:usesFunctionalGroup → :Chemical
    """
    functionalization_id: str  # Primary key
    func_mof_id: str  # Foreign key to FunctionalizedMOF
    functionalization_type: str  # e.g., "AmineFunctionalization", "MetalSubstitution"
    functional_group_id: Optional[str] = None  # Foreign key to ChemicalEntity
    functional_group_name: Optional[str] = None  # Maps to syn:functionalGroupName
    functional_group_smiles: Optional[str] = None  # Maps to syn:functionalGroupSMILES
    functionalization_method: Optional[str] = None  # Maps to syn:functionalizationMethod
    functionalization_degree: Optional[float] = None  # Maps to syn:functionalizationDegree
    data_sources: List[str] = field(default_factory=list)
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# Application/Capability Entities
# ============================================================================

@dataclass
class CapabilityEntity:
    """
    Capability Entity Model
    
    Ontology Class: :Capability (base class)
    Subclasses: :CO2CaptureCapability, :HydrogenStorageCapability, 
                :MethaneStorageCapability, :PhotocatalyticCapability,
                :LuminescentSensingCapability, :DACCapability, :CatalysisCapability
    
    Ontology Properties:
    - :hasValue (xsd:decimal) - Capability value
    
    Relationships:
    - :hasCapability (from :MOF) → :MOF
    - :isRealizedIn → :ApplicationProcess
    
    Linking to MOFs:
    - mof_id MUST match MOFEntity.mof_id (typically CSD code)
    - Use CSD code from data source to link capabilities to MOFs
    """
    capability_id: str  # Primary key
    mof_id: str  # Foreign key to MOF - MUST match MOFEntity.mof_id (typically CSD code)
    capability_type: str  # e.g., "CO2CaptureCapability", "HydrogenStorageCapability"
    value: Optional[float] = None  # Maps to :hasValue
    data_source: str = ""  # For provenance
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ApplicationProcessEntity:
    """
    Application Process Entity Model
    
    Ontology Class: :ApplicationProcess (base class)
    Subclasses: :CatalysisProcess, :GasStorageProcess, :SensingProcess
    
    Relationships:
    - :isRealizedIn (from :Capability) → :Capability
    """
    application_id: str  # Primary key
    application_type: str  # e.g., "CatalysisProcess", "GasStorageProcess"
    description: Optional[str] = None
    data_source: str = ""  # For provenance
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# Data Provenance Entity
# ============================================================================

@dataclass
class DataProvenanceEntity:
    """
    Data Provenance Entity Model
    
    Ontology Class: syn:DataProvenance
    
    Ontology Properties:
    - syn:sourceName (xsd:string) - Name of data source
    - syn:sourceURL (xsd:anyURI) - URL of source
    - syn:dataQuality (xsd:string) - Quality indicator
    - syn:dateAdded (xsd:dateTime) - Date added to KG
    
    Relationships:
    - syn:hasDataSource (from any entity) → Any entity
    """
    provenance_id: str  # Primary key
    source_name: str  # Maps to syn:sourceName
    source_url: Optional[str] = None  # Maps to syn:sourceURL
    data_quality: Optional[str] = None  # Maps to syn:dataQuality
    date_added: str = field(default_factory=lambda: datetime.now().isoformat())  # Maps to syn:dateAdded
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())