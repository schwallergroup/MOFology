"""
Data models for all MOF Knowledge Graph links.

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
from enum import Enum



"""
relationships to be modeled:
- :hasSpaceGroup (from :MOF) → :MOF (many-to-many relationship)
- :hasCrystalSystem (from :MOF) → :MOF (many-to-many relationship)
- :hasStructuralProperty (from :MOF) → :MOF
- :hasPropertyOwner (inverse of :hasProperty) → :MOF
- :hasStructuralPropertyOwner (inverse of :hasStructuralProperty) → :MOF
- :hasComputationalPropertyOwner (inverse of :hasComputationalProperty) → :MOF
- :hasPhysicalPropertyOwner (inverse of :hasPhysicalProperty) → :MOF
- :hasMOF (inverse of syn:hasSynthesisProcess) → :MOF
- syn:hasCondition → syn:SynthesisCondition
- syn:hasProcedure → syn:SynthesisProcedure
- syn:usesSolvent → syn:Solvent
- syn:usesMetalPrecursor → syn:MetalPrecursor
- syn:usesLinkerPrecursor → syn:LinkerPrecursor
- syn:hasCondition (from syn:SynthesisProcess) → syn:SynthesisProcess
- :describedInAbstract (inverse of :hasAbstract) → :MOF
- :describedIn (inverse of :hasPublication) → :MOF
- syn:hasFunctionalization (from :MOF) → :MOF or syn:FunctionalizedMOF
- syn:hasFunctionalizationType → syn:FunctionalizationType (e.g., syn:AmineFunctionalization)
- syn:usesFunctionalGroup → :Chemical
- :hasCapability (from :MOF) → :MOF
- :isRealizedIn → :ApplicationProcess
- syn:hasDataSource (from any entity) → Any entity
"""


# ============================================================================
# Enums for Constrained Values
# ============================================================================

class PropertyType(Enum):
    """Property type classification matching ontology subclasses."""
    STRUCTURAL = "StructuralProperty"
    COMPUTATIONAL = "ComputationalProperty"
    PHYSICAL = "PhysicalProperty"


class FunctionalizationType(Enum):
    """Functionalization types from ontology."""
    AMINE = "AmineFunctionalization"
    METAL_SUBSTITUTION = "MetalSubstitution"
    LINKER_MODIFICATION = "LinkerModification"
    GRAFTING = "Grafting"
    METAL_EXCHANGE = "MetalExchange"


class CapabilityType(Enum):
    """Capability types from ontology."""
    CO2_CAPTURE = "CO2CaptureCapability"
    HYDROGEN_STORAGE = "HydrogenStorageCapability"
    METHANE_STORAGE = "MethaneStorageCapability"
    PHOTOCATALYTIC = "PhotocatalyticCapability"
    LUMINESCENT_SENSING = "LuminescentSensingCapability"
    DAC = "DACCapability"
    CATALYSIS = "CatalysisCapability"


class ApplicationType(Enum):
    """Application process types from ontology."""
    CATALYSIS = "CatalysisProcess"
    GAS_STORAGE = "GasStorageProcess"
    SENSING = "SensingProcess"


# ============================================================================
# Core MOF Component Relationships
# ============================================================================

@dataclass
class HasLinkerRelation:
    """
    MOF → OrganicLinker relationship
    
    Ontology Property: :hasLinker
    Domain: :MOF
    Range: :OrganicLinker
    Inverse: :usedInMOF
    
    Cardinality: MOF must have at least 1 linker (owl:minCardinality 1)
    
    Usage: Links a MOF to its organic linker component(s).
    A MOF can have multiple linkers (e.g., mixed-linker MOFs).
    """
    relation_id: str  # Primary key for this relationship instance
    mof_id: str  # Foreign key to MOFEntity.mof_id (subject)
    linker_id: str  # Foreign key to LinkerEntity.linker_id (object)
    # Optional metadata
    stoichiometry: Optional[float] = None  # Linker stoichiometry in MOF
    is_primary: bool = True  # Whether this is the primary/main linker
    data_source: str = ""
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class HasMetalNodeRelation:
    """
    MOF → MetalCluster relationship
    
    Ontology Property: :hasMetalNode
    Domain: :MOF
    Range: :MetalCluster
    Inverse: :isComponentOf
    
    Usage: Links a MOF to its metal cluster/node component(s).
    A MOF can have multiple metal nodes (e.g., bimetallic MOFs).
    """
    relation_id: str
    mof_id: str  # Foreign key to MOFEntity.mof_id
    cluster_id: str  # Foreign key to MetalClusterEntity.cluster_id
    # Optional metadata
    stoichiometry: Optional[float] = None
    is_primary: bool = True
    data_source: str = ""
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# Property Relationships
# ============================================================================

@dataclass
class HasPropertyRelation:
    """
    MOF → MaterialProperty relationship (generic)
    
    Ontology Property: :hasProperty
    Domain: :MOF
    Range: :MaterialProperty
    Inverse: :hasPropertyOwner
    
    Usage: Generic property link. Use specific subproperties when possible.
    
    Note: The PropertyEntity already contains mof_id as a foreign key,
    so this relation model is mainly useful for:
    1. Explicit relationship tracking with metadata
    2. Cases where you need relationship-level provenance
    3. N-ary relationship scenarios
    
    For most cases, PropertyEntity.mof_id is sufficient.
    """
    relation_id: str
    mof_id: str  # Foreign key to MOFEntity.mof_id
    property_id: str  # Foreign key to PropertyEntity.property_id
    property_type: PropertyType = PropertyType.STRUCTURAL
    data_source: str = ""
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class HasStructuralPropertyRelation:
    """
    MOF → StructuralProperty relationship
    
    Ontology Property: :hasStructuralProperty
    Domain: :MOF
    Range: :StructuralProperty
    Inverse: :hasStructuralPropertyOwner
    
    Examples: Space group, crystal system, unit cell parameters, density
    """
    relation_id: str
    mof_id: str
    property_id: str
    data_source: str = ""
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class HasComputationalPropertyRelation:
    """
    MOF → ComputationalProperty relationship
    
    Ontology Property: :hasComputationalProperty
    Domain: :MOF
    Range: :ComputationalProperty
    Inverse: :hasComputationalPropertyOwner
    
    Examples: Band gap, PLD, LCD, ASA, void fraction, CO2 uptake (computed)
    """
    relation_id: str
    mof_id: str
    property_id: str
    data_source: str = ""
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class HasPhysicalPropertyRelation:
    """
    MOF → PhysicalProperty relationship
    
    Ontology Property: :hasPhysicalProperty
    Domain: :MOF
    Range: :PhysicalProperty
    Inverse: :hasPhysicalPropertyOwner
    
    Examples: Thermal stability, magnetic properties, luminescence
    """
    relation_id: str
    mof_id: str
    property_id: str
    data_source: str = ""
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# Structural Classification Relationships
# ============================================================================

@dataclass
class HasSpaceGroupRelation:
    """
    MOF → SpaceGroup relationship
    
    Ontology Property: :hasStructuralProperty (specialized for space group)
    
    Usage: Links a MOF to its crystallographic space group.
    Many MOFs share the same space group, enabling grouping/querying.
    """
    relation_id: str
    mof_id: str  # Foreign key to MOFEntity.mof_id
    space_group_id: str  # Foreign key to SpaceGroupEntity.space_group_id
    data_source: str = ""
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class HasCrystalSystemRelation:
    """
    MOF → CrystalSystem relationship
    
    Ontology Property: :hasStructuralProperty (specialized for crystal system)
    
    Usage: Links a MOF to its crystal system classification.
    """
    relation_id: str
    mof_id: str
    crystal_system_id: str  # Foreign key to CrystalSystemEntity.crystal_system_id
    data_source: str = ""
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class HasTopologyRelation:
    """
    MOF → Topology relationship
    
    Ontology Property: :hasTopology
    Domain: :MOF
    Range: :Topology
    
    Usage: Links a MOF to its framework topology.
    Enables efficient querying like "all MOFs with pcu topology".
    """
    relation_id: str
    mof_id: str  # Foreign key to MOFEntity.mof_id
    topology_id: str  # Foreign key to TopologyEntity.topology_id
    data_source: str = ""
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class HasLatticeParametersRelation:
    """
    MOF → LatticeParameters relationship
    
    Ontology Property: :hasStructuralProperty (specialized for lattice params)
    
    Usage: Links a MOF to its unit cell lattice parameters.
    Unlike space group/crystal system, lattice params are MOF-specific.
    
    Note: LatticeParameterEntity already contains mof_id, so this is
    mainly for explicit relationship tracking if needed.
    """
    relation_id: str
    mof_id: str
    lattice_param_id: str  # Foreign key to LatticeParameterEntity
    data_source: str = ""
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# Synthesis Relationships
# ============================================================================

@dataclass
class HasSynthesisProcessRelation:
    """
    MOF → SynthesisProcess relationship
    
    Ontology Property: syn:hasSynthesisProcess
    Domain: :MOF
    Range: syn:SynthesisProcess
    Inverse: :hasMOF
    
    Usage: Links a MOF to its synthesis process(es).
    A MOF may have multiple synthesis routes documented.
    """
    relation_id: str
    mof_id: str  # Foreign key to MOFEntity.mof_id
    synthesis_id: str  # Foreign key to SynthesisProcessEntity.synthesis_id
    is_primary_synthesis: bool = True  # Whether this is the main/original synthesis
    data_source: str = ""
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class HasConditionRelation:
    """
    SynthesisProcess → SynthesisCondition relationship
    
    Ontology Property: syn:hasCondition
    Domain: syn:SynthesisProcess
    Range: syn:SynthesisCondition
    
    Usage: Links a synthesis process to its conditions (T, P, time).
    """
    relation_id: str
    synthesis_id: str  # Foreign key to SynthesisProcessEntity
    condition_id: str  # Foreign key to SynthesisConditionEntity
    step_number: Optional[int] = None  # For multi-step syntheses
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class HasProcedureRelation:
    """
    SynthesisProcess → SynthesisProcedure relationship
    
    Ontology Property: syn:hasProcedure
    Domain: syn:SynthesisProcess
    Range: syn:SynthesisProcedure
    
    Usage: Links synthesis process to textual procedure description.
    """
    relation_id: str
    synthesis_id: str
    procedure_id: str  # Foreign key to SynthesisProcedureEntity
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class UsesSolventRelation:
    """
    SynthesisProcess → Solvent relationship
    
    Ontology Property: syn:usesSolvent
    Domain: syn:SynthesisProcess
    Range: syn:Solvent
    Inverse: :usedInSynthesis
    
    Usage: Links synthesis process to solvent(s) used.
    Multiple solvents may be used in a single synthesis.
    """
    relation_id: str
    synthesis_id: str  # Foreign key to SynthesisProcessEntity
    solvent_id: str  # Foreign key to SolventEntity
    # Quantity information
    volume_ml: Optional[float] = None
    ratio: Optional[float] = None  # For mixed solvents
    is_primary: bool = True
    data_source: str = ""
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class UsesMetalPrecursorRelation:
    """
    SynthesisProcess → MetalPrecursor relationship
    
    Ontology Property: syn:usesMetalPrecursor
    Domain: syn:SynthesisProcess
    Range: syn:MetalPrecursor
    Inverse: :usedAsMetalPrecursorIn
    
    Usage: Links synthesis process to metal precursor(s) used.
    """
    relation_id: str
    synthesis_id: str
    precursor_id: str  # Foreign key to ChemicalEntity (MetalPrecursor)
    # Quantity information
    amount_mmol: Optional[float] = None
    amount_g: Optional[float] = None
    data_source: str = ""
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class UsesLinkerPrecursorRelation:
    """
    SynthesisProcess → LinkerPrecursor relationship
    
    Ontology Property: syn:usesLinkerPrecursor
    Domain: syn:SynthesisProcess
    Range: syn:LinkerPrecursor
    Inverse: :usedAsLinkerPrecursorIn
    
    Usage: Links synthesis process to linker precursor(s) used.
    """
    relation_id: str
    synthesis_id: str
    precursor_id: str  # Foreign key to ChemicalEntity (LinkerPrecursor)
    amount_mmol: Optional[float] = None
    amount_g: Optional[float] = None
    data_source: str = ""
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SimilarSynthesisToRelation:
    """
    MOF → MOF relationship (symmetric)
    
    Ontology Property: syn:similarSynthesisTo
    Domain: :MOF
    Range: :MOF
    Note: owl:inverseOf syn:similarSynthesisTo (symmetric)
    
    Usage: Links MOFs that have similar synthesis procedures.
    Useful for synthesis route recommendation.
    """
    relation_id: str
    mof_id_1: str  # First MOF
    mof_id_2: str  # Second MOF
    similarity_score: Optional[float] = None  # 0.0 to 1.0
    similarity_basis: Optional[str] = None  # What makes them similar
    data_source: str = ""
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# Functionalization Relationships
# ============================================================================

@dataclass
class DerivedFromRelation:
    """
    FunctionalizedMOF → MOF relationship
    
    Ontology Property: syn:derivedFrom
    Domain: syn:FunctionalizedMOF
    Range: :MOF
    
    Usage: Links a functionalized MOF to its parent (unfunctionalized) MOF.
    """
    relation_id: str
    func_mof_id: str  # Foreign key to FunctionalizedMOFEntity
    parent_mof_id: str  # Foreign key to parent MOFEntity.mof_id
    data_source: str = ""
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class HasFunctionalizationRelation:
    """
    MOF → Functionalization relationship
    
    Ontology Property: syn:hasFunctionalization
    Domain: :MOF
    Range: syn:Functionalization
    
    Usage: Links a MOF to its functionalization process(es).
    """
    relation_id: str
    mof_id: str  # Foreign key to MOFEntity (can be FunctionalizedMOF)
    functionalization_id: str  # Foreign key to FunctionalizationEntity
    data_source: str = ""
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class HasFunctionalizationTypeRelation:
    """
    Functionalization → FunctionalizationType relationship
    
    Ontology Property: syn:hasFunctionalizationType
    Domain: syn:Functionalization
    Range: syn:FunctionalizationType
    
    Usage: Classifies the type of functionalization applied.
    """
    relation_id: str
    functionalization_id: str
    functionalization_type: FunctionalizationType
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class UsesFunctionalGroupRelation:
    """
    Functionalization → Chemical relationship
    
    Ontology Property: syn:usesFunctionalGroup
    Domain: syn:Functionalization
    Range: :Chemical
    
    Usage: Links functionalization to the functional group chemical used.
    """
    relation_id: str
    functionalization_id: str
    functional_group_id: str  # Foreign key to ChemicalEntity
    data_source: str = ""
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# Publication/Literature Relationships
# ============================================================================

@dataclass
class HasPublicationRelation:
    """
    MOF → Publication relationship
    
    Ontology Property: :hasPublication
    Domain: :MOF
    Range: :Publication
    Inverse: :describedIn
    
    Usage: Links a MOF to publications about it.
    """
    relation_id: str
    mof_id: str  # Foreign key to MOFEntity.mof_id
    publication_id: str  # Foreign key to PublicationEntity
    is_primary_reference: bool = False  # Whether this is the original/discovery paper
    data_source: str = ""
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class HasAbstractRelation:
    """
    MOF → Abstract relationship
    
    Ontology Property: :hasAbstract
    Domain: :MOF
    Range: :Abstract
    Inverse: :describedInAbstract
    
    Usage: Links a MOF to publication abstracts.
    Note: AbstractEntity already has mof_id, so this is mainly for
    explicit relationship tracking if needed.
    """
    relation_id: str
    mof_id: str
    abstract_id: str  # Foreign key to AbstractEntity
    data_source: str = ""
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# Capability/Application Relationships
# ============================================================================

@dataclass
class HasCapabilityRelation:
    """
    MOF → Capability relationship
    
    Ontology Property: :hasCapability
    Domain: :MOF
    Range: :Capability
    
    Usage: Links a MOF to its demonstrated or inferred capabilities.
    Note: Capabilities can be inferred via OWL reasoning rules in ontology.
    """
    relation_id: str
    mof_id: str  # Foreign key to MOFEntity.mof_id
    capability_id: str  # Foreign key to CapabilityEntity
    capability_type: CapabilityType
    is_inferred: bool = False  # True if inferred by reasoning, False if asserted
    inference_basis: Optional[str] = None  # What property/feature led to inference
    data_source: str = ""
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class IsRealizedInRelation:
    """
    Capability → ApplicationProcess relationship
    
    Ontology Property: :isRealizedIn
    Domain: :Capability
    Range: :ApplicationProcess
    
    Usage: Links a capability to application processes where it is realized.
    """
    relation_id: str
    capability_id: str  # Foreign key to CapabilityEntity
    application_id: str  # Foreign key to ApplicationProcessEntity
    application_type: ApplicationType
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# Provenance Relationships
# ============================================================================

@dataclass
class HasDataSourceRelation:
    """
    Any Entity → DataProvenance relationship
    
    Ontology Property: syn:hasDataSource
    Domain: (any entity)
    Range: syn:DataProvenance
    
    Usage: Links any entity to its data provenance information.
    This is a generic relation - most entities track data_source as a field,
    but this allows linking to full DataProvenanceEntity for detailed tracking.
    """
    relation_id: str
    entity_type: str  # Type of entity (e.g., "MOF", "Property", "Synthesis")
    entity_id: str  # Primary key of the entity
    provenance_id: str  # Foreign key to DataProvenanceEntity
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# Transitive/Derived Relationships (from property chains)
# ============================================================================

@dataclass
class DirectlyUsesSolventRelation:
    """
    MOF → Solvent relationship (derived via property chain)
    
    Ontology Property: :directlyUsesSolvent
    Property Chain: syn:hasSynthesisProcess o syn:usesSolvent
    
    Usage: Direct link from MOF to solvent (inferred from synthesis).
    This can be materialized for query performance or computed on-the-fly.
    
    Note: This is derived/inferred - typically computed, not stored.
    """
    relation_id: str
    mof_id: str
    solvent_id: str
    via_synthesis_id: str  # The synthesis process that established this link
    is_materialized: bool = False  # Whether this was explicitly stored
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class HasSynthesisConditionRelation:
    """
    MOF → SynthesisCondition relationship (derived via property chain)
    
    Ontology Property: :hasSynthesisCondition
    Property Chain: syn:hasSynthesisProcess o syn:hasCondition
    
    Usage: Direct link from MOF to synthesis conditions.
    """
    relation_id: str
    mof_id: str
    condition_id: str
    via_synthesis_id: str
    is_materialized: bool = False
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# Component Hierarchy Relationships
# ============================================================================

@dataclass
class HasComponentRelation:
    """
    Generic part-whole relationship (transitive)
    
    Ontology Property: :hasComponent
    Note: owl:TransitiveProperty
    Inverse: :partOf
    
    Usage: Generic hierarchical component relationship.
    Used when specific relationships (hasLinker, hasMetalNode) don't apply.
    """
    relation_id: str
    parent_entity_type: str
    parent_entity_id: str
    component_entity_type: str
    component_entity_id: str
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# Utility: Relationship Registry
# ============================================================================

# Map ontology predicates to relation models for validation/lookup
PREDICATE_REGISTRY = {
    # Core MOF relationships
    ":hasLinker": HasLinkerRelation,
    ":hasMetalNode": HasMetalNodeRelation,
    
    # Property relationships
    ":hasProperty": HasPropertyRelation,
    ":hasStructuralProperty": HasStructuralPropertyRelation,
    ":hasComputationalProperty": HasComputationalPropertyRelation,
    ":hasPhysicalProperty": HasPhysicalPropertyRelation,
    
    # Structural classification
    ":hasSpaceGroup": HasSpaceGroupRelation,
    ":hasCrystalSystem": HasCrystalSystemRelation,
    ":hasTopology": HasTopologyRelation,
    ":hasLatticeParameters": HasLatticeParametersRelation,
    
    # Synthesis relationships
    "syn:hasSynthesisProcess": HasSynthesisProcessRelation,
    "syn:hasCondition": HasConditionRelation,
    "syn:hasProcedure": HasProcedureRelation,
    "syn:usesSolvent": UsesSolventRelation,
    "syn:usesMetalPrecursor": UsesMetalPrecursorRelation,
    "syn:usesLinkerPrecursor": UsesLinkerPrecursorRelation,
    "syn:similarSynthesisTo": SimilarSynthesisToRelation,
    
    # Functionalization
    "syn:derivedFrom": DerivedFromRelation,
    "syn:hasFunctionalization": HasFunctionalizationRelation,
    "syn:hasFunctionalizationType": HasFunctionalizationTypeRelation,
    "syn:usesFunctionalGroup": UsesFunctionalGroupRelation,
    
    # Publications
    ":hasPublication": HasPublicationRelation,
    ":hasAbstract": HasAbstractRelation,
    
    # Capabilities
    ":hasCapability": HasCapabilityRelation,
    ":isRealizedIn": IsRealizedInRelation,
    
    # Provenance
    "syn:hasDataSource": HasDataSourceRelation,
    
    # Derived/chain properties
    ":directlyUsesSolvent": DirectlyUsesSolventRelation,
    ":hasSynthesisCondition": HasSynthesisConditionRelation,
    
    # Generic
    ":hasComponent": HasComponentRelation,
}


# Domain/Range constraints for validation
PREDICATE_CONSTRAINTS = {
    ":hasLinker": {"domain": "MOF", "range": "OrganicLinker"},
    ":hasMetalNode": {"domain": "MOF", "range": "MetalCluster"},
    ":hasProperty": {"domain": "MOF", "range": "MaterialProperty"},
    ":hasStructuralProperty": {"domain": "MOF", "range": "StructuralProperty"},
    ":hasComputationalProperty": {"domain": "MOF", "range": "ComputationalProperty"},
    ":hasPhysicalProperty": {"domain": "MOF", "range": "PhysicalProperty"},
    ":hasTopology": {"domain": "MOF", "range": "Topology"},
    "syn:hasSynthesisProcess": {"domain": "MOF", "range": "SynthesisProcess"},
    "syn:hasCondition": {"domain": "SynthesisProcess", "range": "SynthesisCondition"},
    "syn:usesSolvent": {"domain": "SynthesisProcess", "range": "Solvent"},
    "syn:usesMetalPrecursor": {"domain": "SynthesisProcess", "range": "MetalPrecursor"},
    "syn:usesLinkerPrecursor": {"domain": "SynthesisProcess", "range": "LinkerPrecursor"},
    "syn:derivedFrom": {"domain": "FunctionalizedMOF", "range": "MOF"},
    "syn:hasFunctionalization": {"domain": "MOF", "range": "Functionalization"},
    "syn:hasFunctionalizationType": {"domain": "Functionalization", "range": "FunctionalizationType"},
    "syn:usesFunctionalGroup": {"domain": "Functionalization", "range": "Chemical"},
    ":hasPublication": {"domain": "MOF", "range": "Publication"},
    ":hasAbstract": {"domain": "MOF", "range": "Abstract"},
    ":hasCapability": {"domain": "MOF", "range": "Capability"},
    ":isRealizedIn": {"domain": "Capability", "range": "ApplicationProcess"},
}