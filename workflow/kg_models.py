"""Pydantic models for structured knowledge graph extraction.

Uses OpenAI's structured output feature to guarantee valid typed objects
instead of fragile string parsing.
"""

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """An entity extracted from text."""

    name: str = Field(description="The name of the entity, using title case and complete identifier")
    type: str = Field(description="The entity type (e.g., Person, Organization, Location, Event, Concept)")
    description: str = Field(description="A comprehensive 2-3 sentence description of the entity based on the text")
    keywords: list[str] = Field(
        default_factory=list,
        description="3-5 high-level keywords that characterize this entity (e.g., ['French military', 'emperor', 'revolutionary'])"
    )
    source_doc_id: int | None = Field(
        default=None,
        description="The source document/chunk ID this entity was extracted from (for batch extraction)"
    )


class Relationship(BaseModel):
    """A relationship between two entities."""

    source: str = Field(description="The name of the source entity")
    target: str = Field(description="The name of the target entity")
    type: str = Field(description="The relationship type in camelCase (e.g., WROTE_ABOUT, BORN_IN, LED)")
    description: str = Field(description="A concise description of the relationship")
    strength: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Relationship importance/confidence score from 1-10"
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="2-3 high-level thematic keywords summarizing the relationship (e.g., ['Military leadership', 'Power structure'])"
    )
    source_doc_id: int | None = Field(
        default=None,
        description="The source document/chunk ID this relationship was extracted from (for batch extraction)"
    )


class KnowledgeGraph(BaseModel):
    """A knowledge graph extracted from multiple documents (for batch extraction)."""

    entities: list[Entity] = Field(default_factory=list, description="List of entities found across all documents")
    relationships: list[Relationship] = Field(default_factory=list, description="List of relationships between entities")
