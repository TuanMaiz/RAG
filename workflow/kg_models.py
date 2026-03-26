"""Pydantic models for structured knowledge graph extraction.

Uses OpenAI's structured output feature to guarantee valid typed objects
instead of fragile string parsing.
"""

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """An entity extracted from text."""

    name: str = Field(description="The name of the entity, using title case and complete identifier")
    type: str = Field(description="The entity type (e.g., Person, Organization, Location, Event, Concept)")
    description: str = Field(description="A concise description of the entity based on the text")


class Relationship(BaseModel):
    """A relationship between two entities."""

    source: str = Field(description="The name of the source entity")
    target: str = Field(description="The name of the target entity")
    type: str = Field(description="The relationship type in camelCase (e.g., WROTE_ABOUT, BORN_IN, LED)")
    description: str = Field(description="A concise description of the relationship")


class KnowledgeGraph(BaseModel):
    """A knowledge graph extracted from text."""

    entities: list[Entity] = Field(default_factory=list, description="List of entities found in the text")
    relationships: list[Relationship] = Field(default_factory=list, description="List of relationships between entities")
