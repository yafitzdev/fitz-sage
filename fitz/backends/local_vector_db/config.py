from pathlib import Path
from pydantic import BaseModel, Field


class LocalVectorDBConfig(BaseModel):
    """
    Configuration for local vector database backends.
    """

    path: Path = Field(
        default_factory=lambda: Path.home() / ".fitz" / "vector_db",
        description="Base directory for local vector database storage",
    )

    persist: bool = Field(
        default=True,
        description="Whether to persist the vector index to disk",
    )
