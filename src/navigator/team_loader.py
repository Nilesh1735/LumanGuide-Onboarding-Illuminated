"""
Team configuration loader for the Contextual Team Navigator.

Reads a static YAML team configuration file and converts each team member
into LangChain Document objects suitable for FAISS vector store ingestion.
"""

import os
import logging
from typing import Optional

import yaml
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Default path relative to project root
DEFAULT_TEAM_CONFIG_PATH = os.path.join(os.getcwd(), "data", "team_config.yaml")

# Metadata tag that distinguishes team profiles from regular documents
TEAM_DOC_TYPE = "team_profile"


def load_team_config(path: Optional[str] = None) -> list[dict]:
    """
    Load team member data from a YAML configuration file.

    Args:
        path: Absolute or relative path to the YAML file.
              Defaults to data/team_config.yaml in the current working directory.

    Returns:
        List of team member dictionaries as parsed from the YAML "team" key.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML structure is missing the "team" key or is empty.
    """
    config_path = path or DEFAULT_TEAM_CONFIG_PATH

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Team config not found at {config_path}. "
            "Create a team_config.yaml in the data/ directory."
        )

    with open(config_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict) or "team" not in data:
        raise ValueError(
            f"Invalid team config structure in {config_path}. "
            "Expected a top-level 'team' key containing a list of members."
        )

    team_members = data["team"]
    if not team_members or not isinstance(team_members, list):
        raise ValueError(
            f"'team' key in {config_path} is empty or not a list."
        )

    logger.info(
        "Loaded %d team member(s) from %s",
        len(team_members), config_path,
    )
    return team_members


def team_config_to_documents(
    team_data: list[dict],
    tenant_id: str = "default_tenant",
) -> list[Document]:
    """
    Convert parsed team configuration data into LangChain Document objects.

    Each team member produces two documents:
      1. A **summary document** — name, role, expertise tags, project list.
         Optimized for broad semantic matches (e.g. "Who knows about auth?").
      2. An **expertise detail document** — expanded expertise descriptions.
         Optimized for specific technical queries (e.g. "JWT token refresh flow").

    Args:
        team_data: List of team member dicts from load_team_config().
        tenant_id: Tenant identifier stamped on all documents.

    Returns:
        List of Document objects ready for FAISS ingestion.
    """
    documents: list[Document] = []

    for member in team_data:
        name = member.get("name", "Unknown")
        role = member.get("role", "Unknown")
        email = member.get("email", "")
        slack = member.get("slack_handle", "")
        expertise = member.get("expertise", [])
        projects = member.get("projects", [])
        availability = member.get("availability", "unknown")
        timezone = member.get("timezone", "unknown")

        base_metadata = {
            "doc_type": TEAM_DOC_TYPE,
            "team_member_name": name,
            "tenant_id": tenant_id,
            "role": role,
            "email": email,
            "slack_handle": slack,
            "availability": availability,
            "timezone": timezone,
        }

        # 1. Summary document — broad match target
        project_list = ", ".join(
            f"{p.get('name', '')} ({p.get('role', 'contributor')})"
            for p in projects
        )
        expertise_str = ", ".join(expertise) if expertise else "general"

        summary_content = (
            f"{name} is a {role} with expertise in {expertise_str}. "
            f"Projects: {project_list}. "
            f"Slack: {slack}. Availability: {availability}."
        )

        summary_doc = Document(
            page_content=summary_content,
            metadata={
                **base_metadata,
                "document_variant": "summary",
                "projects": [p.get("name", "") for p in projects],
            },
        )
        documents.append(summary_doc)

        # 2. Expertise detail document — specific match target
        if expertise:
            expertise_details = ". ".join(
                f"{name} has deep expertise in {skill}"
                for skill in expertise
            )
            expertise_detail_doc = Document(
                page_content=expertise_details,
                metadata={
                    **base_metadata,
                    "document_variant": "expertise_detail",
                    "projects": [p.get("name", "") for p in projects],
                },
            )
            documents.append(expertise_detail_doc)

    logger.info(
        "Generated %d Document(s) from %d team member(s)",
        len(documents), len(team_data),
    )
    return documents


def ingest_team_config(
    tenant_id: str = "default_tenant",
    config_path: Optional[str] = None,
) -> int:
    """
    Full ingestion pipeline: load YAML → convert to Documents → add to FAISS.

    Args:
        tenant_id: Tenant identifier for metadata stamping.
        config_path: Optional override for the YAML file path.

    Returns:
        Number of documents ingested.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If the config is malformed.
    """
    from src.rag.retriever_setup import add_documents_to_retriever

    team_data = load_team_config(path=config_path)
    documents = team_config_to_documents(team_data, tenant_id=tenant_id)

    if documents:
        add_documents_to_retriever(documents, tenant_id=tenant_id)

    logger.info(
        "Team Navigator: ingested %d document(s) for %d member(s)",
        len(documents), len(team_data),
    )
    return len(documents)
