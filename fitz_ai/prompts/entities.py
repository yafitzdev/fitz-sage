# fitz_ai/prompts/entities.py
"""
Prompts for entity extraction.

Extracts both domain concepts and named entities from content.

Version: v1 - Initial entity extraction prompt
"""

EXTRACTION_PROMPT = """Extract entities from the following content.

ENTITY TYPES TO EXTRACT:

Domain Concepts:
- class: Class definitions
- function: Function or method definitions
- api: API endpoints or external services
- module: Python modules or packages
- config: Configuration keys or settings
- system: Systems, services, or infrastructure components

Named Entities:
- person: People's names
- organization: Companies, teams, or groups
- location: Physical or logical locations
- product: Products, tools, or technologies
- concept: Abstract concepts or domain terminology

EXTRACTION RULES:
1. Extract only entities explicitly mentioned in the content
2. Each entity needs: name, type, and brief description (1 sentence)
3. Use the most specific type that applies
4. For code: prefer class/function/module over concept
5. Limit to 10 most significant entities

CONTENT:
{content}

Return ONLY a valid JSON array, no markdown formatting:
[{{"name": "EntityName", "type": "entity_type", "description": "Brief context"}}]

If no entities found, return: []"""


EXTRACTION_PROMPT_CODE = """Extract entities from the following code.

ENTITY TYPES TO EXTRACT:
- class: Class definitions and their purpose
- function: Function definitions and their purpose
- method: Important methods on classes
- api: API endpoints, external service calls
- module: Imported modules or packages
- config: Configuration constants or settings

EXTRACTION RULES:
1. Focus on definitions, not usages
2. Include docstring content in descriptions
3. Limit to 10 most significant entities
4. Skip trivial helpers and internal methods

CODE:
{content}

Return ONLY a valid JSON array, no markdown formatting:
[{{"name": "EntityName", "type": "entity_type", "description": "Brief context"}}]

If no entities found, return: []"""


__all__ = ["EXTRACTION_PROMPT", "EXTRACTION_PROMPT_CODE"]
