# fitz_sage/code/prompts.py
"""LLM prompt templates for standalone code retrieval."""

HUB_FILES_HINT = (
    "--- ARCHITECTURAL HUBS (files importing many subsystems "
    "— high likelihood of relevance for integration tasks) ---\n"
    "{hub_files}\n\n"
)

EXPAND_AND_SELECT_PROMPT = (
    "You are selecting files from a codebase to answer a question.\n\n"
    "Question: {query}\n\n"
    "Below is the structural index of every file. Each entry shows "
    "file path, classes, functions, and imports.\n\n"
    "{structural_index}\n\n"
    "{hub_files_hint}"
    "First, generate 3-5 search terms, synonyms, and related concepts for the "
    "question. Then, using those terms, select 5-15 relevant files.\n\n"
    "Include files that:\n"
    "- Contain the code being asked about\n"
    "- Define protocols, base classes, or types used by relevant code\n"
    "- Contain configuration or factory patterns that affect the relevant code\n\n"
    "Err on the side of including MORE files — missing a relevant file is worse "
    "than including an extra one.\n\n"
    "Return JSON:\n"
    '```json\n{{"search_terms": ["term1", "term2"], '
    '"files": ["path/to/file1.py", "path/to/file2.py"]}}\n```'
)

NEIGHBOR_SCREEN_PROMPT = (
    "You are filtering sibling files for relevance to a question.\n\n"
    "Question: {query}\n\n"
    "A relevant file was found in this directory. Below are its sibling files "
    "with their structural info.\n\n"
    "Trigger file: {trigger_file}\n\n"
    "{sibling_index}\n\n"
    "Which of these sibling files are also relevant to the question?\n\n"
    "Return ONLY a JSON array of file paths:\n"
    '```json\n["path/to/file1.py", "path/to/file2.py"]\n```'
)
