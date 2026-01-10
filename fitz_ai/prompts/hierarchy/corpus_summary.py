# fitz_ai/prompts/hierarchy/corpus_summary.py
"""
Prompt for generating corpus-level (Level 0) summaries.

Used to synthesize insights across all group summaries.
Identifies: patterns over time, evolution, cross-cutting themes.

Version: v3 - Adapts to code vs documentation content
"""

PROMPT = """You are synthesizing multiple summaries into a corpus-level overview.
This overview will be retrieved when users ask about the big picture, architecture, or patterns.

TASK: Identify patterns and structure across all the content.

ADAPT TO CONTENT TYPE:

**If this is a CODEBASE:**
- Overall architecture: how do the modules/components fit together?
- Key abstractions: main classes, interfaces, design patterns used
- Data flow: how does information move through the system?
- Entry points: where does execution start, what are the main APIs?

**If this is DOCUMENTATION or DATA:**
- Temporal Patterns: how metrics/themes evolved over time
- Consistent Themes: topics that appear across multiple documents
- Progression: what improved, declined, or emerged as priorities
- Strategic Insights: what do these patterns suggest?

FORMAT: Write 3-4 paragraphs. Be specific and cite which sources support each insight.
Optimize for answering "How does this work?" and "What are the trends?" style questions."""
