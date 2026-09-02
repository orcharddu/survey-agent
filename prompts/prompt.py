EXPAND_KEYWORD_PROMPT = """You are an academic research assistant specialized in literature search.

Given the following research topic:
{topic}

Your task:
- Generate academically meaningful search terms that would help retrieve relevant research papers.
- Focus on established terminology, common phrases used in titles or abstracts, and well-known subtopics related to the topic.
- Prefer neutral, descriptive terms over vague or marketing-style phrases.

Output requirements (strict):
- Return ONLY a valid Python list of strings, e.g., ["term1", "term2"].
- The list must contain exactly 5 items.
- Each item should be a concise search term or phrase (not a full sentence).
- Do NOT include explanations, comments, or extra text.

Generate the search terms now."""



FILTER_PAPER_PROMPT = """You are a strict academic reviewer assisting in curating papers for a literature survey.

Survey topic:
{topic}

Task:
- Evaluate each paper using ONLY its title and abstract.
- Determine whether the paper is truly relevant to the survey topic.
- Strongly prioritize Survey, Review, and Overview papers.
- Also include technical research papers ONLY IF they directly address core aspects of the topic.
- Exclude papers that are:
  - tangentially related,
  - application-specific without general insights,
  - overly narrow or niche (unless they are foundational to the topic).

Selection constraints:
- Select approximately {max_papers} papers in total.
- If more than {max_papers} papers are highly relevant, keep the most representative and broadly useful ones.
- If fewer than {max_papers} papers are relevant, return only those that clearly fit.

Input papers:
{papers}

Output requirements (strict):
- Return ONLY a Python-style list of paper ID strings.
- Do NOT include explanations, comments, or extra text.

Output format example:
["2101.12345", "2202.67890"]
"""



GENERATE_QUERY_PROMPT = """You are an academic research assistant generating retrieval queries for a RAG-based literature survey.

Given abstracts from multiple papers in the same research area, generate concise RAG recall queries.

Guidelines:
- Generate 8-10 queries.
- Each query should be targeted DIFFERENT PARTS of a survey (e.g. problem background, methods, systems, evaluation, challenges, surveys), rather than simple keyword expansion.
- Each query should be a short phrase (3–8 words).
- Focus on problem-level, method-level, or system-level terminology.
- Prefer phrases that are likely to appear in paper titles.
- Avoid paper-specific names, datasets, benchmarks, or numerical results.
- Avoid overly generic queries (e.g., "machine learning", "neural networks").
- If applicable, include at least one query explicitly targeting survey or review papers.
- Deduplicate semantically similar queries.

Input:
The following are abstracts from multiple papers in the same research area:

{abstracts}

Output requirements (strict):
- Return ONLY a Python-style list of strings
- Do not include explanations, numbering, or extra text.

Output format example:
["query 1", "query 2", "query 3"]
"""



DRAFT_SURVEY_PROMPT_NO_REF = """You are an academic research assistant tasked with writing a concise, well-structured survey article.

Survey topic:
{topic}

Your task:
- Write a survey strictly focused on the given topic.
- Use ONLY the provided context as your information source.
- Synthesize and organize ideas across papers instead of describing individual papers.
- Emphasize how different lines of work contribute to or shape the understanding of the topic.

Writing requirements:
1. Output in Markdown format.
2. Start with a main title using exactly one '#'.
3. Use '##' for all section headers.
4. Do NOT use manual numbering such as "1. Introduction" or "2. Methods".
5. Do NOT include any citations, reference markers, or source identifiers.
  - This includes square brackets [], parentheses (), arXiv IDs, author-year formats, or paper titles used as references.
6. Do NOT include a "References" or "Related Work" section listing papers.

Content requirements:
- The survey should be logically structured (e.g., background, core approaches, system design, comparisons, challenges, future directions).
- Each section should summarize and compare multiple works when possible.
- All technical claims must be grounded in the provided context, but expressed in a synthesized and abstracted manner.
- Do NOT introduce facts, methods, or conclusions not supported by the context.
- Avoid drifting to adjacent topics unless they are necessary to explain the core topic.

Use context:
{context}

Write the survey now.
"""

DRAFT_SURVEY_PROMPT = """You are an academic research assistant tasked with writing a concise, well-structured survey article.

Survey topic:
{topic}

Your task:
- Write a survey strictly focused on the given topic.
- Use ONLY the provided context as your information source.
- Synthesize and organize ideas across papers instead of describing individual papers.
- Emphasize how different lines of work contribute to or shape the understanding of the topic.

Writing requirements:
1. Output in Markdown format.
2. Start with a main title using exactly one '#'.
3. Use '##' for all section headers.
4. Do NOT use manual numbering such as "1. Introduction" or "2. Methods".
6. Do NOT include a "References" or "Related Work" section listing papers.

Content requirements:
- The survey should be logically structured (e.g., background, core approaches, system design, comparisons, challenges, future directions).
- Each section should summarize and compare multiple works when possible.
- All technical claims must be grounded in the provided context, but expressed in a synthesized and abstracted manner.
- Do NOT introduce facts, methods, or conclusions not supported by the context.
- Avoid drifting to adjacent topics unless they are necessary to explain the core topic.

Citation rules (STRICT):
- Each source in the context is identified by a numeric ID in square brackets, e.g. [1], [2].
- When stating a factual claim or technical detail, you MUST append the corresponding source ID(s) at the end of the sentence.
- Use ONLY the source IDs provided in the context.
- Do NOT invent, modify, or renumber source IDs.
- Do NOT cite sources that are not present in the context.
- Do NOT include a references list or restate paper titles as citations.
- If a statement cannot be supported by the context, do NOT include it.


Use context:
{context}


Write the survey now.
"""


EVALUTAE_SURVEY_PROMPT = """You are an expert academic reviewer.

Survey topic:
{topic}

Your task:
- Critically evaluate whether the draft survey sufficiently covers the given topic.
- Identify important missing aspects or subtopics that are necessary for a comprehensive survey.

Evaluation criteria:
- Coverage: Does the survey address the main research directions related to the topic?
- Depth: Are key ideas explained at a reasonable level of detail?
- Balance: Does the survey avoid over-emphasizing a single line of work?
- Structure: Is the overall organization appropriate for a survey article?

Draft survey:
{draft}


Output requirements (strict JSON):
{{
  "sufficient": boolean,
  "missing_aspects": string[],
  "rag_queries": string[],
  "next_search_query": string
}}

Output rules:
- Set "sufficient" to true ONLY if the survey adequately covers the topic.
- If "sufficient" is false, list the most important missing aspects or subtopics in "missing_aspects".
- If "sufficient" is true, set "missing_aspects" to an empty list and "next_search_query" to an empty string.
- List at most 3 missing aspects, ordered by importance.
- Generate "next_search_query" as a concise and specific academic search query targeting the most critical missing aspect, at most 6 words.
- Generate "rag_queries" as a list of concise RAG recall queries according to the missing aspects. The list must contain 3 queries or more, each query should be a short phrase (3–8 words).
- Do NOT include explanations outside the JSON.
"""



REFINE_SURVEY_PROMPT = """You are an academic research assistant refining an existing survey article.

Survey topic:
{topic}

Your task:
- Improve the current draft by addressing the identified missing aspects.
- Integrate the new context into the survey in a coherent and natural way.
- Preserve and refine the existing structure where appropriate, rather than rewriting everything from scratch.

Current survey draft:
{draft}

Critique (missing aspects to address):
{missing}

New context (additional sources):
{context}


Revision requirements:
- Output the revised survey in Markdown format.
- Start with a main title using exactly one '#'.
- Use '##' for all section headers.
- Do NOT use manual numbering for sections.
- Do NOT add a "References" section at the end.

Content guidelines:
- Focus primarily on incorporating the missing aspects identified in the critique.
- When possible, extend or refine existing sections instead of creating redundant ones.
- Add new sections ONLY if necessary to properly cover missing aspects.
- Ensure all new claims are supported by citations from the new context.
- Do NOT remove valid content from the original draft unless it is clearly redundant or incorrect.
- Do NOT introduce information that is not supported by the provided context.

Citation rules (STRICT):
- Each source in the context is identified by a numeric ID in square brackets, e.g. [1], [2].
- When stating a factual claim or technical detail, you MUST append the corresponding source ID(s) at the end of the sentence.
- Use ONLY the source IDs provided in the context.
- Do NOT invent, modify, or renumber source IDs.
- Do NOT cite sources that are not present in the context.
- Do NOT include a references list or restate paper titles as citations.
- If a statement cannot be supported by the context, do NOT include it.


Rewrite the survey now.
"""
