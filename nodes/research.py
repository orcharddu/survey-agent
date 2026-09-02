from typing import List, Tuple

import arxiv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from graph.state import AgentState
from providers.knowledge_base import get_knowledge_base
from providers.providers import get_llm
from prompts.prompt import EXPAND_KEYWORD_PROMPT, FILTER_PAPER_PROMPT, GENERATE_QUERY_PROMPT
from utils.chainlit_msg import send_chainlit_message


def expand_keywords(topic: str) -> List[str]:
    logger.info(f"Expanding keywords for '{topic}'")

    llm = get_llm().bind(temperature=0.8)
    prompt = ChatPromptTemplate.from_template(EXPAND_KEYWORD_PROMPT)

    chain = prompt | llm | JsonOutputParser()
    input_prompt = prompt.invoke({"topic": topic})
    logger.debug(f"Input: {input_prompt}")
    try:
        keywords = chain.invoke({"topic": topic})
        logger.debug(f"Output: {keywords}")
        if topic not in keywords:
            keywords.insert(0, topic)
        logger.info(f"Generated Keywords: {keywords}")
    except Exception as e:
        logger.error(f"Keyword extraction failed: {e}, use default topic: {topic}")
        keywords = [topic]
    return keywords

def generate_rag_query(topic: str, papers: List[arxiv.Result]):
    logger.info("Generating queries for future RAG retrieval based on paper abstracts")


    default_queries = [f"{topic} background",
                        f"{topic} taxonomy",
                        f"{topic} methods",
                        f"{topic} evaluation",
                        f"{topic} challenges"]
    if not papers:
        return default_queries

    paper_descriptions = ""
    for p in papers:
        clean_summary = p.summary.replace("\n", " ")[:1000]
        paper_descriptions += f"{clean_summary} \n\n"

    llm = get_llm().bind(temperature=0.3)
    prompt = ChatPromptTemplate.from_template(GENERATE_QUERY_PROMPT)
    input_prompt = prompt.invoke({"abstracts": paper_descriptions})
    logger.debug(f"Input: {input_prompt}")

    chain = prompt | llm | JsonOutputParser()

    try:
        result = chain.invoke({"abstracts": paper_descriptions})
        logger.debug(f"Output: {result}")
        logger.info(f"Generated RAG queries: {result}")
        if not result:
            return default_queries
        return result
    except Exception as e:
        logger.error(f"Generating RAG query failed: {e}, use default topic: {topic}")
        return default_queries


def update_reference_db(state: AgentState, papers: List[arxiv.Result]) -> List[Tuple[str, str]]:
    current_db = state.get("reference_db", [])
    if not isinstance(current_db, list):
        current_db = []

    updated_db = list(current_db)
    existing_ids = {pid for pid, _ in updated_db}

    for p in papers:
        pid = p.entry_id.split('/')[-1]
        if pid not in existing_ids:
            title = f"{p.title} ({p.published.year})"
            updated_db.append((pid, title))
            existing_ids.add(pid)

    return updated_db

def node_research(state: AgentState):
    topic = state["topic"]
    kb_instance = get_knowledge_base(topic=topic)
    logger.info(f"Start researching on topic: '{topic}'")
    send_chainlit_message(f"🚀 **Starting Research on:** `{topic}`")

    # Expand Keywords
    keywords = expand_keywords(topic)
    send_chainlit_message(f"🔎 **Keywords Expanded:**\n{', '.join(keywords)}")

    # Search ArXiv
    send_chainlit_message("📡 **Searching ArXiv...**")
    raw_papers = search_arxiv(keywords, paper_per_keyword=5)

    # Filter Papers
    send_chainlit_message(f"🧐 **Analyzing {len(raw_papers)} papers for relevance...**")
    relevant_papers = filter_papers(topic, raw_papers, max_papers=5)
    send_chainlit_message(f"✅ **Filter Complete:** Kept **{len(relevant_papers)}** relevant papers.")

    # Generate retrieval queries
    rag_queries = generate_rag_query(topic, relevant_papers)

    # Update references
    updated_refs = update_reference_db(state, relevant_papers)

    # Download and process PDFs
    new_count = kb_instance.download_and_process(relevant_papers)
    logger.success(f"Research complete. Added {new_count} new chunks to knowledge base.")
    send_chainlit_message(f"📚 **Research Phase Complete.**\nKnowledge base updated with **{new_count}** new data chunks.")
    return {
        "keywords": keywords,
        "reference_db": updated_refs,
        "rag_queries": rag_queries,
    }


def search_arxiv(keywords: List[str], paper_per_keyword=5):
    logger.info(f"Searching ArXiv for: {keywords}")
    found_papers = []
    seen_ids = set()
    client = arxiv.Client()
    for query in keywords:
        count = 0
        search_query = f'{query} AND (abs:"survey" OR abs:"review" OR abs:"overview" OR ti:"survey" OR ti:"review")'
        search = arxiv.Search(query=search_query, max_results=paper_per_keyword, sort_by=arxiv.SortCriterion.Relevance)
        for r in list(client.results(search)):
            if r.entry_id not in seen_ids and count < paper_per_keyword:
                seen_ids.add(r.entry_id)
                found_papers.append(r)
                count += 1
    return found_papers

def filter_papers(topic: str, papers: List[arxiv.Result], max_papers: int) -> List[arxiv.Result]:
    logger.info(f"Filtering {len(papers)} papers for relevance to topic: '{topic}'")

    if not papers:
        return []

    # Map ID to paper object for retrieval later
    paper_map = {}
    paper_descriptions = ""

    for p in papers:
        # Extract ID from URL
        pid = p.entry_id.split('/')[-1]
        paper_map[pid] = p

        # Clean abstract to save tokens
        clean_summary = p.summary.replace("\n", " ")[:1000]
        paper_descriptions += f"ID: {pid}\nTitle: {p.title}\nAbstract: {clean_summary}\n\n"

    logger.debug(paper_descriptions)
    # Use low temperature for strict decision making
    llm = get_llm().bind(temperature=0.1)

    prompt = ChatPromptTemplate.from_template(FILTER_PAPER_PROMPT)

    chain = prompt | llm | JsonOutputParser()
    input_prompt = prompt.invoke({"topic": topic, "papers": paper_descriptions, "max_papers": max_papers})
    logger.debug(f"Input: {input_prompt}")

    try:
        result = chain.invoke({"topic": topic, "papers": paper_descriptions, "max_papers": max_papers})
        logger.debug(f"Output: {result}")
        relevant_ids = result

        # Filter the original list
        filtered_papers = []
        for pid in relevant_ids:
            if pid in paper_map:
                filtered_papers.append(paper_map[pid])
            else:
                logger.warning(f"LLM hallucinated ID {pid}, ignoring")

    except Exception as e:
        logger.error(f"Error during paper filtering: {e}. Keeping all papers as fallback")
        return papers

    if not filter_papers:
        logger.warning("Filtering removed all papers. Keeping all papers as fallback")
        return papers
    logger.info(f"Relevance Filter: Kept {len(filtered_papers)} / {len(papers)} papers")
    return filtered_papers

