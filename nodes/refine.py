import re
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from graph.state import AgentState
from nodes.draft import retrieve_with_queries
from providers.knowledge_base import get_knowledge_base
from providers.providers import get_llm
from nodes.research import filter_papers, search_arxiv, update_reference_db
from prompts.prompt import (
    REFINE_SURVEY_PROMPT,
)
from utils.chainlit_msg import send_chainlit_message

def generate_bibliography(state: AgentState) -> str:
    reference_db = state.get("reference_db", []).copy()
    if not reference_db:
        return ""
    survey = state['draft']

    matches = re.findall(r'\[(\d+)\]', survey)
    cited_indices = {int(m) for m in matches}

    bib = "\n\n## References\n\n"
    for i, (pid, title) in enumerate(reference_db):
        if i+1 in cited_indices or not cited_indices:
            url = f"https://arxiv.org/abs/{pid}"
            bib += f"- [{i+1}] [{pid}] [{title}]({url})\n"
    return f"{bib}\n *Only references that have been used will be listed.*"


def node_refine(state: AgentState):
    topic = state["topic"]
    draft = state["draft"]
    critique = state["critique"]
    iteration = state["iteration"]
    max_iterations = state["max_iterations"]
    kb_instance = get_knowledge_base(topic=topic)
    missing = critique.get("missing_aspects", [])
    rag_queries = critique.get("rag_queries", [])
    query = critique.get("next_search_query", "")


    logger.info(f"Start refining draft, iteration: {iteration} / {max_iterations}")
    send_chainlit_message(f"✨ **Refining Draft (Iteration {iteration})...**")

    if query:
        # Same process as research stage
        papers = search_arxiv([query], paper_per_keyword=3)
        relevant_papers = filter_papers(query, papers, max_papers=2)
        updated_refs = update_reference_db(state, relevant_papers)
        kb_instance.download_and_process(relevant_papers)

    logger.info(f"Start retrieving from the knowledge base using new queries: {rag_queries}")
    send_chainlit_message(f"🔍 **Digging Deeper:**\n*Searching knowledge base for: {rag_queries}*")
    context_docs = retrieve_with_queries(
        kb_instance,
        rag_queries,
        k_per_query=5,
    )

    context_pieces = []
    ref_map = {pid: i + 1 for i, (pid, _) in enumerate(updated_refs)}
    for d in context_docs:
        source = d.metadata.get("source", "")
        pid = source.split("/")[-1]
        # Find the reference number (default to "?" if not found in current session refs)
        ref_idx = ref_map.get(pid, "?")
        context_pieces.append(f"[{ref_idx}]  (use this ID when citing)\n {d.page_content[:2000]}")
    logger.info(f"Retrieved {len(context_pieces)} chunks of data from knowledge base")
    context_text = "\n\n".join(context_pieces)
    logger.info(f"Start refining the survey on topic: '{topic}'")
    send_chainlit_message("✍️ **Polishing & Updating Content...**")
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(REFINE_SURVEY_PROMPT)

    chain = prompt | llm | StrOutputParser()
    input_prompt = prompt.invoke(
        {
            "topic": topic,
            "draft": draft,
            "missing": str(missing),
            "context": context_text,
        }
    )
    logger.debug(f"Input: {input_prompt}")
    new_draft = chain.invoke(
        {
            "topic": topic,
            "draft": draft,
            "missing": str(missing),
            "context": context_text,
        }
    )
    logger.debug(f"Output: {new_draft}")
    logger.success("Completed refinement of the survey")
    send_chainlit_message("✅ **Refinement Complete.**")
    return {"draft": new_draft, "iteration": iteration + 1, "reference_db": updated_refs}
