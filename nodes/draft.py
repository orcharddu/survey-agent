from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from graph.state import AgentState
from providers.knowledge_base import KnowledgeBase, get_knowledge_base
from providers.providers import get_llm

from prompts.prompt import (
    DRAFT_SURVEY_PROMPT,
)
from utils.chainlit_msg import send_chainlit_message

def retrieve_with_queries(
    kb_instance: KnowledgeBase,
    rag_queries: list[str],
    k_per_query: int = 5,
):
    docs = []
    for q in rag_queries:
        retrieved = kb_instance.retrieve(q, k=k_per_query)
        docs.extend(retrieved)

    unique_docs = {}
    for d in docs:
        doc_id = (
            d.metadata.get("source"),
            d.page_content[:100],
        )
        unique_docs[doc_id] = d
    return list(unique_docs.values())

def node_draft(state: AgentState):

    topic = state["topic"]
    reference_db = state.get("reference_db", []).copy()
    kb_instance = get_knowledge_base(topic=topic)

    rag_queries = state["rag_queries"]

    logger.info(f"Start retrieving from the knowledge base using queries: {rag_queries}")
    send_chainlit_message(f"🔍 **Retrieving Context...**\n*Using queries: {rag_queries}*")
    context_docs = retrieve_with_queries(
        kb_instance,
        rag_queries,
        k_per_query=5,
    )

    context_pieces = []
    ref_map = {pid: i + 1 for i, (pid, _) in enumerate(reference_db)}
    for d in context_docs:
        source = d.metadata.get("source", "")
        pid = source.split("/")[-1]
        # Get the reference number (default to "?" if not found in current session refs)
        ref_idx = ref_map.get(pid, "?")
        context_pieces.append(f"[{ref_idx}]  (use this ID when citing)\n {d.page_content[:2000]}")
    logger.info(f"Retrieved {len(context_pieces)} chunks of data from knowledge base")
    context_text = "\n\n".join(context_pieces)
    logger.info(f"Start writing the first draft survey on topic: '{topic}'")
    send_chainlit_message("✍️ **Drafting Initial Survey...**\n*Synthesizing information from retrieved papers...*")
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(DRAFT_SURVEY_PROMPT)
    chain = prompt | llm | StrOutputParser()
    input_prompt = prompt.invoke({"topic": topic, "context": context_text})
    logger.debug(f"Input: {input_prompt}")
    draft = chain.invoke({"topic": topic, "context": context_text})
    logger.debug(f"Output: {draft}")
    logger.success("Completed first draft of the survey")
    send_chainlit_message("📝 **First Draft Generated.**\n*Sending to Critic for evaluation...*")
    return {"draft": draft, "iteration": 1}
