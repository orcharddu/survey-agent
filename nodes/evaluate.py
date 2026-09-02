from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from graph.state import AgentState
from providers.providers import get_llm
from prompts.prompt import EVALUTAE_SURVEY_PROMPT
from utils.chainlit_msg import send_chainlit_message


def node_evaluate(state: AgentState):
    iteration = state["iteration"]
    max_iterations = state["max_iterations"]
    if iteration > max_iterations:
        return {}
    draft = state["draft"]
    topic = state["topic"]

    logger.info(f"Start evaluating draft of the survey, iteration: {iteration} / {max_iterations}")
    send_chainlit_message(f"🕵️ **Critic:** Evaluating draft (Iteration {iteration})...")

    llm = get_llm().bind(temperature=0.1)

    prompt = ChatPromptTemplate.from_template(EVALUTAE_SURVEY_PROMPT)
    chain = prompt | llm | JsonOutputParser()
    input_prompt = prompt.invoke({"topic": topic, "draft": draft})
    logger.debug(f"Input: {input_prompt}")
    critique = chain.invoke({"topic": topic, "draft": draft})
    logger.debug(f"Output: {critique}")
    if critique.get("sufficient", False):
        logger.success("Completed evaluation, no refinement needed")
        send_chainlit_message("🤔 **Critic Feedback:** No refinement needed.")
    else:
        rag_queries = critique.get("rag_queries", [])
        logger.success(f"Completed evaluation, refinement needed on new queries: {rag_queries}")
        missing = critique.get("missing_aspects", [])
        next_query = critique.get("next_search_query", "None")
        send_chainlit_message(f"🤔 **Critic Feedback:** Refinement needed.\n- **Missing:** {missing}\n- **Next Plan:** Search for `{next_query}`")
    return {"critique": critique}
