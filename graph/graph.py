import os
from langgraph.graph import END, StateGraph
from loguru import logger

from graph.state import AgentState
from nodes.draft import node_draft
from nodes.evaluate import node_evaluate
from nodes.refine import generate_bibliography, node_refine
from nodes.research import node_research
from utils.chainlit_msg import send_chainlit_message

def should_continue(state: AgentState):
    critique = state.get("critique", {})
    iteration = state.get("iteration", 1)
    max_iter = state.get("max_iterations", 3)

    if critique.get("sufficient", False):
        logger.info("Draft is sufficient. Ending.")
        return "end"

    if iteration > max_iter:
        logger.info("Max iterations reached. Ending.")
        return "end"

    logger.info("Continuing to refinement.")
    return "refine"

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("research", node_research)
    workflow.add_node("draft", node_draft)
    workflow.add_node("evaluate", node_evaluate)
    workflow.add_node("refine", node_refine)

    workflow.set_entry_point("research")
    workflow.add_edge("research", "draft")
    workflow.add_edge("draft", "evaluate")

    # Conditional Edge from Evaluation
    workflow.add_conditional_edges(
        "evaluate",
        should_continue,
        {
            "end": END,
            "refine": "refine"
        }
    )
    # Loop back from refine to evaluate
    workflow.add_edge("refine", "evaluate")
    return workflow.compile()

def run_survey_agent(topic: str) -> str:

    app = build_graph()

    logger.info(f"Starting the survey writing workflow for topic: {topic}")
    initial_state = {
        "topic": topic,
        "max_iterations": 2,
        "iteration": 1,
        "keywords": [],
        "draft": "",
        "critique": {},
        "reference_db": []
    }

    final_state = app.invoke(initial_state)
    survey = final_state['draft']
    reference = generate_bibliography(final_state)
    final_survey = f"{survey}{reference}"
    if not os.path.exists("surveys"):
        os.makedirs("surveys")
    filename = os.path.join("surveys", f"survey_{topic.replace(' ', '_')}.md")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(final_survey)
    logger.success(f"Done! Saved to {filename}")
    send_chainlit_message("🎉 **Survey Generation Complete!**")
    return final_survey
