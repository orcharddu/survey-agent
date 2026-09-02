from typing import Any, Dict, List, Tuple, TypedDict


class AgentState(TypedDict):
    topic: str
    keywords: List[str]
    rag_queries: List[str]
    draft: str
    critique: Dict[str, Any]
    iteration: int
    max_iterations: int
    reference_db: List[Tuple[str, str]]
