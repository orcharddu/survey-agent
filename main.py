import logging
import sys

from loguru import logger

logging.basicConfig(level=logging.ERROR)
ocr_logger = logging.getLogger("RapidOCR")
ocr_logger.addHandler(logging.NullHandler())
ocr_logger.setLevel(logging.ERROR)
from graph.graph import run_survey_agent  # noqa: E402


def main():
    """
        run with `uv run ./main.py` for console using
    """
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    # logger.add("app.log", level="SUCCESS")
    topic = input("\nPlease enter a topic for survey generation: ")
    run_survey_agent(topic)


if __name__ == "__main__":
    main()
