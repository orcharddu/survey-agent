import os
import sys
import logging
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")
logging.getLogger("watchfiles.main").setLevel(logging.WARNING)
os.environ["CHAINLIT_PROJECT_DEFAULT_LANG"] = "en-US"
import chainlit as cl  # noqa: E402
from graph.graph import run_survey_agent  # noqa: E402


@cl.on_chat_start
async def start():
    await cl.Message(
        content="👋 **Welcome to the Survey Agent!**\n\nI can help you research academic topics, read papers, and write comprehensive surveys.\n\nPlease enter a **research topic** to get started."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """
        run with `chainlit run app.py -w` for Webpage using
    """
    topic = message.content
    try:
        final_survey = await cl.make_async(run_survey_agent)(topic)
        if final_survey:
            await cl.Message(content=final_survey).send()
        else:
            await cl.Message(content="❌ Finished but no draft found. Please check logs.").send()
    except Exception as e:
        logger.error(f"App Error: {e}")
        await cl.Message(content=f"❌ **Error:** {e}").send()
