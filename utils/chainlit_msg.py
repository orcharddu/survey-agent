try:
    import chainlit as cl
except ImportError:
    cl = None

def send_chainlit_message(content: str):
    if cl:
        try:
            cl.run_sync(cl.Message(content=content).send())
        except Exception:
            pass

