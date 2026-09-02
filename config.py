import os

os.environ["DEEPSEEK_API_KEY"] = ""
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
MODEL = "deepseek-v4-flash"
