import os
from dotenv import load_dotenv
load_dotenv()

# Load Redis URL from environment variable
REDIS_URL = os.getenv("REDIS_URL")
if not REDIS_URL:
    raise ValueError("REDIS_URL environment variable must be set for RedisSaver checkpointer.")