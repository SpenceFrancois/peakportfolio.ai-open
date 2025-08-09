import os
from tiingo import TiingoClient

TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")  # Read from environment
client = TiingoClient({'api_key': TIINGO_API_KEY})
