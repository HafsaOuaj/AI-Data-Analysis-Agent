# Ensure the project source directory is in sys.path
import os
import sys
import logging

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
from tools.data_understanding import load_dataset
from mcp.server.fastmcp import FastMCP

# -----------------------
# LOGGING SETUP
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("visualization")
mcp = FastMCP("visualization")
def main():
    logger.info("Starting MCP server: visualization")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()