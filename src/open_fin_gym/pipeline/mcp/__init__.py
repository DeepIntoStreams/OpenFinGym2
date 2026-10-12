import logging
import os

from alpaca_mcp_server.server import build_server
from fastmcp import Client
from fastmcp.client import BearerAuth
from fastmcp.client.group import ClientGroup

from .yahoo_finance import yfinance_mcp as yfinance_mcp

os.environ["ALPACA_TOOLSETS"] = (
    "assets,stock-data,news,crypto-data,options-data,fixed-income-data"
)

KAGGLE_MCP_URL = "https://www.kaggle.com/mcp/"
CRYPTO_COM_MCP_URL = "https://mcp.crypto.com/market-data/mcp/"
COINDESK_MCP_URL = "https://mcp.coindesk.com/mcp/"
GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/readonly/"

alpaca_mcp_server = build_server()

logger = logging.getLogger(__name__)


def init_mcp_servers() -> ClientGroup:
    servers = {
        "kaggle": Client(KAGGLE_MCP_URL),
        "yahoo": Client(yfinance_mcp),
        "crypto_com": Client(CRYPTO_COM_MCP_URL),
    }

    if "GITHUB_PAT" in os.environ:
        servers["github"] = Client(
            "https://api.githubcopilot.com/mcp/readonly",
            auth=BearerAuth(os.getenv("GITHUB_PAT")),
        )
    else:
        logger.info("GITHUB_PAT not provided, not using Github MCP tool")

    if ("ALPACA_API_KEY" in os.environ) and ("ALPACA_SECRET_KEY" in os.environ):
        servers["alpaca"] = Client(alpaca_mcp_server)
    else:
        logger.info("ALPACA_API_KEY or ALPACA_SECRET_KEY not set, not using Alpaca mcp")

    return ClientGroup(servers)
