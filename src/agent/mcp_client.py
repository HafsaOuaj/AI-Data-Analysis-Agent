import asyncio
import logging
import sys
import time
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ollama import chat
from anthropic import Anthropic
from dotenv import load_dotenv

# Load env variables from .env
load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging(level: str = "DEBUG") -> logging.Logger:
    """Configure and return the root logger with console + file handlers."""
    log_level = getattr(logging, level.upper(), logging.DEBUG)

    logger = logging.getLogger("mcp_client")
    logger.setLevel(log_level)

    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler – always writes DEBUG and above
    fh = logging.FileHandler("mcp_client.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


logger = setup_logging()


# ---------------------------------------------------------------------------
# Thinking / reasoning display helpers
# ---------------------------------------------------------------------------

def _print_thinking(content_block) -> None:
    """
    Pretty-print any thinking / reasoning content found in a response block.

    Ollama (and some Anthropic models) may surface thinking in different ways:
      • A dedicated 'thinking' block (type == 'thinking', attribute .thinking)
      • A text block whose content starts with <think> … </think>
    """
    # --- Dedicated thinking block (e.g. DeepSeek-R1, QwQ, extended-thinking) ---
    if getattr(content_block, "type", None) == "thinking":
        thinking_text = getattr(content_block, "thinking", "") or ""
        if thinking_text.strip():
            _render_thinking_box(thinking_text)
        return

    # --- Inline <think>…</think> tags inside a text block ---
    if getattr(content_block, "type", None) == "text":
        raw = getattr(content_block, "text", "") or ""
        if "<think>" in raw and "</think>" in raw:
            start = raw.index("<think>") + len("<think>")
            end = raw.index("</think>")
            thinking_text = raw[start:end].strip()
            if thinking_text:
                _render_thinking_box(thinking_text)


def _render_thinking_box(text: str) -> None:
    """Print the thinking content inside a visible ASCII box."""
    width = 72
    border = "─" * width
    print(f"\n┌{border}┐")
    print(f"│{'  🧠  THINKING PROCESS':^{width}}│")
    print(f"├{border}┤")
    for line in text.splitlines():
        # Wrap long lines
        while len(line) > width - 2:
            print(f"│ {line[:width-2]} │")
            line = line[width - 2:]
        print(f"│ {line:<{width-2}} │")
    print(f"└{border}┘\n")


# ---------------------------------------------------------------------------
# MCPClient
# ---------------------------------------------------------------------------

class MCPClient:
    def __init__(
        self,
        model_name: str = "qwen3:4b",
        data_path: str = "data/ecommerce_customer_analytics.csv",
        tools_path: str = "src/tools/data_understanding.py",
    ):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        self.model_name = model_name
        self.data_path = data_path
        self.tools: list = []
        self.tools_dict: dict = {}

        logger.debug(
            "MCPClient initialised | model=%s | data_path=%s",
            self.model_name,
            self.data_path,
        )

    # Connection

    async def connect_to_server(self, server_script_path: str) -> None:
        """Connect to an MCP server.

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        logger.info("Connecting to MCP server: %s", server_script_path)

        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None,
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()
        logger.debug("MCP session initialised")

        response = await self.session.list_tools()
        tool_names = [tool.name for tool in response.tools]
        logger.info("Available tools: %s", tool_names)
        print("\nConnected to server with tools:", tool_names)

    # Query processing

    async def process_query(self, query: str):
        logger.info("Processing query: %r", query)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a data analysis assistant. "
                    "Use tools when needed."
                ),
            },
            {
                "role": "user",
                "content": f"{query}\nDataset: {self.data_path}",
            },
        ]

        tool_response = await self.session.list_tools()

        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.inputSchema,
                },
            }
            for t in tool_response.tools
        ]
        while True:

            print("\n🤖 Thinking...\n")

            stream = chat(
                model=self.model_name,
                messages=messages,
                tools=available_tools,
                stream=True,
                think=True
            )

            thinking = ""
            content =""
            tool_calls=[]

            assistant_text = ""
            tool_calls = []
            done_thinking = False

            # STREAM TOKENS LIVE
            for chunk in stream:
                if chunk.message.thinking:
                    thinking +=chunk.message.thinking
                print(thinking, end="", flush=True)
                if chunk.message.content:
                    if not done_thinking:
                        done_thinking =True
                        print('\n')
                    content += chunk.message.content
                    print(chunk.message.content, end="", flush=True)
                if chunk.message.tool_calls:
                    tool_calls.extend(chunk.message.tool_calls)
                    print(chunk.message.tool_calls)
            # append accumulated fields to the messages
            if thinking or content or tool_calls:
                messages.append({'role': 'assistant', 'thinking': thinking, 'content': content, 'tool_calls': tool_calls})
            if not tool_calls:
                break

            for tool_call in tool_calls:

                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments

                print(f"\n🧠 Calling tool: {tool_name}")
                print(f"   args: {tool_args}")

                try:
                    result = await self.session.call_tool(tool_name, tool_args)
                except Exception as e:
                    print(f"❌ Tool error: {e}")
                    result = type("obj", (), {"content": [str(e)]})

                print(f"📦 Tool result received ({len(str(result))} chars)\n")

                messages.append(
                    {
                        "role": "tool",
                        "tool_name": tool_name,
                        "content": str(result),
                    }
                )

        return content

    async def chat_loop(self) -> None:
        """Run an interactive chat loop."""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        logger.info("Chat loop started")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == "quit":
                    logger.info("User requested exit")
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                logger.exception("Unhandled error during query processing: %s", e)
                print(f"\nError: {e}")

    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.debug("Cleaning up MCP session resources")
        await self.exit_stack.aclose()


async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())