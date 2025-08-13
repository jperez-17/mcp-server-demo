# MCP Server Demo

This repository serves as a demonstration of an [MCP (Model Context Protocol)](https://modelcontextprotocol.io/overview) server, utilizing the [MCP Python SDK
](https://pypi.org/project/mcp/). The main objective of the project is to receive a prompt from an MCP client, query external API services to retrieve information, process the data using a language model (LLM) through [LangChain](https://www.langchain.com/langchain), and return a processed response to the MCP client for presentation to the end user.

## Installation

Use the package manager [uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods).

- Download project via git `git@github.com:jperez-17/mcp-server-demo.git` or https with `https://github.com/jperez-17/mcp-server-demo.git`
- Enter into project dir `cd mcp-server-demo`
- Configure `.env` file with the required [environment variables](#environment-variables). You can use `.env.example` as guide to configure your `.env` file (This file MUST be git-ignored).
- Run:

```
python -m venv .venv
source .venv/bin/activate
fastmcp dev server.py
```

## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

```
USER_API_URL=
DEFAULT_USER_ID=

TOKEN_API_URL=
TOKEN_USER_NAME=
TOKEN_USER_PASSWORD=

OPENAI_API_KEY=

PORT=
```
