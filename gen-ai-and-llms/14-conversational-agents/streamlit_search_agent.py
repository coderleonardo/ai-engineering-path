"""Conversational Search Agent — Streamlit Reference App.

Reference implementation. See ../12-langchain-p1/langchain.md for the underlying LangChain
concepts (prompt templates, chains, memory) and
../12-langchain-p1/langchain_chains_memory_rag.ipynb for the create_agent + checkpointer memory
pattern used here outside of a Streamlit context.

Methods covered:
- A tool-using chat agent (`create_agent` + a checkpointer) that decides per-turn whether to call
  a web search tool (`DuckDuckGoSearchRun`)
- Thread-scoped conversational memory persisted across Streamlit reruns via `st.session_state`
- Rendering an agent's intermediate tool calls (name, input, output) in expandable chat elements
- Streaming an agent run (`agent.stream(..., stream_mode="updates")`) to recover those
  intermediate steps as they happen, instead of only inspecting the final response

Use this as a reference when: you need copy-paste-ready code for a Streamlit chat UI backed by a
tool-using LangChain agent with per-session memory.

Don't use this as a reference for: RAG retrieval (see
../12-langchain-p1/langchain_chains_memory_rag.ipynb) — this app only wires up a search tool, no
vector store.

Requires: streamlit, langchain, langchain-openai, langchain-community, langgraph,
duckduckgo-search (not part of the repo's shared pyproject.toml — install separately to run this
app).

Run with: streamlit run streamlit_search_agent.py
"""

import streamlit as st

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.checkpoint.memory import InMemorySaver

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="DSA")
st.title("Conversational Search Agent")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if "messages" not in st.session_state or st.sidebar.button("Reset"):
    st.session_state.messages = [{"role": "assistant", "content": "Como eu posso ajudar você?", "steps": []}]
    # A fresh thread_id disconnects the new conversation from the checkpointer's saved state for
    # the old one, which is what actually resets the agent's memory (clearing the displayed
    # messages alone would not).
    st.session_state.thread_id = str(id(st.session_state.messages))


def extract_tool_steps(messages):
    """Pair each tool call in a langgraph message list with its resulting ToolMessage.

    create_agent's response is a flat list of messages (AI messages with tool_calls interleaved
    with ToolMessages), not the (action, observation) tuples AgentExecutor used to return -- this
    reconstructs an equivalent view for display purposes.
    """
    steps = []
    for message in messages:
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            continue
        for call in tool_calls:
            output = next(
                (m.content for m in messages if getattr(m, "tool_call_id", None) == call["id"]),
                None,
            )
            steps.append({"tool": call["name"], "tool_input": call["args"], "output": output})
    return steps


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        for step in message.get("steps", []):
            with st.expander(f"✅ **{step['tool']}**: {step['tool_input']}"):
                st.write(step["output"])
        st.write(message["content"])

if prompt := st.chat_input(placeholder="Digite uma pergunta para começar!"):
    st.session_state.messages.append({"role": "user", "content": prompt, "steps": []})
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Adicione sua OpenAI API key para continuar.")
        st.stop()

    with st.chat_message("assistant"):
        llm = ChatOpenAI(api_key=openai_api_key, streaming=True)
        search_agent = create_agent(
            llm,
            tools=[DuckDuckGoSearchRun(name="Search")],
            checkpointer=InMemorySaver(),
        )
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        with st.spinner("Pensando..."):
            response = search_agent.invoke({"messages": [{"role": "user", "content": prompt}]}, config)

        answer = response["messages"][-1].content
        steps = extract_tool_steps(response["messages"])

        for step in steps:
            with st.expander(f"✅ **{step['tool']}**: {step['tool_input']}"):
                st.write(step["output"])
        st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer, "steps": steps})
