from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from src.agents.tools import rag_search, get_kpis, generate_report_sections
from langchain_groq import ChatGroq
import os

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

tools = [rag_search, get_kpis, generate_report_sections]

def get_agent_llm():
    # We use qwen/qwen3-32b as requested
    return ChatGroq(
        model_name="qwen/qwen3-32b",
        temperature=0.0
    ).bind_tools(tools)

def filter_messages_for_llm(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Filter out historical tool-call messages and raw tool responses from past turns.
    This prevents the LLM from recycling old context, forcing it to call tools freshly
    for every user query, while preserving conversational turn history and the active 
    turn's tool responses.
    """
    if not messages:
        return []
    
    # Find the last human query index to isolate the current turn
    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_idx = i
            break
            
    if last_human_idx == -1:
        return messages
        
    filtered = []
    for i, msg in enumerate(messages):
        # Always keep messages of the current active turn (at or after the last HumanMessage)
        if i >= last_human_idx:
            filtered.append(msg)
            continue
            
        # For past turns, keep queries and final answers, but skip tool invocation artifacts
        if isinstance(msg, HumanMessage):
            filtered.append(msg)
        elif isinstance(msg, AIMessage):
            # Keep only AIMessages that do not contain tool calls (i.e. final answers)
            if not getattr(msg, "tool_calls", None):
                filtered.append(msg)
        # Exclude ToolMessages from previous turns
    return filtered

def chatbot_node(state: AgentState):
    llm = get_agent_llm()
    filtered_messages = filter_messages_for_llm(state["messages"])
    response = llm.invoke(filtered_messages)
    return {"messages": [response]}

def create_agent_graph():
    from langgraph.prebuilt import ToolNode
    
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("chatbot", chatbot_node)
    
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    
    graph_builder.add_edge(START, "chatbot")
    
    # Define conditional edges
    def route_tools(state: AgentState):
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return END
        
    graph_builder.add_conditional_edges("chatbot", route_tools)
    graph_builder.add_edge("tools", "chatbot")
    
    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)

# Global graph instance
app_graph = create_agent_graph()

def _parse_retry_after(error_msg: str) -> float:
    """
    Parse Groq's Retry-After string (e.g. '9m17.3s', '45.2s', '2m') into seconds.
    Returns 0.0 if not found.
    """
    import re
    match = re.search(r"Please try again in ([0-9.]+m)?([0-9.]+s)?", error_msg)
    if not match:
        return 0.0
    minutes = float(match.group(1)[:-1]) if match.group(1) else 0.0
    seconds = float(match.group(2)[:-1]) if match.group(2) else 0.0
    return minutes * 60 + seconds

def run_agent_query(query: str, thread_id: str = "default") -> str:
    from langchain_core.messages import HumanMessage
    import time
    inputs = {"messages": [HumanMessage(content=query)]}
    config = {"configurable": {"thread_id": thread_id}}

    def _execute() -> str:
        print(f"\n--- [AGENT START] Processing query ---")
        print(f"Query: {query} | Thread ID: {thread_id}")
        final_state = None
        for step in app_graph.stream(inputs, config=config, stream_mode="values"):
            if "messages" in step:
                last_msg = step["messages"][-1]
                print(f"-> [Agent Step] Role: {last_msg.type} | Content Length: {len(str(last_msg.content))} chars")
                if getattr(last_msg, "tool_calls", None):
                    for tc in last_msg.tool_calls:
                        print(f"   Tool Call: {tc['name']} | Args: {tc['args']}")
            final_state = step
        print(f"--- [AGENT COMPLETE] ---\n")
        return final_state["messages"][-1].content

    try:
        return _execute()
    except Exception as e:
        error_msg = str(e)
        if "rate_limit_exceeded" in error_msg or "Rate limit reached" in error_msg:
            wait_secs = _parse_retry_after(error_msg)
            wait_display = f"{int(wait_secs // 60)}m {int(wait_secs % 60)}s" if wait_secs >= 60 else f"{int(wait_secs)}s"

            # Only auto-retry for short waits (≤30s). For longer waits, return immediately
            # so the user isn't stuck staring at a spinner for minutes.
            MAX_AUTO_RETRY_SECS = 30
            if 0 < wait_secs <= MAX_AUTO_RETRY_SECS:
                print(f"[QUOTA] Short wait ({wait_secs:.0f}s). Auto-retrying...")
                import time
                time.sleep(wait_secs + 1)
                try:
                    return _execute()
                except Exception as e2:
                    error_msg2 = str(e2)
                    wait_secs2 = _parse_retry_after(error_msg2)
                    wait_display2 = f"{int(wait_secs2 // 60)}m {int(wait_secs2 % 60)}s" if wait_secs2 >= 60 else f"{int(wait_secs2)}s"
                    return f"🛑 **Execution Paused: API Quota Reached**\n\nThe financial reasoning model is under high demand. Please try again in **{wait_display2}**."
            elif wait_secs > MAX_AUTO_RETRY_SECS:
                print(f"[QUOTA] Long wait required ({wait_secs:.0f}s). Returning error immediately.")
                return f"🛑 **Execution Paused: API Quota Reached**\n\nThe financial reasoning model is under high demand. Please try again in **{wait_display}**."
            # Unknown format
            return f"🛑 **Execution Paused: API Quota Reached**\n\nThe financial reasoning model is under high demand. Please wait ~30 seconds and try again."
        return f"⚠️ **System Error:** {error_msg}"

