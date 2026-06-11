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
    # We use qwen/qwen3-32b
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

def run_agent_query(query: str, thread_id: str = "default") -> str:
    from langchain_core.messages import HumanMessage
    import re
    inputs = {"messages": [HumanMessage(content=query)]}
    config = {"configurable": {"thread_id": thread_id}}
    try:
        print(f"\n--- [AGENT START] Processing query ---")
        print(f"Query: {query} | Thread ID: {thread_id}")
        
        # Stream the graph execution to print node progress
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
    except Exception as e:
        error_msg = str(e)
        if "rate_limit_exceeded" in error_msg or "Rate limit reached" in error_msg:
            match = re.search(r"Please try again in ([0-9.]+[a-zA-Z]+)", error_msg)
            time_msg = match.group(1) if match else "a few seconds"
            return f"🛑 **Execution Paused: API Quota Reached**\n\nThe financial reasoning model has temporarily hit its token limit due to high demand. Please try your query again in **{time_msg}**."
        return f"⚠️ **System Error:** {error_msg}"
