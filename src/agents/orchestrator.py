

from typing import Annotated, TypedDict #Used for defining structured state types.

from langgraph.graph import StateGraph, START, END #Imports LangGraph’s graph system.
#Think of this as creating a flowchart that can execute.

#This tells LangGraph how to append new messages to old messages.
from langgraph.graph.message import add_messages

#Stores conversation history in memory using thread_id.
from langgraph.checkpoint.memory import MemorySaver

#Core classes for messages and tools
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

"""
Message types:

Class	            Meaning
HumanMessage	    user message
AIMessage	        model response
BaseMessage	        parent type for all messages

"""

#Imports RAG tool and other tools
from src.agents.tools import rag_search, get_kpis, generate_report_sections

#Imports the Qwen LLM from Groq
from langchain_groq import ChatGroq

#AgentState class defines the structure of our agent’s memory.
"""
This means the agent’s state contains only one thing:

messages

And every time a node returns new messages, LangGraph will append them instead of replacing them.

So state grows like:

[HumanMessage]
[HumanMessage, AIMessage with tool call]
[HumanMessage, AIMessage with tool call, ToolMessage]
[HumanMessage, AIMessage with tool call, ToolMessage, AIMessage final answer]

Tiny message-train. 🚂

"""
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

tools = [rag_search, get_kpis, generate_report_sections]

def get_agent_llm():
    # We use qwen/qwen3-32b as requested
    return ChatGroq(
        model_name="qwen/qwen3-32b",
        temperature=0.0,
        max_retries=3,
        timeout=120.0
    ).bind_tools(tools)

"""
These are external functions the LLM can call.

Most likely:

Tool	                                    Purpose
rag_search	                search documents/transcripts using RAG
get_kpis	                fetch financial metrics
generate_report_sections	generate structured report blocks

5. LLM Creation
def get_agent_llm():
    return ChatGroq(
        model_name="qwen/qwen3-32b",
        temperature=0.0
    ).bind_tools(tools)

This creates the LLM.

Important parts:

model_name="qwen/qwen3-32b"

Uses Qwen 3 32B through Groq.

temperature=0.0

Makes output more deterministic and less “creative squirrel mode.”

.bind_tools(tools)

This gives the LLM access to your tools.

So the model can respond with either:

Final answer

or:

Call rag_search with these arguments

"""


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
    This extracts wait time from Groq rate limit messages.

    Example:

    Please try again in 9m17.3s

    It converts that into seconds:

    557.3 seconds

"""
    import re
    m_match = re.search(r"([0-9.]+)m\b", error_msg)
    s_match = re.search(r"([0-9.]+)s\b", error_msg)
    ms_match = re.search(r"([0-9.]+)ms\b", error_msg)
    
    total = 0.0
    if m_match: total += float(m_match.group(1)) * 60
    if ms_match: total += float(ms_match.group(1)) / 1000
    elif s_match: total += float(s_match.group(1))
    
    if total == 0.0:
        return 15.0
    return total

#This is the public function your app likely calls.
"""
Example:

run_agent_query("Compare Apple and Nvidia risks")
"""

def run_agent_query(query: str, thread_id: str = "default") -> str:
    from langchain_core.messages import HumanMessage
    import time
    inputs = {"messages": [HumanMessage(content=query)]} #Wraps the user query as a LangChain message.
    config = {"configurable": {"thread_id": thread_id}} #This tells LangGraph which memory thread to use.
    """
    Example:

    thread_id="user_123"

    Each thread has separate conversation history.

    """

    def _execute() -> str: #Actually runs the graph.
        from src.agents.events import emit
        from langchain_core.messages import ToolMessage, AIMessage

        print(f"\n--- [AGENT START] Processing query ---")
        print(f"Query: {query} | Thread ID: {thread_id}")
        
        emit({
            "type": "query_start",
            "query": query,
            "thread_id": thread_id,
            "log": f"Received query: '{query}'"
        })
        
        final_state = None
        for step in app_graph.stream(inputs, config=config, stream_mode="values"):
            if "messages" in step:
                last_msg = step["messages"][-1]
                print(f"-> [Agent Step] Role: {last_msg.type} | Content Length: {len(str(last_msg.content))} chars")
                
                if isinstance(last_msg, AIMessage):
                    if getattr(last_msg, "tool_calls", None):
                        emit({"type": "node_enter", "node": "chatbot", "log": "Chatbot decided to use tools..."})
                        for tc in last_msg.tool_calls:
                            print(f"   Tool Call: {tc['name']} | Args: {tc['args']}")
                            emit({
                                "type": "tool_call",
                                "tool": tc['name'],
                                "args": tc['args'],
                                "log": f"Calling tool {tc['name']}"
                            })
                    else:
                        preview = str(last_msg.content)[:300]
                        emit({
                            "type": "node_exit",
                            "node": "chatbot",
                            "output_preview": preview,
                            "log": "Chatbot generated final response"
                        })
                
                elif isinstance(last_msg, ToolMessage):
                    preview = str(last_msg.content)[:300]
                    emit({
                        "type": "tool_result",
                        "tool": last_msg.name,
                        "output_preview": preview,
                        "log": f"Tool {last_msg.name} returned result"
                    })

            final_state = step
        print(f"--- [AGENT COMPLETE] ---\n")
        emit({
            "type": "query_complete",
            "log": "Query execution complete"
        })
        return final_state["messages"][-1].content #After the graph ends, return the final AI response.

    try:
        return _execute()
    except Exception as e:
        error_msg = str(e)
        if "rate_limit_exceeded" in error_msg or "Rate limit reached" in error_msg:
            wait_secs = _parse_retry_after(error_msg)
            wait_display = f"{int(wait_secs // 60)}m {int(wait_secs % 60)}s" if wait_secs >= 60 else f"{int(wait_secs)}s"

            MAX_AUTO_RETRY_SECS = 45
            if wait_secs <= MAX_AUTO_RETRY_SECS:
                print(f"[QUOTA] Short wait ({wait_secs:.0f}s). Auto-retrying...")
                from src.agents.events import emit
                emit({"type": "ping", "log": f"API rate limit hit. Auto-retrying in {wait_secs:.0f}s..."})
                import time
                time.sleep(wait_secs + 1)
                try:
                    return _execute()
                except Exception as e2:
                    error_msg2 = str(e2)
                    wait_secs2 = _parse_retry_after(error_msg2)
                    wait_display2 = f"{int(wait_secs2 // 60)}m {int(wait_secs2 % 60)}s" if wait_secs2 >= 60 else f"{int(wait_secs2)}s"
                    emit({"type": "error", "log": "Second rate limit hit. Pausing."})
                    return f"🛑 **Execution Paused: API Quota Reached**\n\nThe financial reasoning model is under high demand. Please try again in **{wait_display2}**."
            else:
                print(f"[QUOTA] Long wait required ({wait_secs:.0f}s). Returning error immediately.")
                from src.agents.events import emit
                emit({"type": "error", "log": "API rate limit hit. Wait too long to auto-retry."})
                return f"🛑 **Execution Paused: API Quota Reached**\n\nThe financial reasoning model is under high demand. Please try again in **{wait_display}**."
        
        from src.agents.events import emit
        emit({"type": "error", "log": f"System Error: {error_msg}"})
        return f"⚠️ **System Error:** {error_msg}"

