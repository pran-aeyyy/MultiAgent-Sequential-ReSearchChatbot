import os
import streamlit as st
from groq import Groq
from langchain.memory import ConversationBufferMemory
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import ArxivAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# Set Groq API Key
os.environ["GROQ_API_KEY"] = "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"  # Replace with your actual key
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize Memory
memory = ConversationBufferMemory()

# 📌 Define Tools
web_search = DuckDuckGoSearchRun()
arxiv_search = ArxivAPIWrapper()

tools = [
    Tool(name="Web Search", func=web_search.run, description="Find research papers online."),
    Tool(name="ArXiv Search", func=arxiv_search.run, description="Search academic papers from ArXiv."),
]

# 📌 Function to get LLM response via Groq (Summarization)
def get_llm_response(query, context=""):
    full_query = f"Context: {context}\nUser: {query}\nResponse:"
    response = client.chat.completions.create(
        model= "llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": full_query}],
    )
    return response.choices[0].message.content

# 📌 Agent 1: Web Search Agent
def web_search_agent(topic):
    research_agent = initialize_agent(
        tools,
        ChatGroq(model= "llama-3.3-70b-versatile"),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    search_query = f"Research papers on {topic}"
    results = research_agent.run(search_query)
    
    # Extract first 3 research paper titles
    papers = results.split("\n")[:3]  
    if not papers:
        return "No research papers found."
    
    return papers

# 📌 Agent 2: Paper Selection Agent
def paper_selection_agent(papers):
    context = "\n".join(papers)
    best_paper = get_llm_response(f"From these papers, which is most relevant?\n{context}")
    return best_paper

# 📌 Agent 3: Summarization Agent
def summarization_agent(selected_paper):
    summary = get_llm_response(f"Summarize this research paper: {selected_paper}")
    return summary

# 📌 Streamlit UI
st.title("📚 Sequential Multi-Agent Research Chatbot")

# Step 1: Enter Discussion Topic
st.subheader("Step 1: Enter a Research Topic")
topic = st.text_input("Enter a topic:")

# Step 2: Start Research Button
st.subheader("Step 2: Search for Research Papers")
if st.button("Find Papers & Summarize"):
    if topic:
        # Agent 1: Search for papers
        papers = web_search_agent(topic)
        st.write("### 🔍 Top 3 Research Papers Found:")
        for i, paper in enumerate(papers, 1):
            st.write(f"{i}. {paper}")

        # Agent 2: Select the most relevant paper
        selected_paper = paper_selection_agent(papers)
        st.write("### 🏆 Most Relevant Paper:")
        st.write(selected_paper)

        # Agent 3: Summarize the selected paper
        summary = summarization_agent(selected_paper)
        st.write("### 📝 Research Paper Summary:")
        st.markdown(summary)

        # Store in memory
        memory.save_context({"input": topic}, {"output": summary})

        # Display Conversation History
        st.write("### 📝 Conversation History")
        st.text(memory.load_memory_variables({}).get("history", "No history yet."))
    else:
        st.warning("⚠️ Please enter a topic before starting.")
