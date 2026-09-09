"""
app.py
------
Streamlit chat interface for the AI Student Support Assistant.

Run with:
    streamlit run app.py
"""

import streamlit as st
from agent import StudentSupportAgent

st.set_page_config(page_title="AI Student Support Assistant", page_icon="🎓")

st.title("🎓 AI Student Support Assistant")
st.caption("Answers college-related questions using RAG + Tools + Memory")

with st.sidebar:
    st.header("Student")
    student_id = st.text_input("Student ID / Roll No.", value="guest")
    st.markdown("---")
    st.markdown(
        "**Try asking:**\n"
        "- What is the attendance requirement for exams?\n"
        "- My attendance is 68%, am I eligible?\n"
        "- How do I apply for a bonafide certificate?\n"
        "- What is covered in Unit 4 of CS301?\n"
        "- Contact details for the IT helpdesk?\n"
        "- I want to raise a complaint about my hostel room."
    )
    if st.button("Reset conversation"):
        st.session_state.pop("agent", None)
        st.session_state.pop("messages", None)
        st.rerun()

# Initialize agent (once per student id change)
if "agent" not in st.session_state or st.session_state.get("student_id") != student_id:
    st.session_state.agent = StudentSupportAgent(student_id=student_id)
    st.session_state.student_id = student_id
    st.session_state.messages = []

agent: StudentSupportAgent = st.session_state.agent

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            st.caption(msg["meta"])

# Chat input
user_input = st.chat_input("Ask about regulations, syllabus, FAQs, or notices...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    result = agent.handle_message(user_input)

    if result["source"] == "tool":
        meta = f"🛠️ Tool used: `{result['used_tool']}`"
    elif result["retrieved"]:
        sources = ", ".join(sorted({r["source"] for r in result["retrieved"]}))
        meta = f"📚 Retrieved from: {sources}"
    else:
        meta = "📚 RAG (no strong match found)"

    st.session_state.messages.append({"role": "assistant", "content": result["answer"], "meta": meta})
    with st.chat_message("assistant"):
        st.markdown(result["answer"])
        st.caption(meta)
