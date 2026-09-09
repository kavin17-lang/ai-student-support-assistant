"""
test_agent.py
-------------
Quick command-line demo of the agent (no Streamlit needed). Useful for a
fast sanity check or for recording a terminal demo for your submission.

Run with:
    python test_agent.py
"""

from agent import StudentSupportAgent

SAMPLE_QUERIES = [
    "What is the attendance requirement to be eligible for exams?",
    "My attendance is 68%, am I eligible for exams?",
    "How do I apply for a bonafide certificate?",
    "What topics are covered in Unit 4 of CS301 - Artificial Intelligence?",
    "What are the contact details for the IT helpdesk?",
    "I want to raise a complaint, my hostel room fan is broken.",
]


def main():
    agent = StudentSupportAgent(student_id="demo_student")
    print("=" * 70)
    print("AI STUDENT SUPPORT ASSISTANT — DEMO RUN")
    print("=" * 70)
    for q in SAMPLE_QUERIES:
        result = agent.handle_message(q)
        print(f"\nStudent: {q}")
        print(f"Assistant [{result['source']}]: {result['answer']}")
    print("\n" + "=" * 70)
    print("Conversation memory (last turns):")
    print(agent.conv_memory.get_recent_context())


if __name__ == "__main__":
    main()
