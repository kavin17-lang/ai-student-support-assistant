"""
agent.py
--------
The orchestration layer that ties together RAG + Tools + Memory into one
agent, matching the use-case brief:
"AI Student Support Assistant — answers college-related questions from
regulations, syllabus, FAQs and notices. Key Agent Capabilities: RAG +
Tools + Memory."

Routing logic (simple rule-based "agentic" decision step — this is the
part you can extend with an LLM-based planner/router later):
  1. If the message matches a tool trigger (raise a complaint, escalate,
     ask for office contact info, check attendance eligibility) -> call
     the matching tool.
  2. Otherwise -> retrieve relevant context from the knowledge base (RAG)
     and answer from it.
  3. Every turn is written to conversation memory so follow-up questions
     have context, and key facts (like student id) persist across turns.
"""

import re

from memory_store import ConversationMemory, LongTermMemory
from rag_engine import RAGEngine
from tools import (
    lookup_office_contact,
    raise_support_ticket,
    check_attendance_eligibility,
)


class StudentSupportAgent:
    def __init__(self, student_id: str = "guest"):
        self.student_id = student_id
        self.rag = RAGEngine(data_dir="data")
        self.conv_memory = ConversationMemory(max_turns=8)
        self.long_memory = LongTermMemory()

    # ---- intent routing -------------------------------------------------

    def _detect_tool_intent(self, message: str):
        text = message.lower()

        # Attendance eligibility check, e.g. "my attendance is 68%"
        m = re.search(r"(\d{1,3})\s*%", text)
        if m and "attend" in text:
            pct = float(m.group(1))
            return "attendance", {"attendance_percent": pct}

        # Office contact lookup
        if any(w in text for w in ["contact", "phone number", "email of", "where is the", "office"]):
            for office_name in [
                "admin office", "examination cell", "hostel warden",
                "it helpdesk", "scholarship cell",
            ]:
                if office_name.split()[0] in text:
                    return "office", {"office_name": office_name}

        # Escalation / complaint
        if any(w in text for w in ["complaint", "escalate", "not resolved", "raise a ticket", "talk to a human"]):
            return "escalate", {"description": message}

        return None, {}

    # ---- main entry point -------------------------------------------------

    def handle_message(self, message: str) -> dict:
        """
        Returns a dict:
          {
            "answer": str,
            "source": "tool" | "rag",
            "used_tool": str | None,
            "retrieved": list,   # RAG chunks, if any
          }
        """
        intent, params = self._detect_tool_intent(message)

        if intent == "attendance":
            result = check_attendance_eligibility(**params)
            answer = result["message"]
            self.conv_memory.add_turn(message, answer)
            return {"answer": answer, "source": "tool", "used_tool": "check_attendance_eligibility", "retrieved": []}

        if intent == "office":
            result = lookup_office_contact(**params)
            if "error" in result:
                answer = result["error"]
            else:
                answer = (
                    f"{result['office'].title()} is located at {result['location']} "
                    f"(hours: {result['hours']})."
                )
                if "email" in result:
                    answer += f" You can also email {result['email']}."
            self.conv_memory.add_turn(message, answer)
            return {"answer": answer, "source": "tool", "used_tool": "lookup_office_contact", "retrieved": []}

        if intent == "escalate":
            ticket = raise_support_ticket(self.student_id, category="general", description=params["description"])
            answer = (
                f"I've raised support ticket {ticket['ticket_id']} for you. "
                "A staff member will follow up with you shortly."
            )
            self.conv_memory.add_turn(message, answer)
            return {"answer": answer, "source": "tool", "used_tool": "raise_support_ticket", "retrieved": []}

        # Default: RAG over college documents
        answer, retrieved = self.rag.generate_answer(message)
        self.conv_memory.add_turn(message, answer)
        self.long_memory.remember(self.student_id, "last_topic", message)
        return {"answer": answer, "source": "rag", "used_tool": None, "retrieved": retrieved}
