"""
tools.py
--------
Implements the "Tools" capability. Each tool is a plain Python function the
agent can call for tasks that need an action rather than a document lookup
(e.g. logging a complaint, checking office contact info, escalating to a
human). In a production build these would call real college APIs/ERP
systems; here they are simulated so the whole project runs standalone.
"""

import json
import os
from datetime import datetime

TICKETS_FILE = "data/support_tickets.json"

OFFICE_DIRECTORY = {
    "admin office": {"location": "Room 12, Admin Block", "hours": "9 AM - 5 PM, Mon-Fri"},
    "examination cell": {"location": "Room 5, Admin Block", "hours": "10 AM - 4 PM, Mon-Fri"},
    "hostel warden": {"location": "Hostel Block A, Ground Floor", "hours": "9 AM - 6 PM, all days"},
    "it helpdesk": {"location": "Room 3, IT Block", "hours": "9 AM - 6 PM, Mon-Sat", "email": "ithelpdesk@college.edu"},
    "scholarship cell": {"location": "Admin Block", "hours": "10 AM - 4 PM, Mon-Fri", "email": "scholarships@college.edu"},
}


def lookup_office_contact(office_name: str) -> dict:
    """Tool: look up contact/location details for a named college office."""
    key = office_name.strip().lower()
    for name, info in OFFICE_DIRECTORY.items():
        if key in name or name in key:
            return {"office": name, **info}
    return {"error": f"No contact details found for '{office_name}'."}


def raise_support_ticket(student_id: str, category: str, description: str) -> dict:
    """Tool: log a support ticket / escalation for a human staff member to follow up on."""
    os.makedirs("data", exist_ok=True)
    tickets = []
    if os.path.exists(TICKETS_FILE):
        try:
            with open(TICKETS_FILE, "r", encoding="utf-8") as f:
                tickets = json.load(f)
        except (json.JSONDecodeError, IOError):
            tickets = []

    ticket = {
        "ticket_id": f"TCK-{len(tickets) + 1:04d}",
        "student_id": student_id,
        "category": category,
        "description": description,
        "status": "open",
        "created_at": datetime.utcnow().isoformat(),
    }
    tickets.append(ticket)
    with open(TICKETS_FILE, "w", encoding="utf-8") as f:
        json.dump(tickets, f, indent=2)
    return ticket


def check_attendance_eligibility(attendance_percent: float) -> dict:
    """Tool: applies the 75% attendance regulation to tell a student if they're exam-eligible."""
    if attendance_percent >= 75:
        return {"eligible": True, "message": "You meet the 75% attendance requirement."}
    elif attendance_percent >= 65:
        return {
            "eligible": False,
            "message": (
                "You are below 75% but above 65%. You may apply for "
                "condonation with a valid medical certificate."
            ),
        }
    else:
        return {
            "eligible": False,
            "message": "You are below 65% attendance and not eligible to apply for condonation.",
        }


# Registry so the agent can route a query to the right tool by keyword.
TOOL_KEYWORDS = {
    "contact": lookup_office_contact,
    "office": lookup_office_contact,
    "escalate": raise_support_ticket,
    "complaint": raise_support_ticket,
    "attendance": check_attendance_eligibility,
}
