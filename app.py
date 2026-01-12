import json
import os
import random
import re
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List

# import anthropic
# import openai
import requests

# import src.ranchocordova.chatbot_enhanced as chatbot_enhanced
import torch
from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)

import document_processors.mcp_logic as mcp_logic

# Import both chatbots
# import src.ranchocordova.chatbot as chatbot_original
import src.ranchocordova.chatbot_unified
from document_processors.claude_processor import ClaudeLikeDocumentProcessor
from document_processors.mcp_logic import (
    # call_mcp_tool,
    # call_tool_with_sql,
    detect_visualization_intent,
    # discover_tools,
    # generate_llm_response,
    # generate_table_description,
    generate_visualization,
    # parse_user_query,
)
from document_processors.specific_folder_reader import (
    OperationsDocumentProcessor,
    ProjectDocumentProcessor,
)
from src.ranchocordova.chatbot_unified import _llm, generate_answer, generate_response_streaming, initialize_models

print("🔥 Warming models at startup")
src.ranchocordova.chatbot_unified.initialize_models()

model, tokenizer = src.ranchocordova.chatbot_unified._llm  # ✅ THIS WORKS

inputs = tokenizer("warmup", return_tensors="pt").to(model.device)
with torch.inference_mode():
    model.generate(**inputs, max_new_tokens=1)

print("🔥 Warm-up complete")


request_tracker = defaultdict(list)
request_lock = threading.Lock()
# openai_client = (
#   openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#  if os.getenv("OPENAI_API_KEY")
# else None
# )
# print("🔑  OpenAI client initialized:", openai_client is not None)


app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config["JSON_AS_ASCII"] = False

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

MCP_GDRIVE_URL = os.getenv("MCP_GDRIVE_URL", "http://127.0.0.1:8000")

# Initialize Anthropic client
# try:
#    claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
#    print("Claude client initialized successfully")
# except Exception as e:
#    print(f"Warning: Claude client not initialized: {e}")
claude_client = None

# Cache for file contents and folder structure
file_content_cache = {}
folder_structure_cache = {}

# We use this to for the project management and operation Q&A pages, to read the files
project_processor = None
operations_processor = None


def clean_and_format_response(response_text):
    """Clean up response and ensure proper markdown formatting"""
    import re

    # Remove metadata lines
    metadata_patterns = [
        r"---[\s\S]*?---",
        r"Iterative Analysis Complete.*?\.",
        r"Session total:.*?\.",
        r"Files accessed:.*?\.",
        r"Analysis based on.*?\.",
        r"\d+ iterations using.*?\.",
    ]

    for pattern in metadata_patterns:
        response_text = re.sub(pattern, "", response_text, flags=re.IGNORECASE)

    # Remove ASCII box-drawing characters
    ascii_chars = ["│", "├", "┤", "┼", "─", "┬", "┴", "╋", "║", "═", "╔", "╗", "╚", "╝"]
    for char in ascii_chars:
        response_text = response_text.replace(char, "")

    # Fix malformed tables (pipes without proper spacing)
    # Convert: | Project | Cost to proper markdown table
    lines = response_text.split("\n")
    fixed_lines = []
    in_table = False

    for i, line in enumerate(lines):
        # Detect table lines (contains | but not formatted properly)
        if "|" in line and line.count("|") >= 2:
            cells = [cell.strip() for cell in line.split("|") if cell.strip()]

            if not in_table:
                # First row - add header
                fixed_lines.append("| " + " | ".join(cells) + " |")
                # Add separator
                fixed_lines.append("|" + "|".join(["---" for _ in cells]) + "|")
                in_table = True
            else:
                # Data rows
                fixed_lines.append("| " + " | ".join(cells) + " |")
        else:
            if in_table and line.strip() == "":
                in_table = False
            fixed_lines.append(line)

    response_text = "\n".join(fixed_lines)

    # Remove excessive blank lines
    response_text = re.sub(r"\n{3,}", "\n\n", response_text)

    # Ensure proper spacing around headers
    response_text = re.sub(r"\n(#{1,3}\s)", r"\n\n\1", response_text)
    response_text = re.sub(r"(#{1,3}\s[^\n]+)\n([^\n#])", r"\1\n\n\2", response_text)

    return response_text.strip()


#####################################################
#### ALL THE DIFFERENT PAGES' ROUTES IN THIS APP ####
#####################################################


# Dummy user model
class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password


users = {"admin": User(1, "admin", "password123")}


@login_manager.user_loader
def load_user(user_id):
    for user in users.values():
        if user.id == int(user_id):
            return user
    return None


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if not username or not password:
            flash("Please provide both username and password", "danger")
            return render_template("login.html")

        user = users.get(username)
        if user and user.password == password:
            login_user(user)
            return redirect(
                url_for(
                    "agent_catalog",
                    cat="public_services",
                    dept="ranchocordova",
                    dept_display="City of Rancho Cordova",
                )
            )  # Changed from "dashboard"
        else:
            flash("Invalid username or password", "danger")

    return render_template("login.html")


@app.route("/categories")
@login_required
def categories():
    return render_template("categories.html")


@app.route("/dashboard/<category>")
@login_required
def dashboard(category):
    # Define ALL departments for each category (9 total)
    category_departments = {
        "public_services": [
            {"name": "FTB", "key": "ftb", "icon": "ftb.jpeg"},
            {
                "name": "Rancho Cordova",
                "key": "ranchocordova",
                "icon": "ranchocordova.jpeg",
            },
            {"name": "Dept of Motor Vehicles", "key": "dmv", "icon": "dmv.jpeg"},
            {"name": "City of San Jose", "key": "sanjose", "icon": "sanjose.jpeg"},
            {"name": "Employment Development Dept", "key": "edd", "icon": "edd.jpeg"},
            {"name": "CalPERS", "key": "calpers", "icon": "calpers.jpeg"},
            {"name": "CDFA", "key": "cdfa", "icon": "cdfa.jpeg"},
            {
                "name": "Office of Energy Infrastructure",
                "key": "energy",
                "icon": "energy.jpeg",
            },
            {"name": "Fi$cal", "key": "fiscal", "icon": "fiscal.jpeg"},
        ],
        "energy": [
            {"name": "FTB", "key": "ftb", "icon": "ftb.jpeg"},
            {
                "name": "Rancho Cordova",
                "key": "ranchocordova",
                "icon": "ranchocordova.jpeg",
            },
            {"name": "Dept of Motor Vehicles", "key": "dmv", "icon": "dmv.jpeg"},
            {"name": "City of San Jose", "key": "sanjose", "icon": "sanjose.jpeg"},
            {"name": "Employment Development Dept", "key": "edd", "icon": "edd.jpeg"},
            {"name": "CalPERS", "key": "calpers", "icon": "calpers.jpeg"},
            {"name": "CDFA", "key": "cdfa", "icon": "cdfa.jpeg"},
            {
                "name": "Office of Energy Infrastructure",
                "key": "energy",
                "icon": "energy.jpeg",
            },
            {"name": "Fi$cal", "key": "fiscal", "icon": "fiscal.jpeg"},
        ],
        "health": [
            {"name": "FTB", "key": "ftb", "icon": "ftb.jpeg"},
            {
                "name": "Rancho Cordova",
                "key": "ranchocordova",
                "icon": "ranchocordova.jpeg",
            },
            {"name": "Dept of Motor Vehicles", "key": "dmv", "icon": "dmv.jpeg"},
            {"name": "City of San Jose", "key": "sanjose", "icon": "sanjose.jpeg"},
            {"name": "Employment Development Dept", "key": "edd", "icon": "edd.jpeg"},
            {"name": "CalPERS", "key": "calpers", "icon": "calpers.jpeg"},
            {"name": "CDFA", "key": "cdfa", "icon": "cdfa.jpeg"},
            {
                "name": "Office of Energy Infrastructure",
                "key": "energy",
                "icon": "energy.jpeg",
            },
            {"name": "Fi$cal", "key": "fiscal", "icon": "fiscal.jpeg"},
        ],
    }

    departments = category_departments.get(category, [])
    category_names = {
        "public_services": "Public Services",
        "energy": "Energy",
        "health": "Health",
    }

    return render_template(
        "dashboard.html",
        departments=departments,
        category_name=category_names.get(category, "Unknown"),
        category=category,  # Add this so template can pass it
    )


@app.route("/ftb")
@login_required
def ftb():
    return render_template("ftb.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


### This routes to the modules page ###
@app.route("/modules/<dept>")
@login_required
def modules(dept):
    icons = {
        "ftb": "ftb.jpeg",
        "dmv": "dmv.jpeg",
        "sanjose": "sanjose.jpeg",
        "edd": "edd.jpeg",
        "fiscal": "fiscal.jpeg",
        "ranchocordova": "ranchocordova.jpeg",
        "calpers": "calpers.jpeg",
        "cdfa": "cdfa.jpeg",
        "energy": "energy.jpeg",
    }

    display_names = {
        "ftb": "FTB",
        "dmv": "DMV",
        "sanjose": "San Jose",
        "edd": "EDD",
        "fiscal": "Fi$cal",
        "ranchocordova": "Rancho Cordova",
        "calpers": "CalPERS",
        "cdfa": "CDFA",
        "energy": "Energy",
    }

    modules_list = [
        {"name": "Agent Catalog", "icon": "qa.png", "route": "qa_page"},
        {"name": "Workflow", "icon": "workflow.png", "route": "oops"},
        {"name": "Transaction", "icon": "transaction.png", "route": "oops"},
        {"name": "Insights", "icon": "insights.png", "route": "insights"},
        {"name": "Data Management", "icon": "datamanagement.png", "route": "oops"},
        {"name": "Voice Agent", "icon": "voiceagent.png", "route": "voice_agent"},
    ]

    company_icon = icons.get(dept, "default.jpeg")
    company_name = display_names.get(dept, "Department")
    category = request.args.get("cat", "public_services")

    return render_template(
        "modules.html",
        company_icon=company_icon,
        company_name=company_name,
        modules=modules_list,
        category=category,
        dept=dept,  # Add this
        dept_display=display_names.get(dept, dept),  # Add this
    )


# Department-specific agent configurations
DEPARTMENT_AGENTS = {
    "ftb": [
        {"name": "Enterprise Q&A:<br>PMO Agent", "route": "qa_projects"},
        {"name": "Enterprise Q&A:<br>Operational Agent", "route": "qa_operations"},
        {"name": "Refund & Filing<br>Assistant", "route": "oops"},
        {"name": "Notice Explainer &<br>Resolution Agent", "route": "oops"},
        {"name": "Payment Plan &<br>Collections Agent", "route": "oops"},
        {"name": "Identity Verification &<br>Fraud Triage Agent", "route": "oops"},
        {"name": "Correspondence Intake,<br>Summarization & Routing", "route": "oops"},
        {"name": "Case Dossier &<br>Audit Prep Agent", "route": "oops"},
    ],
    "ranchocordova": [
        {
            "name": "Energy Efficiency<br>Agent",
            "route": "rancho_energy",
            "icon": "energy_agent_ranchocordovapng",
        },
        {
            "name": "Customer Service<br>Agent",
            "route": "rancho_customer_service",
            "icon": "customer_service_ranchocordova.png",
        },
    ],
    # Add more departments as needed
}


@app.route("/agent_catalog/<cat>/<dept>")
@login_required
def agent_catalog(cat, dept):
    dept_display_names = {
        "ftb": "FTB",
        "ranchocordova": "City of Rancho Cordova",
        "dmv": "DMV",
        "sanjose": "San Jose",
        "edd": "EDD",
        "calpers": "CalPERS",
        "cdfa": "CDFA",
        "energy": "Office of Energy Infrastructure",
        "fiscal": "Fi$cal",
    }

    dept_display = dept_display_names.get(dept, dept.upper())

    # Get agents for this department
    agents = DEPARTMENT_AGENTS.get(dept, [])

    return render_template(
        "agent_catalog.html",
        category=cat,
        dept=dept,
        dept_display=dept_display,
        agents=agents,
    )


@app.route("/rancho_energy")
@login_required
def rancho_energy():
    """Rancho Cordova Energy Efficiency Agent"""
    category = request.args.get("cat", "public_services")
    dept = "ranchocordova"
    dept_display = "City of Rancho Cordova"
    agent_name = "Energy Efficiency Agent"
    return render_template(
        "rancho_agent_chat_enhanced.html",  # ✅ ENHANCED
        category=category,
        dept=dept,
        dept_display=dept_display,
        agent_name=agent_name,
        agent_type="energy",
    )


@app.route("/rancho_customer_service")
@login_required
def rancho_customer_service():
    """Rancho Cordova Customer Service Agent"""
    category = request.args.get("cat", "public_services")
    dept = "ranchocordova"
    dept_display = "City of Rancho Cordova"
    agent_name = "Customer Service Agent"
    return render_template(
        "rancho_agent_chat_enhanced.html",  # ✅ ENHANCED (same template!)
        category=category,
        dept=dept,
        dept_display=dept_display,
        agent_name=agent_name,
        agent_type="customer_service",
    )


@app.route("/rancho_agent_api", methods=["POST"])
@login_required
def rancho_agent_api():
    """API endpoint for Rancho Cordova chatbots with visualization support"""
    try:
        data = request.json
        query = data.get("query", "")
        agent_type = data.get("agent_type", "")

        if not query or not query.strip():
            return jsonify({"answer": "Please provide a valid question."})

        # generate_answer now auto-detects agent type and returns visualization
        result = generate_answer(query, agent_type)

        return jsonify(
            {
                "answer": result.get("answer", ""),
                "visualization": result.get("visualization", None),
                "source": "rancho_cordova",
            }
        )

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return jsonify(
            {
                "answer": f"I encountered an error: {str(e)}",
                "visualization": None,
                "source": "error",
            }
        ), 500


@app.route("/rancho_agent_api_stream", methods=["POST"])
@login_required
def rancho_agent_api_stream():
    """
    Streaming API endpoint for Rancho Cordova chatbots.
    Uses Server-Sent Events (SSE) to stream tokens in real-time.
    """
    data = request.json
    query = data.get("query", "")
    agent_type = data.get("agent_type", "")

    if not query or not query.strip():
        return jsonify({"error": "Please provide a valid question."}), 400

    def generate():
        try:
            for token in generate_response_streaming(query, use_rag=True, agent_type=agent_type):
                # Send each token as a Server-Sent Event
                yield f"data: {token}\n\n"
            # Signal completion
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )

def extract_and_format_table(response_text):
    """Extract table data from response text and format as HTML"""
    lines = response_text.split("\n")
    table_lines = []

    # Look for lines that contain table-like data (with | separators)
    for line in lines:
        if "|" in line and len(line.split("|")) >= 3:
            # Clean up the line
            cleaned_line = line.strip()
            if cleaned_line and not all(c in "|-= " for c in cleaned_line):
                table_lines.append(cleaned_line)

    if len(table_lines) >= 2:  # Need at least header + 1 data row
        try:
            # Process the first line as headers
            headers = [h.strip() for h in table_lines[0].split("|") if h.strip()]

            # Process remaining lines as data
            rows = []
            for line in table_lines[1:]:
                cells = [c.strip() for c in line.split("|") if c.strip()]
                if len(cells) == len(headers):  # Only include properly formatted rows
                    rows.append(cells)

            if rows:
                # Build HTML table
                headers_html = "".join([f"<th>{h}</th>" for h in headers])
                rows_html = "".join(
                    [
                        f"<tr>{''.join([f'<td>{cell}</td>' for cell in row])}</tr>"
                        for row in rows
                    ]
                )

                return f"""
                    <div class="mt-3 table-responsive">
                        <table class="table table-hover table-striped" id="operations-table">
                            <thead class="table-dark">
                                <tr>{headers_html}</tr>
                            </thead>
                            <tbody>{rows_html}</tbody>
                        </table>
                    </div>
                """
        except Exception as e:
            print(f"Table extraction error: {e}")

    return None


### This routes to the voice agent page ###
@app.route("/voice_agent")
@login_required
def voice_agent():
    """Voice Agent Page"""
    return render_template("voice_agent.html")


#####################################################
#### RELATED TO SESSION CACHE AND FOR INSPECTION ####
#####################################################
# Add a simple cache inspection endpoint
@app.route("/inspect_cache")
@login_required
def inspect_cache():
    cache_info = {
        "cache_size": len(file_content_cache),
        "cached_file_ids": list(file_content_cache.keys()),
    }
    return jsonify(cache_info)


# Enhanced cache clearing and debugging endpoints
@app.route("/clear_cache", methods=["POST"])
@login_required
def clear_cache():
    global file_content_cache, folder_structure_cache
    file_content_cache = {}
    folder_structure_cache = {}
    return jsonify({"status": "All caches cleared"})


@app.route("/cache_status")
@login_required
def cache_status():
    return jsonify(
        {
            "cached_files": len(file_content_cache),
            "cached_folders": len(folder_structure_cache),
            "cache_keys": list(file_content_cache.keys())[
                :10
            ],  # First 10 for debugging
        }
    )


# Add endpoint to view session context
@app.route("/session_context", methods=["GET"])
@login_required
def view_session_context():
    global document_processor

    if document_processor is None:
        return jsonify({"status": "No active session"})

    return jsonify(
        {
            "session_active": True,
            "files_mentioned": len(
                document_processor.session_context["files_mentioned"]
            ),
            "topics_discussed": document_processor.session_context["topics_discussed"],
            "session_summary": document_processor.session_context["session_summary"],
            "conversation_length": len(document_processor.conversation_history),
        }
    )


@app.route("/debug_routes", methods=["GET"])
def debug_routes():
    """Show all available routes"""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append(f"{rule.endpoint}: {rule.rule} [{', '.join(rule.methods)}]")
    return "<br>".join(sorted(routes))


@app.route("/test_viz")
def test_viz():
    import pandas as pd

    from src.ranchocordova.viz import generate_simple_visualization

    energy_df = pd.read_csv("src/ranchocordova/data/Energy.txt")
    viz = generate_simple_visualization("Show me energy forecast", energy_df, None)

    return f'<html><body><h1>Test</h1><img src="{viz}" style="width:100%; max-width:800px;"></body></html>'


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
