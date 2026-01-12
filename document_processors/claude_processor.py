import hashlib
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Any, Dict, List

import anthropic
import openai
import requests


class APIProvider(Enum):
    CLAUDE = "claude"
    GPT4 = "gpt4"
    GPT35 = "gpt3.5"


class RobustAPIClient:
    def __init__(self, claude_client, openai_client, available_tools):
        self.claude_client = claude_client
        self.openai_client = openai_client
        self.available_tools = available_tools
        self.openai_functions = self._convert_claude_tools_to_openai()
        self.response_cache = {}
        self.provider_status = {
            APIProvider.CLAUDE: {"available": True, "last_error": None},
            APIProvider.GPT4: {"available": True, "last_error": None},
            APIProvider.GPT35: {"available": True, "last_error": None},
        }

    def _convert_claude_tools_to_openai(self):
        """Convert Claude tools to OpenAI function format"""
        functions = []
        for tool in self.available_tools:
            function = {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            }
            functions.append(function)
        return functions

    def call_with_multi_fallback(self, **kwargs):
        """Try Claude first, then GPT with tool calling"""
        providers = [APIProvider.CLAUDE, APIProvider.GPT4, APIProvider.GPT35]

        # Check cache
        cache_key = hashlib.md5(str(kwargs).encode()).hexdigest()
        if cache_key in self.response_cache:
            print("Using cached response")
            return self.response_cache[cache_key]

        for provider in providers:
            if not self.provider_status[provider]["available"]:
                continue

            try:
                print(f"Trying {provider.value}...")

                if provider == APIProvider.CLAUDE:
                    response = self._call_claude(**kwargs)
                else:
                    model = (
                        "gpt-4-turbo"
                        if provider == APIProvider.GPT4
                        else "gpt-3.5-turbo"
                    )
                    filtered_kwargs = {k: v for k, v in kwargs.items() if k != "model"}
                    response = self._gpt_function_call(model, **filtered_kwargs)

                # Success - cache and return
                self.response_cache[cache_key] = response
                self.provider_status[provider]["available"] = True
                return response

            except Exception as e:
                error_code = str(e)
                print(f"{provider.value} failed: {error_code}")

                if "429" in error_code or "529" in error_code:
                    self.provider_status[provider]["available"] = False
                    self._schedule_provider_reset(provider, 300)  # 5 min cooldown
                continue

        # All failed - return fallback
        return self._create_fallback_response(kwargs.get("messages", []))

    def _call_claude(self, **kwargs):
        """Claude with retry"""
        max_retries = 2
        for attempt in range(max_retries):
            try:
                return self.claude_client.messages.create(**kwargs)
            except Exception as e:
                if ("429" in str(e) or "529" in str(e)) and attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                raise e

    def _gpt_function_call(self, model, **kwargs):
        """GPT with function calling - aggressive version"""
        messages = []

        # Add aggressive system prompt addition
        if kwargs.get("system"):
            aggressive_addon = "\n\nIMPORTANT: Be extremely thorough. If one file doesn't have information, immediately try 2-3 more files. Never give up easily."
            kwargs["system"] = kwargs["system"] + aggressive_addon
            messages.append({"role": "system", "content": kwargs["system"]})

        for msg in kwargs.get("messages", []):
            content = self._convert_claude_message(msg.get("content", ""))
            messages.append({"role": msg["role"], "content": content})

        # Increase max_tokens for more detailed responses
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            functions=self.openai_functions if self.openai_functions else None,
            function_call="auto" if self.openai_functions else None,
            max_tokens=kwargs.get("max_tokens", 6000),  # Increased from 4000
            temperature=0.2,  # Slightly higher for more creative file searching
        )

        return self._convert_gpt_to_claude_format(response)

    def _call_gpt_with_tools(self, model, **kwargs):
        """GPT with function calling"""
        messages = []
        if kwargs.get("system"):
            messages.append({"role": "system", "content": kwargs["system"]})

        for msg in kwargs.get("messages", []):
            content = self._convert_claude_message(msg.get("content", ""))
            messages.append({"role": msg["role"], "content": content})

        gpt_kwargs = {k: v for k, v in kwargs.items() if k != "model"}

        response = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            functions=self.openai_functions if self.openai_functions else None,
            function_call="auto" if self.openai_functions else None,
            max_tokens=gpt_kwargs.get("max_tokens", 4000),
            temperature=gpt_kwargs.get("temperature", 0.1),
        )

        return self._convert_gpt_to_claude_format(response)

    def _convert_claude_message(self, content):
        """Convert Claude structured content to text"""
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        parts.append(part["text"])
                    elif part.get("type") == "tool_result":
                        parts.append(f"Tool result: {part.get('content', '')}")
            return "\n".join(parts)
        return str(content)

    def _convert_gpt_to_claude_format(self, gpt_response):
        """Convert GPT response to Claude-compatible format"""
        choice = gpt_response.choices[0]

        class GPTResponse:
            def __init__(self, gpt_choice):
                self.content = []

                if gpt_choice.message.content:
                    self.content.append(GPTContent(gpt_choice.message.content))

                if gpt_choice.message.function_call:
                    self.content.append(GPTToolUse(gpt_choice.message.function_call))
                    self.stop_reason = "tool_use"
                else:
                    self.stop_reason = "end_turn"

        class GPTContent:
            def __init__(self, text):
                self.type = "text"
                self.text = text

        class GPTToolUse:
            def __init__(self, function_call):
                self.type = "tool_use"
                self.id = f"gpt_{int(time.time() * 1000)}"
                self.name = function_call.name
                self.input = json.loads(function_call.arguments)

        return GPTResponse(choice)

    def _create_fallback_response(self, messages):
        """Emergency fallback when all APIs fail"""

        class FallbackResponse:
            def __init__(self, text):
                self.content = [FallbackContent(text)]
                self.stop_reason = "end_turn"

        class FallbackContent:
            def __init__(self, text):
                self.type = "text"
                self.text = text

        return FallbackResponse(
            "I'm experiencing technical difficulties with AI services. Please try again shortly."
        )

    def _schedule_provider_reset(self, provider, delay):
        """Re-enable provider after delay"""
        import threading

        def reset():
            time.sleep(delay)
            self.provider_status[provider]["available"] = True
            print(f"{provider.value} re-enabled")

        threading.Thread(target=reset, daemon=True).start()


def call_claude_with_retry_and_fallback(claude_client, **kwargs):
    """Call Claude API with retry logic and GPT fallback"""
    max_retries = 3
    base_delay = 1

    # Try caching first
    cache_key = hashlib.md5(str(kwargs).encode()).hexdigest()
    if cache_key in claude_response_cache:
        print("Using cached Claude response")
        return claude_response_cache[cache_key]

    # Try Claude with retries
    for attempt in range(max_retries):
        try:
            response = claude_client.messages.create(**kwargs)
            # Cache successful response
            claude_response_cache[cache_key] = response
            return response

        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                print(f"Rate limit hit, retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                continue
            elif "429" in str(e) and attempt == max_retries - 1:
                # Final attempt failed - try GPT fallback
                print("Claude rate limited, falling back to GPT-4...")
                return call_gpt_fallback(claude_client, **kwargs)

            else:
                raise e

    # If all retries failed, try GPT
    return call_gpt_fallback(claude_client, **kwargs)


def call_gpt_fallback(claude_client, **kwargs):  # Added claude_client parameter
    """Fallback to GPT-4 when Claude fails"""
    if not openai_client:
        raise Exception("OpenAI API not configured and Claude failed")

    try:
        # Convert Claude format to OpenAI format
        messages = []
        if kwargs.get("system"):
            messages.append({"role": "system", "content": kwargs["system"]})

        # Convert Claude messages to OpenAI format
        for msg in kwargs.get("messages", []):
            if isinstance(msg.get("content"), list):
                # Handle tool results - convert to text
                content_parts = []
                for part in msg["content"]:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            content_parts.append(part["text"])
                        elif part.get("type") == "tool_result":
                            content_parts.append(
                                f"Tool result: {part.get('content', '')}"
                            )
                content = "\n".join(content_parts)
            else:
                content = msg.get("content", "")

            messages.append({"role": msg["role"], "content": content})

        # Call GPT-4
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 4000),
            temperature=kwargs.get("temperature", 0.1),
        )

        # Convert back to Claude-like response format
        class MockResponse:
            def __init__(self, gpt_response):
                self.content = [MockContent(gpt_response.choices[0].message.content)]
                self.stop_reason = "end_turn"

        class MockContent:
            def __init__(self, text):
                self.type = "text"
                self.text = text

        return MockResponse(response)

    except Exception as e:
        print(f"GPT fallback also failed: {e}")
        raise Exception("Both Claude and GPT APIs failed")


class ClaudeLikeDocumentProcessor:
    def __init__(self, http_server_url, claude_client):
        self.http_server_url = http_server_url
        self.claude_client = claude_client
        self.openai_client = (
            openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            if os.getenv("OPENAI_API_KEY")
            else None
        )
        self.last_api_call = 0
        self.min_delay = 0.2
        self.conversation_history = []
        self._discovery_lock = False
        self._discovery_complete = False
        self.session_context = {
            "files_mentioned": {},  # Track files that have been discussed
            "topics_discussed": [],  # Track conversation topics
            "user_preferences": {},  # Learn user patterns
            "session_summary": "",  # Running summary of conversation - ADD COMMA HERE
            "extracted_data_by_query": {},  # COLON not equals
            "visualization_ready_data": {},  # COLON not equals, fix spelling
            "conversation_flow": [],  # COLON not equals
            "last_query_data": None,  # COLON not equals
            "cumulative_project_data": [],  # Store all projects found
            "cumulative_employee_data": [],  # Store all employees found
            "current_query_type": None,  # Track what we're collecting
            "document_catalog": {},
            "total_documents_available": 0,
        }
        self.last_table_data = None  # Store parsed table for visualization
        self.available_tools = [
            {
                "name": "search_files",
                "description": "Search for files in Google Drive by name or content type. Uses enhanced fallback search strategies.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query - can be partial matches, will try broader searches if needed",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum results",
                            "default": 20,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "read_file",
                "description": "Read complete content from a specific file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_id": {
                            "type": "string",
                            "description": "Google Drive file ID",
                        },
                        "focus_area": {
                            "type": "string",
                            "description": "Optional: specific aspect to focus on (e.g., 'budget', 'timeline', 'personnel')",
                        },
                    },
                    "required": ["file_id"],
                },
            },
            {
                "name": "get_file_metadata",
                "description": "Get metadata about files without reading full content",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_pattern": {
                            "type": "string",
                            "description": "Pattern to match files",
                            "default": "",
                        }
                    },
                },
            },
            {
                "name": "recall_session_context",
                "description": "Access previous conversation context and discussed topics",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "context_type": {
                            "type": "string",
                            "enum": [
                                "files_mentioned",
                                "topics_discussed",
                                "full_summary",
                            ],
                            "description": "Type of context to retrieve",
                        }
                    },
                    "required": ["context_type"],
                },
            },
        ]
        self.api_client = RobustAPIClient(
            claude_client, self.openai_client, self.available_tools
        )

    def update_session_context(self, query, files_accessed, topics):
        """Update session context with new information"""

        # Update files mentioned
        for file_info in files_accessed:
            file_id = file_info.get("id") or file_info.get("file_id")
            if file_id:
                self.session_context["files_mentioned"][file_id] = {
                    "name": file_info.get(
                        "name", file_info.get("file_name", "unknown")
                    ),
                    "last_accessed": "current_session",
                    "context": f"Accessed when user asked: {query[:100]}",
                }

        # Update topics
        query_topics = self.extract_topics_from_query(query)
        for topic in query_topics:
            if topic not in self.session_context["topics_discussed"]:
                self.session_context["topics_discussed"].append(topic)

        # Update session summary
        if len(self.session_context["topics_discussed"]) > 0:
            self.session_context["session_summary"] = (
                f"User has been asking about: {', '.join(self.session_context['topics_discussed'])}. Files accessed: {len(self.session_context['files_mentioned'])} total."
            )

    def extract_topics_from_query(self, query):
        """Extract topics from user query"""
        topics = []
        query_lower = query.lower()

        # Common project-related topics
        topic_keywords = {
            "projects": ["project", "initiative", "development"],
            "budget": ["budget", "cost", "expense", "financial"],
            "employees": ["employee", "staff", "team", "personnel"],
            "incidents": ["incident", "issue", "problem", "bug"],
            "reports": ["report", "analysis", "summary"],
            "timeline": ["timeline", "schedule", "deadline", "milestone"],
        }

        for topic, keywords in topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                topics.append(topic)

        return topics

    # Add this method to ClaudeLikeDocumentProcessor class
    def _debug_table_structure(self, response_text):
        """Debug helper to see table structure"""
        lines = response_text.split("\n")
        table_lines = [
            line for line in lines if "|" in line and len(line.split("|")) >= 3
        ]

        if table_lines:
            print("\n=== TABLE STRUCTURE DEBUG ===")
            for i, line in enumerate(table_lines[:5]):  # First 5 rows
                cells = [cell.strip() for cell in line.split("|")]
                print(f"Row {i}: {cells}")
            print("=== END TABLE DEBUG ===\n")

    def parse_table_structure(self, response_text):
        """
        Parse markdown table preserving COMPLETE structure.
        This replaces all the regex nonsense.

        Returns:
            dict: {
                "headers": ["Column1", "Column2", ...],
                "rows": [
                    {"Column1": "value1", "Column2": "value2", ...},
                    ...
                ],
                "raw_table": "original markdown table text"
            }
        """
        lines = response_text.split("\n")

        headers = []
        rows = []
        table_lines = []
        in_table = False

        for line in lines:
            # Check if this line is part of a table
            if "|" in line:
                stripped = line.strip()

                # Skip empty lines and separator rows (---)
                if not stripped or all(c in "|-= " for c in stripped):
                    if in_table:
                        table_lines.append(line)
                    continue

                # Split into cells
                cells = [c.strip() for c in line.split("|")]
                # Remove empty first/last cells (from leading/trailing pipes)
                cells = [c for c in cells if c.strip()]

                if not cells:
                    continue

                # First row with actual content = headers
                if not headers:
                    headers = cells
                    in_table = True
                    table_lines.append(line)
                    print(f"📋 Found table with {len(headers)} columns: {headers}")
                else:
                    # Data row - must match header count
                    # Data row - handle mismatched columns gracefully
                    if len(cells) == len(headers):
                        # Perfect match - use as-is
                        row_dict = dict(zip(headers, cells))
                        rows.append(row_dict)
                        table_lines.append(line)
                    elif len(cells) < len(headers):
                        # Row has fewer cells than headers - pad with "N/A"
                        print(
                            f"⚠️ Row has {len(cells)} cells, padding to {len(headers)} columns"
                        )

                        # Pad cells with N/A to match header count
                        padded_cells = cells + ["N/A"] * (len(headers) - len(cells))
                        row_dict = dict(zip(headers, padded_cells))
                        rows.append(row_dict)
                        table_lines.append(line)

                        print(
                            f"   Padded row: {list(row_dict.keys())[:5]} = {list(row_dict.values())[:5]}"
                        )
                    elif len(cells) > len(headers):
                        # Row has more cells than headers - truncate
                        print(
                            f"⚠️ Row has {len(cells)} cells, truncating to {len(headers)} columns"
                        )

                        truncated_cells = cells[: len(headers)]
                        row_dict = dict(zip(headers, truncated_cells))
                        rows.append(row_dict)
                        table_lines.append(line)
                    else:
                        print(f"❌ Skipping invalid row: {cells}")
            elif in_table and not line.strip():
                # Empty line after table = end of table
                break

        if not headers or not rows:
            print("❌ No valid table found in response")
            return None

        print(f"✅ Parsed table: {len(headers)} columns × {len(rows)} rows")

        return {
            "headers": headers,
            "rows": rows,
            "raw_table": "\n".join(table_lines),
            "row_count": len(rows),
        }

    def table_to_html(self, table_data):
        """
        Convert parsed table data to clean HTML.

        Args:
            table_data: Output from parse_table_structure()

        Returns:
            str: HTML table string
        """
        if not table_data:
            return None

        headers = table_data["headers"]
        rows = table_data["rows"]

        # Build HTML
        headers_html = "".join([f"<th>{h}</th>" for h in headers])

        rows_html = ""
        for row in rows:
            cells_html = "".join([f"<td>{row.get(h, 'N/A')}</td>" for h in headers])
            rows_html += f"<tr>{cells_html}</tr>\n"

        html = f"""
        <div class="table-responsive mt-3">
            <table class="table table-hover table-striped table-bordered">
                <thead class="table-dark">
                    <tr>{headers_html}</tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """

        return html

    def validate_visualization_alignment(self, table_data, viz_html):
        """
        Validate that visualization headers match table headers.
        """
        if not table_data or not viz_html:
            return True

        original_headers = table_data["headers"]

        # Extract headers from viz HTML (simple check)
        viz_headers = []
        for header in original_headers:
            if header in viz_html:
                viz_headers.append(header)

        if len(viz_headers) < len(original_headers):
            print(f"⚠️ VALIDATION WARNING:")
            print(f"   Original headers: {original_headers}")
            print(f"   Found in viz: {viz_headers}")
            print(f"   Missing: {set(original_headers) - set(viz_headers)}")
            return False

        print(
            f"✅ Validation passed: All {len(original_headers)} headers present in visualization"
        )
        return True

    def extract_visualization_data(self, table_data, user_query):
        """
        Extract RELEVANT data for visualization from parsed table.

        This replaces all the regex pattern matching nonsense.

        Args:
            table_data: Output from parse_table_structure()
            user_query: The original user question

        Returns:
            dict: Ready for visualization with proper context
        """
        if not table_data:
            return None

        headers = table_data["headers"]
        rows = table_data["rows"]

        # Identify the "identifier" column (usually first column with names)
        identifier_col = headers[0]  # Usually "Project Name", "Employee Name", etc.

        # Find numeric columns
        numeric_columns = []
        for header in headers[1:]:  # Skip first column (names)
            # Check if this column contains numeric data
            sample_values = [row.get(header, "") for row in rows[:3]]

            is_numeric = False
            for val in sample_values:
                # Try to parse as number
                clean_val = (
                    str(val).replace("$", "").replace(",", "").replace("%", "").strip()
                )
                try:
                    float(clean_val)
                    is_numeric = True
                    break
                except (ValueError, AttributeError):
                    continue

            if is_numeric:
                numeric_columns.append(header)

        print(f"📊 Found numeric columns: {numeric_columns}")

        # Extract data
        viz_data = {
            "identifier_column": identifier_col,
            "numeric_columns": numeric_columns,
            "data": [],
        }

        for row in rows:
            item = {"name": row.get(identifier_col, "Unknown"), "values": {}}

            for num_col in numeric_columns:
                raw_value = row.get(num_col, "")

                # Parse numeric value
                clean_value = (
                    str(raw_value)
                    .replace("$", "")
                    .replace(",", "")
                    .replace("%", "")
                    .strip()
                )

                # Handle special cases
                if clean_value.startswith("-"):
                    sign = -1
                    clean_value = clean_value[1:]
                else:
                    sign = 1

                try:
                    numeric_value = float(clean_value) * sign
                    item["values"][num_col] = numeric_value
                except (ValueError, AttributeError):
                    item["values"][num_col] = 0

            viz_data["data"].append(item)

        print(f"✅ Extracted visualization data for {len(viz_data['data'])} items")
        print(f"   Columns: {numeric_columns}")

        return viz_data

    def _clean_extraction_name(self, name):
        """Clean names from table extractions"""
        # Remove markdown and table formatting
        name = re.sub(r"\*+", "", name)
        name = re.sub(r"[|_\-=]{2,}", "", name)
        name = name.strip()

        # Remove common table headers/metadata
        skip_patterns = [
            r"employee\s*name",
            r"project\s*name",
            r"name",
            r"score",
            r"rating",
            r"productivity",
            r"efficiency",
            r"budget",
            r"cost",
        ]

        for pattern in skip_patterns:
            if re.match(pattern, name.lower()):
                return None

        return name

    def _is_meaningful_data(self, name, value, pattern_type):
        """Validate if this is actual business data"""
        if not name or len(name) < 3 or len(name) > 30:
            return False

        # Check for obvious metadata
        metadata_terms = ["scale", "total", "range", "from", "to", "chart", "table"]
        if any(term in name.lower() for term in metadata_terms):
            return False

        # Value range validation by data type
        if "employee" in pattern_type.lower():
            return 0 <= value <= 5 or 0 <= value <= 100  # Ratings or percentages
        elif "project" in pattern_type.lower():
            return value >= 100000  # Reasonable project budget minimum

        return value > 0

    def _extract_table_column_data(
        self, response_text, name_column_index=0, value_column_index=7
    ):
        """Extract specific columns from markdown tables"""
        import re

        lines = response_text.split("\n")
        data = {}

        for line in lines:
            if "|" in line and not all(c in "|-= " for c in line):
                cells = [c.strip().strip("*") for c in line.split("|") if c.strip()]

                # Skip header and separator rows
                if len(cells) > value_column_index and cells[0] not in [
                    "",
                    "Project Name",
                    "Employee Name",
                ]:
                    try:
                        name = cells[name_column_index]
                        value_str = (
                            cells[value_column_index].replace("$", "").replace(",", "")
                        )
                        value = float(value_str)

                        if len(name) > 3 and value > 1000:
                            data[name] = value
                    except (ValueError, IndexError):
                        continue

        return data

    def _clean_name(self, name):
        """Clean up extracted names and separate status information"""
        import re

        # Remove newlines and clean up
        name = name.replace("\n", " ").strip()

        # Remove status words from project names
        status_words = ["completed", "cancelled", "in progress", "on hold", "active"]
        for status in status_words:
            name = re.sub(rf"\b{status}\b", "", name, flags=re.IGNORECASE)

        # Clean up formatting
        name = name.strip().strip("*").strip('"').strip("'")
        name = re.sub(r"\s+", " ", name)  # Multiple spaces to single
        name = name.replace("**", "").replace("|", "")

        return name.strip()

    def _extract_project_with_status(self, text_match):
        """Extract project name and status from mixed text"""
        import re

        statuses = {
            "completed": "Completed",
            "cancelled": "Cancelled",
            "in progress": "In Progress",
            "on hold": "On Hold",
            "active": "Active",
        }

        # Find status in the text
        found_status = "Unknown"
        clean_name = text_match

        for status_key, status_value in statuses.items():
            if status_key in text_match.lower():
                found_status = status_value
                # Remove status from name
                clean_name = re.sub(
                    rf"\b{status_key}\b", "", text_match, flags=re.IGNORECASE
                )
                break

        clean_name = self._clean_name(clean_name)

        return clean_name, found_status

    def _parse_value(self, value_str):
        """Parse various value formats"""
        value_str = value_str.replace(",", "").replace("$", "")

        if "%" in value_str:
            return float(value_str.replace("%", ""))
        elif "M" in value_str or "million" in value_str.lower():
            return float(value_str.replace("M", "").replace("million", "")) * 1000000
        elif "K" in value_str or "thousand" in value_str.lower():
            return float(value_str.replace("K", "").replace("thousand", "")) * 1000
        else:
            return float(value_str)

    def _is_valid_data_point(self, name, value):
        """Check if this is valid visualization data"""
        # Skip metadata and system messages
        skip_terms = [
            "iterative",
            "session",
            "analysis",
            "files accessed",
            "total found",
            "search strategy",
            "claude",
            "gpt",
            "api",
            "tool",
            "iteration",
        ]

        if any(term in name.lower() for term in skip_terms):
            return False

        # Must have reasonable name length and value
        if len(name) < 3 or len(name) > 50:
            return False

        # Value should be reasonable (not 0 or 1 for most business data)
        if isinstance(value, (int, float)) and (value < 0 or value > 100000000):
            return False

        return True

    def _categorize_data(self, data, response_context):
        """Automatically categorize data based on content"""
        data_keys = " ".join(data.keys()).lower()

        # Employee/People metrics
        if any(
            term in response_context
            for term in [
                "employee",
                "staff",
                "productivity",
                "performance",
                "availability",
            ]
        ):
            if any(
                term in data_keys for term in ["score", "rating", "%", "productivity"]
            ):
                return "employee_metrics"
            else:
                return "employee_data"

        # Incident/Issue tracking
        elif any(
            term in response_context
            for term in ["incident", "issue", "bug", "ticket", "problem"]
        ):
            if any(
                term in data_keys
                for term in ["open", "closed", "pending", "critical", "high", "low"]
            ):
                return "incident_status"
            else:
                return "incident_metrics"

        # Project data
        elif any(
            term in response_context
            for term in ["project", "initiative", "budget", "timeline"]
        ):
            if any(term in data_keys for term in ["budget", "cost", "$"]):
                return "project_budgets"
            else:
                return "project_data"

        # Financial data
        elif any(
            term in data_keys for term in ["$", "budget", "cost", "revenue", "expense"]
        ):
            return "financial_data"

        # Performance metrics (percentages)
        elif any(
            str(v)
            for v in data.values()
            if isinstance(v, (int, float)) and 0 < v <= 100
        ):
            return "performance_metrics"

        # Default categorization
        else:
            return "general_data"

    def _extract_and_store_response_data(self, user_query, final_response):
        """Enhanced data storage with proper type separation"""

        print(f"\n>>> EXTRACTING DATA - Response length: {len(final_response)}")
        print(f">>> First 300 chars: {final_response[:300]}")

        # Determine current data type
        query_lower = user_query.lower()
        current_data_type = "general"

        if any(
            term in query_lower
            for term in ["employee", "productivity", "efficiency", "staff", "worker"]
        ):
            current_data_type = "employee"
        elif any(
            term in query_lower for term in ["project", "budget", "cost", "initiative"]
        ):
            current_data_type = "project"
        elif any(
            term in query_lower for term in ["incident", "issue", "bug", "ticket"]
        ):
            current_data_type = "incident"

        print(f"📊 Current data type: {current_data_type}")

        # Check if we're switching data types
        last_data_type = getattr(self, "_last_data_type", None)
        if last_data_type and last_data_type != current_data_type:
            print(
                f"🔄 Switching from {last_data_type} to {current_data_type} - clearing old data"
            )
            # Clear old visualization data when switching types
            self.session_context["visualization_ready_data"] = {}
            self.session_context["last_query_data"] = None

        # Store current data type
        self._last_data_type = current_data_type

        # Extract new data using existing methods
        viz_data = None
        try:
            if self.last_table_data:
                viz_data = self.extract_visualization_data(
                    self.last_table_data, user_query
                )
                if viz_data:
                    print(
                        f">>> EXTRACTED VIZ DATA: {viz_data.get('numeric_columns', [])}"
                    )
            else:
                print("⚠️ No table data available for visualization")
        except Exception as e:
            print(f"❌ Visualization extraction failed: {e}")
            import traceback

            traceback.print_exc()
            viz_data = None

        if viz_data:
            # The new format from extract_visualization_data is:
            # {"identifier_column": "...", "numeric_columns": [...], "data": [...]}
            # Store it directly - no need for pattern matching
            typed_key = f"{current_data_type}_{len(self.session_context.get('extracted_data_by_query', {}))}"
            self.session_context["extracted_data_by_query"][typed_key] = viz_data
            self.session_context["last_query_data"] = viz_data

            print(
                f"📊 Stored {current_data_type} data with {len(viz_data.get('data', []))} items"
            )
            print(f"   Numeric columns: {viz_data.get('numeric_columns', [])}")
        else:
            print("❌ No visualization data extracted")

        # Store conversation pair for context
        self.session_context["conversation_flow"].append(
            {
                "query": user_query,
                "response_preview": final_response[:200] + "...",
                "data_extracted": bool(viz_data),
                "data_keys": list(viz_data.keys()) if viz_data else [],
            }
        )

        print(
            f"💾 Total stored queries with data: {len([q for q in self.session_context['extracted_data_by_query'].values() if q])}"
        )

        # if 'list' in user_query.lower() or 'all' in user_query.lower():
        #    if 'project' in user_query.lower():
        #        # Extract project names from response
        #        projects = self._extract_project_list(final_response)
        #        self.session_context["cumulative_project_data"].extend(projects)
        #    elif 'employee' in user_query.lower():
        #        employees = self._extract_employee_list(final_response)
        #        self.session_context["cumulative_employee_data"].extend(employees)

    def _resolve_contextual_references(self, query):
        """Resolve 'them', 'it', 'those' references to previous data"""

        query_lower = query.lower()
        pronouns = ["them", "those", "it", "this", "these"]

        if any(pronoun in query_lower for pronoun in pronouns):
            if self.session_context.get("last_query_data"):
                # Build context about what the pronouns refer to
                last_data = self.session_context["last_query_data"]
                data_description = []

                for key, data in last_data.items():
                    if isinstance(data, dict):
                        data_description.append(f"{key}: {list(data.keys())}")

                context_note = f"\n\nCONTEXT: The user is referring to data from the previous query: {', '.join(data_description)}"
                return query + context_note

        return query

    ### initiate document awareness while opening the chatbot
    def _initialize_document_awareness(self):
        """Fast, single-call document discovery with lock"""

        print(
            f"🔍 Discovery called - Lock: {self._discovery_lock}, Complete: {self._discovery_complete}"
        )

        # CHECK IF ALREADY DONE
        if self._discovery_complete:
            print("📦 Discovery already complete, using cached results")
            return self.session_context.get("document_catalog", {})

        # CHECK IF CURRENTLY RUNNING
        if self._discovery_lock:
            print("⏳ Discovery already in progress, waiting...")
            max_wait = 10
            waited = 0
            while self._discovery_lock and waited < max_wait:
                time.sleep(0.5)
                waited += 0.5
                print(f"  ... waited {waited}s")

            if self._discovery_lock:
                print("❌ Discovery timeout - forcing completion")
                self._discovery_lock = False
                self._discovery_complete = True

            return self.session_context.get("document_catalog", {})

        # SET LOCK
        self._discovery_lock = True
        print("🔒 Discovery lock acquired")

        try:
            print("🔍 Starting SINGLE discovery search...")

            # Use threading instead of signal for timeout
            from concurrent.futures import (
                ThreadPoolExecutor,
            )
            from concurrent.futures import (
                TimeoutError as FutureTimeoutError,
            )

            def do_search():
                print("  → Calling execute_tool with query='project'")
                return self.execute_tool(
                    "search_files", {"query": "project", "max_results": 20}
                )

            # Execute with 15 second timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(do_search)
                try:
                    initial_search = future.result(timeout=15)  # 15 second timeout
                    print(
                        f"  ✓ Search returned: {initial_search.get('success')}, {len(initial_search.get('files', []))} files"
                    )
                except FutureTimeoutError:
                    print("❌ Search timed out after 15 seconds")
                    initial_search = {"success": False}

            if not initial_search.get("success") or not initial_search.get("files"):
                print(
                    "⚠️ Initial search failed or returned no files - using empty catalog"
                )
                discovered_files = {"general": {}}
                total_files_found = 0
            else:
                print(
                    f"  → Categorizing {len(initial_search.get('files', []))} files..."
                )
                discovered_files = self._auto_categorize_files(
                    initial_search.get("files", [])
                )
                total_files_found = sum(
                    len(files) for files in discovered_files.values()
                )
                print(f"  ✓ Categorized into {len(discovered_files)} categories")

            # Cache results
            self.session_context["document_catalog"] = discovered_files
            self.session_context["total_documents_available"] = total_files_found
            self.session_context["discovery_timestamp"] = time.time()

            # Create file type index
            if discovered_files:
                self._create_file_type_index(discovered_files)
                catalog_summary = self._create_document_catalog_summary(
                    discovered_files
                )
                self.session_context["session_summary"] = catalog_summary

            print(f"✅ Discovery complete: {total_files_found} files")

            # MARK AS COMPLETE
            self._discovery_complete = True

            return discovered_files

        except Exception as e:
            print(f"❌ Discovery error: {e}")
            import traceback

            traceback.print_exc()
            # Return empty catalog on error
            self.session_context["document_catalog"] = {}
            self.session_context["total_documents_available"] = 0
            self._discovery_complete = True  # Mark as complete to prevent retry loops
            return {}
        finally:
            print("🔓 Discovery lock released")
            self._discovery_lock = False

    def _quick_fallback_discovery(self):
        """Ultra-fast fallback with minimal searches"""
        print("🔄 Using quick fallback discovery...")

        # Only 3 targeted searches instead of 7+
        quick_terms = ["project", "report", "data"]
        discovered = {}

        for term in quick_terms:
            try:
                result = self.execute_tool(
                    "search_files",
                    {
                        "query": term,
                        "max_results": 10,  # Only 10 per term
                    },
                )
                if result.get("success") and result.get("files"):
                    discovered[term] = {
                        f["id"]: {
                            "name": f["name"],
                            "type": f.get("mimeType", "unknown"),
                            "relevance_score": 5,
                        }
                        for f in result.get("files", [])
                    }
            except Exception as e:
                print(f"⚠️ Fallback search for '{term}' failed: {e}")
                continue

        return discovered

    ### to discover file category, while initializing document awareness
    def _auto_categorize_files(self, files):
        """Ultra-fast categorization"""
        print(f"  📂 Categorizing {len(files)} files...")

        categories = {}

        # Simple keyword matching - NO REGEX for speed
        for file in files:
            name = file.get("name", "").lower()
            file_id = file["id"]

            # Simple keyword checks (FAST)
            if "project" in name or "plan" in name:
                cat = "projects"
            elif "budget" in name or "cost" in name:
                cat = "financial"
            elif "report" in name or "analysis" in name:
                cat = "reports"
            elif "tax" in name or "form" in name:
                cat = "tax"
            elif ".csv" in name or ".xlsx" in name:
                cat = "data"
            else:
                cat = "other"

            if cat not in categories:
                categories[cat] = {}

            categories[cat][file_id] = {
                "name": file["name"],
                "type": file.get("mimeType", "unknown"),
                "relevance_score": 5,
            }

        print(f"  ✓ Created {len(categories)} categories")
        return categories

    ### works with the initiate document awareness()
    def _fallback_discovery(self):
        """Fallback if dynamic discovery fails"""
        basic_terms = ["project", "data", "report"]
        discovered = {}

        for term in basic_terms:
            result = self.execute_tool(
                "search_files", {"query": term, "max_results": 10}
            )
            if result.get("success"):
                discovered[term] = {
                    f["id"]: {
                        "name": f["name"],
                        "type": f.get("type", "unknown"),
                        "relevance_score": 5,
                    }
                    for f in result.get("files", [])
                }

        return discovered

    def _create_file_type_index(self, discovered_files):
        """Create an index by file type for faster lookups"""
        file_type_index = {}

        for category, files in discovered_files.items():
            for file_id, file_info in files.items():
                file_type = file_info.get("type", "unknown")
                if file_type not in file_type_index:
                    file_type_index[file_type] = []
                file_type_index[file_type].append(
                    {"id": file_id, "name": file_info["name"], "category": category}
                )

        self.session_context["file_type_index"] = file_type_index

    def _discover_files_by_category(self, search_terms, max_files_per_term=15):
        """Discover files with parallel execution"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        category_files = {}
        seen_file_ids = set()

        def search_term(term):
            """Single search operation"""
            try:
                result = self.execute_tool(
                    "search_files", {"query": term, "max_results": max_files_per_term}
                )

                if result.get("success") and result.get("files"):
                    return [(file_info, term) for file_info in result["files"]]
            except Exception as e:
                print(f"⚠️ Error searching '{term}': {e}")
            return []

        # Execute searches in parallel (max 5 concurrent)
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_term = {
                executor.submit(search_term, term): term for term in search_terms
            }

            for future in as_completed(future_to_term):
                for file_info, term in future.result():
                    file_id = file_info["id"]
                    if file_id not in seen_file_ids:
                        category_files[file_id] = {
                            "name": file_info["name"],
                            "type": file_info.get("type", "unknown"),
                            "size": file_info.get("size", "unknown"),
                            "found_via_term": term,
                            "relevance_score": self._calculate_relevance_score(
                                file_info["name"], search_terms
                            ),
                        }
                        seen_file_ids.add(file_id)

        # Sort by relevance
        return dict(
            sorted(
                category_files.items(),
                key=lambda x: x[1]["relevance_score"],
                reverse=True,
            )
        )

    def _calculate_relevance_score(self, filename, search_terms):
        """Calculate relevance score based on filename match with search terms"""
        score = 0
        filename_lower = filename.lower()

        for term in search_terms:
            if term in filename_lower:
                score += 10
            # Partial matches
            for word in term.split():
                if word in filename_lower:
                    score += 5

        # Bonus for common file indicators
        indicators = ["report", "analysis", "data", "summary", "plan"]
        for indicator in indicators:
            if indicator in filename_lower:
                score += 3

        return score

    def _create_document_catalog_summary(self, discovered_files):
        """Create a comprehensive summary of discovered documents"""
        summary_parts = ["📚 DOCUMENT CATALOG SUMMARY:\n"]

        for category, files in discovered_files.items():
            if files:
                summary_parts.append(f"**{category.upper()}** ({len(files)} files):")

                # Show top 5 most relevant files per category
                top_files = list(files.items())[:5]
                for file_id, file_info in top_files:
                    summary_parts.append(
                        f"  • {file_info['name']} (score: {file_info['relevance_score']})"
                    )

                if len(files) > 5:
                    summary_parts.append(f"  ... and {len(files) - 5} more files")
                summary_parts.append("")

        total_files = sum(len(files) for files in discovered_files.values())
        summary_parts.append(
            f"📊 Total: {total_files} documents across {len(discovered_files)} categories"
        )

        return "\n".join(summary_parts)

    def _get_relevant_files_for_query(self, user_query):
        """Get files relevant to the user's query from the document catalog"""
        if "document_catalog" not in self.session_context:
            return []

        query_lower = user_query.lower()
        relevant_files = []

        # Score files based on query relevance
        for category, files in self.session_context["document_catalog"].items():
            category_relevance = 0

            # Check if query relates to this category
            category_keywords = {
                "projects": ["project", "initiative", "development"],
                "incidents": ["incident", "issue", "problem", "bug", "error"],
                "reports": ["report", "analysis", "summary"],
                "plans": ["plan", "strategy", "timeline"],
                "budgets": ["budget", "cost", "financial"],
                "operations": ["operations", "operational"],
                "data": ["data", "dataset", "metrics"],
            }

            if category in category_keywords:
                for keyword in category_keywords[category]:
                    if keyword in query_lower:
                        category_relevance += 10

            # If category is relevant, add its files
            if category_relevance > 0:
                for file_id, file_info in files.items():
                    file_query_score = 0
                    filename_lower = file_info["name"].lower()

                    # Score based on filename matching query terms
                    query_words = query_lower.split()
                    for word in query_words:
                        if len(word) > 3:  # Skip short words
                            if word in filename_lower:
                                file_query_score += 15

                    total_score = (
                        file_info["relevance_score"]
                        + file_query_score
                        + category_relevance
                    )

                    if total_score > 10:  # Threshold for relevance
                        relevant_files.append(
                            {
                                "file_id": file_id,
                                "name": file_info["name"],
                                "category": category,
                                "total_score": total_score,
                            }
                        )

        # Sort by total score and return top 10
        relevant_files.sort(key=lambda x: x["total_score"], reverse=True)
        return relevant_files[:10]

    def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        provider_used: str = "claude",
        tool_use_id: str = None,
    ) -> Dict[str, Any]:
        """Execute tool with timeout protection"""

        print(f"🔧 TOOL: {tool_name} with args: {arguments}")

        try:
            # Use ThreadPoolExecutor for timeout instead of signal
            from concurrent.futures import (
                ThreadPoolExecutor,
            )
            from concurrent.futures import (
                TimeoutError as FutureTimeoutError,
            )

            def execute():
                if tool_name == "search_files":
                    return self._search_files(arguments)
                elif tool_name == "read_file":
                    result = self._read_file(arguments)
                    if provider_used.startswith("gpt") and result.get("success"):
                        result = self._enhance_with_gpt_analysis(result, arguments)
                    return result
                elif tool_name == "get_file_metadata":
                    return self._get_file_metadata(arguments)
                elif tool_name == "recall_session_context":
                    return self._recall_session_context(arguments)
                else:
                    return {"error": f"Unknown tool: {tool_name}"}

            # Execute with timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(execute)
                try:
                    result = future.result(timeout=20)  # 20 second timeout
                    print(f"  ✓ Tool completed: {result.get('success', False)}")
                except FutureTimeoutError:
                    print(f"  ⏱️ Tool {tool_name} timed out after 20 seconds")
                    result = {"error": f"Tool execution timed out", "success": False}

        except Exception as e:
            print(f"  ❌ Tool error: {e}")
            result = {"error": f"Tool execution failed: {str(e)}", "success": False}

        # Append Claude-style tool_result if needed
        if tool_use_id:
            self.conversation_history.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": json.dumps(result),
                        }
                    ],
                }
            )

        return result

    def _enhance_with_gpt_analysis(self, file_result, original_args):
        """Add GPT-specific analysis when GPT calls tools"""
        if not self.openai_client:
            return file_result

        try:
            content = file_result.get("content", "")[:2000]  # Limit for context

            analysis_prompt = f"""Analyze this document briefly:

            File: {file_result.get("file_name", "Document")}
            Content: {content}

            Provide:
            1. Key topics (2-3 items)
            2. Important data/numbers
            3. Main insights
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": analysis_prompt}],
                max_tokens=300,
                temperature=0.1,
            )

            file_result["gpt_analysis"] = response.choices[0].message.content
            return file_result

        except Exception as e:
            print(f"GPT analysis failed: {e}")
            return file_result

    def _batch_search_files(self, queries, max_results=20):
        """Execute multiple searches in parallel"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def single_search(query):
            try:
                return self.execute_tool(
                    "search_files", {"query": query, "max_results": max_results}
                )
            except:
                return {"success": False, "files": []}

        all_files = []
        seen_ids = set()

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(single_search, q) for q in queries]

            for future in as_completed(futures):
                result = future.result()
                if result.get("success"):
                    for f in result.get("files", []):
                        if f["id"] not in seen_ids:
                            all_files.append(f)
                            seen_ids.add(f["id"])

        return all_files

    def _search_files(self, args):
        """Enhanced search with timeout"""
        query = args.get("query", "")
        max_results = args.get("max_results", 20)

        print(f"  🔍 Searching for: '{query}' (max {max_results})")

        try:
            response = requests.post(
                f"{self.http_server_url}/call_tool",
                json={
                    "name": "search_gdrive_files",
                    "arguments": {"query": query, "max_results": max_results},
                },
                timeout=15,  # 15 SECOND TIMEOUT
            )

            files_found = []
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    files_found = result.get("data", [])
                    print(f"  ✓ Found {len(files_found)} files")
                else:
                    print(
                        f"  ⚠️ Search unsuccessful: {result.get('error', 'unknown error')}"
                    )
            else:
                print(f"  ⚠️ HTTP {response.status_code}")

            return {
                "success": len(files_found) > 0,
                "files": [
                    {
                        "id": f["id"],
                        "name": f["name"],
                        "type": f.get("mimeType", "unknown"),
                        "size": f.get("size", "unknown"),
                        "modified": f.get("modifiedTime", "unknown"),
                    }
                    for f in files_found
                ],
                "total_found": len(files_found),
                "search_strategy_used": "single_search",
            }

        except requests.Timeout:
            print("  ⏱️ Search request timed out after 15s")
            return {"success": False, "files": [], "total_found": 0, "error": "timeout"}
        except Exception as e:
            print(f"  ❌ Search error: {e}")
            return {"success": False, "files": [], "total_found": 0, "error": str(e)}

    def _read_file(self, args):
        """Read file with optional focus area"""
        file_id = args.get("file_id")
        focus_area = args.get("focus_area")

        response = requests.post(
            f"{self.http_server_url}/call_tool",
            json={"name": "read_gdrive_file", "arguments": {"file_id": file_id}},
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                file_data = result.get("data", {})
                content_info = file_data.get("content", {})

                # Extract content intelligently
                extracted = self._extract_focused_content(content_info, focus_area)

                return {
                    "success": True,
                    "file_name": file_data.get("file_name", "unknown"),
                    "file_type": content_info.get("type", "unknown"),
                    "content": extracted["content"],
                    "metadata": extracted["metadata"],
                    "focus_applied": focus_area is not None,
                }

        return {"success": False, "error": "Failed to read file"}

    def _extract_focused_content(self, content_info, focus_area):
        """Extract content with optional focus area"""
        file_type = content_info.get("type", "unknown")

        content = ""
        full_content = ""

        if file_type in ["csv", "excel"]:
            data = content_info.get("data", [])
            columns = content_info.get("columns", [])

            if focus_area:
                # Filter data based on focus area
                focused_data = []
                focus_lower = focus_area.lower()

                for record in data:
                    # Check if any field relates to focus area
                    record_text = " ".join(str(v) for v in record.values() if v).lower()
                    if focus_lower in record_text:
                        focused_data.append(record)

                # If focus yields results, use it; otherwise use all data
                relevant_data = focused_data if focused_data else data
            else:
                relevant_data = data

            for i, record in enumerate(relevant_data[:100], 1):  # Increased limit
                record_parts = []
                for k, v in record.items():
                    if v is not None and str(v).strip():
                        record_parts.append(f"{k}: {v}")
                content += f"Record {i}: {', '.join(record_parts)}\n"

            if len(relevant_data) > 100:  # Changed from 50
                content += f"\n⚠️  [{len(relevant_data) - 100} additional records not shown - this is PARTIAL DATA]\n"
                content += f"Total records in file: {len(relevant_data)}\n"
                content += f"Records provided above: 100\n"

            # Diagnostic logging for numerical accuracy
            print(f"📊 DATA EXTRACTION SUMMARY:")
            print(f"   Total records in file: {len(data)}")
            print(f"   Records sent to LLM: {min(len(relevant_data), 100)}")
            print(f"   ⚠️  Data truncated: {len(relevant_data) > 100}")
            if len(relevant_data) > 100:
                print(
                    f"   ⚠️  WARNING: LLM only sees {100}/{len(relevant_data)} records!"
                )

            # Show sample numbers from first few records
            sample_numbers = []
            for record in relevant_data[:5]:
                for v in record.values():
                    if isinstance(v, (int, float)):
                        sample_numbers.append(str(v))
            if sample_numbers:
                print(f"   Sample numbers in data: {', '.join(sample_numbers[:10])}")

            return {
                "content": content,
                "metadata": {
                    "total_records": len(data),
                    "focused_records": len(relevant_data),
                    "records_shown": min(len(relevant_data), 100),  # NEW
                    "data_truncated": len(relevant_data) > 100,  # NEW
                    "columns": columns,
                    "focus_applied": focus_area is not None,
                },
            }

        elif file_type == "pdf":
            full_content = content_info.get("content", "")

            if focus_area:
                # Extract sections related to focus area
                lines = full_content.split("\n")
                focused_lines = []
                context_lines = 2  # Lines before/after match

                for i, line in enumerate(lines):
                    if focus_area.lower() in line.lower():
                        start = max(0, i - context_lines)
                        end = min(len(lines), i + context_lines + 1)
                        focused_lines.extend(lines[start:end])
                        focused_lines.append("---")

                if focused_lines:
                    content = f"Content focused on '{focus_area}':\n\n" + "\n".join(
                        focused_lines
                    )
                else:
                    content = full_content[:5000]  # Fallback to beginning
            else:
                content = full_content

            return {
                "content": content,
                "metadata": {
                    "total_length": len(full_content),
                    "pages": content_info.get("num_pages", 0),
                    "focus_applied": focus_area is not None,
                },
            }

        else:
            # Handle other file types
            full_content = str(content_info.get("content", ""))
            return {
                "content": full_content[:5000],  # Reasonable limit
                "metadata": {"total_length": len(full_content), "focus_applied": False},
            }

    def _get_file_metadata(self, args):
        """Get file metadata without reading content"""
        file_pattern = args.get("file_pattern", "")

        response = requests.post(
            f"{self.http_server_url}/call_tool",
            json={
                "name": "search_gdrive_files",
                "arguments": {"query": file_pattern, "max_results": 100},
            },
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                files = result.get("data", [])

                metadata = {
                    "total_files": len(files),
                    "file_types": {},
                    "files_by_type": {},
                    "files": [],
                }

                for file in files:
                    mime_type = file.get("mimeType", "unknown")
                    file_type = self._mime_to_readable_type(mime_type)

                    metadata["file_types"][file_type] = (
                        metadata["file_types"].get(file_type, 0) + 1
                    )

                    if file_type not in metadata["files_by_type"]:
                        metadata["files_by_type"][file_type] = []

                    metadata["files_by_type"][file_type].append(
                        {
                            "id": file["id"],
                            "name": file["name"],
                            "size": file.get("size", "unknown"),
                        }
                    )

                    metadata["files"].append(
                        {"id": file["id"], "name": file["name"], "type": file_type}
                    )

                return {"success": True, "metadata": metadata}

        return {"success": False, "error": "Failed to get metadata"}

    def _mime_to_readable_type(self, mime_type):
        """Convert MIME type to readable format"""
        mapping = {
            "text/csv": "CSV",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "Excel",
            "application/pdf": "PDF",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "Word",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": "PowerPoint",
            "text/plain": "Text",
            "application/json": "JSON",
            "text/xml": "XML",
        }
        return mapping.get(mime_type, "Other")

    def _recall_session_context(self, args):
        """Recall session context for conversational continuity"""
        context_type = args.get("context_type", "full_summary")

        if context_type == "files_mentioned":
            return {
                "success": True,
                "files_mentioned": self.session_context["files_mentioned"],
                "total_files": len(self.session_context["files_mentioned"]),
            }
        elif context_type == "topics_discussed":
            return {
                "success": True,
                "topics": self.session_context["topics_discussed"],
                "total_topics": len(self.session_context["topics_discussed"]),
            }
        elif context_type == "full_summary":
            return {
                "success": True,
                "session_summary": self.session_context["session_summary"],
                "files_count": len(self.session_context["files_mentioned"]),
                "topics_count": len(self.session_context["topics_discussed"]),
                "files_mentioned": list(self.session_context["files_mentioned"].keys()),
                "topics_discussed": self.session_context["topics_discussed"],
            }

        return {"success": False, "error": "Invalid context type"}

    def _aggressive_file_search(self, query_intent, max_files=5):
        """Aggressively search and read multiple files for comprehensive information"""

        search_terms = []

        if "project" in query_intent.lower() and "member" in query_intent.lower():
            search_terms = ["project", "team", "member", "personnel", "staff"]
        elif "employee" in query_intent.lower():
            search_terms = ["employee", "staff", "personnel", "team", "people"]
        elif "budget" in query_intent.lower():
            search_terms = ["budget", "cost", "financial", "expense", "money"]

        all_file_data = []

        for term in search_terms:
            search_result = self.execute_tool(
                "search_files", {"query": term, "max_results": 10}
            )

            if search_result.get("success") and search_result.get("files"):
                for file_info in search_result["files"][
                    :3
                ]:  # Read top 3 files per search term
                    file_data = self.execute_tool(
                        "read_file", {"file_id": file_info["id"]}
                    )
                    if file_data.get("success"):
                        all_file_data.append(
                            {
                                "file_name": file_info["name"],
                                "content": file_data.get("content", ""),
                                "search_term": term,
                            }
                        )

                    if len(all_file_data) >= max_files:
                        break

            if len(all_file_data) >= max_files:
                break

        return all_file_data

    def build_enhanced_system_message(self, user_query, context_info, relevant_files):
        """Build comprehensive system message with formatting rules"""

        # Build discovery status message
        total_docs = self.session_context.get("total_documents_available", 0)
        if total_docs > 0:
            discovery_status = f"\n📚 DOCUMENT CATALOG: {total_docs} documents discovered and available for search."
        else:
            discovery_status = "\n⚠️ No documents cataloged yet - use search_files to find relevant documents."

        # Build relevant files info - handle both list and dict formats
        relevant_files_info = ""
        if relevant_files:
            relevant_files_info = "\n\nRELEVANT FILES FOR THIS QUERY:\n"

            # Handle if relevant_files is a list
            if isinstance(relevant_files, list):
                for file_info in relevant_files[:10]:  # Top 10
                    if isinstance(file_info, dict):
                        file_name = file_info.get("name", "Unknown")
                        file_id = file_info.get("id", "N/A")
                        relevance = file_info.get("relevance_score", 0)
                        relevant_files_info += (
                            f"- {file_name} (ID: {file_id}, Score: {relevance})\n"
                        )
                    else:
                        relevant_files_info += f"- {file_info}\n"

            # Handle if relevant_files is a dict
            elif isinstance(relevant_files, dict):
                for file_id, file_info in list(relevant_files.items())[:10]:  # Top 10
                    if isinstance(file_info, dict):
                        file_name = file_info.get("name", "Unknown")
                        relevance = file_info.get("relevance_score", 0)
                        relevant_files_info += (
                            f"- {file_name} (ID: {file_id}, Score: {relevance})\n"
                        )
                    else:
                        relevant_files_info += f"- {file_info} (ID: {file_id})\n"

            relevant_files_info += (
                "\n👆 Start by reading these files using read_file tool."
            )

        system_message = f"""You are an aggressive, thorough document analyst with access to Google Drive tools.

        🔢 CRITICAL NUMERICAL ACCURACY RULES (READ FIRST):
        ===================================================
        ⚠️  **NEVER GUESS OR APPROXIMATE NUMERICAL VALUES**
        ⚠️  **ONLY report numbers that appear EXACTLY in the source files**
        ⚠️  If a file says "150 employees", report EXACTLY 150 (not "around 150" or "approximately 150")
        ⚠️  If doing math calculations:
            - Show your work: "Project A: 50 + Project B: 75 = Total: 125"
            - Double-check your arithmetic
            - State when you're calculating vs. reading
        ⚠️  **CRITICAL**: You may only see PARTIAL data (e.g., 50 out of 500 records)
            - ALWAYS mention this limitation
            - Say: "Based on the 50 records provided (out of 500 total)..."
            - Do NOT claim totals unless you've seen all data
        ⚠️  Preserve decimal places exactly as shown in source
        ⚠️  When counting items, count ONLY what you actually read

        EXAMPLES OF CORRECT REPORTING:
        -------------------------------
        ✅ "The file shows 23 active projects"
        ✅ "Based on the 50 records shown (out of 200 total), 15 are marked as completed"
        ✅ "Calculating total: 45 + 67 + 23 = 135"
        ✅ "The exact value in the file is $1,234.56"

        ❌ WRONG:
        -------------------------------
        ❌ "There are approximately 20 projects" (when file says 23)
        ❌ "Around 150 employees" (when file says exactly 147)
        ❌ "Based on the sample, there are probably 400 total" (extrapolating without basis)
        ❌ "About $1,200" (when file says $1,234.56)

        MANDATORY VERIFICATION BEFORE STATING NUMBERS:
        ----------------------------------------------
        Before reporting ANY number, ask yourself:
        1. "Did I see this EXACT number in a file?" → If yes, report it
        2. "Am I calculating this?" → If yes, show calculation
        3. "Am I guessing/approximating?" → If yes, DON'T report it, read more files instead
        4. "Am I looking at partial data?" → If yes, state the limitation clearly

            MANDATORY BEHAVIOR:
            - Always read multiple files when information isn't found in the first file
            - When a file appears empty or lacks expected data, immediately try 2-3 more files
            - Use different search terms if initial searches don't yield results
            - Extract ANY relevant information from files, even if incomplete
            - Never conclude "no information found" without reading at least 3 different files
            - Be persistent and thorough - the user expects detailed answers

            RESPONSE FORMATTING REQUIREMENTS (CRITICAL):

            🎯 DATA SELECTION PRINCIPLE:

            When creating tables, the entities in the first column should match
            the subject of the user's question.

            Before presenting any table, verify:
            "The user asked about ___. Does my table show ___?"

            If yes → present it
            If no → find the correct data section and rebuild the table

            This applies to any query and any entity type - always match the table
            to what was actually asked.

            ✅ Always format responses in clean, readable Markdown
            ✅ Use proper Markdown tables with alignment:
               | Column 1 | Column 2 | Column 3 |
               |----------|----------|----------|
               | Data 1   | Data 2   | Data 3   |

            📋 CRITICAL: TABLE CREATION RULES
            ==================================

            🚨 ABSOLUTE REQUIREMENTS - NO EXCEPTIONS:
            =========================================

            1. **COLUMN COUNT CONSISTENCY:**
               - If you create 5 headers, EVERY row MUST have exactly 5 cells
               - If you don't have data for a column, use "N/A", "--", or "Not Available"
               - NEVER create a row with fewer cells than headers
               - NEVER create a row with more cells than headers

            2. **ONLY CREATE COLUMNS YOU HAVE DATA FOR:**
               - If you only have: Name, Budget, Cost, Variance, Progress
               - Then create ONLY 5 columns: | Name | Budget | Cost | Variance | Progress |
               - DO NOT add extra columns like "Manager" or "Region" if you don't have that data!

            3. **VERIFICATION BEFORE SUBMITTING:**
               - Count the pipes (|) in your header row
               - Count the pipes (|) in each data row
               - They MUST be identical!
               - Example:
                 Header: | A | B | C |  → 4 pipes
                 Row 1:  | 1 | 2 | 3 |  → 4 pipes ✅
                 Row 2:  | X | Y |     → 3 pipes ❌ WRONG!

            4. **NO PLACEHOLDER ROWS:**
               - DO NOT create rows like: | -- | -- | -- |
               - If you don't have data for an item, don't create a row for it!
               - Only create rows for items you actually have data about

            ✅ CORRECT EXAMPLE:
            | Project Name | Budget | Actual Cost | Variance |
            |--------------|--------|-------------|----------|
            | Rhinestone   | $3.6M  | $8.4M       | -$148K   |
            | Blue Bird    | $4.3M  | $9.1M       | -$85K    |

            ❌ WRONG EXAMPLE:
            | Project Name | Budget | Cost | Variance | Manager | Dept | Region |
            |--------------|--------|------|----------|---------|------|--------|
            | --           | --     |      |          |         |      |        |
            | Rhinestone   | $3.6M  | $8.4M| -$148K   | 85%     |      |        |
                                                      ↑ WRONG - Inconsistent columns!

            ⚠️ If you're unsure about a column, DON'T CREATE IT!
            ⚠️ Only create columns for data you ACTUALLY HAVE!

            🚨 CRITICAL TABLE RULES (MUST FOLLOW):
            ======================================
            1. COUNT your header columns (example: 3 columns above)
            2. EVERY data row must have EXACTLY THE SAME number of columns
            3. NEVER skip cells - use "N/A", "--", or "Not specified" for empty values
            4. Each row must have the same number of "|" separators

            ✅ CORRECT:
            | Name | Age | City |
            |------|-----|------|
            | John | 30  | NYC  |
            | Jane | 25  | LA   |
            | Bob  | N/A | SF   |  ← Empty age shown as "N/A"

            ❌ WRONG:
            | Name | Age | City |
            |------|-----|------|
            | John | 30  |       ← WRONG! Missing City column
            | Jane |             ← WRONG! Missing Age and City

            ⚠️ Before submitting your response, visually count the "|" symbols in each row!
            ⚠️ Header: 4 pipes | → Data rows: MUST also have 4 pipes |

            If you're creating a table with 8 columns, EVERY row must show all 8 values!

            ✅ Use headers for organization:
               ## Main Section
               ### Subsection

            ✅ Use bullet points for lists:
               - Item 1
               - Item 2

            ✅ Use bold for emphasis: **Important Text**

            ✅ Keep responses clean and scannable
            ✅ NO ASCII art, NO box-drawing characters (│ ┤ ├ ─ ┼)
            ✅ NO raw pipe characters without proper table formatting
            ✅ Always complete tables - never truncate mid-row

            ❌ NEVER use plain text tables with pipes like this:
               | Project | Cost
               | Alpha | $100

            ❌ NEVER use box-drawing characters: ├──┤ │ ─

            RESPONSE COMPLETION REQUIREMENTS:
            - ALWAYS provide complete responses - never truncate tables, lists, or analysis
            - If creating tables, include ALL data found, not just samples
            - Finish all sentences and close all markdown formatting properly
            - When showing project/employee data, include complete listings
            - End responses with proper conclusions, not mid-sentence cutoffs

            {discovery_status}

            Available tools:
            {json.dumps(self.available_tools, indent=2)}

            AGGRESSIVE SEARCH PROTOCOL:
            1. READ the highest-scoring relevant files listed below
            2. If no useful data found, IMMEDIATELY search for alternative files
            3. Try different file types (.csv, .pdf, .xlsx, .docx)
            4. Use varied search terms
            5. Extract partial information and combine from multiple sources
            6. Never stop after 1-2 files - always try at least 3-4 files

            {context_info}
            {relevant_files_info}

            EXTRACTION RULES:
            - Extract names, roles, numbers, dates from ANY file that contains them
            - **Report numbers EXACTLY as they appear** - no rounding, no approximating
            - If a file has partial data, note it and search for complementary files
            - Combine information from multiple sources to give complete answers
            - When files seem empty, try reading them anyway
            - **When you only see partial data, explicitly state this limitation**

            Current query: {user_query}

            Be thorough, persistent, and NUMERICALLY ACCURATE. Provide complete, well-formatted Markdown responses with EXACT numbers from files."""

        return system_message

    def process_query_iteratively(self, user_query):
        """Process query using iterative tool calling with session memory"""

        print(f"\n{'=' * 60}")
        print(f"🎯 NEW QUERY: {user_query}")
        print(
            f"Discovery status - Complete: {self._discovery_complete}, Lock: {self._discovery_lock}"
        )
        print(f"{'=' * 60}\n")

        # Add user query to conversation history
        self.conversation_history.append({"role": "user", "content": user_query})

        enhanced_query = self._resolve_contextual_references(user_query)

        # ONLY initialize if not already done
        if not self._discovery_complete:
            print("⚠️ Discovery not complete, initializing...")
            self._initialize_document_awareness()
        else:
            print("✓ Discovery already complete, skipping initialization")

        # Get files relevant to this specific query
        relevant_files = self._get_relevant_files_for_query(user_query)

        # Build context-aware system message
        context_info = ""
        if self.session_context["files_mentioned"]:
            context_info += f"\nPREVIOUS FILES ACCESSED THIS SESSION: {list(self.session_context['files_mentioned'].values())}"

        if self.session_context["topics_discussed"]:
            context_info += f"\nTOPICS DISCUSSED THIS SESSION: {', '.join(self.session_context['topics_discussed'])}"

        if self.session_context["session_summary"]:
            context_info += (
                f"\nSESSION SUMMARY: {self.session_context['session_summary']}"
            )

        if relevant_files:
            context_info += f"\n\nFILES MOST RELEVANT TO CURRENT QUERY:"
        for file in relevant_files[:5]:  # Show top 5 relevant files
            context_info += f"\n• {file['name']} (ID: {file['file_id']}) - Score: {file['total_score']}"

        # Enhanced system message with memory
        system_message = self.build_enhanced_system_message(
            user_query, context_info, relevant_files
        )

        max_iterations = 3  # 10
        iteration = 0
        files_accessed_this_query = []
        successful_files = 0  # ✅ initialize to avoid UnboundLocalError

        while iteration < max_iterations:
            iteration += 1
            print(f"\n--- ITERATION {iteration} ---")

            try:
                now = time.time()
                elapsed = now - self.last_api_call
                if elapsed < self.min_delay:
                    sleep_time = self.min_delay - elapsed
                    print(f"Rate limiting: waiting {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)

                response = self.api_client.call_with_multi_fallback(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4000,
                    temperature=0.1,
                    system=system_message,
                    tools=self.available_tools,
                    messages=self.conversation_history,
                )

                self.last_api_call = time.time()

                # Add API response to conversation (handling both Claude and GPT responses)
                assistant_message = {"role": "assistant", "content": []}

                for content_block in response.content:
                    if hasattr(content_block, "type") and content_block.type == "text":
                        assistant_message["content"].append(
                            {"type": "text", "text": content_block.text}
                        )
                    elif (
                        hasattr(content_block, "type")
                        and content_block.type == "tool_use"
                    ):
                        assistant_message["content"].append(
                            {
                                "type": "tool_use",
                                "id": content_block.id,
                                "name": content_block.name,
                                "input": content_block.input,
                            }
                        )

                self.conversation_history.append(assistant_message)

                # Check if we need to use tools (Claude tool_use or GPT needs tools)
                if (
                    hasattr(response, "stop_reason")
                    and response.stop_reason == "tool_use"
                ) or self._response_needs_tools(response):
                    # Execute tool calls with robust error handling
                    tool_results = []

                    for content_block in response.content:
                        if (
                            hasattr(content_block, "type")
                            and content_block.type == "tool_use"
                        ):
                            tool_name = content_block.name
                            tool_args = content_block.input
                            tool_use_id = content_block.id

                            # Execute tool with error handling
                            try:
                                result = self.execute_tool(
                                    tool_name,
                                    tool_args,
                                )

                                # Track files accessed for session context
                                if tool_name == "read_file" and "file_id" in tool_args:
                                    files_accessed_this_query.append(
                                        {
                                            "id": tool_args["file_id"],
                                            "context": f"Read via {tool_name}",
                                            "status": "success",
                                        }
                                    )
                                elif (
                                    tool_name == "search_files"
                                    and result.get("success")
                                    and result.get("files")
                                ):
                                    for file_info in result["files"][
                                        :3
                                    ]:  # Track first 3 found files
                                        files_accessed_this_query.append(
                                            {
                                                "id": file_info["id"],
                                                "name": file_info["name"],
                                                "context": f"Found via search: {tool_args.get('query', '')}",
                                                "status": "success",
                                            }
                                        )

                                tool_results.append(
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": tool_use_id,
                                        "content": json.dumps(result, indent=2),
                                    }
                                )

                            except Exception as tool_error:
                                print(f"Tool execution error: {tool_error}")
                                # Provide graceful fallback for tool errors
                                tool_results.append(
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": tool_use_id,
                                        "content": json.dumps(
                                            {
                                                "error": f"Tool execution failed: {str(tool_error)}",
                                                "fallback_message": "Unable to access this resource at the moment. Please try again later.",
                                            },
                                            indent=2,
                                        ),
                                    }
                                )

                    # Add tool results as a single user message
                    if tool_results:
                        self.conversation_history.append(
                            {"role": "user", "content": tool_results}
                        )

                    # Continue the conversation
                    continue

                else:
                    # Analysis complete - update session context and return final response
                    query_topics = self.extract_topics_from_query(user_query)
                    self.update_session_context(
                        user_query, files_accessed_this_query, query_topics
                    )

                    final_response = ""
                    for content_block in response.content:
                        if (
                            hasattr(content_block, "type")
                            and content_block.type == "text"
                        ):
                            final_response += content_block.text
                        elif hasattr(
                            content_block, "text"
                        ):  # Fallback for GPT responses
                            final_response += content_block.text

                    if final_response and final_response.strip():
                        self._partial_response = final_response

                    if (
                        iteration >= max_iterations or len(final_response) > 500
                    ):  # Substantial response
                        self._extract_and_store_response_data(
                            user_query, final_response
                        )

                    # ADD FOLLOW-UP LOGIC HERE:
                    # Detect if this is a list/aggregation query
                    is_list_query = any(
                        word in user_query.lower()
                        for word in ["list", "all", "show me", "what are", "how many"]
                    )

                    # For list queries, be more aggressive about gathering data
                    if is_list_query and iteration < 8:
                        successful_files = len(
                            [
                                f
                                for f in files_accessed_this_query
                                if f.get("status") == "success"
                            ]
                        )

                        # Need at least 5-6 files for comprehensive lists
                        if successful_files < 6:
                            follow_up = {
                                "role": "user",
                                "content": f"Continue searching and reading more files. You've accessed {successful_files} files so far - please access at least 6 total to ensure comprehensive coverage. Search with different terms if needed.",
                            }
                            self.conversation_history.append(follow_up)
                            print(
                                f"Requesting more file access for list query ({successful_files}/6 files)..."
                            )
                            continue

                    # Check if response is too brief (for first few iterations)
                    if iteration <= 3 and len(final_response.strip()) < 100:
                        successful_files = len(
                            [
                                f
                                for f in files_accessed_this_query
                                if f.get("status") == "success"
                            ]
                        )
                        if successful_files < 3:
                            follow_up = {
                                "role": "user",
                                "content": "The previous response was too brief. Please search more files and provide more detailed information.",
                            }
                            self.conversation_history.append(follow_up)
                            print(
                                f"Adding follow-up push for more detailed analysis..."
                            )
                            continue

                    # Check for "no information found" type responses
                    if any(
                        phrase in final_response.lower()
                        for phrase in [
                            "no information",
                            "no relevant",
                            "no data",
                            "not found",
                        ]
                    ):
                        if iteration <= 3:
                            follow_up = {
                                "role": "user",
                                "content": "You said no information was found, but please try harder. Search with different terms and read more files. There should be data available in the document catalog.",
                            }
                            self.conversation_history.append(follow_up)
                            print(f"Pushing back on 'no information found' response...")
                            continue

                    # Original metadata and return code continues here...
                    provider_status = (
                        self._get_provider_status()
                        if hasattr(self, "api_client")
                        else "Standard API"
                    )
                    successful_files = len(
                        [
                            f
                            for f in files_accessed_this_query
                            if f.get("status") == "success"
                        ]
                    )

                    metadata = f"\n\n---\nIterative Analysis Complete: {iteration} iterations using {provider_status}."
                    if files_accessed_this_query:
                        metadata += f" Files accessed: {successful_files}/{len(files_accessed_this_query)} successful."
                    metadata += f" Session total: {len(self.session_context['files_mentioned'])} files, {len(self.session_context['topics_discussed'])} topics discussed."

                    safe_response = (
                        final_response + metadata
                        if "final_response" in locals()
                        else "No response generated."
                    )
                    if not isinstance(safe_response, str):
                        safe_response = str(safe_response)

                    # NEW: Parse and store table structure before returning
                    if "|" in safe_response:
                        try:
                            self.last_table_data = self.parse_table_structure(
                                safe_response
                            )
                            if self.last_table_data:
                                print(
                                    f"✅ Stored table with {len(self.last_table_data['headers'])} columns for visualization"
                                )
                        except Exception as e:
                            print(f"⚠️ Failed to parse table: {e}")
                            self.last_table_data = None
                    else:
                        self.last_table_data = None

                    return safe_response

            except Exception as e:
                error_msg = str(e)
                print(f"Error in iteration {iteration}: {error_msg}")

                # Handle specific error types gracefully
                if "429" in error_msg:
                    return f"I'm currently experiencing high demand. Your analysis was partially completed ({iteration} iterations). Please try again in a few moments for a complete analysis."
                elif "529" in error_msg or "502" in error_msg or "503" in error_msg:
                    return f"I'm experiencing temporary service issues. Your analysis was partially completed ({iteration} iterations). Please try again shortly."
                else:
                    # Provide partial results if we got some analysis done
                    partial_response = f"Analysis encountered an error after {iteration} iterations: {error_msg}"
                    if files_accessed_this_query:
                        successful_files = len(
                            [
                                f
                                for f in files_accessed_this_query
                                if f.get("status") == "success"
                            ]
                        )
                        partial_response += f" However, I was able to access {successful_files} files before the error occurred."
                    return partial_response

        # Reached max iterations
        successful_files = len(
            [f for f in files_accessed_this_query if f.get("status") == "success"]
        )

        summary_prompt = {
            "role": "user",
            "content": f"You've now processed {successful_files} files across {iteration} iterations. Based on ALL the information gathered in this conversation, provide a complete answer to the original question: '{user_query}'\n\nCreate a comprehensive response with tables, lists, and all relevant data you've found. Do not say you need more files - synthesize what you have.",
        }
        self.conversation_history.append(summary_prompt)
        try:
            final_response = self.api_client.call_with_multi_fallback(
                model="claude-opus-4-20250514",
                max_tokens=4000,
                temperature=0.1,
                system=system_message,
                messages=self.conversation_history,
            )

            final_text = ""
            for content_block in final_response.content:
                if hasattr(content_block, "type") and content_block.type == "text":
                    final_text += content_block.text

            if final_text and len(final_text.strip()) > 50:
                # Validate numerical accuracy (PATCH 4)
                import re

                numbers_in_response = re.findall(r"\b\d+\.?\d*\b", final_text)
                print(f"🔍 FINAL RESPONSE VALIDATION:")
                print(f"   Numbers found in response: {numbers_in_response[:20]}")
                print(f"   Response length: {len(final_text)} characters")
                print(f"   Files accessed: {successful_files}")

                # NEW: Parse and store table before returning
                full_response = (
                    final_text
                    + f"\n\n[Analysis based on {successful_files} files across {iteration} iterations]"
                )

                if "|" in final_text:
                    try:
                        self.last_table_data = self.parse_table_structure(final_text)
                        if self.last_table_data:
                            print(
                                f"✅ Stored table with {len(self.last_table_data['headers'])} columns"
                            )
                    except Exception as e:
                        print(f"⚠️ Failed to parse table: {e}")
                        self.last_table_data = None
                else:
                    self.last_table_data = None

                return full_response

        except Exception as e:
            print(f"Final synthesis call failed: {e}")

        # Fallback if synthesis fails
        if (
            hasattr(self, "_partial_response")
            and self._partial_response
            and len(self._partial_response.strip()) > 20
        ):
            # NEW: Parse table from partial response
            if "|" in self._partial_response:
                try:
                    self.last_table_data = self.parse_table_structure(
                        self._partial_response
                    )
                except Exception as e:
                    print(f"⚠️ Failed to parse partial table: {e}")
                    self.last_table_data = None
            else:
                self.last_table_data = None

            return self._partial_response
        else:
            return f"Analysis completed after {max_iterations} iterations with {successful_files} files processed. Unable to generate final summary."

    def _response_needs_tools(self, response):
        """Check if GPT response indicates it needs tools (for GPT fallback compatibility)"""
        if not hasattr(response, "content"):
            return False

        for content_block in response.content:
            if hasattr(content_block, "text"):
                text = content_block.text.lower()
                # Look for indicators that GPT is trying to use tools
                tool_indicators = [
                    "search for",
                    "need to find",
                    "let me look for",
                    "i should check",
                ]
                if any(indicator in text for indicator in tool_indicators):
                    return True
        return False

    def _get_provider_status(self):
        """Get current API provider status for metadata"""
        if hasattr(self, "api_client") and hasattr(self.api_client, "provider_status"):
            active_providers = []
            for provider, status in self.api_client.provider_status.items():
                if status["available"]:
                    active_providers.append(provider.value)
            return f"Multi-API ({', '.join(active_providers)} available)"
        return "Standard API"


def get_cached_claude_response(claude_client, system_message, conversation_history):
    """Get cached Claude response or make new API call"""
    # Create hash of the conversation for caching
    cache_key = hashlib.md5(
        f"{system_message}{json.dumps(conversation_history)}".encode()
    ).hexdigest()

    if cache_key in claude_response_cache:
        print("Using cached Claude response")
        return claude_response_cache[cache_key]

    # Make API call with retry logic
    response = call_claude_with_retry(
        claude_client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=4000,
        temperature=0.1,
        system=system_message,
        messages=conversation_history,
    )

    # Cache the response
    claude_response_cache[cache_key] = response

    return response
