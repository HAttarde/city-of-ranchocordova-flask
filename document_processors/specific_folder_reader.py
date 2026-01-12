import re
import json
import time
import requests
from typing import List, Dict, Any
import anthropic
import hashlib
import time
import random
import openai
import os
from enum import Enum
import openai
from document_processors.claude_processor import ClaudeLikeDocumentProcessor


# This is the project documents accessor
class ProjectDocumentProcessor(ClaudeLikeDocumentProcessor):
    def __init__(self, http_server_url, claude_client):
        super().__init__(http_server_url, claude_client)
        self.project_folder_id = "1_RuIezT1KN8miQ_3_167rkCXP9ZoZq35"
        self._discovery_cache = {
            "initialized": False,
            "total_files": 0,
            "files": [],
            "last_refresh": None,
            "by_name": {},
            "subfolders_count": 0,
        }

    def _search_files(self, args):
        """Search with RECURSIVE folder scoping (includes subfolders)"""
        query = args.get("query", "")
        max_results = args.get("max_results", 20)
        apply_filter = args.get("apply_category_filter", False)  # Default: no filter

        print(f"  🔍 Searching: '{query}' (filter: {'ON' if apply_filter else 'OFF'})")

        try:
            # Use discovery cache for consistency
            cache = self.ensure_discovery()

            if not cache:
                print("  ❌ Discovery failed")
                return {"success": False, "files": [], "total_found": 0}

            all_files = cache["files"]
            print(
                f"  📂 Working with {len(all_files)} cached files from {cache['subfolders_count'] + 1} folders"
            )

            # Filter by query if provided
            if query and query.strip():
                query_lower = query.lower()
                filtered_files = [
                    f for f in all_files if query_lower in f.get("name", "").lower()
                ]
                print(
                    f"  🔍 Query filter: {len(all_files)} → {len(filtered_files)} files"
                )
            else:
                filtered_files = all_files

            # Apply category filter only if explicitly requested
            if apply_filter:
                category_files = self._apply_category_filter(filtered_files)
                print(
                    f"  🏷️ Category filter: {len(filtered_files)} → {len(category_files)} files"
                )
            else:
                category_files = filtered_files
                print(f"  ✓ No category filter applied")

            print(f"  ✅ Final result: {len(category_files)} files")

            return {
                "success": True,
                "files": [
                    {
                        "id": f["id"],
                        "name": f["name"],
                        "type": f.get("mimeType", "unknown"),
                        "size": f.get("size", "unknown"),
                        "modified": f.get("modifiedTime", "unknown"),
                    }
                    for f in category_files[:max_results]
                ],
                "total_found": len(category_files),
                "search_method": "cached_recursive_discovery",
            }

        except Exception as e:
            print(f"  ❌ Search error: {e}")
            import traceback

            traceback.print_exc()
            return {"success": False, "files": [], "total_found": 0}

    def _get_all_subfolders(self, parent_folder_id, max_retries=2):
        """
        Recursively get all subfolder IDs under a parent folder.

        Args:
            parent_folder_id: The ID of the parent folder
            max_retries: Number of times to retry on timeout

        Returns:
            list: List of all subfolder IDs (recursive)
        """
        subfolders = []

        for attempt in range(max_retries):
            try:
                print(
                    f"     → Attempt {attempt + 1}/{max_retries} to discover subfolders in {parent_folder_id}"
                )

                response = requests.post(
                    f"{self.http_server_url}/call_tool",
                    json={
                        "name": "list_folder_contents",
                        "arguments": {
                            "folder_id": parent_folder_id,
                            "max_results": 1000,
                        },
                    },
                    timeout=30,  # Increased timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        items = result.get("data", [])

                        # Find folders
                        folders = [
                            item
                            for item in items
                            if item.get("mimeType")
                            == "application/vnd.google-apps.folder"
                        ]

                        print(f"     ✅ Found {len(folders)} subfolders")

                        # Add folder IDs and recursively search them
                        for folder in folders:
                            folder_id = folder["id"]
                            folder_name = folder.get("name", "Unknown")
                            print(
                                f"        ↳ Subfolder: {folder_name} (ID: {folder_id})"
                            )
                            subfolders.append(folder_id)

                            # Recursively get subfolders of subfolders
                            sub_subfolders = self._get_all_subfolders(
                                folder_id, max_retries=max_retries
                            )
                            subfolders.extend(sub_subfolders)

                        # Success - break retry loop
                        break
                    else:
                        print(
                            f"     ⚠️ API returned success=False: {result.get('error', 'Unknown error')}"
                        )
                else:
                    print(f"     ⚠️ HTTP {response.status_code}: {response.text[:200]}")

            except requests.Timeout:
                print(f"     ⏱️ Timeout on attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    print(f"     🔄 Retrying...")
                    import time

                    time.sleep(2)  # Wait 2 seconds before retry
                else:
                    print(f"     ❌ All retry attempts exhausted")
            except Exception as e:
                print(f"     ❌ Error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    import time

                    time.sleep(2)
                else:
                    import traceback

                    traceback.print_exc()

        return subfolders

    def ensure_discovery(self, force_refresh=False):
        """
        Ensure files are discovered and cached (INCLUDING SUBFOLDERS).
        This is the KEY method that solves the inconsistency problem.

        Args:
            force_refresh: If True, re-scan even if cache exists

        Returns:
            dict: Discovery cache with all files
        """
        import time

        # Return cache if already initialized and not forcing refresh
        if self._discovery_cache["initialized"] and not force_refresh:
            print(
                f"✅ Using cached discovery: {self._discovery_cache['total_files']} files from {self._discovery_cache['subfolders_count'] + 1} folders"
            )
            return self._discovery_cache

        print(
            f"🔍 Running folder discovery for {self.project_folder_id} (including subfolders)..."
        )

        try:
            # Step 1: Get all subfolders recursively
            subfolders = self._get_all_subfolders(self.project_folder_id)
            all_folder_ids = [self.project_folder_id] + subfolders
            print(
                f"  📂 Total folders to scan: {len(all_folder_ids)} (1 root + {len(subfolders)} subfolders)"
            )

            # Step 2: Collect files from ALL folders
            all_files = []
            for i, current_folder_id in enumerate(all_folder_ids, 1):
                print(
                    f"  📂 [{i}/{len(all_folder_ids)}] Scanning folder: {current_folder_id}"
                )

                response = requests.post(
                    f"{self.http_server_url}/call_tool",
                    json={
                        "name": "list_folder_contents",
                        "arguments": {
                            "folder_id": current_folder_id,
                            "max_results": 1000,
                        },
                    },
                    timeout=30,
                )

                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        items = result.get("data", [])
                        # Only keep files, not folders
                        files = [
                            f
                            for f in items
                            if f.get("mimeType") != "application/vnd.google-apps.folder"
                        ]
                        all_files.extend(files)
                        print(f"     → Found {len(files)} files")
                else:
                    print(f"     ⚠️ HTTP {response.status_code}")

            print(
                f"  📊 Total files collected (before deduplication): {len(all_files)}"
            )

            # Step 3: Remove duplicates (same file might appear multiple times)
            seen_ids = set()
            unique_files = []
            for f in all_files:
                if f["id"] not in seen_ids:
                    seen_ids.add(f["id"])
                    unique_files.append(f)

            print(f"  📊 Unique files after deduplication: {len(unique_files)}")

            # Build lookup dictionary for fast access by name
            by_name = {f["name"]: f for f in unique_files}

            # Update cache
            self._discovery_cache = {
                "initialized": True,
                "total_files": len(unique_files),
                "files": unique_files,
                "last_refresh": time.time(),
                "by_name": by_name,
                "subfolders_count": len(subfolders),
            }

            print(
                f"✅ Discovery complete: {len(unique_files)} files across {len(all_folder_ids)} folders"
            )
            print(f"   Sample files: {[f['name'] for f in unique_files[:3]]}")

            return self._discovery_cache

        except Exception as e:
            print(f"❌ Discovery error: {e}")
            import traceback

            traceback.print_exc()
            return None

    def list_all_files(self, force_refresh=False):
        """
        Get all files in the folder (including subfolders) without filtering.
        This is the RELIABLE way to list projects.

        Args:
            force_refresh: If True, re-scan the folder

        Returns:
            list: All files across all folders
        """
        cache = self.ensure_discovery(force_refresh)
        return cache["files"] if cache else []

    def get_file_count(self, force_refresh=False):
        """
        Get accurate file count across all folders.

        Args:
            force_refresh: If True, re-scan the folder

        Returns:
            int: Total number of files
        """
        cache = self.ensure_discovery(force_refresh)
        return cache["total_files"] if cache else 0

    def get_discovery_info(self):
        """
        Get information about the discovery cache.
        Useful for debugging.

        Returns:
            dict: Discovery cache info
        """
        return {
            "initialized": self._discovery_cache["initialized"],
            "total_files": self._discovery_cache["total_files"],
            "subfolders_count": self._discovery_cache["subfolders_count"],
            "last_refresh": self._discovery_cache["last_refresh"],
            "has_cache": self._discovery_cache["initialized"],
        }

    def _apply_category_filter(self, files):
        """Filter for project-related files"""
        filtered = []
        for f in files:
            name_lower = f.get("name", "").lower()
            if any(
                kw in name_lower
                for kw in [
                    "project",
                    "plan",
                    "milestone",
                    "timeline",
                    "budget",
                    "resource",
                    "task",
                    "deliverable",
                    "roadmap",
                ]
            ):
                filtered.append(f)
        return filtered

    def _get_file_metadata(self, args):
        """Get file metadata - FOLDER SCOPED version"""
        file_pattern = args.get("file_pattern", "")

        print(f"  📋 Getting metadata from folder {self.project_folder_id}")

        response = requests.post(
            f"{self.http_server_url}/call_tool",
            json={
                "name": "search_gdrive_files",
                "arguments": {
                    "query": file_pattern,
                    "max_results": 1000,
                    "folder_id": self.project_folder_id,  # ← KEY!
                },
            },
            timeout=15,
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                files = result.get("data", [])

                # Filter and verify
                filtered_files = self._apply_category_filter(files)
                verified_files = [
                    f
                    for f in filtered_files
                    if self.project_folder_id in f.get("parents", [])
                ]

                print(f"  ✅ Metadata: {len(verified_files)} files")

                metadata = {
                    "total_files": len(verified_files),
                    "file_types": {},
                    "files_by_type": {},
                    "files": [],
                }

                for file in verified_files:
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

    def _initialize_document_awareness(self):
        """Project discovery with STRICT folder scoping"""

        if self._discovery_complete:
            print("📦 Project discovery already complete")
            return self.session_context.get("document_catalog", {})

        if self._discovery_lock:
            print("⏳ Project discovery in progress...")
            max_wait = 10
            waited = 0
            while self._discovery_lock and waited < max_wait:
                time.sleep(0.5)
                waited += 0.5
            return self.session_context.get("document_catalog", {})

        self._discovery_lock = True

        try:
            print(f"🔍 Project discovery in folder: {self.project_folder_id}")

            # Search ONLY in this folder
            result = self._search_files({"query": "", "max_results": 50})

            if not result.get("success") or not result.get("files"):
                print("⚠️ No project files found in folder")
                discovered_files = {"projects": {}}
            else:
                discovered_files = self._categorize_project_files(
                    result.get("files", [])
                )

            total_files_found = sum(len(files) for files in discovered_files.values())

            self.session_context["document_catalog"] = discovered_files
            self.session_context["total_documents_available"] = total_files_found
            self.session_context["discovery_timestamp"] = time.time()

            print(f"✅ Project discovery: {total_files_found} files in folder")

            self._discovery_complete = True

            return discovered_files

        finally:
            self._discovery_lock = False

    def _categorize_project_files(self, files):
        """Fast categorization for project files"""
        categories = {"projects": [], "budgets": [], "timelines": [], "resources": []}

        for file in files:
            name = file.get("name", "").lower()

            if "budget" in name or "cost" in name:
                categories["budgets"].append(file)
            elif "timeline" in name or "schedule" in name:
                categories["timelines"].append(file)
            elif "resource" in name or "team" in name:
                categories["resources"].append(file)
            else:
                categories["projects"].append(file)

        result = {}
        for cat, file_list in categories.items():
            if file_list:
                result[cat] = {
                    f["id"]: {
                        "name": f["name"],
                        "type": f.get("mimeType", "unknown"),
                        "relevance_score": 10,
                    }
                    for f in file_list
                }

        return result


# Operations-specific document processor class
class OperationsDocumentProcessor(ClaudeLikeDocumentProcessor):
    def __init__(self, http_server_url, claude_client):
        super().__init__(http_server_url, claude_client)
        # KEEP the folder ID - it's useful for scoped searches
        self.operations_folder_id = "116wKTVLaOkK6QRyozG6cx9SBloxUWTro"

    def _initialize_document_awareness(self):
        """Tax-specific discovery with lock to prevent duplicates"""

        # CHECK IF ALREADY DONE (prevents repeated calls)
        if self._discovery_complete:
            print("📦 Tax discovery already complete")
            return self.session_context.get("document_catalog", {})

        # CHECK IF CURRENTLY RUNNING (prevents parallel calls)
        if self._discovery_lock:
            print("⏳ Tax discovery in progress, waiting...")
            max_wait = 30
            waited = 0
            while self._discovery_lock and waited < max_wait:
                time.sleep(0.5)
                waited += 0.5
            return self.session_context.get("document_catalog", {})

        # SET LOCK
        self._discovery_lock = True

        try:
            print("🔍 Initializing tax document discovery...")

            # SINGLE targeted search instead of multiple
            search_args = {
                "query": "",  # Specific search term, NOT empty string
                "max_results": 50,
            }

            # Use folder scope if available
            if self.operations_folder_id:
                search_args["folder_id"] = self.operations_folder_id

            result = self.execute_tool("search_files", search_args)

            if not result.get("success") or not result.get("files"):
                print("⚠️ Tax search returned no files")
                discovered_files = {"tax_forms": {}}
            else:
                # Categorize the files we found
                discovered_files = self._categorize_tax_files(result.get("files", []))

            total_files_found = sum(len(files) for files in discovered_files.values())

            # Cache results
            self.session_context["document_catalog"] = discovered_files
            self.session_context["total_documents_available"] = total_files_found
            self.session_context["discovery_timestamp"] = time.time()

            # Create summary
            catalog_summary = self._create_tax_document_summary(discovered_files)
            self.session_context["session_summary"] = catalog_summary

            print(f"✅ Tax discovery complete: {total_files_found} files")

            # MARK AS COMPLETE
            self._discovery_complete = True

            return discovered_files

        except Exception as e:
            print(f"❌ Tax discovery error: {e}")
            import traceback

            traceback.print_exc()
            return {}
        finally:
            # ALWAYS RELEASE LOCK
            self._discovery_lock = False

    def _create_tax_document_summary(self, discovered_files):
        """Create summary for tax documents"""
        summary_parts = ["📊 TAX DOCUMENT CATALOG:\n"]

        for category, files in discovered_files.items():
            if files:
                cat_name = {
                    "form_100": "Corporate Tax (Form 100)",
                    "form_540": "Individual Tax (Form 540)",
                    "form_568": "LLC Tax (Form 568)",
                    "schedules": "Tax Schedules",
                    "general_tax": "General Tax Documents",
                }.get(category, category.upper())

                summary_parts.append(f"**{cat_name}** ({len(files)} files):")

                top_files = list(files.items())[:3]
                for file_id, file_info in top_files:
                    summary_parts.append(f"  • {file_info['name']}")

                if len(files) > 3:
                    summary_parts.append(f"  ... and {len(files) - 3} more files")
                summary_parts.append("")

        return "\n".join(summary_parts)

    def _categorize_tax_files(self, files):
        """Fast pattern-based categorization for tax files"""
        categories = {
            "form_100": [],
            "form_540": [],
            "form_568": [],
            "schedules": [],
            "general_tax": [],
        }

        for file in files:
            name = file.get("name", "").lower()

            if "100" in name or "corporate" in name:
                categories["form_100"].append(file)
            elif "540" in name or "individual" in name:
                categories["form_540"].append(file)
            elif "568" in name or "3522" in name or "llc" in name:
                categories["form_568"].append(file)
            elif "schedule" in name:
                categories["schedules"].append(file)
            else:
                categories["general_tax"].append(file)

        # Convert to expected format
        result = {}
        for cat, file_list in categories.items():
            if file_list:
                result[cat] = {
                    f["id"]: {
                        "name": f["name"],
                        "type": f.get("mimeType", "unknown"),
                        "relevance_score": 10,
                    }
                    for f in file_list
                }

        return result

    def _discover_tax_documents_dynamically(self):
        """Discover with folder scoping if available"""
        # Build search arguments
        search_args = {"query": "", "max_results": 100}

        # Add folder ID to scope search if available (OPTIONAL)
        if hasattr(self, "operations_folder_id") and self.operations_folder_id:
            search_args["folder_id"] = self.operations_folder_id

        result = self.execute_tool("search_files", search_args)

        if not result.get("success"):
            return {}

        # Auto-categorize by pattern matching
        files = result.get("files", [])
        categories = {
            "corporate_tax": [],
            "individual_tax": [],
            "llc_tax": [],
            "schedules": [],
            "general_tax": [],
        }

        for file in files:
            name = file.get("name", "").lower()

            # Pattern matching (dynamic, not hardcoded)
            if re.search(r"form\s*100|corporate", name):
                categories["corporate_tax"].append(file)
            elif re.search(r"form\s*540|individual", name):
                categories["individual_tax"].append(file)
            elif re.search(r"form\s*568|form\s*3522|llc", name):
                categories["llc_tax"].append(file)
            elif re.search(r"schedule", name):
                categories["schedules"].append(file)
            else:
                categories["general_tax"].append(file)

        # Convert to expected format
        discovered = {}
        for category, file_list in categories.items():
            if file_list:
                discovered[category] = {
                    f["id"]: {
                        "name": f["name"],
                        "type": f.get("mimeType", "unknown"),
                        "relevance_score": 10,
                    }
                    for f in file_list
                }

        return discovered

    def _apply_category_filter(self, files):
        return files

    def _search_files(self, args):
        """Search with STRICT folder scoping"""
        query = args.get("query", "")
        max_results = args.get("max_results", 20)

        # Use the appropriate folder_id for this processor
        # For ProjectDocumentProcessor: self.project_folder_id
        # For OperationsDocumentProcessor: self.operations_folder_id
        folder_id = self.operations_folder_id  # or self.operations_folder_id

        print(f"  🔍 Searching in folder {folder_id}: '{query}'")

        try:
            # METHOD 1: List folder contents directly (MOST RELIABLE)
            print("  📂 Trying folder listing method...")
            response = requests.post(
                f"{self.http_server_url}/call_tool",
                json={
                    "name": "list_folder_contents",
                    "arguments": {"folder_id": folder_id, "max_results": 1000},
                },
                timeout=10,
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    all_files = result.get("data", [])
                    print(f"  📂 Folder listing: {len(all_files)} files")

                    # Filter by query
                    if query:
                        query_lower = query.lower()
                        query_words = query_lower.split()  # Split into words
                        filtered_files = []

                        for f in all_files:
                            filename_lower = f.get("name", "").lower()
                            # File matches if it contains ANY word from the query
                            if any(word in filename_lower for word in query_words):
                                filtered_files.append(f)
                    else:
                        filtered_files = all_files

                    # Apply category filter (project-specific or tax-specific)
                    category_files = self._apply_category_filter(filtered_files)

                    # CRITICAL: Verify files are actually in the target folder
                    verified_files = []

                    print(
                        f"  🔍 DEBUG: After category filter: {len(category_files)} files"
                    )
                    if category_files:
                        sample = category_files[0]
                        print(f"       Sample: {sample.get('name')}")
                        print(f"       Parents: {sample.get('parents')}")

                    for f in category_files:
                        parents = f.get("parents", [])
                        if folder_id in parents:
                            verified_files.append(f)
                        else:
                            print(
                                f"  🚫 REJECTED: '{f.get('name')}' (parents: {parents})"
                            )

                    category_files = verified_files
                    print(f"  ✅ Parent-verified: {len(category_files)} files")

                    print(
                        f"  🔍 DEBUG: After category filter: {len(category_files)} files"
                    )
                    if category_files:
                        print(f"       Sample: {sample.get('name')}")
                        print(f"       Parents: {sample.get('parents')}")

                    print(f"  ✓ Filtered to {len(category_files)} relevant files")

                    return {
                        "success": True,
                        "files": [
                            {
                                "id": f["id"],
                                "name": f["name"],
                                "type": f.get("mimeType", "unknown"),
                                "size": f.get("size", "unknown"),
                                "modified": f.get("modifiedTime", "unknown"),
                            }
                            for f in category_files[:max_results]
                        ],
                        "total_found": len(category_files),
                        "search_method": "folder_listing",
                    }

            # METHOD 2: Fallback to search with folder_id
            print("  🔍 Falling back to search method...")
            response = requests.post(
                f"{self.http_server_url}/call_tool",
                json={
                    "name": "search_gdrive_files",
                    "arguments": {
                        "query": query,
                        "max_results": max_results,
                        "folder_id": folder_id,  # CRITICAL: Pass folder_id
                    },
                },
                timeout=10,
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    files = result.get("data", [])

                    category_files = self._apply_category_filter(files)
                    # CRITICAL: Verify files are actually in the target folder
                    verified_files = []
                    for f in category_files:
                        parents = f.get("parents", [])
                        if folder_id in parents:
                            verified_files.append(f)
                        else:
                            print(
                                f"  🚫 REJECTED: '{f.get('name')}' (parents: {parents})"
                            )

                    category_files = verified_files
                    print(f"  ✅ Parent-verified: {len(category_files)} files")

                    print(f"  ✓ Search found {len(category_files)} files")

                    return {
                        "success": True,
                        "files": [
                            {
                                "id": f["id"],
                                "name": f["name"],
                                "type": f.get("mimeType", "unknown"),
                                "size": f.get("size", "unknown"),
                                "modified": f.get("modifiedTime", "unknown"),
                            }
                            for f in category_files
                        ],
                        "total_found": len(category_files),
                        "search_method": "search_with_folder_id",
                    }
                else:
                    print(f"  ⚠️ Search failed: {result.get('error')}")
            else:
                print(f"  ⚠️ HTTP {response.status_code}")

            return {"success": False, "files": [], "total_found": 0}

        except requests.Timeout:
            print("  ⏱️ Search timed out")
            return {"success": False, "files": [], "total_found": 0}
        except Exception as e:
            print(f"  ❌ Search error: {e}")
            import traceback

            traceback.print_exc()
            return {"success": False, "files": [], "total_found": 0}

    def _get_file_metadata(self, args):
        """Get file metadata - FOLDER SCOPED version"""
        file_pattern = args.get("file_pattern", "")

        print(f"  📋 Getting metadata from folder {self.project_folder_id}")

        response = requests.post(
            f"{self.http_server_url}/call_tool",
            json={
                "name": "search_gdrive_files",
                "arguments": {
                    "query": file_pattern,
                    "max_results": 1000,
                    "folder_id": self.project_folder_id,  # ← KEY!
                },
            },
            timeout=15,
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                files = result.get("data", [])

                # Filter and verify
                filtered_files = self._apply_category_filter(files)
                verified_files = [
                    f
                    for f in filtered_files
                    if self.project_folder_id in f.get("parents", [])
                ]

                print(f"  ✅ Metadata: {len(verified_files)} files")

                metadata = {
                    "total_files": len(verified_files),
                    "file_types": {},
                    "files_by_type": {},
                    "files": [],
                }

                for file in verified_files:
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
