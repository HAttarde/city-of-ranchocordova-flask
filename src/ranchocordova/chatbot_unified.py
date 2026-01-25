import os
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .data_loader import get_data_loader
from .vector_store import get_vector_store
from .groq_client import get_groq_client

# Import visualization module
from .visualizations import generate_visualization

# Globals
_groq = None
_embedder = None
_vector_store = None  # ChromaDB vector store
_energy_df = None
_cs_df = None
_dept_df = None


def initialize_models():
    """Load embedder, KB and dataframes once. Uses ChromaDB for persistent storage and Groq API for LLM."""
    print("##### CALLING initialize_models()\n")
    global _groq, _embedder, _vector_store
    global _energy_df, _cs_df, _dept_df

    if _groq is not None:
        return

    print("Loading Rancho Cordova models with Groq API + ChromaDB support...")

    # Initialize Groq client (no local model needed!)
    _groq = get_groq_client()
    
    # Embedder still runs locally (lightweight, CPU-friendly)
    _embedder = SentenceTransformer("all-MiniLM-L6-v2")

    base_path = os.path.join(os.path.dirname(__file__), "data")
    _energy_df = pd.read_csv(os.path.join(base_path, "Energy.txt"))
    _cs_df = pd.read_csv(os.path.join(base_path, "CustomerService.txt"))

    # Load department data if available
    dept_file = os.path.join(base_path, "Department-city of Rancho Cordova.txt")
    if os.path.exists(dept_file):
        _dept_df = pd.read_csv(dept_file)
    else:
        _dept_df = pd.DataFrame()

    print("Loading enhanced energy datasets with PDF support...")
    loader = get_data_loader()
    print("✅ Enhanced datasets loaded")

    # ========================================================================
    # Web Scraping - Auto-crawl on startup if cache is empty or stale
    # ========================================================================
    
    try:
        from .web_scraper import get_scraper, SITE_CONFIGS
        scraper = get_scraper()
        stats = scraper.get_cache_stats()
        total_pages = sum(s.get("pages", 0) for s in stats.values())
        
        # Check which sites need crawling
        stale_sites = [site for site in SITE_CONFIGS if scraper.needs_refresh(site)]
        
        if total_pages > 0 and not stale_sites:
            print(f"✅ Web cache loaded ({total_pages} cached pages)")
        else:
            # Auto-crawl if cache is empty or stale
            if stale_sites:
                print(f"🌐 Auto-crawling {len(stale_sites)} website(s)... (this may take a few minutes)")
                for site_key in stale_sites:
                    site_name = SITE_CONFIGS[site_key]["name"]
                    try:
                        print(f"  🔍 Crawling {site_name}...")
                        pages_scraped = scraper.crawl_site(site_key)
                        print(f"  ✓ {site_name}: {pages_scraped} pages scraped")
                    except Exception as e:
                        print(f"  ⚠️ Error crawling {site_key}: {e}")
                
                # Update stats after crawling
                stats = scraper.get_cache_stats()
                total_pages = sum(s.get("pages", 0) for s in stats.values())
                print(f"✅ Web scraping complete ({total_pages} total cached pages)")
            else:
                print("⚠️ No websites configured for scraping")
    except Exception as e:
        print(f"⚠️ Web scraping skipped: {e}")

    # ========================================================================
    # ChromaDB Vector Store - Persistent Embeddings
    # ========================================================================
    
    _vector_store = get_vector_store()
    _vector_store.set_embedder(_embedder)
    
    # Get list of source files for change detection
    source_files = _get_source_files(base_path)
    
    # Check if we need to rebuild embeddings
    collection_exists = _vector_store.initialize()
    needs_rebuild = not collection_exists or _vector_store.needs_rebuild(source_files)
    
    if needs_rebuild:
        print("🔄 Building embeddings (source files changed or first run)...")
        
        # Build chunk KB
        chunks = []

        # ENERGY TABLE (original)
        for _, row in _energy_df.iterrows():
            chunks.append(
                f"ENERGY_RECORD | "
                f"CustomerID={row['CustomerID']} | "
                f"AccountType={row['AccountType']} | "
                f"Month={row['Month']} | "
                f"EnergyConsumption_kWh={row['EnergyConsumption_kWh']}"
            )

        # CUSTOMER SERVICE (original)
        for _, row in _cs_df.iterrows():
            text_row = " | ".join([f"{col}={row[col]}" for col in _cs_df.columns])
            chunks.append(f"CS_RECORD | {text_row}")

        # DEPARTMENTS (original)
        if not _dept_df.empty:
            for _, row in _dept_df.iterrows():
                text_row = " | ".join([f"{col}={row[col]}" for col in _dept_df.columns])
                chunks.append(f"DEPT_RECORD | {text_row}")

        # DYNAMIC: Extract knowledge from actual CSV files and PDFs
        chunks.extend(_extract_benchmark_insights(base_path))
        chunks.extend(_extract_tou_rate_insights(base_path))
        chunks.extend(_extract_rebate_insights(base_path))
        chunks.extend(_extract_pdf_knowledge())
        
        # WEB CONTENT: Add cached web scraping content
        try:
            from .web_scraper import get_web_chunks
            web_chunks = get_web_chunks()
            if web_chunks:
                chunks.extend(web_chunks)
                print(f"  ✓ Added {len(web_chunks)} web content chunks")
        except Exception as e:
            print(f"  ⚠️ Error loading web chunks: {e}")

        print(f"✅ Total RAG chunks: {len(chunks)}")

        # Add to ChromaDB and persist
        _vector_store.add_chunks(chunks)
        _vector_store.save_hashes(source_files)
        
        print("✅ Embeddings saved to ChromaDB")
    else:
        print(f"✅ Loaded {_vector_store.get_chunk_count()} cached embeddings from ChromaDB")

    print("✅ Rancho models initialized with Groq API + ChromaDB support.")


def _get_source_files(base_path: str) -> list:
    """Get list of source files to monitor for changes."""
    files = []
    data_files = [
        "Energy.txt",
        "CustomerService.txt",
        "Department-city of Rancho Cordova.txt",
        "CA_Benchmarks.csv",
        "SMUD_TOU_Rates.csv",
        "SMUD_Rebates.csv",
        "2024AnnualReport_5YearSummary.pdf",
        "CEC-200-2021-005-PO.pdf",
    ]
    for filename in data_files:
        file_path = os.path.join(base_path, filename)
        if os.path.exists(file_path):
            files.append(file_path)
    return files


# ============================================================================
# NEW: PDF Knowledge Extraction
# ============================================================================


def _extract_pdf_knowledge() -> list:
    """
    Extract knowledge chunks from PDF documents.
    Creates searchable chunks from annual reports and technical documents.
    """
    chunks = []
    loader = get_data_loader()
    pdf_contents = loader.get_all_pdf_contents()

    if not pdf_contents:
        print("  ⚠️  No PDF documents found")
        return chunks

    print(f"\n📄 Extracting knowledge from {len(pdf_contents)} PDF documents...")

    for filename, pdf_data in pdf_contents.items():
        doc_type = _identify_document_type(filename)

        # Split text into manageable chunks (by paragraphs or sections)
        text = pdf_data["text"]

        # Split by double newlines (paragraphs) or sections
        sections = re.split(r"\n\n+", text)

        chunk_count = 0
        for i, section in enumerate(sections):
            section = section.strip()

            # Only include substantial sections (more than 50 characters)
            if len(section) > 50:
                # Create a searchable chunk
                chunk = (
                    f"PDF_DOCUMENT | "
                    f"Source={filename} | "
                    f"Type={doc_type} | "
                    f"Section={i + 1} | "
                    f"Content={section[:1000]}"  # Limit to 1000 chars per chunk
                )
                chunks.append(chunk)
                chunk_count += 1

        print(f"  ✓ Extracted {chunk_count} chunks from {filename}")

    print(f"  ✓ Total PDF chunks: {len(chunks)}")
    return chunks


def _identify_document_type(filename: str) -> str:
    """Identify the type of document based on filename"""
    filename_lower = filename.lower()

    if "annual" in filename_lower or "report" in filename_lower:
        return "Annual_Report"
    elif "cec" in filename_lower or "california" in filename_lower:
        return "Technical_Standard"
    elif "manual" in filename_lower:
        return "Manual"
    elif "policy" in filename_lower:
        return "Policy_Document"
    else:
        return "General_Document"


# ============================================================================
# DYNAMIC CSV EXTRACTION (Benchmark, TOU, Rebates)
# ============================================================================


def _extract_benchmark_insights(base_path: str) -> list:
    """Dynamically extract utility comparison insights from CSV."""
    chunks = []
    csv_path = os.path.join(base_path, "CA_Benchmarks.csv")

    if not os.path.exists(csv_path):
        print("  ⚠️  CA_Benchmarks.csv not found")
        return chunks

    try:
        df = pd.read_csv(csv_path)

        # Create a chunk for each utility/home type combination
        for _, row in df.iterrows():
            chunks.append(
                f"UTILITY_COMPARISON | "
                f"Utility={row['Utility_or_CCA']} | "
                f"Type={row.get('Utility_Type', 'N/A')} | "
                f"Home_Type={row['Home_Type']} | "
                f"Avg_Monthly_kWh={row['Avg_Monthly_Usage_kWh']} | "
                f"Avg_Annual_kWh={row['Avg_Annual_Usage_kWh']} | "
                f"Rate_per_kWh=${row['Avg_Rate_usd_per_kWh']} | "
                f"Avg_Monthly_Bill=${row['Est_Avg_Monthly_Bill_usd']}"
            )

        # Create comparison insights (SMUD vs others)
        smud_data = df[df["Utility_or_CCA"] == "SMUD"]
        if not smud_data.empty:
            smud_avg_rate = smud_data["Avg_Rate_usd_per_kWh"].mean()

            # Compare with PG&E
            pge_data = df[df["Utility_or_CCA"] == "PG&E"]
            if not pge_data.empty:
                pge_avg_rate = pge_data["Avg_Rate_usd_per_kWh"].mean()
                savings_pct = (pge_avg_rate - smud_avg_rate) / pge_avg_rate * 100

                chunks.append(
                    f"UTILITY_SAVINGS | "
                    f"Comparison=SMUD_vs_PGE | "
                    f"SMUD_Rate=${smud_avg_rate:.3f}/kWh | "
                    f"PGE_Rate=${pge_avg_rate:.3f}/kWh | "
                    f"Savings={savings_pct:.0f}% | "
                    f"Description=SMUD residential customers save approximately {savings_pct:.0f}% on electricity "
                    f"rates compared to PG&E. SMUD's average rate is ${smud_avg_rate:.2f}/kWh vs PG&E's ${pge_avg_rate:.2f}/kWh."
                )

        print(f"  ✓ Extracted {len(chunks)} benchmark insights")

    except Exception as e:
        print(f"  ⚠️  Error processing benchmarks: {e}")

    return chunks


def _extract_tou_rate_insights(base_path: str) -> list:
    """Dynamically extract TOU rate information from CSV."""
    chunks = []
    csv_path = os.path.join(base_path, "SMUD_TOU_Rates.csv")

    if not os.path.exists(csv_path):
        print("  ⚠️  SMUD_TOU_Rates.csv not found")
        return chunks

    try:
        df = pd.read_csv(csv_path)

        # Create chunks for each rate period
        for _, row in df.iterrows():
            chunks.append(
                f"TOU_RATE | "
                f"Plan={row['plan']} | "
                f"Period={row['period']} | "
                f"Season={row['season']} | "
                f"DayType={row['day_type']} | "
                f"Hours={row['start_time']}-{row['end_time']} | "
                f"Rate=${row['rate_per_kwh_usd']}/kWh"
            )

        # Create peak/off-peak summaries
        if "period" in df.columns:
            peak_rates = df[df["period"].str.contains("Peak", case=False, na=False)]
            offpeak_rates = df[df["period"].str.contains("Off", case=False, na=False)]

            if not peak_rates.empty and not offpeak_rates.empty:
                avg_peak = peak_rates["rate_per_kwh_usd"].mean()
                avg_offpeak = offpeak_rates["rate_per_kwh_usd"].mean()
                savings_pct = (avg_peak - avg_offpeak) / avg_peak * 100

                chunks.append(
                    f"TOU_SAVINGS | "
                    f"Peak_Rate=${avg_peak:.3f}/kWh | "
                    f"OffPeak_Rate=${avg_offpeak:.3f}/kWh | "
                    f"Savings={savings_pct:.0f}% | "
                    f"Description=You can save up to {savings_pct:.0f}% by shifting energy usage from peak "
                    f"hours to off-peak hours. Peak rate is ${avg_peak:.2f}/kWh "
                    f"vs off-peak ${avg_offpeak:.2f}/kWh."
                )

        print(f"  ✓ Extracted {len(chunks)} TOU rate insights")

    except Exception as e:
        print(f"  ⚠️  Error processing TOU rates: {e}")

    return chunks


def _extract_rebate_insights(base_path: str) -> list:
    """Dynamically extract rebate program information from CSV."""
    chunks = []
    csv_path = os.path.join(base_path, "SMUD_Rebates.csv")

    if not os.path.exists(csv_path):
        print("  ⚠️  SMUD_Rebates.csv not found")
        return chunks

    try:
        df = pd.read_csv(csv_path)

        # Create chunks for each rebate program
        for _, row in df.iterrows():
            chunks.append(
                f"REBATE_PROGRAM | "
                f"CustomerType={row['customer_type']} | "
                f"Category={row['program_category']} | "
                f"Program={row['rebate_name']} | "
                f"Technologies={row['eligible_technologies']} | "
                f"Amount={row['typical_rebate_range_usd']} | "
                f"URL={row.get('program_url', 'N/A')} | "
                f"Notes={row.get('notes', 'N/A')}"
            )

        # Create category summaries
        if "program_category" in df.columns:
            categories = (
                df.groupby("program_category")
                .agg({"rebate_name": "count"})
                .reset_index()
            )
            categories.columns = ["program_category", "count"]

            for _, cat in categories.iterrows():
                category_data = df[df["program_category"] == cat["program_category"]]
                chunks.append(
                    f"REBATE_SUMMARY | "
                    f"Category={cat['program_category']} | "
                    f"Programs={cat['count']} | "
                    f"Description=SMUD offers {cat['count']} rebate program(s) in the "
                    f"{cat['program_category']} category for various energy efficiency upgrades."
                )

        print(f"  ✓ Extracted {len(chunks)} rebate insights")

    except Exception as e:
        print(f"  ⚠️  Error processing rebates: {e}")

    return chunks


# ============================================================================
# RAG Retrieval
# ============================================================================


def retrieve_top_k(query: str, k: int = 5) -> list:
    """Retrieve top-k most relevant chunks for a query using ChromaDB."""
    global _vector_store
    
    if _vector_store is None:
        initialize_models()

    return _vector_store.query(query, k=k)


# ============================================================================
# Agent Type Detection
# ============================================================================


def detect_agent_type(prompt: str) -> str:
    """
    Determine which type of agent should handle this request.
    Returns: 'energy', 'customer_service', 'visualization', or 'general'
    """
    prompt_lower = prompt.lower()

    # Customer service keywords (expanded)
    cs_keywords = [
        "pothole",
        "water bill",
        "trash",
        "garbage",
        "recycling",
        "permit",
        "complaint",
        "report",
        "request",
        "service",
        "problem",
        "issue",
        "fix",
        "repair",
        "maintenance",
        "street",
        "sidewalk",
        "park",
        "community",
    ]
    if any(kw in prompt_lower for kw in cs_keywords):
        return "customer_service"

    # Visualization keywords
    viz_keywords = [
        "chart",
        "graph",
        "plot",
        "visual",
        "show me",
        "display",
        "compare",
        "trend",
        "pattern",
        "distribution",
    ]
    if any(kw in prompt_lower for kw in viz_keywords):
        return "visualization"

    # Energy keywords
    energy_keywords = [
        "energy",
        "electricity",
        "power",
        "kwh",
        "bill",
        "consumption",
        "usage",
        "rate",
        "peak",
        "off-peak",
        "rebate",
        "save",
        "savings",
        "appliance",
        "smud",
        "pge",
        "utility",
        "thermostat",
        "hvac",
    ]
    if any(kw in prompt_lower for kw in energy_keywords):
        return "energy"

    return "general"


# ============================================================================
# System Prompts
# ============================================================================

# Common scope instruction for all agents
SCOPE_INSTRUCTION = """
IMPORTANT SCOPE LIMITATION:
You are ONLY able to answer questions related to:
- City of Rancho Cordova (city services, departments, permits, complaints, etc.)
- SMUD (Sacramento Municipal Utility District) - electricity rates, rebates, energy efficiency
- Energy consumption and sustainability in the Rancho Cordova area

If a user asks about topics OUTSIDE this scope (e.g., other cities, general knowledge, coding, 
politics, sports, or unrelated subjects), politely decline and redirect them. Example response:
"I'm specifically designed to help with City of Rancho Cordova services and SMUD energy-related 
questions. I'm not able to assist with [topic]. Is there anything about Rancho Cordova city 
services or your SMUD electricity account I can help you with?"
"""

ENERGY_SYSTEM_PROMPT = f"""You are an energy efficiency expert for the City of Rancho Cordova and SMUD (Sacramento Municipal Utility District).
Provide helpful, accurate information about energy usage, rates, rebates, and savings tips.
Use the provided context to give specific, data-driven answers.
Be concise but thorough. Format your response with bullet points or numbered lists when appropriate.
{SCOPE_INSTRUCTION}"""

CUSTOMER_SERVICE_SYSTEM_PROMPT = f"""You are a customer service representative for the City of Rancho Cordova.
Help residents with city services, complaints, and requests. Be helpful and direct them to
the appropriate department if needed.
Be friendly, professional, and solution-oriented.
{SCOPE_INSTRUCTION}"""

VISUALIZATION_SYSTEM_PROMPT = f"""You are a data visualization assistant for Rancho Cordova and SMUD energy data.
Help create charts and visualizations to understand energy data patterns and trends.
When describing visualizations, be clear about what the chart shows and key insights.
{SCOPE_INSTRUCTION}"""

GENERAL_SYSTEM_PROMPT = f"""You are a helpful assistant specifically for City of Rancho Cordova residents.
Provide accurate, concise answers to questions about city services, SMUD energy, and local resources.
If you're unsure about something, say so and suggest where they might find more information.
{SCOPE_INSTRUCTION}"""


# ============================================================================
# Generate Response
# ============================================================================


def generate_response(prompt: str, use_rag: bool = True, agent_type: str = None) -> str:
    """
    Generate chatbot response using Groq API with optional RAG.

    Args:
        prompt: User's question or request
        use_rag: Whether to use retrieval-augmented generation
        agent_type: Override automatic agent type detection
    """
    global _groq
    
    if _groq is None:
        initialize_models()

    # Detect agent type if not specified
    if agent_type is None:
        agent_type = detect_agent_type(prompt)

    print(f"🤖 Agent Type: {agent_type}")

    # Build context
    if use_rag:
        context_chunks = retrieve_top_k(prompt, k=4)
        context = "\n".join(context_chunks)
        print(f"📚 Retrieved {len(context_chunks)} relevant chunks")
    else:
        context = ""

    # Build system message based on agent type
    if agent_type == "energy":
        system_msg = ENERGY_SYSTEM_PROMPT
    elif agent_type == "customer_service":
        system_msg = CUSTOMER_SERVICE_SYSTEM_PROMPT
    elif agent_type == "visualization":
        system_msg = VISUALIZATION_SYSTEM_PROMPT
    else:
        system_msg = GENERAL_SYSTEM_PROMPT

    # Build the full prompt
    if context:
        user_prompt = f"""Using the following information:
{context}

Question: {prompt}

Answer:"""
    else:
        user_prompt = f"""Question: {prompt}

Answer:"""

    # Generate response using Groq API
    response = _groq.generate_response(
        system_message=system_msg,
        user_message=user_prompt,
        max_tokens=512,
        temperature=0.7,
    )

    return response.strip()


def generate_response_streaming(prompt: str, use_rag: bool = True, agent_type: str = None):
    """
    Generate chatbot response with streaming (yields tokens as they're generated).
    Use this for real-time UI updates.

    Args:
        prompt: User's question or request
        use_rag: Whether to use retrieval-augmented generation
        agent_type: Override automatic agent type detection
    
    Yields:
        str: Tokens as they are generated
    """
    global _groq
    
    if _groq is None:
        initialize_models()

    # Detect agent type if not specified
    if agent_type is None:
        agent_type = detect_agent_type(prompt)

    print(f"🤖 Agent Type: {agent_type} (streaming)")

    # Build context
    if use_rag:
        context_chunks = retrieve_top_k(prompt, k=4)
        context = "\n".join(context_chunks)
        print(f"📚 Retrieved {len(context_chunks)} relevant chunks")
    else:
        context = ""

    # Use domain-specific system prompts
    if agent_type == "energy":
        system_msg = ENERGY_SYSTEM_PROMPT
    elif agent_type == "customer_service":
        system_msg = CUSTOMER_SERVICE_SYSTEM_PROMPT
    elif agent_type == "visualization":
        system_msg = VISUALIZATION_SYSTEM_PROMPT
    else:
        system_msg = GENERAL_SYSTEM_PROMPT

    # Build the user prompt with context
    if context:
        user_prompt = f"""Using the following information:
{context}

Question: {prompt}

Answer:"""
    else:
        user_prompt = f"""Question: {prompt}

Answer:"""

    # Stream response using Groq API
    for token in _groq.generate_response_streaming(
        system_message=system_msg,
        user_message=user_prompt,
        max_tokens=512,
        temperature=0.7,
    ):
        yield token


# ============================================================================
# Main Chat Function
# ============================================================================


def chat(user_message: str, conversation_history: list = None) -> dict:
    """
    Main chat interface with enhanced PDF support.

    Args:
        user_message: The user's input
        conversation_history: Optional list of previous messages

    Returns:
        dict with 'response', 'agent_type', 'context_used'
    """
    global _groq
    
    if _groq is None:
        initialize_models()

    agent_type = detect_agent_type(user_message)

    # Check if asking about specific documents
    if any(
        keyword in user_message.lower()
        for keyword in ["pdf", "document", "report", "annual"]
    ):
        loader = get_data_loader()
        pdfs = loader.get_all_pdf_contents()
        if pdfs:
            pdf_names = ", ".join(pdfs.keys())
            user_message += f" (Available documents: {pdf_names})"

    response = generate_response(user_message, use_rag=True, agent_type=agent_type)

    return {"response": response, "agent_type": agent_type, "context_used": True}


# ============================================================================
# Backward Compatibility for Flask App
# ============================================================================


def generate_answer(prompt: str, agent_type: str = None) -> dict:
    """
    Generate answer with LLM-driven visualization support.
    
    Uses the LLM to:
    1. Analyze if visualization is needed
    2. Determine the best chart type
    3. Generate Chart.js configuration for frontend rendering
    """
    global _groq, _energy_df, _cs_df
    
    if _groq is None:
        initialize_models()

    if agent_type is None or agent_type == "":
        agent_type = detect_agent_type(prompt)

    print(f"🤖 Agent Type: {agent_type}")
    print(f"📝 Query: {prompt}")

    # ------------------------------------------------------------------
    # PRE-CHECK: Only analyze visualization for relevant queries
    # ------------------------------------------------------------------
    
    chart_config = None
    prompt_lower = prompt.lower()
    
    # Keywords that suggest a visualization might be needed
    viz_keywords = [
        "chart", "graph", "plot", "show", "visualize", "display",
        "trend", "compare", "comparison", "breakdown", "distribution",
        "forecast", "predict", "over time", "volume"
    ]
    
    needs_viz_check = any(kw in prompt_lower for kw in viz_keywords)
    
    if needs_viz_check:
        print("📊 Query contains visualization keywords - analyzing...")
        # ------------------------------------------------------------------
        # LLM-DRIVEN VISUALIZATION (Chart.js)
        # ------------------------------------------------------------------
        
        try:
            from .llm_visualization import generate_llm_visualization
            
            chart_config = generate_llm_visualization(
                query=prompt,
                energy_df=_energy_df,
                cs_df=_cs_df,
            )
            
            if chart_config:
                print(f"✅ Generated Chart.js config: {chart_config.get('type', 'unknown')} chart")
        except Exception as e:
            print(f"⚠️ Visualization generation failed: {e}")
            import traceback
            traceback.print_exc()
            chart_config = None
    else:
        print("ℹ️ No visualization keywords detected - skipping chart analysis")


    # ------------------------------------------------------------------
    # GENERATE TEXT RESPONSE
    # ------------------------------------------------------------------

    if chart_config:
        # If we have a chart, generate a brief explanation
        response_text = _generate_viz_explanation(prompt, chart_config)
    else:
        # Normal text response with RAG
        response_text = generate_response(prompt, use_rag=True, agent_type=agent_type)

    return {"answer": response_text, "chart_config": chart_config}


def _generate_viz_explanation(prompt: str, chart_config: dict) -> str:
    """Generate a brief explanation for the visualization."""
    prompt_lower = prompt.lower()
    chart_type = chart_config.get("type", "chart")
    title = chart_config.get("options", {}).get("plugins", {}).get("title", {}).get("text", "")
    
    if "forecast" in prompt_lower:
        return f"Here's the energy consumption forecast based on recent trends. {title}"
    elif "trend" in prompt_lower:
        return f"This chart shows how the data changes over time. {title}"
    elif "compare" in prompt_lower or "vs" in prompt_lower:
        return f"Here's a comparison visualization. {title}"
    elif "reason" in prompt_lower or "breakdown" in prompt_lower or "distribution" in prompt_lower:
        return f"This breakdown shows the distribution of categories. {title}"
    elif "volume" in prompt_lower:
        return f"This chart displays the volume data over the selected period. {title}"
    else:
        return f"Here's the visualization you requested: {title}"


# ============================================================================
# Compatibility - expose _llm as None for any code that checks it
# ============================================================================
_llm = None  # Kept for backward compatibility checks
