"""
agents.py — Manufacturing Supplier Finder — CrewAI Agent Orchestration
=======================================================================
All three errors from previous versions are fixed here:

  FIX 1 — "No module named 'fastapi'"
     Was caused by the GroqLLMShim approach triggering LiteLLM proxy imports.
     Solution: use the native "groq/<model>" string for Agent(llm=...).
     LiteLLM's BASE package (bundled with crewai) supports Groq natively.
     Only the [proxy] extra needs fastapi — we never install that.

  FIX 2 — "OPENAI_API_KEY is required"
     CrewAI validates an OpenAI key at startup even when you never use OpenAI.
     Solution: set os.environ["OPENAI_API_KEY"] = "not-used" in code,
     right after loading .env, before any crewai import. This satisfies
     the validator without touching OpenAI at all — your GROQ_API_KEY
     is what actually gets used for every LLM call.

  FIX 3 — "RateLimitError: tokens per minute exceeded"
     CrewAI's agent loop makes hidden planning/reasoning calls on top of
     your 3 tool calls, exhausting Groq's free 12k TPM limit.
     Solution: _groq_call_with_retry() wraps every tool-level Groq call.
     It reads the "try again in Xs" hint from the error and sleeps exactly
     that long before retrying (up to RETRY_MAX times).

Architecture:
  Crew (Process.sequential)
    ├── ResearcherAgent
    │     ├── QueryParserTool      (1 LLM call, 100 tokens)
    │     ├── SupplierScraperTool  (0 LLM calls — pure HTTP scraping)
    │     └── SupplierExtractTool  (1 LLM call, 3000 tokens)
    └── WriterAgent                (1 LLM call, 3000 tokens)
          context=[research_task]  ← CrewAI native hand-off

Modes:
  CLI:    python agents.py "Find aluminum suppliers in India"
  Server: python agents.py --server
"""

import os, sys, re, json, time, textwrap, threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Type


# ─────────────────────────────────────────────────────────────
# STEP 1 — load .env
# ─────────────────────────────────────────────────────────────

def load_dotenv(path: str = ".env"):
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v

load_dotenv()

# STEP 2 — satisfy CrewAI's OpenAI validator BEFORE any crewai import.
# This dummy value is never sent to OpenAI; all real calls use GROQ_API_KEY.
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "not-used-groq-is-the-llm"


# ─────────────────────────────────────────────────────────────
# IMPORTS  (crewai imported after env is fully set up)
# ─────────────────────────────────────────────────────────────

from groq import Groq
from groq import RateLimitError as GroqRateLimitError
from scraper import ScraperEngine, ScraperConfig

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

GROQ_MODEL  = os.getenv("GROQ_MODEL", "llama3-70b-8192")
PORT        = int(os.getenv("PORT",   "5000"))
HOST        = os.getenv("HOST",       "0.0.0.0")
DEBUG       = os.getenv("DEBUG",      "false").lower() == "true"

# LiteLLM Groq provider string — no fastapi, no proxy, no OpenAI
CREW_LLM    = f"groq/{GROQ_MODEL}"

# Rate-limit retry settings (tunable via .env)
RETRY_MAX       = int(os.getenv("GROQ_RETRY_MAX",    "6"))
RETRY_BASE_WAIT = float(os.getenv("GROQ_RETRY_WAIT", "6.0"))  # seconds


# ─────────────────────────────────────────────────────────────
# RATE-LIMIT RETRY WRAPPER
# ─────────────────────────────────────────────────────────────

def _groq_call_with_retry(fn, *args, **kwargs) -> str:
    """
    Wraps any callable that makes a Groq SDK call.
    On GroqRateLimitError, reads the 'try again in Xs' hint from the
    error body and sleeps that long (+1.5s buffer) before retrying.
    Falls back to linear backoff (6s, 12s, 18s, …) if no hint found.
    All other exceptions propagate immediately.
    """
    for attempt in range(1, RETRY_MAX + 1):
        try:
            return fn(*args, **kwargs)
        except GroqRateLimitError as exc:
            if attempt == RETRY_MAX:
                raise
            wait = RETRY_BASE_WAIT * attempt          # default linear backoff
            m    = re.search(r"try again in ([\d.]+)s", str(exc))
            if m:
                wait = float(m.group(1)) + 1.5        # Groq hint + safety buffer
            print(f"  [Groq] Rate limit — sleeping {wait:.1f}s "
                  f"(attempt {attempt}/{RETRY_MAX})")
            time.sleep(wait)
        except Exception:
            raise
    raise RuntimeError("Groq retry limit exceeded")


# ─────────────────────────────────────────────────────────────
# LLM HELPER  (used inside tools — direct SDK, with retry)
# ─────────────────────────────────────────────────────────────

def call_groq(client: Groq, system: str, user: str,
              max_tokens: int = 2048, temperature: float = 0.3) -> str:
    def _call(_):
        return client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        ).choices[0].message.content.strip()

    return _groq_call_with_retry(_call, None)


def parse_json_llm(raw: str) -> list | dict | None:
    """Strip markdown fences and parse the first JSON block found."""
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        return json.loads(clean)
    except Exception:
        m = re.search(r"(\[.*\]|\{.*\})", clean, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    return None


# ─────────────────────────────────────────────────────────────
# STREAM LOGGER
# ─────────────────────────────────────────────────────────────

class StreamLogger:
    """Dual-output: terminal print + SSE queue for the chatbot UI."""
    def __init__(self, queue=None):
        self.queue = queue
        self.lines = []

    def _emit(self, entry: dict):
        self.lines.append(entry)
        print(entry.get("message", str(entry)))
        if self.queue:
            self.queue.put(json.dumps(entry))

    def log(self, msg: str, level: str = "info"):
        self._emit({"type": "log", "level": level, "message": msg,
                    "ts": datetime.utcnow().isoformat()})

    def suppliers(self, data: list):
        self._emit({"type": "suppliers", "data": data})

    def done(self, report: str, meta: dict):
        self._emit({"type": "done", "report": report, "meta": meta})

    def error(self, msg: str):
        self._emit({"type": "error", "message": msg})
        print(f"[ERROR] {msg}", file=sys.stderr)

    def suppliers_raw(self, data): pass   # scraper.py compat


# ─────────────────────────────────────────────────────────────
# SHARED PIPELINE STATE
# ─────────────────────────────────────────────────────────────

@dataclass
class PipelineState:
    session_id:        str  = field(default_factory=lambda: f"MFG-{int(time.time())}")
    user_query:        str  = ""
    parsed_product:    str  = ""
    parsed_location:   str  = ""
    scrape_summary:    str  = ""
    raw_results:       list = field(default_factory=list)
    sources_used:      list = field(default_factory=list)
    handoff_done:      bool = False
    handoff_timestamp: str  = ""
    final_report:      str  = ""
    errors:            list = field(default_factory=list)

    def mark_handoff(self):
        self.handoff_done      = True
        self.handoff_timestamp = datetime.utcnow().isoformat()


# ══════════════════════════════════════════════════════════════
# CREWAI TOOLS
# Private dependencies (groq client, state, logger) are stored
# via object.__setattr__ to bypass Pydantic's field validation.
# ══════════════════════════════════════════════════════════════

def _set(obj, name, val): object.__setattr__(obj, name, val)
def _get(obj, name):      return object.__getattribute__(obj, name)


# ── Pydantic input schemas ─────────────────────────────────────

class QueryInput(BaseModel):
    query: str = Field(..., description="Raw user procurement query")

class ScrapeInput(BaseModel):
    product:  str = Field(..., description="Product or material to source")
    location: str = Field(..., description="Target country or region (empty string if none)")

class ExtractInput(BaseModel):
    product:  str = Field(..., description="Product or material")
    location: str = Field(..., description="Target location")
    context:  str = Field(..., description="Raw scraped text returned by supplier_scraper")


# ── Tool 1: Query Parser ───────────────────────────────────────

class QueryParserTool(BaseTool):
    """
    1 LLM call, max_tokens=100.
    Parses free-text query → {product, location}.
    """
    name:        str = "query_parser"
    description: str = (
        "Parse a manufacturing procurement query to extract the product/material "
        "and the target country or region. "
        "Input: the raw user query string. "
        "Returns JSON: {\"product\": \"...\", \"location\": \"...\"}."
    )
    args_schema: Type[BaseModel] = QueryInput

    def __init__(self, groq_client: Groq, state: PipelineState, logger: StreamLogger):
        super().__init__()
        _set(self, "_groq",   groq_client)
        _set(self, "_state",  state)
        _set(self, "_logger", logger)

    def _run(self, query: str) -> str:
        groq   = _get(self, "_groq")
        state  = _get(self, "_state")
        logger = _get(self, "_logger")

        logger.log("  [Tool:QueryParser] Parsing query…", "info")
        raw = call_groq(
            groq,
            system=(
                "You are a manufacturing procurement assistant. "
                "Extract ONLY the product/material and target country or region. "
                "Return ONLY valid JSON: {\"product\": \"...\", \"location\": \"...\"} "
                "Use empty string for location if none is mentioned."
            ),
            user=query,
            max_tokens=100,
        )
        parsed   = parse_json_llm(raw)
        product  = parsed.get("product",  query) if isinstance(parsed, dict) else query
        location = parsed.get("location", "")    if isinstance(parsed, dict) else ""

        state.parsed_product  = product
        state.parsed_location = location
        logger.log(f"  [Tool:QueryParser] product={product!r}  location={location!r}", "success")
        return json.dumps({"product": product, "location": location})


# ── Tool 2: Supplier Scraper ───────────────────────────────────

class SupplierScraperTool(BaseTool):
    """
    0 LLM calls — pure HTTP scraping via ScraperEngine.
    Hits web search APIs + B2B directories + deep-scrapes top pages.
    Returns condensed text context for the extractor.
    """
    name:        str = "supplier_scraper"
    description: str = (
        "Search the web and B2B directories (IndiaMART, Alibaba, ThomasNet, etc.) "
        "to collect raw supplier data for a product and location. "
        "Input: product and location strings. "
        "Returns a text block to pass directly to supplier_extractor."
    )
    args_schema: Type[BaseModel] = ScrapeInput

    def __init__(self, scraper: ScraperEngine, state: PipelineState, logger: StreamLogger):
        super().__init__()
        _set(self, "_scraper", scraper)
        _set(self, "_state",   state)
        _set(self, "_logger",  logger)

    def _run(self, product: str, location: str) -> str:
        scraper = _get(self, "_scraper")
        state   = _get(self, "_state")
        logger  = _get(self, "_logger")

        logger.log(f"  [Tool:SupplierScraper] Scraping {product!r} / {location!r}…", "info")
        scraper.logger = logger
        raw = scraper.run(product, location)

        sources = list({d["source"] for d in raw.get("dir_suppliers", [])})
        if raw.get("search_hits"):
            sources.insert(0, "Web Search")
        state.sources_used = sources

        context = scraper.build_llm_context(raw, limit=8000)
        state.scrape_summary = context
        logger.log(
            f"  [Tool:SupplierScraper] "
            f"{len(raw.get('dir_suppliers', []))} directory entries, "
            f"{len(raw.get('search_hits', []))} search hits", "success"
        )
        return context


# ── Tool 3: Supplier Extractor ─────────────────────────────────

class SupplierExtractTool(BaseTool):
    """
    1 LLM call, max_tokens=3000.
    Converts raw scrape context → structured JSON supplier list.
    Marks the formal hand-off on PipelineState.
    """
    name:        str = "supplier_extractor"
    description: str = (
        "Extract structured supplier records from raw scraped text using an LLM. "
        "Input: product, location, and the context string from supplier_scraper. "
        "Returns JSON: {\"suppliers\": [...], \"sources\": [...]}."
    )
    args_schema: Type[BaseModel] = ExtractInput

    def __init__(self, groq_client: Groq, state: PipelineState, logger: StreamLogger):
        super().__init__()
        _set(self, "_groq",   groq_client)
        _set(self, "_state",  state)
        _set(self, "_logger", logger)

    def _run(self, product: str, location: str, context: str) -> str:
        groq   = _get(self, "_groq")
        state  = _get(self, "_state")
        logger = _get(self, "_logger")

        logger.log("  [Tool:SupplierExtractor] Extracting structured suppliers…", "info")
        raw_resp = call_groq(
            groq,
            system=textwrap.dedent("""
                You are a manufacturing procurement data extractor.

                From the web scrape data below, extract REAL supplier / manufacturer records.
                Return a JSON array where each object has EXACTLY these fields
                (use empty string "" if unknown — NEVER omit a field):

                [
                  {
                    "name":           "Company legal name",
                    "location":       "City, Country",
                    "products":       "What they manufacture or supply",
                    "contact":        "Phone number or email",
                    "website":        "https://... or empty",
                    "certifications": "ISO 9001, CE, etc. or empty",
                    "moq":            "Minimum order quantity or empty",
                    "source":         "Which directory/site this came from"
                  }
                ]

                Rules:
                - Return ONLY the JSON array. No explanation, no markdown.
                - Extract at least 5 and up to 15 suppliers.
                - Merge duplicates — same company appearing twice = one entry.
                - Do NOT invent names or contacts not found in the data.
                - Prefer entries with real contact info (phone/email).
            """),
            user=f"Product: {product}\nTarget Location: {location}\n\nSCRAPED DATA:\n{context}",
            max_tokens=3000,
        )

        suppliers = parse_json_llm(raw_resp)
        clean = []
        if isinstance(suppliers, list):
            required = {"name","location","products","contact","website",
                        "certifications","moq","source"}
            for s in suppliers:
                if isinstance(s, dict) and s.get("name"):
                    for k in required:
                        s.setdefault(k, "")
                    clean.append(s)

        state.raw_results = clean
        state.mark_handoff()   # ← formal hand-off to WriterAgent

        logger.log(
            f"  [Tool:SupplierExtractor] {len(clean)} suppliers extracted — "
            f"hand-off marked ✓ ({state.handoff_timestamp})", "success"
        )
        logger.suppliers(clean)
        return json.dumps({"suppliers": clean, "sources": state.sources_used}, indent=2)


# ══════════════════════════════════════════════════════════════
# CREWAI AGENT BUILDERS
# llm=CREW_LLM  →  "groq/llama3-70b-8192"
# LiteLLM routes this to Groq using GROQ_API_KEY from env.
# No fastapi, no proxy, no OpenAI calls.
# ══════════════════════════════════════════════════════════════

def build_researcher_agent(tools: list) -> Agent:
    return Agent(
        role="Manufacturing Procurement Researcher",
        goal=(
            "Find and extract comprehensive, accurate supplier data for the requested "
            "product and location by searching the web and B2B directories."
        ),
        backstory=(
            "You are an expert manufacturing procurement researcher with 15 years of "
            "experience sourcing suppliers across India, China, Europe and North America. "
            "You always follow a strict 3-step process: first parse the query with "
            "query_parser, then gather raw data with supplier_scraper, then produce "
            "structured records with supplier_extractor. You pass each tool's output "
            "directly as input to the next tool."
        ),
        tools=tools,
        llm=CREW_LLM,
        verbose=True,
        allow_delegation=False,   # no sub-agent spawning = fewer hidden LLM calls
        max_iter=4,               # 3 tool calls + 1 final reasoning step
    )


def build_writer_agent() -> Agent:
    return Agent(
        role="Senior Procurement Report Analyst",
        goal=(
            "Transform structured supplier data from the Researcher into a polished, "
            "executive-grade Supplier Sourcing Report with recommendations and risk assessment."
        ),
        backstory=(
            "You are a senior manufacturing procurement analyst who writes clear, factual "
            "reports grounded exclusively in the data provided. You never invent company "
            "names, contacts, or certifications."
        ),
        tools=[],              # Writer is LLM-only — no tools needed
        llm=CREW_LLM,
        verbose=True,
        allow_delegation=False,
        max_iter=2,
    )


# ══════════════════════════════════════════════════════════════
# CREWAI TASK BUILDERS
# ══════════════════════════════════════════════════════════════

def build_research_task(agent: Agent, user_query: str) -> Task:
    return Task(
        description=textwrap.dedent(f"""
            A manufacturing procurement team submitted this query:
            "{user_query}"

            Execute these three steps IN ORDER using your tools:

            STEP 1 — Call query_parser with the query above.
                     Note the returned product and location values.

            STEP 2 — Call supplier_scraper with the product and location from Step 1.
                     Note the returned context text.

            STEP 3 — Call supplier_extractor with the product and location from Step 1
                     AND the context text from Step 2.

            Return the exact JSON output of supplier_extractor as your final answer.
            Do not add any commentary outside the JSON.
        """),
        expected_output='JSON object: {"suppliers": [...], "sources": [...]}',
        agent=agent,
    )


def build_writing_task(agent: Agent, state: PipelineState, research_task: Task) -> Task:
    return Task(
        description=textwrap.dedent(f"""
            You have received structured supplier data from the Researcher Agent (see context).

            Write a comprehensive, professional Supplier Sourcing Report.
            Use PLAIN TEXT ONLY — no markdown, no *, no #, no bullet dashes.
            Copy the separator lines exactly as shown:

            ================================================================
            SUPPLIER SOURCING REPORT
            ================================================================
            QUERY SUMMARY
            Product  : [from data]
            Location : [from data]
            Sources  : [from data]
            Date     : {datetime.utcnow().strftime("%B %d, %Y UTC")}
            Session  : {state.session_id}

            ----------------------------------------------------------------
            EXECUTIVE SUMMARY
            ----------------------------------------------------------------
            [3-4 sentences: market context, key finding, top recommendation]

            ----------------------------------------------------------------
            SUPPLIER DIRECTORY
            ----------------------------------------------------------------
            [Number each supplier. Write "N/A" for unknown fields.]

            1. Company Name
               Location       : ...
               Products       : ...
               Contact        : ...
               Website        : ...
               Certifications : ...
               MOQ            : ...
               Source         : ...

            [continue for all suppliers]

            ----------------------------------------------------------------
            MARKET INSIGHTS
            ----------------------------------------------------------------
            [4-5 sentences on pricing trends, lead times, risks, regional landscape]

            ----------------------------------------------------------------
            TOP 3 RECOMMENDATIONS
            ----------------------------------------------------------------
            1. [Supplier Name] — [one specific justification sentence]
            2. [Supplier Name] — [one specific justification sentence]
            3. [Supplier Name] — [one specific justification sentence]

            ----------------------------------------------------------------
            RISK ASSESSMENT
            ----------------------------------------------------------------
            HIGH:   [risk description]
            MEDIUM: [risk description]
            LOW:    [risk description]

            ----------------------------------------------------------------
            NEXT STEPS
            ----------------------------------------------------------------
            1. [action]
            2. [action]
            3. [action]
            4. [action]
            5. [action]

            ================================================================
            END OF REPORT
            ================================================================

            RULES: Base ALL content on supplied data. Do NOT invent anything.
            Use actual company names from the data. Keep tone executive and concise.
        """),
        expected_output=(
            "A complete plain-text Supplier Sourcing Report following the exact structure above."
        ),
        agent=agent,
        context=[research_task],   # ← CrewAI native hand-off: Researcher output → Writer input
    )


# ══════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════

class ManufacturingOrchestrator:
    """
    Wires everything together:
      ScraperEngine (scraper.py)
        → 3 CrewAI BaseTool subclasses
          → ResearcherAgent (Task 1, sequential)
            → [CrewAI context hand-off]
              → WriterAgent (Task 2, sequential)
                → PipelineState + StreamLogger + JSON report file
    """

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            sys.exit(
                "\n❌  GROQ_API_KEY not set.\n"
                "    Add GROQ_API_KEY=gsk_... to your .env file\n"
                "    Free key → https://console.groq.com\n"
            )
        self.groq_client = Groq(api_key=api_key)
        self.scraper_cfg = ScraperConfig(
            tavily_key=os.getenv("TAVILY_API_KEY"),
            serper_key=os.getenv("SERPER_API_KEY"),
        )

    def run(self, user_query: str, logger: StreamLogger = None) -> PipelineState:
        logger = logger or StreamLogger()
        state  = PipelineState(user_query=user_query)
        t0     = time.time()

        # ── Header ──────────────────────────────────────────────
        logger.log(f"SESSION : {state.session_id}",  "system")
        logger.log(f"QUERY   : {user_query}",         "system")
        logger.log(f"MODEL   : {GROQ_MODEL}",         "system")
        logger.log(f"LLM     : {CREW_LLM} via LiteLLM (no OpenAI, no fastapi)", "system")
        logger.log(f"RETRY   : up to {RETRY_MAX}x with backoff", "system")
        if self.scraper_cfg.has_tavily:
            logger.log("  Search: Tavily ✓",             "success")
        if self.scraper_cfg.has_serper:
            logger.log("  Search: Serper ✓",             "success")
        logger.log("  Search: DuckDuckGo ✓ (fallback)", "info")

        # ── Per-run objects ──────────────────────────────────────
        scraper = ScraperEngine(self.scraper_cfg)

        tools = [
            QueryParserTool(self.groq_client, state, logger),
            SupplierScraperTool(scraper, state, logger),
            SupplierExtractTool(self.groq_client, state, logger),
        ]

        # ── Agents & Tasks ───────────────────────────────────────
        logger.log("══════════ BUILDING CREW ══════════", "agent")
        researcher_agent = build_researcher_agent(tools)
        writer_agent     = build_writer_agent()

        research_task = build_research_task(researcher_agent, user_query)
        writing_task  = build_writing_task(writer_agent, state, research_task)

        # ── Crew (sequential: Researcher → Writer) ───────────────
        crew = Crew(
            agents=[researcher_agent, writer_agent],
            tasks=[research_task, writing_task],
            process=Process.sequential,
            verbose=True,
        )

        # ── Kickoff ──────────────────────────────────────────────
        logger.log("══════════ RESEARCHER AGENT ══════════", "agent")
        try:
            crew_output = crew.kickoff()
        except Exception as e:
            state.errors.append(str(e))
            logger.error(f"Crew error: {e}")
            crew_output = None

        # ── Extract report ───────────────────────────────────────
        logger.log("══════════ WRITER AGENT ══════════", "agent")
        if crew_output:
            raw_out = str(crew_output)
            if "SUPPLIER SOURCING REPORT" in raw_out:
                state.final_report = raw_out[raw_out.find("====="):].strip()
            else:
                state.final_report = raw_out.strip()
            logger.log("  [Writer] Report generated ✓", "success")
        else:
            state.errors.append("Crew produced no output")
            logger.error("No output from Crew")

        # ── Persist JSON ─────────────────────────────────────────
        elapsed = round(time.time() - t0, 1)
        logger.log(f"Pipeline complete in {elapsed}s", "success")

        out = {
            "session_id":      state.session_id,
            "query":           state.user_query,
            "product":         state.parsed_product,
            "location":        state.parsed_location,
            "sources_used":    state.sources_used,
            "suppliers_found": len(state.raw_results),
            "suppliers":       state.raw_results,
            "report":          state.final_report,
            "elapsed_seconds": elapsed,
            "errors":          state.errors,
        }
        fname = f"report_{state.session_id}.json"
        try:
            with open(fname, "w") as f:
                json.dump(out, f, indent=2)
            logger.log(f"Saved → {fname}")
        except Exception as e:
            logger.log(f"Could not save JSON: {e}", "warn")

        logger.done(state.final_report, {
            "session_id":      state.session_id,
            "elapsed_seconds": elapsed,
            "suppliers_found": len(state.raw_results),
            "sources_used":    state.sources_used,
        })
        return state


# ══════════════════════════════════════════════════════════════
# FLASK SERVER  (SSE streaming for chatbot UI)
# ══════════════════════════════════════════════════════════════

def create_app():
    try:
        from flask import Flask, request, Response, jsonify, send_from_directory
        from flask_cors import CORS
    except ImportError:
        sys.exit("pip install flask flask-cors")

    import queue as _queue

    app          = Flask(__name__, static_folder="static", static_url_path="")
    orchestrator = ManufacturingOrchestrator()
    CORS(app)

    @app.route("/")
    def index():
        return send_from_directory("static", "index.html")

    @app.route("/api/query", methods=["POST"])
    def query_endpoint():
        body       = request.get_json(force=True)
        user_query = (body.get("query") or "").strip()
        if not user_query:
            return jsonify({"error": "query is required"}), 400

        q      = _queue.Queue()
        logger = StreamLogger(queue=q)

        def run_pipeline():
            try:
                orchestrator.run(user_query, logger)
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
            finally:
                q.put(None)   # sentinel — tells generate() to stop

        threading.Thread(target=run_pipeline, daemon=True).start()

        def generate():
            while True:
                item = q.get()
                if item is None:
                    break
                yield f"data: {item}\n\n"

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    @app.route("/api/health")
    def health():
        cfg = orchestrator.scraper_cfg
        return jsonify({
            "status":      "ok",
            "model":       GROQ_MODEL,
            "crew_llm":    CREW_LLM,
            "port":        PORT,
            "tavily":      cfg.has_tavily,
            "serper":      cfg.has_serper,
            "ddg":         True,
            "directories": [
                k.replace("use_", "") for k, v in cfg.__dict__.items()
                if k.startswith("use_") and v
            ],
        })

    return app


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--server" in sys.argv:
        print(f"\n  🚀  Manufacturing Agent Server (CrewAI + Groq)")
        print(f"  URL   → http://localhost:{PORT}")
        print(f"  Model → {CREW_LLM}\n")
        create_app().run(host=HOST, port=PORT, debug=DEBUG, threaded=True)
    else:
        args       = [a for a in sys.argv[1:] if not a.startswith("--")]
        user_query = " ".join(args) if args else input("Query: ").strip()
        if not user_query:
            user_query = "Find aluminum suppliers in India"
        ManufacturingOrchestrator().run(user_query)
