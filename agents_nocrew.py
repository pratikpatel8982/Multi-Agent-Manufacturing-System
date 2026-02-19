"""
agents.py — Manufacturing Supplier Finder — Agent Orchestration
================================================================
Imports ScraperEngine from scraper.py.
All scraping logic lives in scraper.py; all LLM + agent logic lives here.

Modes:
  CLI:    python agents.py "Find aluminum suppliers in India"
  Server: python agents.py --server
"""

import os, sys, re, json, time, textwrap, threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# .env LOADER
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


# ─────────────────────────────────────────────────────────────
# IMPORTS (after .env is loaded)
# ─────────────────────────────────────────────────────────────

from groq import Groq
from scraper import ScraperEngine, ScraperConfig   # ← separated module


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

GROQ_MODEL = os.getenv("GROQ_MODEL",  "llama-3.3-70b-versatile")
PORT       = int(os.getenv("PORT",    "5000"))
HOST       = os.getenv("HOST",        "0.0.0.0")
DEBUG      = os.getenv("DEBUG",       "false").lower() == "true"


# ─────────────────────────────────────────────────────────────
# STOP SIGNAL  (per-session cancellation)
# ─────────────────────────────────────────────────────────────

# Maps session_id → threading.Event. Set the event to abort that pipeline.
_stop_events: dict = {}
_stop_lock = threading.Lock()

# In-memory report store — avoids writing files to disk
# Maps session_id → report dict. Capped at 50 entries (oldest evicted).
_report_store: dict = {}
_REPORT_STORE_MAX = 50

def _store_report(session_id: str, data: dict):
    if len(_report_store) >= _REPORT_STORE_MAX:
        oldest = next(iter(_report_store))
        _report_store.pop(oldest, None)
    _report_store[session_id] = data

def register_stop(session_id: str) -> threading.Event:
    ev = threading.Event()
    with _stop_lock:
        _stop_events[session_id] = ev
    return ev

def request_stop(session_id: str) -> bool:
    with _stop_lock:
        ev = _stop_events.get(session_id)
    if ev:
        ev.set()
        return True
    return False

def cleanup_stop(session_id: str):
    with _stop_lock:
        _stop_events.pop(session_id, None)


# ─────────────────────────────────────────────────────────────
# STREAM LOGGER
# ─────────────────────────────────────────────────────────────

class StreamLogger:
    """Dual-output: terminal + SSE queue for the chatbot frontend."""
    def __init__(self, queue=None):
        self.queue = queue
        self.lines = []

    def _emit(self, entry: dict):
        self.lines.append(entry)
        msg = entry.get("message", str(entry))
        print(msg)
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

    # scraper.py compatibility — same interface
    def suppliers_raw(self, data): pass


# ─────────────────────────────────────────────────────────────
# SHARED STATE  (hand-off object between agents)
# ─────────────────────────────────────────────────────────────

@dataclass
class PipelineState:
    session_id:        str  = field(default_factory=lambda: f"MFG-{int(time.time())}")
    user_query:        str  = ""
    parsed_product:    str  = ""
    parsed_location:   str  = ""
    scrape_summary:    str  = ""          # raw LLM context from scraper
    raw_results:       list = field(default_factory=list)   # structured suppliers
    sources_used:      list = field(default_factory=list)   # which sources fired
    handoff_done:      bool = False
    handoff_timestamp: str  = ""
    final_report:      str  = ""
    errors:            list = field(default_factory=list)
    stopped:           bool = False
    stop_event:        object = field(default=None, repr=False)

    def mark_handoff(self):
        self.handoff_done      = True
        self.handoff_timestamp = datetime.utcnow().isoformat()

    def is_stopped(self) -> bool:
        return self.stopped or (self.stop_event is not None and self.stop_event.is_set())


# ─────────────────────────────────────────────────────────────
# LLM HELPER
# ─────────────────────────────────────────────────────────────

class RateLimitSkip(Exception):
    """Raised when Groq rate limit is hit — caller should skip gracefully."""
    pass

class PipelineStopped(Exception):
    """Raised when the user pressed Stop."""
    pass

def call_groq(client: Groq, system: str, user: str,
              max_tokens: int = 2048, temperature: float = 0.3,
              stop_event: threading.Event = None) -> str:
    """
    Call Groq LLM. On rate-limit (429 / TPD / TPM), raises RateLimitSkip
    so the caller can continue the pipeline with a fallback instead of
    crashing. Other errors propagate normally.
    """
    if stop_event and stop_event.is_set():
        raise PipelineStopped()
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        msg = str(e)
        if "429" in msg or "rate_limit" in msg.lower() or "rate limit" in msg.lower():
            raise RateLimitSkip(msg)
        raise


def parse_json_llm(raw: str) -> list | dict | None:
    """Strip markdown fences and parse JSON from LLM output."""
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        return json.loads(clean)
    except Exception:
        # Try extracting first [...] or {...} block
        m = re.search(r"(\[.*\]|\{.*\})", clean, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    return None


# ══════════════════════════════════════════════════════════════
# AGENT 1 — RESEARCHER
# ══════════════════════════════════════════════════════════════

class ResearcherAgent:
    """
    Responsibilities:
    1. Parse user query → product + location (via LLM)
    2. Delegate scraping to ScraperEngine (scraper.py)
    3. Run LLM extraction over raw scrape data → structured suppliers list
    4. Mark hand-off on PipelineState
    """

    def __init__(self, groq_client: Groq, scraper: ScraperEngine):
        self.llm     = groq_client
        self.scraper = scraper

    # ── Step 1: Parse ─────────────────────────────────────────
    def parse_query(self, query: str, logger: StreamLogger,
                    stop_event=None) -> tuple[str, str]:
        logger.log("  [Researcher] Parsing query…", "info")
        try:
            raw = call_groq(
                self.llm,
                system=(
                    "You are a manufacturing procurement assistant. "
                    "Extract ONLY the product/material and the target country or region from the query. "
                    "Return ONLY valid JSON: {\"product\": \"...\", \"location\": \"...\"}\n"
                    "If no location is mentioned, use empty string."
                ),
                user=query,
                max_tokens=100,
                stop_event=stop_event,
            )
        except RateLimitSkip:
            logger.log("  [Researcher] LLM quota reached — inferring product from query text", "warn")
            # Best-effort fallback: treat full query as product, no location
            words = query.strip().split()
            product  = " ".join(words[:6]) if words else query
            location = ""
            for loc_hint in ["in ", "from ", "at "]:
                if loc_hint in query.lower():
                    idx = query.lower().index(loc_hint)
                    location = query[idx + len(loc_hint):].strip()
                    product  = query[:idx].strip()
                    break
            logger.log(f"  [Researcher] Fallback — Product={product!r}  Location={location!r}", "info")
            return product, location
        parsed = parse_json_llm(raw)
        if isinstance(parsed, dict):
            product  = parsed.get("product",  query)
            location = parsed.get("location", "")
        else:
            product, location = query, ""
        logger.log(f"  [Researcher] Product={product!r}  Location={location!r}", "success")
        return product, location

    # ── Step 2: Scrape ─────────────────────────────────────────
    def gather(self, product: str, location: str,
               logger: StreamLogger) -> tuple[dict, list]:
        """Returns (raw_scrape_dict, sources_used_list)."""
        raw    = self.scraper.run(product, location)
        sources = list({d["source"] for d in raw.get("dir_suppliers", [])})
        if raw.get("search_hits"):
            sources.insert(0, "Web Search")
        return raw, sources

    # ── Step 3: Extract structured suppliers ───────────────────
    def extract_suppliers(self, product: str, location: str,
                          raw: dict, logger: StreamLogger,
                          stop_event=None) -> list:
        logger.log("  [Researcher] Extracting structured suppliers with LLM…")
        context = self.scraper.build_llm_context(raw, limit=8000)

        try:
            raw_resp = call_groq(
                self.llm,
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
                    - Merge duplicates — if same company appears twice, use one entry.
                    - Do NOT invent names or contacts not found in the data.
                    - Prefer entries with real contact info (phone/email).
                """),
                user=f"Product: {product}\nTarget Location: {location}\n\nSCRAPED DATA:\n{context}",
                max_tokens=3000,
                stop_event=stop_event,
            )
        except RateLimitSkip:
            logger.log("  [Researcher] LLM quota reached — using directory data directly", "warn")
            # Fall back to directory results without LLM structuring
            fallback = []
            required = {"name","location","products","contact","website","certifications","moq","source"}
            for s in raw.get("dir_suppliers", []):
                if isinstance(s, dict) and s.get("name"):
                    entry = {k: s.get(k, "") for k in required}
                    fallback.append(entry)
            logger.log(f"  [Researcher] Fallback — {len(fallback)} suppliers from directories", "info")
            return fallback

        suppliers = parse_json_llm(raw_resp)
        if isinstance(suppliers, list):
            # Validate each has required keys
            clean = []
            required = {"name","location","products","contact","website",
                        "certifications","moq","source"}
            for s in suppliers:
                if isinstance(s, dict) and s.get("name"):
                    for k in required:
                        s.setdefault(k, "")
                    clean.append(s)
            return clean
        logger.log("  [Researcher] LLM returned no parseable JSON — returning empty list", "warn")
        return []

    # ── Main entry ─────────────────────────────────────────────
    def run(self, state: PipelineState, logger: StreamLogger) -> PipelineState:
        logger.log("══════════ RESEARCHER AGENT ══════════", "agent")
        ev = state.stop_event

        if state.is_stopped(): return state

        product, location       = self.parse_query(state.user_query, logger, stop_event=ev)
        state.parsed_product    = product
        state.parsed_location   = location

        if state.is_stopped(): return state

        raw, sources            = self.gather(product, location, logger)
        state.sources_used      = sources
        state.scrape_summary    = self.scraper.build_llm_context(raw, limit=4000)

        if state.is_stopped(): return state

        suppliers               = self.extract_suppliers(product, location, raw, logger, stop_event=ev)
        state.raw_results       = suppliers

        logger.log(f"  [Researcher] {len(suppliers)} suppliers found — handing off to Writer…", "success")
        logger.suppliers(suppliers)
        state.mark_handoff()
        return state


# ══════════════════════════════════════════════════════════════
# AGENT 2 — WRITER
# ══════════════════════════════════════════════════════════════

class WriterAgent:
    """
    Receives PipelineState after hand-off.
    Synthesises a professional procurement report using Groq LLM.
    """

    def __init__(self, groq_client: Groq):
        self.llm = groq_client

    def run(self, state: PipelineState, logger: StreamLogger) -> PipelineState:
        logger.log("══════════ WRITER AGENT ══════════", "agent")
        logger.log(f"  Hand-off received at {state.handoff_timestamp}")
        logger.log(f"  Sources used: {', '.join(state.sources_used) or 'n/a'}")

        if state.is_stopped():
            logger.log("  [Writer] Pipeline stopped — skipping report", "warn")
            return state

        if not state.handoff_done:
            state.errors.append("Hand-off not marked complete")
            logger.log("  [Writer] Hand-off incomplete — skipping report", "warn")
            return state

        if not state.raw_results:
            logger.log("  [Writer] No suppliers found — writing generic report", "warn")

        try:
          report = call_groq(
            self.llm,
            system=textwrap.dedent("""
                You are a senior manufacturing procurement analyst.
                Write a comprehensive, professional Supplier Sourcing Report.
                Use PLAIN TEXT ONLY — no markdown, no *, no #, no bullet dashes.
                Use the exact structure below (copy the separator lines exactly):

                ================================================================
                SUPPLIER SOURCING REPORT
                ================================================================
                QUERY SUMMARY
                Product  : ...
                Location : ...
                Sources  : ...
                Date     : ...
                Session  : ...

                ----------------------------------------------------------------
                EXECUTIVE SUMMARY
                ----------------------------------------------------------------
                Write 3–4 sentences: market context, key finding, top recommendation.

                ----------------------------------------------------------------
                SUPPLIER DIRECTORY
                ----------------------------------------------------------------
                (Number each supplier. Fill every field; write "N/A" if unknown.)

                1. Company Name
                   Location       : ...
                   Products       : ...
                   Contact        : ...
                   Website        : ...
                   Certifications : ...
                   MOQ            : ...
                   Source         : ...

                2. (next supplier)
                ...

                ----------------------------------------------------------------
                MARKET INSIGHTS
                ----------------------------------------------------------------
                Write 4–5 sentences covering pricing trends, lead times,
                key risks, and regional supply landscape.

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

                RULES:
                - Base ALL content on the supplier data provided.
                - Do NOT invent suppliers, contacts, or certifications.
                - Be specific — use actual company names from the data.
                - Keep tone executive and concise.
            """),
            user=textwrap.dedent(f"""
                Query    : {state.user_query}
                Product  : {state.parsed_product}
                Location : {state.parsed_location}
                Sources  : {", ".join(state.sources_used)}
                Session  : {state.session_id}
                Date     : {datetime.utcnow().strftime("%B %d, %Y UTC")}

                STRUCTURED SUPPLIER DATA (from Researcher Agent):
                {json.dumps(state.raw_results, indent=2)}

                ADDITIONAL MARKET CONTEXT (raw scrape summary):
                {state.scrape_summary[:2000]}
            """),
              max_tokens=3000,
              temperature=0.25,
          )
        except RateLimitSkip:
            logger.log("  [Writer] LLM quota reached — report skipped, supplier data is available above", "warn")
            # Build a minimal plain-text report from structured data only
            lines = [
                "================================================================",
                "SUPPLIER SOURCING REPORT  (quota limit reached — summary only)",
                "================================================================",
                f"QUERY    : {state.user_query}",
                f"PRODUCT  : {state.parsed_product}",
                f"LOCATION : {state.parsed_location}",
                f"SOURCES  : {', '.join(state.sources_used)}",
                "",
                "----------------------------------------------------------------",
                "SUPPLIER DIRECTORY",
                "----------------------------------------------------------------",
            ]
            for i, s in enumerate(state.raw_results, 1):
                lines.append(f"{i}. {s.get('name','Unknown')}")
                for k in ("location","products","contact","website","certifications","moq","source"):
                    v = s.get(k,"")
                    if v:
                        lines.append(f"   {k.capitalize():<15}: {v}")
            lines += ["", "================================================================",
                      "NOTE: Full narrative report unavailable — LLM token quota reached.",
                      "      Upgrade Groq plan or wait for quota reset to get the full report.",
                      "================================================================"]
            report = "\n".join(lines)

        state.final_report = report
        logger.log("  [Writer] Report generated ✓", "success")
        return state


# ══════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════

class ManufacturingOrchestrator:
    """
    Wires together ScraperEngine → ResearcherAgent → WriterAgent.
    Manages PipelineState, logging, and JSON output.
    """

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            sys.exit(
                "\n❌  GROQ_API_KEY not set.\n"
                "    Add GROQ_API_KEY=gsk_... to your .env file\n"
                "    Free key → https://console.groq.com\n"
            )

        groq_client = Groq(api_key=api_key)

        # Build scraper config from env
        cfg = ScraperConfig(
            tavily_key=os.getenv("TAVILY_API_KEY"),
            serper_key=os.getenv("SERPER_API_KEY"),
        )

        scraper          = ScraperEngine(cfg)   # no logger yet — passed per-run
        self.groq        = groq_client
        self.scraper_cfg = cfg
        self.researcher  = ResearcherAgent(groq_client, scraper)
        self.writer      = WriterAgent(groq_client)

    def run(self, user_query: str, logger: StreamLogger = None) -> PipelineState:
        logger = logger or StreamLogger()

        # Inject logger into scraper at runtime
        self.researcher.scraper.logger = logger

        state            = PipelineState(user_query=user_query)
        state.stop_event = register_stop(state.session_id)
        t0               = time.time()

        logger.log(f"SESSION : {state.session_id}", "system")
        logger.log(f"QUERY   : {user_query}",       "system")
        logger.log(f"MODEL   : {GROQ_MODEL}",       "system")
        # Send session_id to frontend immediately so Stop button can use it
        logger._emit({"type": "session", "session_id": state.session_id})
        if self.scraper_cfg.has_tavily:
            logger.log("  Search: Tavily ✓", "success")
        if self.scraper_cfg.has_serper:
            logger.log("  Search: Serper ✓", "success")
        logger.log("  Search: DuckDuckGo ✓ (fallback)", "info")

        try:
            state = self.researcher.run(state, logger)
            if not state.is_stopped():
                state = self.writer.run(state, logger)
        finally:
            cleanup_stop(state.session_id)

        if state.is_stopped():
            logger.log("Pipeline stopped by user.", "warn")
            logger._emit({"type": "stopped"})

        elapsed = round(time.time() - t0, 1)
        logger.log(f"Pipeline complete in {elapsed}s", "success")

        # Store report in memory for download — no local file saved
        _store_report(state.session_id, {
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
        })

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

    @app.route("/api/stop", methods=["POST"])
    def stop_endpoint():
        body       = request.get_json(force=True)
        session_id = (body.get("session_id") or "").strip()
        if not session_id:
            return jsonify({"error": "session_id required"}), 400
        found = request_stop(session_id)
        return jsonify({"stopped": found, "session_id": session_id})

    @app.route("/api/download/<session_id>")
    def download_endpoint(session_id):
        """Download the report as a plain-text .txt file."""
        from flask import make_response
        data = _report_store.get(session_id)
        if not data:
            return jsonify({"error": "Report not found. It may have expired."}), 404
        report_text = data.get("report", "No report available.")
        # Build a clean txt with metadata header
        header = (
            f"MFG AGENT — SUPPLIER SOURCING REPORT\n"
            f"{'='*64}\n"
            f"Session  : {data['session_id']}\n"
            f"Query    : {data['query']}\n"
            f"Product  : {data['product']}\n"
            f"Location : {data['location']}\n"
            f"Sources  : {', '.join(data['sources_used'])}\n"
            f"Suppliers: {data['suppliers_found']}\n"
            f"Duration : {data['elapsed_seconds']}s\n"
            f"{'='*64}\n\n"
        )
        full_text = header + report_text
        fname = f"supplier_report_{session_id}.txt"
        resp = make_response(full_text)
        resp.headers["Content-Type"] = "text/plain; charset=utf-8"
        resp.headers["Content-Disposition"] = f'attachment; filename="{fname}"'
        return resp

    @app.route("/api/download-json/<session_id>")
    def download_json_endpoint(session_id):
        """Download the full structured data as JSON."""
        from flask import make_response
        data = _report_store.get(session_id)
        if not data:
            return jsonify({"error": "Report not found. It may have expired."}), 404
        fname = f"supplier_report_{session_id}.json"
        resp = make_response(json.dumps(data, indent=2, ensure_ascii=False))
        resp.headers["Content-Type"] = "application/json; charset=utf-8"
        resp.headers["Content-Disposition"] = f'attachment; filename="{fname}"'
        return resp

    @app.route("/api/query", methods=["POST"])
    def query_endpoint():
        body = request.get_json(force=True)
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
                q.put(None)  # sentinel

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
            "status":       "ok",
            "model":        GROQ_MODEL,
            "port":         PORT,
            "tavily":       cfg.has_tavily,
            "serper":       cfg.has_serper,
            "ddg_fallback": True,
            "directories":  [
                k.replace("use_","") for k, v in cfg.__dict__.items()
                if k.startswith("use_") and v
            ],
        })

    return app


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--server" in sys.argv:
        print(f"\n  🚀  Manufacturing Agent Server")
        print(f"  URL   → http://localhost:{PORT}")
        print(f"  Model → {GROQ_MODEL}\n")
        create_app().run(host=HOST, port=PORT, debug=DEBUG, threaded=True)
    else:
        args       = [a for a in sys.argv[1:] if not a.startswith("--")]
        user_query = " ".join(args) if args else input("Query: ").strip()
        if not user_query:
            user_query = "Find aluminum suppliers in India"
        ManufacturingOrchestrator().run(user_query)