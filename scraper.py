"""
scraper.py — Multi-Source Manufacturing Supplier Scraper
=========================================================
Standalone module. Import into agents.py or run directly.

Sources supported (all optional — gracefully skips if key missing):
  Search APIs   : Tavily (best), SerpAPI, DuckDuckGo (free fallback)
  B2B Directories: IndiaMART, Alibaba, TradeIndia, ExportersIndia,
                   Made-in-China, GlobalSources, ThomasNet, Europages,
                   Kompass
  Generic       : BeautifulSoup deep-scrape on any URL

Usage:
  from scraper import ScraperEngine, ScraperConfig, StreamLogger
  engine = ScraperEngine(config, logger)
  results = engine.run(product="aluminum", location="India")
"""

import os, re, json, time, random
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote_plus, urljoin, urlparse

import requests
from bs4 import BeautifulSoup


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

@dataclass
class ScraperConfig:
    # API keys (read from env if not passed directly)
    tavily_key:  Optional[str] = field(default_factory=lambda: os.getenv("TAVILY_API_KEY"))
    serper_key:  Optional[str] = field(default_factory=lambda: os.getenv("SERPER_API_KEY"))

    # Behaviour
    timeout:      int  = int(os.getenv("TIMEOUT",      "12"))
    max_results:  int  = int(os.getenv("MAX_RESULTS",  "10"))
    scrape_limit: int  = int(os.getenv("SCRAPE_LIMIT", "5"))
    delay:        float = 0.4          # polite delay between requests
    max_workers:  int  = 4             # concurrent scrape threads

    # Which sources to enable
    use_tavily:       bool = True
    use_serper:       bool = True
    use_ddg:          bool = True       # always-free fallback
    use_indiamart:    bool = True
    use_tradeindia:   bool = True
    use_exportersindia: bool = True
    use_alibaba:      bool = True
    use_madeinchina:  bool = True
    use_globalsources:bool = True
    use_thomasnet:    bool = True
    use_europages:    bool = True
    use_kompass:      bool = True

    @property
    def has_tavily(self) -> bool:
        return bool(self.tavily_key) and self.use_tavily

    @property
    def has_serper(self) -> bool:
        return bool(self.serper_key) and self.use_serper


# ─────────────────────────────────────────────────────────────
# LOGGER STUB  (compatible with agents.py StreamLogger)
# ─────────────────────────────────────────────────────────────

class StreamLogger:
    """Minimal logger — replace with agents.StreamLogger in production."""
    def log(self, msg: str, level: str = "info"):
        tag = {"warn": "⚠", "error": "✗", "success": "✓",
               "agent": "◈", "system": "◇"}.get(level, "·")
        print(f"  {tag} {msg}")

    def suppliers(self, data: list): pass
    def done(self, report: str, meta: dict): pass
    def error(self, msg: str): self.log(msg, "error")


# ─────────────────────────────────────────────────────────────
# HTTP HELPER
# ─────────────────────────────────────────────────────────────

ROTATE_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
]

def _headers() -> dict:
    return {
        "User-Agent": random.choice(ROTATE_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
    }

def _get(url: str, timeout: int = 12, params: dict = None) -> Optional[requests.Response]:
    try:
        r = requests.get(url, headers=_headers(), timeout=timeout, params=params)
        r.raise_for_status()
        return r
    except Exception:
        return None

def _post(url: str, payload: dict, extra_headers: dict = None, timeout: int = 12) -> Optional[dict]:
    hdrs = {"Content-Type": "application/json", **(extra_headers or {})}
    try:
        r = requests.post(url, json=payload, headers=hdrs, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def _clean_text(soup: BeautifulSoup, limit: int = 4000) -> str:
    for tag in soup(["script","style","nav","footer","header","form","aside","iframe"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:limit]


# ─────────────────────────────────────────────────────────────
# RESULT SCHEMA
# ─────────────────────────────────────────────────────────────

def _sup(name="", location="", products="", contact="",
         website="", certifications="", moq="", source="", **kw) -> dict:
    """Normalise a supplier dict to a consistent schema."""
    return {
        "name":           name.strip(),
        "location":       location.strip(),
        "products":       products.strip(),
        "contact":        contact.strip(),
        "website":        website.strip(),
        "certifications": certifications.strip(),
        "moq":            moq.strip(),
        "source":         source,
    }


# ─────────────────────────────────────────────────────────────
# ━━━  SEARCH LAYER  ━━━
# Returns [{title, url, snippet}] — raw search hits
# ─────────────────────────────────────────────────────────────

class SearchLayer:
    def __init__(self, cfg: ScraperConfig, logger: StreamLogger):
        self.cfg    = cfg
        self.logger = logger

    # ── 1. Tavily AI Search (best quality, 1000 free req/month) ──
    def tavily(self, query: str) -> list:
        if not self.cfg.has_tavily:
            return []
        self.logger.log(f"  [Tavily] {query!r}")
        data = _post(
            "https://api.tavily.com/search",
            {
                "api_key":        self.cfg.tavily_key,
                "query":          query,
                "search_depth":   "advanced",
                "include_answer": False,
                "max_results":    self.cfg.max_results,
            },
        )
        if not data:
            return []
        results = []
        for r in data.get("results", []):
            results.append({
                "title":   r.get("title", ""),
                "url":     r.get("url",   ""),
                "snippet": r.get("content", r.get("snippet", "")),
            })
        self.logger.log(f"  [Tavily] {len(results)} results", "success")
        return results

    # ── 2. SerpAPI / Serper.dev (2500 free req) ──────────────────
    def serper(self, query: str) -> list:
        if not self.cfg.has_serper:
            return []
        self.logger.log(f"  [Serper] {query!r}")
        data = _post(
            "https://google.serper.dev/search",
            {"q": query, "num": self.cfg.max_results},
            extra_headers={"X-API-KEY": self.cfg.serper_key},
        )
        if not data:
            return []
        results = []
        for r in data.get("organic", []):
            results.append({
                "title":   r.get("title",   ""),
                "url":     r.get("link",    ""),
                "snippet": r.get("snippet", ""),
            })
        self.logger.log(f"  [Serper] {len(results)} results", "success")
        return results

    # ── 3. DuckDuckGo (always free, no key) ──────────────────────
    def ddg(self, query: str) -> list:
        if not self.cfg.use_ddg:
            return []
        self.logger.log(f"  [DDG] {query!r}")
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                hits = list(ddgs.text(query, max_results=self.cfg.max_results))
            results = [{"title":   h.get("title",""),
                        "url":     h.get("href", ""),
                        "snippet": h.get("body", "")} for h in hits]
            self.logger.log(f"  [DDG] {len(results)} results", "success")
            return results
        except Exception as e:
            self.logger.log(f"  [DDG] {e}", "warn")
            return []

    def run(self, queries: list[str]) -> list:
        """Run all enabled search sources for a list of queries, deduplicated."""
        seen, all_results = set(), []
        for q in queries:
            for fn in [self.tavily, self.serper, self.ddg]:
                for r in fn(q):
                    url = r.get("url","")
                    if url and url not in seen:
                        seen.add(url)
                        all_results.append(r)
                time.sleep(0.1)
            # Stop after getting enough unique URLs
            if len(all_results) >= self.cfg.max_results * 3:
                break
        return all_results


# ─────────────────────────────────────────────────────────────
# ━━━  DIRECTORY SCRAPERS  ━━━
# Each returns [{name, location, contact, products, source, …}]
# ─────────────────────────────────────────────────────────────

class DirectoryScrapers:
    def __init__(self, cfg: ScraperConfig, logger: StreamLogger):
        self.cfg    = cfg
        self.logger = logger

    # ── IndiaMART ─────────────────────────────────────────────
    def indiamart(self, product: str, location: str) -> list:
        if not self.cfg.use_indiamart:
            return []
        q   = f"{product} {location}".strip()
        url = f"https://www.indiamart.com/search.mp?ss={quote_plus(q)}"
        self.logger.log(f"  [IndiaMART] {url}")
        r = _get(url, self.cfg.timeout)
        if not r:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        out  = []
        # Try multiple card selectors (IndiaMART updates layout often)
        selectors = [
            ".prdlist > li", ".gstVrfd", ".plist",
            "[data-pid]", ".lcf", ".card-body",
        ]
        cards = []
        for sel in selectors:
            cards = soup.select(sel)
            if cards:
                break
        if not cards:
            # Fallback: grab all h3/h4 with company-like names
            cards = soup.select("h3, h4, .producttitle, b.title")

        for card in cards[:10]:
            name    = card.select_one("h3,h4,.company-name,.sup-name,.title,b")
            contact = card.select_one("[href^='tel'],.tel,.mob,.phone")
            desc    = card.select_one("p,.desc,.product-name,.prodname")
            loc_el  = card.select_one(".location,.loc,.city,.address")

            name    = name.get_text(strip=True)    if name    else (card.get_text(strip=True)[:60] if isinstance(card, str) else "")
            contact = contact.get_text(strip=True) if contact else ""
            desc    = desc.get_text(strip=True)    if desc    else product
            loc_val = loc_el.get_text(strip=True)  if loc_el  else location

            if name and len(name) > 3 and not name.lower().startswith("indiamart"):
                out.append(_sup(
                    name=name, location=loc_val, products=desc,
                    contact=contact, source="IndiaMART",
                    website="https://www.indiamart.com"
                ))
        self.logger.log(f"  [IndiaMART] {len(out)} suppliers", "success")
        return out

    # ── TradeIndia ────────────────────────────────────────────
    def tradeindia(self, product: str, location: str) -> list:
        if not self.cfg.use_tradeindia:
            return []
        q   = f"{product} {location}".strip().replace(" ", "+")
        url = f"https://www.tradeindia.com/search/?category=Products&search_string={q}"
        self.logger.log(f"  [TradeIndia] {url}")
        r = _get(url, self.cfg.timeout)
        if not r:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        out  = []
        cards = soup.select(".bxCnt, .product-box, .cat-list-item, li.listing-item")
        for card in cards[:8]:
            name    = card.select_one("h2,h3,.comp-name,.company-name,a.seller")
            desc    = card.select_one("p,.prod-name,.description")
            contact = card.select_one(".tel,.mob,[href^='tel']")
            loc_el  = card.select_one(".location,.city,.state")
            name    = name.get_text(strip=True) if name else ""
            if name and len(name) > 3:
                out.append(_sup(
                    name=name,
                    location=(loc_el.get_text(strip=True) if loc_el else location),
                    products=(desc.get_text(strip=True)    if desc    else product),
                    contact= (contact.get_text(strip=True) if contact else ""),
                    source="TradeIndia",
                    website="https://www.tradeindia.com",
                ))
        self.logger.log(f"  [TradeIndia] {len(out)} suppliers", "success")
        return out

    # ── ExportersIndia ────────────────────────────────────────
    def exportersindia(self, product: str, location: str) -> list:
        if not self.cfg.use_exportersindia:
            return []
        q   = product.lower().replace(" ", "-")
        url = f"https://www.exportersindia.com/search-result/default.aspx?q={quote_plus(product)}"
        self.logger.log(f"  [ExportersIndia] Searching: {product}")
        r = _get(url, self.cfg.timeout)
        if not r:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        out  = []
        cards = soup.select(".companyList li, .seller-list li, .result-list li")
        for card in cards[:8]:
            name  = card.select_one("h2,h3,h4,.company-name,a.comp")
            desc  = card.select_one("p,.product,.desc")
            loc_el= card.select_one(".location,.city")
            name  = name.get_text(strip=True) if name else ""
            if name and len(name) > 3:
                out.append(_sup(
                    name=name,
                    location=(loc_el.get_text(strip=True) if loc_el else location),
                    products=(desc.get_text(strip=True)   if desc   else product),
                    source="ExportersIndia",
                    website="https://www.exportersindia.com",
                ))
        self.logger.log(f"  [ExportersIndia] {len(out)} suppliers", "success")
        return out

    # ── Alibaba ───────────────────────────────────────────────
    def alibaba(self, product: str, location: str) -> list:
        if not self.cfg.use_alibaba:
            return []
        q   = f"{product} supplier {location}".strip()
        url = f"https://www.alibaba.com/trade/search?SearchText={quote_plus(q)}&Country={quote_plus(location)}"
        self.logger.log(f"  [Alibaba] {q!r}")
        r = _get(url, self.cfg.timeout)
        if not r:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        out  = []
        # Alibaba has heavy JS; grab static anchors with company info
        selectors = [
            ".organic-list-offer-outter",
            ".J-offer-wrapper",
            ".m-gallery-product-item-v2",
            "article",
        ]
        cards = []
        for sel in selectors:
            cards = soup.select(sel)
            if cards:
                break
        if not cards:
            # Lightweight fallback — grab h2 text blocks
            for h in soup.select("h2,h3"):
                txt = h.get_text(strip=True)
                if 3 < len(txt) < 120 and "alibaba" not in txt.lower():
                    out.append(_sup(name=txt, products=product,
                                    location=location, source="Alibaba"))
                    if len(out) >= 6:
                        break
            self.logger.log(f"  [Alibaba] {len(out)} (static fallback)", "success")
            return out

        for card in cards[:8]:
            name   = card.select_one(".organic-gallery-offer-outter__title,.subject,h2,h3")
            loc_el = card.select_one(".supplier-location,.country")
            name   = name.get_text(strip=True) if name else ""
            if name and len(name) > 3:
                out.append(_sup(
                    name=name,
                    location=(loc_el.get_text(strip=True) if loc_el else location),
                    products=product, source="Alibaba",
                    website="https://www.alibaba.com",
                ))
        self.logger.log(f"  [Alibaba] {len(out)} suppliers", "success")
        return out

    # ── Made-in-China ─────────────────────────────────────────
    def madeinchina(self, product: str, location: str) -> list:
        if not self.cfg.use_madeinchina:
            return []
        url = f"https://www.made-in-china.com/multi-search/{quote_plus(product)}/F1/pg-1.html"
        self.logger.log(f"  [Made-in-China] {product!r}")
        r = _get(url, self.cfg.timeout)
        if not r:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        out  = []
        for card in soup.select(".prod-info, .srprd-box, .srp-product")[:8]:
            name  = card.select_one(".company-name,.cname,h2,h3")
            desc  = card.select_one(".prod-name,.prod-title,p")
            name  = name.get_text(strip=True) if name else ""
            if name and len(name) > 3:
                out.append(_sup(
                    name=name, products=(desc.get_text(strip=True) if desc else product),
                    location="China", source="Made-in-China",
                    website="https://www.made-in-china.com",
                ))
        self.logger.log(f"  [Made-in-China] {len(out)} suppliers", "success")
        return out

    # ── Global Sources ────────────────────────────────────────
    def globalsources(self, product: str, location: str) -> list:
        if not self.cfg.use_globalsources:
            return []
        url = (f"https://www.globalsources.com/manufacturers/"
               f"{product.lower().replace(' ','-')}.html")
        self.logger.log(f"  [GlobalSources] {product!r}")
        r = _get(url, self.cfg.timeout)
        if not r:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        out  = []
        for card in soup.select(".supplier-info, .supInfo, [class*='supplier']")[:8]:
            name  = card.select_one("h2,h3,.company-name,a[class*='company']")
            loc_el= card.select_one(".country,.location")
            name  = name.get_text(strip=True) if name else ""
            if name and len(name) > 3:
                out.append(_sup(
                    name=name,
                    location=(loc_el.get_text(strip=True) if loc_el else "Asia"),
                    products=product, source="GlobalSources",
                    website="https://www.globalsources.com",
                ))
        self.logger.log(f"  [GlobalSources] {len(out)} suppliers", "success")
        return out

    # ── ThomasNet (North America / industrial) ────────────────
    def thomasnet(self, product: str, location: str) -> list:
        if not self.cfg.use_thomasnet:
            return []
        url = f"https://www.thomasnet.com/search/?searchTerm={quote_plus(product)}&what={quote_plus(product)}&where={quote_plus(location)}"
        self.logger.log(f"  [ThomasNet] {product!r} / {location!r}")
        r = _get(url, self.cfg.timeout)
        if not r:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        out  = []
        for card in soup.select(".supplier-card, .profile-card, article.result")[:8]:
            name    = card.select_one("h2,h3,.company-name,.supplier-name,a[data-testid='company-name']")
            desc    = card.select_one("p,.description,.summary")
            loc_el  = card.select_one(".location,.city,.state")
            cert_el = card.select_one(".certifications,.cert")
            name    = name.get_text(strip=True) if name else ""
            if name and len(name) > 3:
                out.append(_sup(
                    name=name,
                    location=(loc_el.get_text(strip=True) if loc_el else location),
                    products=(desc.get_text(strip=True)    if desc   else product),
                    certifications=(cert_el.get_text(strip=True) if cert_el else ""),
                    source="ThomasNet",
                    website="https://www.thomasnet.com",
                ))
        self.logger.log(f"  [ThomasNet] {len(out)} suppliers", "success")
        return out

    # ── Europages (EU focused) ────────────────────────────────
    def europages(self, product: str, location: str) -> list:
        if not self.cfg.use_europages:
            return []
        url = f"https://www.europages.co.uk/companies/{quote_plus(product)}.html"
        self.logger.log(f"  [Europages] {product!r}")
        r = _get(url, self.cfg.timeout)
        if not r:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        out  = []
        for card in soup.select(".company-card, .epg-item, article.company")[:8]:
            name    = card.select_one("h2,h3,.company-name,a.company-link")
            loc_el  = card.select_one(".country,.location,.city")
            desc    = card.select_one("p,.activity,.description")
            name    = name.get_text(strip=True) if name else ""
            if name and len(name) > 3:
                out.append(_sup(
                    name=name,
                    location=(loc_el.get_text(strip=True) if loc_el else "Europe"),
                    products=(desc.get_text(strip=True)   if desc   else product),
                    source="Europages",
                    website="https://www.europages.co.uk",
                ))
        self.logger.log(f"  [Europages] {len(out)} suppliers", "success")
        return out

    # ── Kompass (global B2B directory) ────────────────────────
    def kompass(self, product: str, location: str) -> list:
        if not self.cfg.use_kompass:
            return []
        url = f"https://www.kompass.com/searchCompanies?text={quote_plus(product)}&country={quote_plus(location)}"
        self.logger.log(f"  [Kompass] {product!r}")
        r = _get(url, self.cfg.timeout)
        if not r:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        out  = []
        for card in soup.select(".result-company, .company-item, .card-company")[:8]:
            name    = card.select_one("h2,h3,.company-name,a.company-link")
            loc_el  = card.select_one(".country,.location")
            desc    = card.select_one("p,.activity,.description")
            name    = name.get_text(strip=True) if name else ""
            if name and len(name) > 3:
                out.append(_sup(
                    name=name,
                    location=(loc_el.get_text(strip=True) if loc_el else location),
                    products=(desc.get_text(strip=True)   if desc   else product),
                    source="Kompass",
                    website="https://www.kompass.com",
                ))
        self.logger.log(f"  [Kompass] {len(out)} suppliers", "success")
        return out

    def run_all(self, product: str, location: str) -> list:
        """Run all directory scrapers that make sense for the location."""
        loc_lower = location.lower()
        scrapers  = []

        # Always-useful global sources
        scrapers += [self.alibaba, self.globalsources, self.kompass]

        # Region-specific
        if any(x in loc_lower for x in ("india","mumbai","delhi","gujarat","maharashtra","bangalore")):
            scrapers += [self.indiamart, self.tradeindia, self.exportersindia]
        if any(x in loc_lower for x in ("china","taiwan","hong kong","shenzhen","guangzhou")):
            scrapers += [self.madeinchina, self.globalsources]
        if any(x in loc_lower for x in ("germany","france","italy","spain","uk","europe","netherlands","poland")):
            scrapers += [self.europages]
        if any(x in loc_lower for x in ("usa","us","united states","canada","north america","")):
            scrapers += [self.thomasnet]

        # If no region matched, add all
        if not any(x in loc_lower for x in ("india","china","taiwan","germany","france","italy","spain","uk","europe","usa","us")):
            scrapers += [self.indiamart, self.madeinchina, self.europages, self.thomasnet]

        # Deduplicate scraper functions
        seen, unique = set(), []
        for fn in scrapers:
            if fn.__name__ not in seen:
                seen.add(fn.__name__)
                unique.append(fn)

        results = []
        for fn in unique:
            try:
                results.extend(fn(product, location))
                time.sleep(self.cfg.delay)
            except Exception as e:
                self.logger.log(f"  [{fn.__name__}] Error: {e}", "warn")

        return results


# ─────────────────────────────────────────────────────────────
# ━━━  PAGE SCRAPER  ━━━
# Deep-scrapes individual URLs from search results
# ─────────────────────────────────────────────────────────────

class PageScraper:
    def __init__(self, cfg: ScraperConfig, logger: StreamLogger):
        self.cfg    = cfg
        self.logger = logger

    def scrape(self, url: str) -> dict:
        """Return {url, title, text, emails, phones, links}."""
        self.logger.log(f"  [Page] {url}")
        r = _get(url, self.cfg.timeout)
        if not r:
            return {"url": url, "text": "", "emails": [], "phones": [], "links": []}

        soup = BeautifulSoup(r.text, "html.parser")

        # Extract emails
        raw_text = r.text
        emails   = list(set(re.findall(
            r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}", raw_text
        )))[:5]

        # Extract phone numbers (international patterns)
        phones = list(set(re.findall(
            r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}", raw_text
        )))[:5]

        # Extract outbound links (same-domain company pages)
        domain = urlparse(url).netloc
        links  = []
        for a in soup.select("a[href]")[:30]:
            href = a["href"]
            if href.startswith("/"):
                href = urljoin(url, href)
            if domain in href and href != url:
                links.append(href)

        title = soup.title.string.strip() if soup.title else ""
        text  = _clean_text(soup, limit=4000)

        return {"url": url, "title": title, "text": text,
                "emails": emails, "phones": phones, "links": links[:10]}

    def scrape_many(self, urls: list) -> list:
        """Scrape multiple URLs with threading."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = []
        limit   = min(len(urls), self.cfg.scrape_limit)
        with ThreadPoolExecutor(max_workers=self.cfg.max_workers) as ex:
            futures = {ex.submit(self.scrape, u): u for u in urls[:limit]}
            for f in as_completed(futures):
                try:
                    results.append(f.result())
                except Exception:
                    pass
        return results


# ─────────────────────────────────────────────────────────────
# ━━━  SCRAPER ENGINE  ━━━
# Orchestrates all layers; returns unified raw data for LLM
# ─────────────────────────────────────────────────────────────

class ScraperEngine:
    def __init__(self, cfg: ScraperConfig = None, logger: StreamLogger = None):
        self.cfg    = cfg    or ScraperConfig()
        self.logger = logger or StreamLogger()
        self.search = SearchLayer(self.cfg, self.logger)
        self.dirs   = DirectoryScrapers(self.cfg, self.logger)
        self.pages  = PageScraper(self.cfg, self.logger)

    def _build_queries(self, product: str, location: str) -> list:
        loc = f" {location}" if location else ""
        return [
            f"{product} suppliers{loc}",
            f"{product} manufacturers{loc} contact email",
            f"buy {product}{loc} wholesale",
            f"{product} factory{loc} ISO certified",
            f"{product} distributors{loc} MOQ price",
        ]

    def run(self, product: str, location: str) -> dict:
        """
        Full scrape pipeline.
        Returns {
          "search_hits":     [{ title, url, snippet }],
          "page_data":       [{ url, title, text, emails, phones }],
          "dir_suppliers":   [{ name, location, contact, ... }],
        }
        """
        self.logger.log(f"══ Scraper starting: {product!r} / {location!r} ══", "agent")

        # 1. Search APIs
        queries     = self._build_queries(product, location)
        search_hits = self.search.run(queries)
        self.logger.log(f"  Total search hits: {len(search_hits)}", "success")

        # 2. Deep-scrape top URLs
        urls        = [h["url"] for h in search_hits if h.get("url")]
        page_data   = self.pages.scrape_many(urls)

        # 3. B2B directories
        dir_results = self.dirs.run_all(product, location)
        self.logger.log(f"  Directory entries: {len(dir_results)}", "success")

        return {
            "search_hits":   search_hits,
            "page_data":     page_data,
            "dir_suppliers": dir_results,
        }

    def build_llm_context(self, raw: dict, limit: int = 8000) -> str:
        """Flatten raw scrape data into a single string for LLM consumption."""
        parts = []
        for h in raw["search_hits"][:12]:
            parts.append(f"[SEARCH] {h['title']}\nURL: {h['url']}\n{h['snippet']}")
        for p in raw["page_data"]:
            blob = p["text"][:1200]
            if p["emails"]:
                blob += f"\nEmails found: {', '.join(p['emails'])}"
            if p["phones"]:
                blob += f"\nPhones found: {', '.join(p['phones'])}"
            parts.append(f"[PAGE] {p['url']}\n{blob}")
        for d in raw["dir_suppliers"]:
            parts.append(
                f"[DIRECTORY:{d['source']}] {d['name']}\n"
                f"Location: {d['location']} | Products: {d['products']}\n"
                f"Contact: {d['contact']} | Website: {d['website']}"
            )
        text = "\n\n---\n\n".join(parts)
        return text[:limit]


# ─────────────────────────────────────────────────────────────
# CLI  (run standalone: python scraper.py "aluminum" "India")
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Load .env
    from pathlib import Path
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                if k.strip() not in os.environ:
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")

    args     = sys.argv[1:]
    product  = args[0] if len(args) > 0 else "aluminum"
    location = args[1] if len(args) > 1 else "India"

    cfg    = ScraperConfig()
    logger = StreamLogger()
    engine = ScraperEngine(cfg, logger)
    raw    = engine.run(product, location)

    print(f"\n{'='*60}")
    print(f"  RESULTS: {len(raw['dir_suppliers'])} directory suppliers")
    print(f"           {len(raw['search_hits'])} search hits")
    print(f"           {len(raw['page_data'])} pages scraped")
    print(f"{'='*60}\n")

    for s in raw["dir_suppliers"][:5]:
        print(f"  • {s['name']} ({s['source']}) — {s['location']}")
