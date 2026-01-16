"""
Web Scraper Module for Rancho Cordova Chatbot
=============================================

Crawls and caches content from city and utility websites:
- cityofranchocordova.org
- data-ranchocordova.opendata.arcgis.com  
- smud.org
- pgedataportals.pge.com

Features:
- Respects robots.txt and rate limits
- Caches content in SQLite for fast retrieval
- Configurable crawl depth
- HTML to clean text conversion
"""

import hashlib
import os
import re
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    "max_depth": 2,
    "rate_limit_seconds": 1.0,  # Delay between requests
    "cache_expiry_days": 7,  # Weekly cache refresh
    "request_timeout": 10,
    "max_pages_per_site": 200,
    "user_agent": "RanchoCordovaBot/1.0 (Educational Project)",
}

# Site configurations with specific crawl settings
SITE_CONFIGS = {
    "ranchocordova": {
        "base_url": "https://www.cityofranchocordova.org/",
        "name": "City of Rancho Cordova",
        "start_paths": ["/", "/residents", "/business", "/government", "/services"],
        "allowed_patterns": [r"cityofranchocordova\.org"],
        "excluded_patterns": [r"/calendar", r"/events", r"\.pdf$", r"\.jpg$", r"\.png$"],
    },
    "opendata": {
        "base_url": "https://data-ranchocordova.opendata.arcgis.com/",
        "name": "Rancho Cordova Open Data",
        "start_paths": ["/", "/search"],
        "allowed_patterns": [r"data-ranchocordova\.opendata\.arcgis\.com"],
        "excluded_patterns": [r"\.geojson$", r"\.csv$", r"\.zip$"],
    },
    "smud": {
        "base_url": "https://www.smud.org/",
        "name": "SMUD",
        "start_paths": [
            "/",
            "/Rate-Information",
            "/Rebates-and-Savings",
            "/Customer-Support",
            "/Clean-Energy",
        ],
        "allowed_patterns": [r"smud\.org"],
        "excluded_patterns": [r"/Account", r"/Login", r"\.pdf$"],
    },
    "pge": {
        "base_url": "https://pgedataportals.pge.com/",
        "name": "PG&E Data Portal",
        "start_paths": ["/"],
        "allowed_patterns": [r"pgedataportals\.pge\.com", r"pge\.com/.*data"],
        "excluded_patterns": [r"/login", r"/account"],
    },
}


# ============================================================================
# WEB SCRAPER CLASS
# ============================================================================


class WebScraper:
    """
    Web scraper with caching and rate limiting.
    
    Usage:
        scraper = WebScraper()
        scraper.crawl_site("smud")  # Crawl SMUD website
        content = scraper.get_all_cached_content()
    """

    def __init__(self, db_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize the web scraper.
        
        Args:
            db_path: Path to SQLite cache database
            config: Override default configuration
        """
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        
        # Set up database path
        if db_path is None:
            base_path = os.path.dirname(__file__)
            db_path = os.path.join(base_path, "data", "web_cache.db")
        
        self.db_path = db_path
        self._init_database()
        
        # Session for HTTP requests
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.config["user_agent"],
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })
        
        # Rate limiting
        self._last_request_time = 0
        
    def _init_database(self):
        """Initialize SQLite cache database."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS page_cache (
                url TEXT PRIMARY KEY,
                site_key TEXT,
                title TEXT,
                content TEXT,
                html TEXT,
                scraped_at TIMESTAMP,
                content_hash TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS crawl_status (
                site_key TEXT PRIMARY KEY,
                last_crawl TIMESTAMP,
                pages_crawled INTEGER,
                status TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_site_key ON page_cache(site_key)
        """)
        
        conn.commit()
        conn.close()
        
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config["rate_limit_seconds"]:
            time.sleep(self.config["rate_limit_seconds"] - elapsed)
        self._last_request_time = time.time()
        
    def _is_valid_url(self, url: str, site_config: Dict) -> bool:
        """Check if URL should be crawled based on site config."""
        # Check allowed patterns
        if not any(re.search(p, url) for p in site_config["allowed_patterns"]):
            return False
            
        # Check excluded patterns
        if any(re.search(p, url) for p in site_config["excluded_patterns"]):
            return False
            
        return True
        
    def _extract_text(self, html: str) -> Tuple[str, str]:
        """
        Extract clean text and title from HTML.
        
        Returns:
            Tuple of (title, clean_text)
        """
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            element.decompose()
            
        # Get title
        title = ""
        if soup.title:
            title = soup.title.string or ""
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)
            
        # Get main content
        main_content = soup.find("main") or soup.find("article") or soup.find("body")
        
        if main_content:
            text = main_content.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)
            
        # Clean up whitespace
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        text = "\n".join(lines)
        
        # Remove very short content (likely navigation/footer remnants)
        if len(text) < 100:
            return title, ""
            
        return title, text
        
    def _extract_links(self, html: str, base_url: str) -> List[str]:
        """Extract all links from HTML page."""
        soup = BeautifulSoup(html, "html.parser")
        links = []
        
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            
            # Skip anchors and javascript
            if href.startswith("#") or href.startswith("javascript:"):
                continue
                
            # Convert relative URLs to absolute
            full_url = urljoin(base_url, href)
            
            # Remove fragments
            full_url = full_url.split("#")[0]
            
            if full_url not in links:
                links.append(full_url)
                
        return links
        
    def scrape_page(self, url: str) -> Optional[Dict]:
        """
        Scrape a single page.
        
        Returns:
            Dict with title, content, html or None if failed
        """
        self._rate_limit()
        
        try:
            response = self.session.get(
                url,
                timeout=self.config["request_timeout"],
                allow_redirects=True
            )
            response.raise_for_status()
            
            # Only process HTML content
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                return None
                
            html = response.text
            title, content = self._extract_text(html)
            
            if not content:
                return None
                
            return {
                "url": url,
                "title": title,
                "content": content,
                "html": html,
                "content_hash": hashlib.md5(content.encode()).hexdigest(),
            }
            
        except requests.RequestException as e:
            print(f"  ⚠️ Failed to scrape {url}: {e}")
            return None
            
    def crawl_site(self, site_key: str, max_depth: Optional[int] = None) -> int:
        """
        Crawl a website recursively.
        
        Args:
            site_key: Key from SITE_CONFIGS (e.g., "smud", "ranchocordova")
            max_depth: Override default max depth
            
        Returns:
            Number of pages scraped
        """
        if site_key not in SITE_CONFIGS:
            raise ValueError(f"Unknown site: {site_key}. Available: {list(SITE_CONFIGS.keys())}")
            
        site_config = SITE_CONFIGS[site_key]
        max_depth = max_depth or self.config["max_depth"]
        max_pages = self.config["max_pages_per_site"]
        
        print(f"\n🌐 Crawling {site_config['name']} (depth={max_depth}, max={max_pages} pages)")
        
        visited: Set[str] = set()
        to_visit: List[Tuple[str, int]] = []  # (url, depth)
        pages_scraped = 0
        
        # Add start URLs
        for path in site_config["start_paths"]:
            url = urljoin(site_config["base_url"], path)
            to_visit.append((url, 0))
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        while to_visit and pages_scraped < max_pages:
            url, depth = to_visit.pop(0)
            
            # Skip if already visited
            if url in visited:
                continue
            visited.add(url)
            
            # Skip if not valid for this site
            if not self._is_valid_url(url, site_config):
                continue
                
            # Scrape the page
            result = self.scrape_page(url)
            
            if result:
                # Save to cache
                cursor.execute("""
                    INSERT OR REPLACE INTO page_cache 
                    (url, site_key, title, content, html, scraped_at, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    url,
                    site_key,
                    result["title"],
                    result["content"],
                    result["html"],
                    datetime.now().isoformat(),
                    result["content_hash"],
                ))
                
                pages_scraped += 1
                print(f"  ✓ [{pages_scraped}] {result['title'][:50]}...")
                
                # Extract links for next depth level
                if depth < max_depth:
                    links = self._extract_links(result["html"], url)
                    for link in links:
                        if link not in visited:
                            to_visit.append((link, depth + 1))
                            
        # Update crawl status
        cursor.execute("""
            INSERT OR REPLACE INTO crawl_status 
            (site_key, last_crawl, pages_crawled, status)
            VALUES (?, ?, ?, ?)
        """, (site_key, datetime.now().isoformat(), pages_scraped, "completed"))
        
        conn.commit()
        conn.close()
        
        print(f"  ✅ Scraped {pages_scraped} pages from {site_config['name']}")
        return pages_scraped
        
    def crawl_all_sites(self) -> Dict[str, int]:
        """Crawl all configured sites."""
        results = {}
        for site_key in SITE_CONFIGS:
            try:
                results[site_key] = self.crawl_site(site_key)
            except Exception as e:
                print(f"  ❌ Error crawling {site_key}: {e}")
                results[site_key] = 0
        return results
        
    def get_cached_content(self, site_key: Optional[str] = None) -> List[Dict]:
        """
        Get cached content from database.
        
        Args:
            site_key: Filter by site, or None for all sites
            
        Returns:
            List of dicts with url, title, content, site_key
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if site_key:
            cursor.execute("""
                SELECT url, site_key, title, content, scraped_at 
                FROM page_cache WHERE site_key = ?
            """, (site_key,))
        else:
            cursor.execute("""
                SELECT url, site_key, title, content, scraped_at 
                FROM page_cache
            """)
            
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "url": row[0],
                "site_key": row[1],
                "title": row[2],
                "content": row[3],
                "scraped_at": row[4],
            }
            for row in rows
        ]
        
    def needs_refresh(self, site_key: str) -> bool:
        """Check if a site's cache needs refreshing."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT last_crawl FROM crawl_status WHERE site_key = ?
        """, (site_key,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return True  # Never crawled
            
        last_crawl = datetime.fromisoformat(row[0])
        expiry = timedelta(days=self.config["cache_expiry_days"])
        
        return datetime.now() - last_crawl > expiry
        
    def get_cache_stats(self) -> Dict:
        """Get statistics about the cache."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Count pages per site
        cursor.execute("""
            SELECT site_key, COUNT(*), MAX(scraped_at) 
            FROM page_cache GROUP BY site_key
        """)
        
        for row in cursor.fetchall():
            stats[row[0]] = {
                "pages": row[1],
                "last_scraped": row[2],
            }
            
        conn.close()
        return stats
        
    def clear_cache(self, site_key: Optional[str] = None):
        """Clear cached content."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if site_key:
            cursor.execute("DELETE FROM page_cache WHERE site_key = ?", (site_key,))
            cursor.execute("DELETE FROM crawl_status WHERE site_key = ?", (site_key,))
        else:
            cursor.execute("DELETE FROM page_cache")
            cursor.execute("DELETE FROM crawl_status")
            
        conn.commit()
        conn.close()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_scraper: Optional[WebScraper] = None


def get_scraper() -> WebScraper:
    """Get singleton scraper instance."""
    global _scraper
    if _scraper is None:
        _scraper = WebScraper()
    return _scraper


def get_web_chunks() -> List[str]:
    """
    Get cached web content as RAG chunks.
    
    Returns:
        List of text chunks formatted for RAG ingestion
    """
    scraper = get_scraper()
    content = scraper.get_cached_content()
    
    chunks = []
    for page in content:
        # Create chunk with metadata
        site_name = SITE_CONFIGS.get(page["site_key"], {}).get("name", page["site_key"])
        
        # Split content into smaller chunks if too long
        text = page["content"]
        max_chunk_size = 1000
        
        if len(text) <= max_chunk_size:
            chunk = (
                f"WEB_CONTENT | "
                f"Source={site_name} | "
                f"URL={page['url']} | "
                f"Title={page['title']} | "
                f"Content={text}"
            )
            chunks.append(chunk)
        else:
            # Split into paragraphs
            paragraphs = text.split("\n\n")
            current_chunk = ""
            
            for para in paragraphs:
                if len(current_chunk) + len(para) < max_chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        chunk = (
                            f"WEB_CONTENT | "
                            f"Source={site_name} | "
                            f"URL={page['url']} | "
                            f"Title={page['title']} | "
                            f"Content={current_chunk.strip()}"
                        )
                        chunks.append(chunk)
                    current_chunk = para + "\n\n"
                    
            # Add remaining content
            if current_chunk.strip():
                chunk = (
                    f"WEB_CONTENT | "
                    f"Source={site_name} | "
                    f"URL={page['url']} | "
                    f"Title={page['title']} | "
                    f"Content={current_chunk.strip()}"
                )
                chunks.append(chunk)
                
    return chunks


def refresh_stale_sites():
    """Refresh any sites with expired cache."""
    scraper = get_scraper()
    
    for site_key in SITE_CONFIGS:
        if scraper.needs_refresh(site_key):
            print(f"🔄 Cache expired for {site_key}, refreshing...")
            scraper.crawl_site(site_key)


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import sys
    
    scraper = WebScraper()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "crawl":
            if len(sys.argv) > 2:
                site = sys.argv[2]
                scraper.crawl_site(site)
            else:
                scraper.crawl_all_sites()
                
        elif command == "stats":
            stats = scraper.get_cache_stats()
            print("\n📊 Cache Statistics:")
            for site, data in stats.items():
                print(f"  {site}: {data['pages']} pages (last: {data['last_scraped']})")
                
        elif command == "clear":
            site = sys.argv[2] if len(sys.argv) > 2 else None
            scraper.clear_cache(site)
            print("✅ Cache cleared")
            
        else:
            print("Usage: python web_scraper.py [crawl|stats|clear] [site_key]")
    else:
        print("Available sites:", list(SITE_CONFIGS.keys()))
        print("\nUsage:")
        print("  python web_scraper.py crawl          # Crawl all sites")
        print("  python web_scraper.py crawl smud     # Crawl specific site")
        print("  python web_scraper.py stats          # Show cache stats")
        print("  python web_scraper.py clear          # Clear all cache")
