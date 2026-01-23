"""
Download Web Content Script
============================
Pre-downloads web content from city/utility websites to speed up app startup.
Similar to download_models.py - run this BEFORE starting the app.

Usage:
    python download_web_content.py
"""

import sys
import os

# Add the project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("\n🌐 Pre-downloading web content from city/utility websites...")
    print("This may take 5-10 minutes on first run.\n")
    
    try:
        from src.ranchocordova.web_scraper import WebScraper, SITE_CONFIGS
        
        scraper = WebScraper()
        
        # Check current cache status
        stats = scraper.get_cache_stats()
        total_cached = sum(s.get("pages", 0) for s in stats.values())
        
        if total_cached > 0:
            print(f"📦 Found existing cache with {total_cached} pages")
            
            # Check if any sites need refresh
            stale_sites = [site for site in SITE_CONFIGS if scraper.needs_refresh(site)]
            
            if not stale_sites:
                print("✅ All sites are up-to-date (cached within last 7 days)")
                print("   Use --force to re-download anyway")
                
                if "--force" not in sys.argv:
                    return
                    
            print(f"🔄 {len(stale_sites)} site(s) need refresh: {', '.join(stale_sites)}")
        else:
            print("📦 No cached content found - downloading all sites...")
            stale_sites = list(SITE_CONFIGS.keys())
        
        # Crawl each site
        total_pages = 0
        for site_key in stale_sites:
            site_name = SITE_CONFIGS[site_key]["name"]
            print(f"\n🔍 Crawling {site_name}...")
            
            try:
                pages = scraper.crawl_site(site_key)
                total_pages += pages
            except Exception as e:
                print(f"  ⚠️ Error crawling {site_key}: {e}")
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"✅ Web content download complete!")
        print(f"   Total pages scraped: {total_pages}")
        
        # Show cache stats
        stats = scraper.get_cache_stats()
        print(f"\n📊 Cache Summary:")
        for site_key, data in stats.items():
            site_name = SITE_CONFIGS.get(site_key, {}).get("name", site_key)
            print(f"   {site_name}: {data.get('pages', 0)} pages")
        
        print(f"{'='*50}\n")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure beautifulsoup4 is installed: pip install beautifulsoup4")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
