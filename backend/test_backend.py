#!/usr/bin/env python3
"""
Quick test script to verify backend components work
Run locally before deploying to droplet
"""

import sys
import requests
from datetime import datetime

def test_api_endpoints(base_url="http://localhost:8000"):
    """Test all API endpoints"""
    print("=" * 60)
    print("Testing Options Scanner API")
    print("=" * 60)
    
    tests = [
        ("Health Check", "/"),
        ("Top Opportunities", "/api/top-opportunities?limit=10"),
        ("Market Sentiment", "/api/market-sentiment"),
        ("Stock Analysis NVDA", "/api/stock/NVDA"),
        ("Available Expiries", "/api/expiries"),
        ("Available Symbols", "/api/symbols"),
        ("Signal Types", "/api/signal-types"),
        ("Scan Status", "/api/scan-status"),
    ]
    
    results = []
    
    for name, endpoint in tests:
        url = f"{base_url}{endpoint}"
        try:
            response = requests.get(url, timeout=10)
            status = "✅ PASS" if response.status_code == 200 else f"❌ FAIL ({response.status_code})"
            results.append((name, status, response.status_code))
            print(f"{status} - {name}")
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    print(f"  → Returned {len(data)} items")
                elif isinstance(data, dict):
                    print(f"  → Keys: {', '.join(list(data.keys())[:5])}")
        except requests.exceptions.ConnectionError:
            results.append((name, "❌ FAIL (Connection)", 0))
            print(f"❌ FAIL - {name} (API not running)")
        except Exception as e:
            results.append((name, f"❌ FAIL ({str(e)})", 0))
            print(f"❌ FAIL - {name}: {e}")
        print()
    
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for _, status, _ in results if "PASS" in status)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    print()
    
    return passed == total


def test_database_connection():
    """Test PostgreSQL connection"""
    print("=" * 60)
    print("Testing Database Connection")
    print("=" * 60)
    
    try:
        import psycopg2
        
        conn = psycopg2.connect(
            host="localhost",
            database="options_scanner",
            user="options_user",
            password="your_secure_password",
            port=5432
        )
        
        cur = conn.cursor()
        cur.execute("SELECT version();")
        version = cur.fetchone()
        print(f"✅ PostgreSQL Connected: {version[0][:50]}...")
        
        # Test tables exist
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = [row[0] for row in cur.fetchall()]
        print(f"✅ Tables found: {', '.join(tables)}")
        
        # Test views exist
        cur.execute("""
            SELECT table_name 
            FROM information_schema.views 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        views = [row[0] for row in cur.fetchall()]
        print(f"✅ Views found: {', '.join(views)}")
        
        cur.close()
        conn.close()
        
        print("✅ Database test PASSED")
        return True
        
    except ImportError:
        print("❌ psycopg2 not installed")
        print("   Install with: pip install psycopg2-binary")
        return False
    except Exception as e:
        print(f"❌ Database test FAILED: {e}")
        print("   Possible issues:")
        print("   - PostgreSQL not running")
        print("   - Database not created")
        print("   - Wrong credentials")
        print("   - Schema not initialized")
        return False


def test_scanner_components():
    """Test scanner components can be imported"""
    print("=" * 60)
    print("Testing Scanner Components")
    print("=" * 60)
    
    tests = [
        ("SchwabClient", "from src.api.schwab_client import SchwabClient"),
        ("Scanner Worker", "import backend.scanner_worker"),
        ("API Service", "import backend.api_service"),
    ]
    
    passed = 0
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"✅ {name} imported successfully")
            passed += 1
        except Exception as e:
            print(f"❌ {name} import failed: {e}")
    
    print()
    if passed == len(tests):
        print("✅ All components test PASSED")
        return True
    else:
        print(f"❌ Components test FAILED ({passed}/{len(tests)})")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "OPTIONS SCANNER BACKEND TEST SUITE" + " " * 14 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    results = {}
    
    # Test 1: Components
    results['components'] = test_scanner_components()
    print()
    
    # Test 2: Database
    results['database'] = test_database_connection()
    print()
    
    # Test 3: API
    print("Testing API (make sure it's running on localhost:8000)...")
    print("Start with: uvicorn backend.api_service:app --reload")
    input("Press Enter when API is running, or Ctrl+C to skip...")
    results['api'] = test_api_endpoints()
    
    # Final summary
    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name.title()}")
    
    print()
    all_passed = all(results.values())
    if all_passed:
        print("✅ ALL TESTS PASSED - Ready to deploy!")
    else:
        print("❌ SOME TESTS FAILED - Fix issues before deploying")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
