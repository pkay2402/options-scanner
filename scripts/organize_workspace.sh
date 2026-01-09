#!/bin/bash
# Organize workspace files into proper folders

cd /Users/piyushkhaitan/schwab/options

# Create directory structure
mkdir -p docs/guides docs/deployment docs/scanners archive/old-scripts archive/html-reports misc

# Move guide/documentation MD files
mv *GUIDE.md docs/guides/ 2>/dev/null || true
mv *README.md docs/guides/ 2>/dev/null || true

# Move deployment docs
mv *DEPLOYMENT*.md docs/deployment/ 2>/dev/null || true
mv *SETUP*.md docs/deployment/ 2>/dev/null || true

# Move scanner docs
mv *SCANNER*.md docs/scanners/ 2>/dev/null || true

# Move comparison, summary, architecture docs
mv *COMPARISON.md docs/guides/ 2>/dev/null || true
mv *SUMMARY.md docs/guides/ 2>/dev/null || true
mv *ARCHITECTURE.md docs/guides/ 2>/dev/null || true
mv *INDEX.md docs/guides/ 2>/dev/null || true
mv *PACKAGE.md docs/guides/ 2>/dev/null || true
mv *QUICK_REF.md docs/guides/ 2>/dev/null || true

# Move other documentation
mv *ANALYSIS.md docs/guides/ 2>/dev/null || true
mv *OPTIMIZATION.md docs/guides/ 2>/dev/null || true
mv *MIGRATION.md docs/guides/ 2>/dev/null || true
mv *REDESIGN*.md docs/guides/ 2>/dev/null || true
mv *CHECKLIST.md docs/guides/ 2>/dev/null || true
mv *DIAGRAM.md docs/guides/ 2>/dev/null || true
mv *MOCKUP.md docs/guides/ 2>/dev/null || true
mv *MANAGEMENT.md docs/guides/ 2>/dev/null || true
mv *AUTOMATION.md docs/guides/ 2>/dev/null || true
mv *EXPLAINED.md docs/guides/ 2>/dev/null || true

# Move HTML reports
mv *.html archive/html-reports/ 2>/dev/null || true

# Move test files
mv test_*.py tests/ 2>/dev/null || true
mv debug_*.py tests/ 2>/dev/null || true

# Move old scanner scripts (but keep scripts folder intact)
mv boundary_scanner.py archive/old-scripts/ 2>/dev/null || true
mv flow_scanner.py archive/old-scripts/ 2>/dev/null || true
mv max_gamma_scanner.py archive/old-scripts/ 2>/dev/null || true
mv opportunity_scanner.py archive/old-scripts/ 2>/dev/null || true
mv newsletter_generator.py archive/old-scripts/ 2>/dev/null || true
mv generate_html_report.py archive/old-scripts/ 2>/dev/null || true

# Move misc files
mv *.txt misc/ 2>/dev/null || true
mv *.rtf misc/ 2>/dev/null || true
mv *.zip misc/ 2>/dev/null || true

# Move backup files
mv *.backup archive/old-scripts/ 2>/dev/null || true

# Move standalone dashboard files
mv immediate_dashboard.py archive/old-scripts/ 2>/dev/null || true
mv index_positioning.py archive/old-scripts/ 2>/dev/null || true
mv Main_Dashboard.py.backup archive/old-scripts/ 2>/dev/null || true

# Move shell scripts to scripts folder
mv *.sh scripts/ 2>/dev/null || true
mv launch_*.sh scripts/ 2>/dev/null || true

echo "✓ Workspace organized!"
echo ""
echo "New structure:"
echo "  docs/guides/          - All documentation"
echo "  docs/deployment/      - Deployment guides"
echo "  docs/scanners/        - Scanner documentation"
echo "  archive/old-scripts/  - Deprecated scripts"
echo "  archive/html-reports/ - Old HTML reports"
echo "  misc/                 - Text files and misc"
echo "  tests/                - Test scripts"
