#!/usr/bin/env python3
"""Script to add cycle scanner endpoint to API server"""

# Read the file
with open('scripts/api_server.py', 'r') as f:
    lines = f.readlines()

# Find insertion point (before market_sentiment endpoint)
insert_idx = None
for i, line in enumerate(lines):
    if '@app.route(\'/api/market_sentiment\')' in line:
        insert_idx = i
        break

if insert_idx is None:
    print('Error: Could not find insertion point')
    exit(1)

# New endpoint code
new_code = '''@app.route('/api/cycle_scanner')
def get_cycle_scanner():
    """
    Get Cycle Scanner results (peaks and bottoms using Ehlers methodology)
    Query params:
        - filter: all (default), peak, bottom, approaching_peak, approaching_bottom, buy, sell
        - limit: number of results (default 20)
    """
    filter_type = request.args.get('filter', 'all')
    limit = int(request.args.get('limit', 20))
    
    try:
        import json
        results_file = Path(__file__).parent.parent / 'data' / 'cycle_signals.json'
        
        if not results_file.exists():
            return jsonify({
                'success': False,
                'error': 'No cycle scanner data available',
                'data': []
            })
        
        with open(results_file, 'r') as f:
            scanner_data = json.load(f)
        
        # Filter data based on filter_type
        if filter_type == 'all':
            data = (scanner_data.get('peak', []) + 
                   scanner_data.get('bottom', []) + 
                   scanner_data.get('approaching_peak', []) + 
                   scanner_data.get('approaching_bottom', []))
        elif filter_type == 'buy':
            data = scanner_data.get('bottom', []) + scanner_data.get('approaching_bottom', [])
        elif filter_type == 'sell':
            data = scanner_data.get('peak', []) + scanner_data.get('approaching_peak', [])
        elif filter_type in ['peak', 'bottom', 'approaching_peak', 'approaching_bottom']:
            data = scanner_data.get(filter_type, [])
        else:
            data = []
        
        # Apply limit
        data = data[:limit] if limit > 0 else data
        
        return jsonify({
            'success': True,
            'filter': filter_type,
            'count': len(data),
            'scan_time': scanner_data.get('metadata', {}).get('scan_time'),
            'data': data
        })
    
    except Exception as e:
        logger.error(f'Error loading cycle scanner data: {e}')
        return jsonify({
            'success': False,
            'error': str(e),
            'data': []
        })

'''

# Insert new code
lines.insert(insert_idx, new_code + '\n')

# Write back
with open('scripts/api_server.py', 'w') as f:
    f.writelines(lines)

print(f'✅ Successfully added cycle scanner endpoint at line {insert_idx}')
