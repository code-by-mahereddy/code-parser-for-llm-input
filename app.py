from flask import Flask, render_template, request, jsonify
import os
import sys
import codeanalyser
import logging

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# WSGI app for production servers (Render, Heroku, etc.)
wsgi_app = app

@app.route('/')
def index():
    try:
        html_path = os.path.join(os.path.dirname(__file__), 'index.html')
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read(), 200, {'Content-Type': 'text/html; charset=utf-8'}
    except FileNotFoundError:
        logger.error("index.html not found")
        return jsonify({'error': 'Frontend not found'}), 404

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'message': 'API is running'})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Log the incoming request
        logger.info(f"Received request: {request.get_json()}")
        
        data = request.get_json()
        if not data:
            logger.error("No JSON data received")
            return jsonify({'error': 'No JSON data received'}), 400
            
        source_code = data.get('source_code', '')
        function_name = data.get('function_name', '')
        
        logger.info(f"Source code length: {len(source_code)}")
        logger.info(f"Function name: '{function_name}'")
        
        if not source_code:
            logger.error("No source code provided")
            return jsonify({'error': 'Source code is required'}), 400
            
        if not function_name:
            logger.error("No function name provided")
            return jsonify({'error': 'Function name is required'}), 400
        
        # Analyze the code
        logger.info("Starting analysis...")
        result = codeanalyser.generate_llm_input(source_code, function_name)
        
        logger.info("Analysis completed successfully")
        return jsonify({
            'success': True,
            'payload': result['payload'],
            'prompt': result['prompt'],
            'raw_json': result['raw_json']
        })
        
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except codeanalyser.AnalyzerConfig as e:
        logger.error(f"Configuration error: {str(e)}")
        return jsonify({'error': 'Configuration error'}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Analysis failed: Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    logger.info(f"Starting Flask app on port {port} (debug={debug})")
    app.run(host='0.0.0.0', port=port, debug=debug)