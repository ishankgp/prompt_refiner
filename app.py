import os
import logging
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from openai import OpenAI
from dotenv import load_dotenv
import json
import time
from datetime import datetime

load_dotenv()

def validate_environment():
    """Validate required environment variables are present"""
    required_vars = ['OPENAI_API_KEY']
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing required environment variables: {missing}")
    
    # Log successful validation (without exposing secrets)
    logging.info("Environment validation successful")

# Validate environment on startup
try:
    validate_environment()
except ValueError as e:
    logging.error(f"Environment validation failed: {e}")
    raise

class Config:
    """Application configuration class"""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    MAX_ITERATIONS = int(os.environ.get('MAX_ITERATIONS', '5'))
    OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'outputs')
    LOG_DIR = os.environ.get('LOG_DIR', 'logs')
    DEFAULT_MODEL = os.environ.get('DEFAULT_MODEL', 'gpt-4o')
    DEFAULT_TEMPERATURE = float(os.environ.get('DEFAULT_TEMPERATURE', '1.0'))
    DEFAULT_MAX_TOKENS = int(os.environ.get('DEFAULT_MAX_TOKENS', '16000'))
    
    @classmethod
    def validate(cls):
        """Validate configuration values"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        if not 0 <= cls.DEFAULT_TEMPERATURE <= 2:
            raise ValueError("DEFAULT_TEMPERATURE must be between 0 and 2")
        if cls.DEFAULT_MAX_TOKENS <= 0:
            raise ValueError("DEFAULT_MAX_TOKENS must be positive")

# Validate configuration
Config.validate()

def validate_refinement_params(data):
    """Validate refinement parameters"""
    required_fields = ['prompt']
    for field in required_fields:
        if not data.get(field):
            raise ValueError(f"Missing required field: {field}")
        if not isinstance(data[field], str) or len(data[field].strip()) == 0:
            raise ValueError(f"Field '{field}' must be a non-empty string")
    
    # Validate optional numeric parameters
    if 'temperature' in data:
        try:
            temp = float(data['temperature'])
            if not 0 <= temp <= 2:
                raise ValueError("Temperature must be between 0 and 2")
        except (ValueError, TypeError):
            raise ValueError("Temperature must be a valid number")
    
    if 'max_tokens' in data:
        try:
            tokens = int(data['max_tokens'])
            if tokens <= 0 or tokens > 100000:
                raise ValueError("max_tokens must be between 1 and 100000")
        except (ValueError, TypeError):
            raise ValueError("max_tokens must be a valid positive integer")
    
    if 'max_iterations' in data:
        try:
            iterations = int(data['max_iterations'])
            if iterations <= 0 or iterations > 20:
                raise ValueError("max_iterations must be between 1 and 20")
        except (ValueError, TypeError):
            raise ValueError("max_iterations must be a valid positive integer")

def sanitize_input(text):
    """Sanitize user input to prevent potential security issues"""
    if not isinstance(text, str):
        return str(text)
    
    # Remove potentially dangerous characters while preserving content
    # This is a basic sanitization - adjust based on your needs
    import re
    
    # Remove null bytes and control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Limit length to prevent memory issues
    max_length = 1000000  # 1MB of text
    if len(text) > max_length:
        text = text[:max_length]
        logging.warning(f"Input truncated to {max_length} characters")
    
    return text.strip()

def load_default_attachment():
    """Load the default attachment from webpage.txt"""
    try:
        with open('webpage.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info("Default attachment (webpage.txt) loaded successfully")
        return content
    except FileNotFoundError:
        logger.warning("webpage.txt not found, no default attachment loaded")
        return ""
    except Exception as e:
        logger.error(f"Error loading webpage.txt: {e}")
        return ""

def load_default_evaluation_criteria():
    """Load the default evaluation criteria from prompt_refinement.md"""
    try:
        with open('prompt_refinement.md', 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info("Default evaluation criteria (prompt_refinement.md) loaded successfully")
        return content
    except FileNotFoundError:
        logger.warning("prompt_refinement.md not found, will use fallback evaluation criteria")
        return ""
    except Exception as e:
        logger.error(f"Error loading prompt_refinement.md: {e}")
        return ""

def get_model_params(model, temperature, max_tokens):
    """Get appropriate parameters for different OpenAI models"""
    # GPT-5 and newer models use max_completion_tokens instead of max_tokens
    if model.startswith('gpt-5'):
        return {
            "model": model,
            "temperature": temperature,
            "max_completion_tokens": max_tokens
        }
    else:
        return {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

def get_smart_content_limit(model, attachments, task_type="general"):
    """
    Intelligently determine how much content to include based on model capabilities
    """
    model_limits = {
        'gpt-5': 500000,      # ~125k tokens
        'gpt-4o': 500000,     # ~125k tokens  
        'gpt-4o-mini': 500000, # ~125k tokens
        'gpt-4-turbo': 500000, # ~125k tokens
        'gpt-4': 30000,       # ~8k tokens
        'gpt-3.5-turbo': 60000, # ~16k tokens
    }
    
    overhead_chars = {
        'critique': 8000,     
        'refinement': 12000,  
        'scoring': 6000,      
        'general': 10000
    }
    
    base_limit = model_limits.get(model, 30000)
    overhead = overhead_chars.get(task_type, 10000)
    available_chars = base_limit - overhead
    
    if len(attachments) <= available_chars:
        logger.info(f"Using full attachment content ({len(attachments)} chars) for {task_type}")
        return attachments
    else:
        logger.warning(f"Truncating attachment from {len(attachments)} to {available_chars} chars for {model}/{task_type}")
        return attachments[:available_chars]

def llm_score_refinement(current_prompt, critique, refinement_history, attachments, model, client):
    """
    Use LLM to score the current refinement iteration across multiple dimensions
    """
    previous_scores = []
    if len(refinement_history) >= 2:
        for item in refinement_history[-3:]:
            if 'llm_scores' in item:
                previous_scores.append(item['llm_scores'].get('overall_readiness', 0))
    
    delta_context = ""
    if previous_scores:
        delta_context = f"\nRECENT READINESS SCORES: {previous_scores} (look for diminishing improvements)"
    
    scoring_prompt = f"""
You are evaluating a prompt refinement iteration. Analyze the current state and provide structured scoring.

SOURCE CONTENT (for reference):
{get_smart_content_limit(model, attachments, "scoring") if attachments else "No source content provided"}

CURRENT PROMPT BEING EVALUATED:
{current_prompt}

LATEST REFINEMENT CRITIQUE:
{critique}

REFINEMENT HISTORY: {len(refinement_history)} iterations completed{delta_context}

Score this refinement on a 0-10 scale across these dimensions:

1. CONTENT_ACCURACY (0-10): How well does the prompt reflect source content?
2. COMPLETENESS (0-10): Covers all essential elements?
3. AUDIO_CLARITY (0-10): Professional tone, clear signposting, proper audio flow?
4. PHARMA_COMPLIANCE (0-10): Appropriate for healthcare professionals?
5. REFINEMENT_QUALITY (0-10): How effective was the latest refinement iteration?
6. DIMINISHING_RETURNS (0-10): Are recent changes becoming minimal? (10 = very small improvements)

Strategic Assessment:
7. DIRECTIONAL_CORRECTNESS (0-10): Is the refinement approach heading in the right direction?
8. OVERALL_READINESS (0-10): Ready for production use?

Decision Points:
- STOP_REFINEMENT: YES/NO
- CHANGE_APPROACH: YES/NO  
- CONFIDENCE: 0-10

Respond in EXACTLY this format:
CONTENT_ACCURACY: [score]
COMPLETENESS: [score]
AUDIO_CLARITY: [score]  
PHARMA_COMPLIANCE: [score]
REFINEMENT_QUALITY: [score]
DIMINISHING_RETURNS: [score]
DIRECTIONAL_CORRECTNESS: [score]
OVERALL_READINESS: [score]
STOP_REFINEMENT: [YES/NO]
CHANGE_APPROACH: [YES/NO]
CONFIDENCE: [score]
REASONING: [2-3 sentences explaining key factors]
"""

    try:
        # Use temperature 1.0 for GPT-5 (only supported value), 0.3 for others
        scoring_temperature = 1.0 if model.startswith('gpt-5') else 0.3
        
        response = make_openai_request(
            messages=[{"role": "user", "content": scoring_prompt}],
            model=model,
            temperature=scoring_temperature,
            max_tokens=1500
        )
        
        evaluation_text = response.choices[0].message.content
        return parse_llm_scores(evaluation_text, previous_scores)
        
    except Exception as e:
        logger.error(f"Error in LLM scoring: {e}")
        # Force immediate flush
        for handler in logger.handlers:
            handler.flush()
        return {
            'satisfied': "SATISFIED" in critique.upper(),
            'scores': {},
            'stop_refinement': False,
            'change_approach': False,
            'reasoning': f"LLM scoring failed: {e}, used fallback",
            'confidence': 5.0,
            'overall_readiness': 5.0
        }

def parse_llm_scores(evaluation_text, previous_scores):
    """Parse structured LLM evaluation response"""
    scores = {}
    lines = evaluation_text.strip().split('\n')
    
    score_fields = [
        'CONTENT_ACCURACY', 'COMPLETENESS', 'AUDIO_CLARITY', 'PHARMA_COMPLIANCE',
        'REFINEMENT_QUALITY', 'DIMINISHING_RETURNS', 'DIRECTIONAL_CORRECTNESS', 
        'OVERALL_READINESS', 'CONFIDENCE'
    ]
    
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().upper()
            value = value.strip()
            
            if key in score_fields:
                try:
                    scores[key.lower()] = float(value)
                except:
                    scores[key.lower()] = 0.0
            elif key == 'STOP_REFINEMENT':
                scores['stop_refinement'] = value.upper() in ['YES', 'TRUE']
            elif key == 'CHANGE_APPROACH':
                scores['change_approach'] = value.upper() in ['YES', 'TRUE']
            elif key == 'REASONING':
                scores['reasoning'] = value
    
    # Calculate satisfaction
    overall_readiness = scores.get('overall_readiness', 0)
    confidence = scores.get('confidence', 0)
    diminishing_returns = scores.get('diminishing_returns', 0)
    
    # Check for score improvement stagnation
    improvement_stagnant = False
    if previous_scores and len(previous_scores) >= 2:
        recent_improvements = [previous_scores[i] - previous_scores[i-1] for i in range(1, len(previous_scores))]
        if all(improvement < 0.5 for improvement in recent_improvements):
            improvement_stagnant = True
    
    # Sophisticated satisfaction logic
    is_satisfied = (
        scores.get('stop_refinement', False) or
        (overall_readiness >= 8.5 and confidence >= 7.0) or
        (overall_readiness >= 7.5 and diminishing_returns >= 8.0) or
        (improvement_stagnant and overall_readiness >= 7.0)
    )
    
    return {
        'satisfied': is_satisfied,
        'scores': scores,
        'stop_refinement': scores.get('stop_refinement', False),
        'change_approach': scores.get('change_approach', False),
        'reasoning': scores.get('reasoning', 'No reasoning provided'),
        'confidence': confidence,
        'overall_readiness': overall_readiness,
        'improvement_stagnant': improvement_stagnant
    }

# Setup logging
def setup_app_logging():
    """Setup basic app logging (console only)"""
    console_handler = logging.StreamHandler()
    console_handler.flush = lambda: console_handler.stream.flush() if console_handler.stream else None
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        handlers=[console_handler],
        force=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=== PROMPT REFINER APPLICATION STARTED ===")
    logger.info(f"OpenAI API Key configured: {'Yes' if os.environ.get('OPENAI_API_KEY') else 'No'}")
    
    return logger

def setup_session_logging(session_id):
    """Setup session-specific logging for each refinement session"""
    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    # Create a unique log file for this session
    log_filename = f"logs/prompt_refiner_{session_id}.log"
    
    # Create file handler with immediate flushing
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.flush = lambda: file_handler.stream.flush() if file_handler.stream else None
    
    # Get the existing logger and add the file handler
    logger = logging.getLogger(__name__)
    
    # Remove any existing file handlers to avoid duplicate logs
    logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.FileHandler)]
    
    # Add the new session file handler
    file_handler.setLevel(logging.INFO)
    if hasattr(file_handler, 'stream'):
        file_handler.stream.reconfigure(line_buffering=True)
    logger.addHandler(file_handler)
    
    logger.info(f"=== PROMPT REFINER SESSION STARTED ===")
    logger.info(f"Session ID: {session_id}")
    logger.info(f"Log file: {log_filename}")
    
    # Force immediate write
    file_handler.flush()
    
    return log_filename

logger = setup_app_logging()

# Session storage for last run results
last_session_results = {
    'session_id': None,
    'refined_prompt': None,
    'history': [],
    'satisfied': False,
    'final_scores': {},
    'metadata': {},
    'timestamp': None
}

def save_session_results(session_data):
    """Save session results for frontend display"""
    global last_session_results
    last_session_results.update(session_data)
    last_session_results['timestamp'] = datetime.now().isoformat()
    logger.info(f"Session results saved for session: {session_data.get('session_id')}")

def load_most_recent_session():
    """Load the most recent session from saved files"""
    try:
        outputs_dir = "outputs"
        if not os.path.exists(outputs_dir):
            return None
            
        # Get all refined_prompt_*.json files
        files = [f for f in os.listdir(outputs_dir) if f.startswith("refined_prompt_") and f.endswith(".json")]
        if not files:
            return None
            
        # Sort by filename (timestamp) to get the most recent
        files.sort(reverse=True)
        most_recent = files[0]
        
        filepath = os.path.join(outputs_dir, most_recent)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Transform the file data to match the expected session format
        session_data = {
            'session_id': data.get('metadata', {}).get('session_id'),
            'refined_prompt': data.get('final_refined_prompt'),
            'final_prompt': data.get('final_refined_prompt'),
            'history': data.get('refinement_history', []),
            'satisfied': data.get('metadata', {}).get('satisfied', False),
            'final_scores': data.get('metadata', {}).get('final_scores', {}),
            'metadata': data.get('metadata', {}),
            'timestamp': data.get('timestamp'),
            'model': data.get('metadata', {}).get('model'),
            'temperature': data.get('metadata', {}).get('temperature'),
            'max_tokens': data.get('metadata', {}).get('max_tokens'),
            'iterations_completed': data.get('metadata', {}).get('iterations'),
            'total_tokens': data.get('metadata', {}).get('total_tokens'),
            'original_prompt': data.get('original_prompt')
        }
        
        logger.info(f"Loaded most recent session from: {most_recent}")
        return session_data
        
    except Exception as e:
        logger.error(f"Error loading most recent session: {e}")
        return None

def get_last_session_results():
    """Get the last session results"""
    # This function will now always try to load the most recent session from a file,
    # ensuring that the latest data is available even after a server restart.
    # The in-memory `last_session_results` will be updated by this.
    
    file_data = load_most_recent_session()
    if file_data:
        # Update the global in-memory session data as well
        global last_session_results
        last_session_results.update(file_data)
        return file_data
    
    # Return empty results if nothing found in files or memory
    return {
        'error': 'No session data found',
        'session_id': None,
        'refined_prompt': None,
        'history': [],
        'satisfied': False,
        'final_scores': {},
        'metadata': {},
        'timestamp': None
    }

app = Flask(__name__)
app.config['SECRET_KEY'] = 'prompt_refiner_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def make_openai_request(messages, model, temperature, max_tokens, timeout=60):
    """
    Make an OpenAI API request with proper error handling
    """
    import openai
    
    try:
        params = get_model_params(model, temperature, max_tokens)
        
        response = client.chat.completions.create(
            messages=messages,
            timeout=timeout,
            **params
        )
        return response
        
    except openai.RateLimitError as e:
        logger.error(f"OpenAI rate limit exceeded: {e}")
        raise ValueError("API rate limit exceeded. Please try again later.")
        
    except openai.AuthenticationError as e:
        logger.error(f"OpenAI authentication error: {e}")
        raise ValueError("Invalid API key. Please check your OPENAI_API_KEY.")
        
    except openai.APIConnectionError as e:
        logger.error(f"OpenAI connection error: {e}")
        raise ValueError("Unable to connect to OpenAI API. Please check your internet connection.")
        
    except openai.APITimeoutError as e:
        logger.error(f"OpenAI timeout error: {e}")
        raise ValueError("Request timed out. Please try again.")
        
    except openai.BadRequestError as e:
        logger.error(f"OpenAI bad request error: {e}")
        raise ValueError(f"Invalid request: {e}")
        
    except openai.APIStatusError as e:
        logger.error(f"OpenAI API status error: {e}")
        raise ValueError(f"API error (status {e.status_code}): {e}")
        
    except Exception as e:
        logger.error(f"Unexpected error in OpenAI request: {e}")
        raise ValueError(f"Unexpected error: {e}")

def save_refined_prompt(original_prompt, refined_prompt, history, metadata, logs=None):
    """Save the refined prompt to a file with timestamp"""
    try:
        if not os.path.exists("outputs"):
            os.makedirs("outputs")
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"refined_prompt_{timestamp}.json"
        filepath = os.path.join("outputs", filename)
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "original_prompt": original_prompt,
            "final_refined_prompt": refined_prompt,
            "metadata": metadata,
            "refinement_history": history,
            "session_logs": logs or []
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved refined prompt to: {filename}")
        print(f"💾 Saved refined prompt to: {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        print(f"❌ Error saving file: {e}")
        return None

@app.route('/')
def index():
    default_attachment = load_default_attachment()
    default_evaluation_criteria = load_default_evaluation_criteria()
    return render_template('index.html', 
                         default_attachment=default_attachment,
                         default_evaluation_criteria=default_evaluation_criteria,
                         has_default_criteria=bool(default_evaluation_criteria))

@app.route('/api/health')
def health_check():
    """Simple health check endpoint"""
    logger.info("Health check successful")
    return jsonify({"status": "ok"})

@app.route('/api/last-session')
def get_last_session():
    """API endpoint to get last session results"""
    logger.info("API call to /api/last-session received.")
    results = get_last_session_results()
    logger.info(f"Returning session data from /api/last-session: {results}")
    return jsonify(results)

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected via WebSocket')
    print('🔌 Client connected')

@socketio.on('disconnect')  
def handle_disconnect():
    logger.info('Client disconnected from WebSocket')
    print('🔌 Client disconnected')

@socketio.on('start_refinement')
def handle_refinement(data):
    """WebSocket handler for real-time refinement streaming"""
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session_logs = []
    
    # Setup session-specific logging
    log_filename = setup_session_logging(session_id)
    
    def log_and_emit(message, emoji='📝', level='info'):
        """Helper function to log and emit simultaneously with immediate flush"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'emoji': emoji,
            'level': level
        }
        session_logs.append(log_entry)
        
        if level == 'error':
            logger.error(f"[{session_id}] {message}")
            emit('error', {'message': message})
        else:
            logger.info(f"[{session_id}] {message}")
            emit('progress', {'message': f'{emoji} {message}', 'emoji': emoji})
        
        # Force immediate flush of all log handlers
        for handler in logger.handlers:
            handler.flush()
        
        print(f"{emoji} {message}")
        # Force stdout flush as well
        import sys
        sys.stdout.flush()
    
    log_and_emit("Starting refinement process via WebSocket", '🚀')
    
    # Validate environment
    if not os.environ.get("OPENAI_API_KEY"):
        log_and_emit('Missing OPENAI_API_KEY. Set it in your .env file.', '❌', 'error')
        return
    
    # Validate and sanitize input parameters
    try:
        validate_refinement_params(data)
        
        # Sanitize inputs
        prompt = sanitize_input(data.get('prompt', ''))
        criteria = sanitize_input(data.get('criteria', ''))
        attachments = data.get('attachments', '')
        if attachments:
            attachments = sanitize_input(attachments)
            
        log_and_emit("Input validation successful", '✅')
        
    except ValueError as e:
        log_and_emit(f'Input validation failed: {e}', '❌', 'error')
        return
    except Exception as e:
        log_and_emit(f'Unexpected validation error: {e}', '❌', 'error')
        return
        
    # Extract parameters with defaults from config
    try:
        initial_prompt = prompt  # Use sanitized version
        model = data.get('model', Config.DEFAULT_MODEL)
        temperature = float(data.get('temperature', Config.DEFAULT_TEMPERATURE))
        max_tokens = int(data.get('max_tokens', Config.DEFAULT_MAX_TOKENS))
        max_iterations = int(data.get('max_iterations', Config.MAX_ITERATIONS))
        iterate_until_satisfied = data.get('iterate_until_satisfied', False)
        evaluation_criteria_override = criteria if criteria else ''
    except (TypeError, ValueError) as e:
        log_and_emit(f"Invalid parameter types in request: {e}", '❌', 'error')
        return
        
    log_and_emit(f"Session ID: {session_id}", '🆔')
    log_and_emit(f"Starting refinement with {model}", '🚀')
    log_and_emit(f"Initial prompt: {initial_prompt[:100]}{'...' if len(initial_prompt) > 100 else ''}", '📝')
    
    # Clamp ranges
    temperature = max(0.0, min(2.0, temperature))
    if not iterate_until_satisfied:
        max_iterations = max(1, min(10, max_iterations))
    else:
        # Set a reasonable upper limit even for "iterate until satisfied"
        max_iterations = 25  # Safety limit to prevent runaway iterations
    
    if iterate_until_satisfied:
        log_and_emit(f"Parameters: temp={temperature}, tokens={max_tokens}, iterating until satisfied (max {max_iterations})", '⚙️')
    else:
        log_and_emit(f"Parameters: temp={temperature}, tokens={max_tokens}, max_iter={max_iterations}", '⚙️')
    
    if not initial_prompt:
        log_and_emit('Prompt is required', '❌', 'error')
        return
        
    current_prompt = initial_prompt
    history = []
    iterations_done = 0
    satisfied = False
    
    # Load evaluation criteria ONCE at the beginning (not in every iteration)
    if evaluation_criteria_override:
        log_and_emit("Using your custom evaluation criteria", '📋')
        evaluation_criteria = evaluation_criteria_override
    else:
        log_and_emit("Loading default pharma/audio expert evaluation criteria...", '🎯')
        default_criteria = load_default_evaluation_criteria()
        
        if default_criteria:
            evaluation_criteria = default_criteria
            log_and_emit("Using prompt_refinement.md as evaluation criteria", '📋')
        else:
            log_and_emit("Generating fallback evaluation criteria...", '🎯')
            
            criteria_generation = f"""
            Given the following user prompt, generate a concise "review prompt" to critique and identify gaps in the prompt.
            Focus on clarity, completeness, missing constraints, target audience, format/structure, and edge cases.

            Initial Prompt:
            "{current_prompt}"

            Attachments:
            "{attachments}"

            Output only the review prompt text, no preface or explanation.
            """

            try:
                log_and_emit("Calling OpenAI API for evaluation criteria...", '💬')
                
                response = make_openai_request(
                    messages=[
                        {"role": "system", "content": "You are a precise assistant that crafts prompt evaluation checklists."},
                        {"role": "user", "content": criteria_generation}
                    ],
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                evaluation_criteria = response.choices[0].message.content.strip()
                log_and_emit(f"Evaluation criteria generated: {evaluation_criteria[:100]}...", '✅')
            except Exception as e:
                log_and_emit(f"Error generating evaluation criteria: {e}", '❌', 'error')
                return

    for i in range(max_iterations):
        iteration_num = i + 1
        if iterate_until_satisfied:
            log_and_emit(f"Starting iteration {iteration_num} (iterating until satisfied)", '🔄')
        else:
            log_and_emit(f"Starting iteration {iteration_num}/{max_iterations}", '🔄')
        
        # 1. Critique using loaded evaluation criteria
        log_and_emit("Analyzing current prompt with LLM scoring...", '🔍')
        
        critique_prompt = f"""
        You are evaluating a SET OF INSTRUCTIONS (a prompt) that tells someone how to create content, NOT evaluating the content itself.

        EVALUATION CRITERIA:
        {evaluation_criteria}

        INSTRUCTIONS/PROMPT TO EVALUATE:
        {current_prompt}

        SOURCE CONTENT FOR CONTEXT:
        {get_smart_content_limit(model, attachments, "critique") if attachments else "No attachments provided"}

        Your task: Critique these INSTRUCTIONS. Are they clear enough? Do they specify what content should be included? Are there missing requirements? Do they provide enough guidance for someone to follow them effectively?

        Focus your critique on the quality and completeness of the INSTRUCTIONS themselves, not on any content that might be created by following those instructions.

        Provide detailed critique and specific suggestions for improving the INSTRUCTIONS:
        """

        try:
            log_and_emit("Calling OpenAI API for analysis...", '💬')
            
            response = make_openai_request(
                messages=[
                    {"role": "system", "content": "You are a pharma and audio narration expert providing detailed critiques."},
                    {"role": "user", "content": critique_prompt}
                ],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            critique = response.choices[0].message.content.strip()
            log_and_emit(f"Analysis complete: {critique[:100]}...", '✅')
        except Exception as e:
            log_and_emit(f"Error generating critique: {e}", '❌', 'error')
            return

        # LLM-based satisfaction scoring
        log_and_emit("🧠 Evaluating satisfaction with LLM scoring...", '🎯')
        
        satisfaction_result = llm_score_refinement(
            current_prompt, critique, history, attachments, model, client
        )

        # Store iteration with LLM scores
        history_item = {
            'prompt': current_prompt,
            'evaluation_criteria': evaluation_criteria,
            'critique': critique,
            'llm_scores': satisfaction_result['scores'],
            'iteration': iteration_num
        }
        history.append(history_item)

        # Display scoring results
        scores = satisfaction_result['scores']
        log_and_emit(f"📊 Overall Readiness: {satisfaction_result['overall_readiness']:.1f}/10 | Confidence: {satisfaction_result['confidence']:.1f}/10", '📊')
        log_and_emit(f"📈 Content: {scores.get('content_accuracy', 0):.1f} | Complete: {scores.get('completeness', 0):.1f} | Audio: {scores.get('audio_clarity', 0):.1f} | Pharma: {scores.get('pharma_compliance', 0):.1f}", '📊')
        
        # Check for approach change recommendation
        if satisfaction_result['change_approach']:
            log_and_emit("🔄 LLM recommends changing refinement approach", '⚠️')
            log_and_emit(f"💡 Reasoning: {satisfaction_result['reasoning']}", '🧠')
            break

        iterations_done = iteration_num
        if satisfaction_result['satisfied']:
            if satisfaction_result['stop_refinement']:
                log_and_emit("✅ LLM recommends stopping - quality criteria met!", '🎉')
            elif satisfaction_result['improvement_stagnant']:
                log_and_emit("🎯 Improvement stagnation detected - stopping refinement", '📈')
            else:
                log_and_emit("⭐ High quality threshold achieved!", '🎉')
            
            log_and_emit(f"💡 Final reasoning: {satisfaction_result['reasoning']}", '🧠')
            satisfied = True
            break
        elif iterate_until_satisfied and iteration_num == max_iterations:
            log_and_emit(f"Reached safety limit ({max_iterations} iterations) while iterating until satisfied", '⚠️')
        elif not iterate_until_satisfied and i == max_iterations - 1:
            # This will be the last iteration for fixed max_iterations mode
            pass

        # 2. Refine prompt based on critique
        log_and_emit("Refining prompt based on LLM feedback...", '🔧')
        
        refinement_prompt = f"""
        You are a prompt engineering expert. Your task is to REFINE THE INSTRUCTIONS/PROMPT below, NOT to execute them.

        The user has provided a PROMPT (set of instructions) that needs improvement. You must refine these INSTRUCTIONS to make them clearer and more effective, but DO NOT execute the instructions to create content.

        CRITIQUE AND FEEDBACK TO ADDRESS:
        {critique}

        CURRENT PROMPT INSTRUCTIONS TO REFINE:
        {current_prompt}

        CONTEXT FOR UNDERSTANDING THE PROMPT:
        {get_smart_content_limit(model, attachments, "refinement") if attachments else "No source content provided"}

        YOUR TASK: Improve the PROMPT INSTRUCTIONS above based on the critique. Make the instructions clearer, more specific, and more complete. Return ONLY the improved prompt instructions, not any content created by following those instructions.

        REFINED PROMPT INSTRUCTIONS:
        """

        try:
            log_and_emit("Calling OpenAI API for prompt refinement...", '💬')
            
            response = make_openai_request(
                messages=[
                    {"role": "system", "content": "You are an expert at refining prompt instructions. You improve the INSTRUCTIONS themselves, you do not execute the instructions to create content. Always return improved instructions/prompts, never the content those instructions would generate."},
                    {"role": "user", "content": refinement_prompt}
                ],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            current_prompt = response.choices[0].message.content.strip()
            log_and_emit(f"Iteration {iteration_num} refinement complete", '✨')
        except Exception as e:
            log_and_emit(f"Error refining prompt: {e}", '❌', 'error')
            return

    if not satisfied:
        if iterate_until_satisfied:
            log_and_emit(f"Reached safety limit ({max_iterations} iterations) without satisfaction", '⏹️')
        else:
            log_and_emit(f"Reached max iterations ({max_iterations})", '⏹️')

    log_and_emit(f"🏁 Refinement complete! Total iterations: {iterations_done}", '🎯')

    # Enhanced metadata with LLM scoring results
    final_scores = history[-1].get('llm_scores', {}) if history else {}
    metadata = {
        'model': model,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'iterations': iterations_done,
        'satisfied': satisfied,
        'iterate_until_satisfied': iterate_until_satisfied,
        'session_id': session_id,
        'final_scores': final_scores,
        'final_readiness': final_scores.get('overall_readiness', 0),
        'final_confidence': final_scores.get('confidence', 0),
        'used_default_criteria': not bool(evaluation_criteria_override)
    }
    
    filename = save_refined_prompt(initial_prompt, current_prompt, history, metadata, session_logs)
    
    if filename:
        log_and_emit(f"Saved to file: {filename}", '💾')
    
    # Save session results for frontend display
    session_results = {
        'session_id': session_id,
        'refined_prompt': current_prompt,
        'history': history,
        'satisfied': satisfied,
        'final_scores': final_scores,
        'metadata': metadata,
        'iterations': iterations_done,
        'model': model
    }
    save_session_results(session_results)
    
    # Send final result with scores
    result = {
        'refined_prompt': current_prompt,
        'final_prompt': current_prompt,  # Add for frontend compatibility
        'original_prompt': initial_prompt,  # Add original prompt
        'history': history,
        'iterations': iterations_done,
        'satisfied': satisfied,
        'model': model,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'saved_file': filename,
        'session_id': session_id,
        'final_scores': final_scores,
        'scores': final_scores,  # Add for frontend compatibility
        'final_readiness': final_scores.get('overall_readiness', 0),
        'final_confidence': final_scores.get('confidence', 0)
    }
    
    emit('complete', result)
    log_and_emit(f"Refinement complete: {iterations_done} iterations, satisfied={satisfied}", '🏁')
    logger.info(f"[{session_id}] FINAL REFINED PROMPT:\n{'='*50}\n{current_prompt}\n{'='*50}")
    
    # Force final flush and cleanup session logging
    for handler in logger.handlers:
        handler.flush()
        # Remove session-specific file handlers to prevent accumulation
        if isinstance(handler, logging.FileHandler) and handler.baseFilename.endswith(f'{session_id}.log'):
            logger.removeHandler(handler)
            handler.close()

if __name__ == '__main__':
    try:
        logger.info("Starting Flask-SocketIO server...")
        logger.info(f"Host: 0.0.0.0, Port: 5000, Debug: False")
        logger.info("Server configuration: Production mode with enhanced error handling")
        
        # Use Flask-SocketIO's run method instead of app.run for WebSocket support
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=False,
            use_reloader=False,
            allow_unsafe_werkzeug=True,
            log_output=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

if __name__ == '__main__':
    logger.info("Starting Flask-SocketIO server...")
    logger.info(f"Host: 0.0.0.0, Port: 5000, Debug: False")
    logger.info("Server configuration: Production mode with enhanced error handling")
    socketio.run(app, debug=False, port=5000, host='0.0.0.0')
