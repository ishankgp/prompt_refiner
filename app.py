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

# Setup logging
def setup_logging():
    """Setup comprehensive logging for analysis"""
    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    # Create a unique log file for each run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/prompt_refiner_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"=== PROMPT REFINER SESSION STARTED ===")
    logger.info(f"Log file: {log_filename}")
    logger.info(f"OpenAI API Key configured: {'Yes' if os.environ.get('OPENAI_API_KEY') else 'No'}")
    
    return logger

logger = setup_logging()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'prompt_refiner_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

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
    
    def log_and_emit(message, emoji='📝', level='info'):
        """Helper function to log and emit simultaneously"""
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
        
        print(f"{emoji} {message}")
    
    log_and_emit("Starting refinement process via WebSocket", '🚀')
    
    if not os.environ.get("OPENAI_API_KEY"):
        log_and_emit('Missing OPENAI_API_KEY. Set it in your .env file.', '❌', 'error')
        return
        
    initial_prompt = data.get('prompt')
    attachments = data.get('attachments') or ""
    model = data.get('model', 'gpt-5')
    
    log_and_emit(f"Session ID: {session_id}", '🆔')
    log_and_emit(f"Starting refinement with {model}", '🚀')
    log_and_emit(f"Initial prompt: {initial_prompt[:100]}{'...' if len(initial_prompt) > 100 else ''}", '📝')
    
    try:
        temperature = float(data.get('temperature', 1.0))
        max_tokens = int(data.get('max_tokens', 6000))
        max_iterations = int(data.get('max_iterations', 5))
        iterate_until_satisfied = data.get('iterate_until_satisfied', False)
    except (TypeError, ValueError):
        temperature, max_tokens, max_iterations = 1.0, 6000, 5
        iterate_until_satisfied = False
        log_and_emit(f"Invalid parameters provided, using defaults", '⚠️')
        
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
    
    evaluation_criteria_override = (data.get('evaluation_criteria') or '').strip()
    
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
                api_params = get_model_params(model, temperature, max_tokens)
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a precise assistant that crafts prompt evaluation checklists."},
                        {"role": "user", "content": criteria_generation}
                    ],
                    **api_params
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
        
        # 1. Critique using loaded evaluation criteria (criteria already loaded before loop)
        log_and_emit("Analyzing current prompt...", '🔍')
        
        critique_prompt = f"""
        Using the following evaluation criteria, critique the user's prompt and list concrete improvements.
        If no improvements are necessary, respond with a single line containing exactly: SATISFIED

        Evaluation Criteria:
        "{evaluation_criteria}"

        Initial Prompt:
        "{current_prompt}"

        Attachments:
        "{attachments}"

        Provide the critique:
        """

        try:
            log_and_emit("Calling OpenAI API for analysis...", '💬')
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that critiques prompts."},
                    {"role": "user", "content": critique_prompt}
                ]
            )
            critique = response.choices[0].message.content.strip()
            log_and_emit(f"Analysis complete: {critique[:100]}...", '✅')
        except Exception as e:
            log_and_emit(f"Error generating critique: {e}", '❌', 'error')
            return

        history.append({
            'prompt': current_prompt,
            'evaluation_criteria': evaluation_criteria,
            'critique': critique
        })

        iterations_done = i + 1
        if 'SATISFIED' in critique:
            satisfied = True
            log_and_emit("Prompt is now satisfactory!", '🎉')
            break
        elif iterate_until_satisfied and i == max_iterations - 1:
            log_and_emit(f"Reached safety limit ({max_iterations} iterations) while iterating until satisfied", '⚠️')
        elif not iterate_until_satisfied and i == max_iterations - 1:
            # This will be the last iteration for fixed max_iterations mode
            pass

        # 3. Refine
        log_and_emit("Refining prompt based on feedback...", '🔧')
        
        refinement_prompt = f"""
        Refine the user's prompt based on the critique below. Keep the intent but improve clarity, specificity, constraints, and output formatting.
        Return only the refined prompt, no commentary.

        Critique:
        "{critique}"

        Initial Prompt:
        "{current_prompt}"

        Attachments:
        "{attachments}"

        Refined Prompt:
        """

        try:
            log_and_emit("Calling OpenAI API for refinement...", '💬')
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that refines prompts."},
                    {"role": "user", "content": refinement_prompt}
                ]
            )
            current_prompt = response.choices[0].message.content.strip()
            log_and_emit(f"Iteration {iteration_num} complete", '✨')
        except Exception as e:
            log_and_emit(f"Error refining prompt: {e}", '❌', 'error')
            return

    if not satisfied:
        if iterate_until_satisfied:
            log_and_emit(f"Reached safety limit ({max_iterations} iterations) without satisfaction", '⏹️')
        else:
            log_and_emit(f"Reached max iterations ({max_iterations})", '⏹️')

    # Save to file
    metadata = {
        'model': model,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'iterations': iterations_done,
        'satisfied': satisfied,
        'iterate_until_satisfied': iterate_until_satisfied,
        'session_id': session_id
    }
    
    filename = save_refined_prompt(initial_prompt, current_prompt, history, metadata, session_logs)
    
    if filename:
        log_and_emit(f"Saved to file: {filename}", '💾')
    
    # Send final result
    result = {
        'refined_prompt': current_prompt,
        'history': history,
        'iterations': iterations_done,
        'satisfied': satisfied,
        'model': model,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'saved_file': filename,
        'session_id': session_id
    }
    
    emit('complete', result)
    log_and_emit(f"Refinement complete: {iterations_done} iterations, satisfied={satisfied}", '🏁')
    logger.info(f"[{session_id}] FINAL REFINED PROMPT:\n{'='*50}\n{current_prompt}\n{'='*50}")

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5001, host='127.0.0.1')
