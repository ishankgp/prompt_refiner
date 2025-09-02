import os
import logging
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from openai import OpenAI
from dotenv import load_dotenv
import json
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

def load_default_review_prompt():
    """Load the default review prompt from prompt_refinement.md"""
    try:
        with open('prompt_refinement.md', 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info("Default review prompt (prompt_refinement.md) loaded successfully")
        return content
    except FileNotFoundError:
        logger.warning("prompt_refinement.md not found, will use fallback review prompt")
        return ""
    except Exception as e:
        logger.error(f"Error loading prompt_refinement.md: {e}")
        return ""

def llm_score_refinement(current_prompt, critique, refinement_history, attachments, model, client):
    """
    Use LLM to score the current refinement iteration across multiple dimensions
    Returns satisfaction decision and scoring details
    """
    
    # Calculate score deltas from history
    previous_scores = []
    if len(refinement_history) >= 2:
        for item in refinement_history[-3:]:  # Look at last 3 iterations
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

Your task: Score this refinement on a 0-10 scale across these dimensions:

1. CONTENT_ACCURACY (0-10): How well does the prompt reflect source content? Any factual errors or hallucinations?
2. COMPLETENESS (0-10): Covers all essential elements? Missing critical pharma details?
3. AUDIO_CLARITY (0-10): Professional tone, clear signposting, proper audio flow for HCP listeners?
4. PHARMA_COMPLIANCE (0-10): Appropriate for healthcare professionals? Regulatory considerations?
5. REFINEMENT_QUALITY (0-10): How effective was the latest refinement iteration?
6. DIMINISHING_RETURNS (0-10): Are recent changes becoming minimal? (10 = very small improvements)

Strategic Assessment:
7. DIRECTIONAL_CORRECTNESS (0-10): Is the refinement approach heading in the right direction?
8. OVERALL_READINESS (0-10): Ready for production use by healthcare professionals?

Decision Points:
- STOP_REFINEMENT: YES/NO (should we stop refining?)
- CHANGE_APPROACH: YES/NO (should we try a different refinement strategy?)
- CONFIDENCE: 0-10 (how confident are you in this assessment?)

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
REASONING: [2-3 sentences explaining the key factors in your decision]
"""

    try:
        api_params = get_model_params(model, 0.3, 1500)  # Low temperature for consistent scoring
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": scoring_prompt}],
            **api_params
        )
        
        evaluation_text = response.choices[0].message.content
        return parse_llm_scores(evaluation_text, previous_scores)
        
    except Exception as e:
        logger.error(f"Error in LLM scoring: {e}")
        # Fallback to simple satisfaction check
        return {
            'satisfied': "SATISFIED" in critique.upper(),
            'scores': {},
            'stop_refinement': False,
            'change_approach': False,
            'reasoning': f"LLM scoring failed: {e}, used fallback",
            'confidence': 5.0
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
    
    # Calculate satisfaction based on multiple criteria
    overall_readiness = scores.get('overall_readiness', 0)
    confidence = scores.get('confidence', 0)
    diminishing_returns = scores.get('diminishing_returns', 0)
    directional_correctness = scores.get('directional_correctness', 0)
    
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
        (improvement_stagnant and overall_readiness >= 7.0) or
        (directional_correctness < 6.0 and scores.get('change_approach', False))
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
    and task requirements
    """
    # Model context limits (in characters, roughly 4 chars per token)
    model_limits = {
        'gpt-5': 500000,      # ~125k tokens
        'gpt-4o': 500000,     # ~125k tokens  
        'gpt-4o-mini': 500000, # ~125k tokens
        'gpt-4-turbo': 500000, # ~125k tokens
        'gpt-4': 30000,       # ~8k tokens
        'gpt-3.5-turbo': 60000, # ~16k tokens
        'o1-preview': 500000,
        'o1-mini': 500000
    }
    
    # Reserve space for prompt structure, response, etc.
    overhead_chars = {
        'critique': 8000,     # Room for prompt structure + response
        'refinement': 12000,  # More room for detailed refinement 
        'scoring': 6000,      # Less room needed for scoring
        'general': 10000
    }
    
    base_limit = model_limits.get(model, 30000)  # Default to conservative limit
    overhead = overhead_chars.get(task_type, 10000)
    available_chars = base_limit - overhead
    
    if len(attachments) <= available_chars:
        logger.info(f"Using full attachment content ({len(attachments)} chars) for {task_type}")
        return attachments
    else:
        logger.warning(f"Truncating attachment from {len(attachments)} to {available_chars} chars for {model}/{task_type}")
        return attachments[:available_chars]

app = Flask(__name__)
app.config['SECRET_KEY'] = 'prompt_refiner_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize OpenAI client with error handling
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment variables")
        client = None
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    client = None

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
    default_review_prompt = load_default_review_prompt()
    return render_template('index.html', 
                         default_attachment=default_attachment,
                         default_review_prompt=default_review_prompt,
                         has_default_review_prompt=bool(default_review_prompt))

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
    
    if not client:
        log_and_emit('OpenAI client not initialized. Check OPENAI_API_KEY in .env file.', '❌', 'error')
        return
    
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
    
    review_prompt_override = (data.get('review_prompt') or '').strip()
    
    if not initial_prompt or not initial_prompt.strip():
        log_and_emit('Prompt is required and cannot be empty', '❌', 'error')
        return
        
    # Sanitize inputs
    initial_prompt = initial_prompt.strip()
    attachments = (attachments or "").strip()
    
    current_prompt = initial_prompt
    history = []
    iterations_done = 0
    satisfied = False
    
    # Load review prompt ONCE at the beginning (not in every iteration)
    if review_prompt_override:
        log_and_emit("Using your custom review prompt", '📋')
        review_prompt = review_prompt_override
    else:
        log_and_emit("Loading default pharma/audio expert review prompt...", '🎯')
        default_review = load_default_review_prompt()
        
        if default_review:
            review_prompt = default_review
            log_and_emit("Using prompt_refinement.md as review criteria", '📋')
        else:
            log_and_emit("Generating fallback review criteria...", '🎯')
            
            review_prompt_generation = f"""
            Given the following user prompt, generate a concise "review prompt" to critique and identify gaps in the prompt.
            Focus on clarity, completeness, missing constraints, target audience, format/structure, and edge cases.

            Initial Prompt:
            "{current_prompt}"

            Attachments:
            "{get_smart_content_limit(model, attachments, "general") if attachments else "No attachments provided"}"

            Output only the review prompt text, no preface or explanation.
            """

            try:
                log_and_emit("Calling OpenAI API for review criteria...", '💬')
                api_params = get_model_params(model, temperature, max_tokens)
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a precise assistant that crafts prompt review checklists."},
                        {"role": "user", "content": review_prompt_generation}
                    ],
                    **api_params
                )
                review_prompt = response.choices[0].message.content.strip()
                log_and_emit(f"Review criteria generated: {review_prompt[:100]}...", '✅')
            except Exception as e:
                log_and_emit(f"Error generating review prompt: {e}", '❌', 'error')
                return

    for i in range(max_iterations):
        iteration_num = i + 1
        if iterate_until_satisfied:
            log_and_emit(f"Starting iteration {iteration_num} (iterating until satisfied)", '🔄')
        else:
            log_and_emit(f"Starting iteration {iteration_num}/{max_iterations}", '🔄')
        
        # 1. Critique and LLM Scoring (review_prompt already loaded)
        log_and_emit("Analyzing current prompt with LLM scoring...", '🔍')
        
        critique_prompt = f"""
        You are evaluating a SET OF INSTRUCTIONS (a prompt) that tells someone how to create content, NOT evaluating the content itself.

        REVIEW CRITERIA:
        {review_prompt}

        INSTRUCTIONS/PROMPT TO EVALUATE:
        {current_prompt}

        SOURCE CONTENT FOR CONTEXT:
        {get_smart_content_limit(model, attachments, "critique") if attachments else "No attachments provided"}

        Your task: Critique these INSTRUCTIONS. Are they clear enough? Do they specify what content should be included? Are there missing requirements? Do they provide enough guidance for someone to follow them effectively?

        Focus your critique on the quality and completeness of the INSTRUCTIONS themselves, not on any content that might be created by following those instructions.

        Provide detailed critique and specific suggestions for improving the INSTRUCTIONS:
        """

        try:
            log_and_emit("Calling OpenAI API for critique analysis...", '💬')
            api_params = get_model_params(model, temperature, max_tokens)
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a pharma and audio narration expert providing detailed critiques."},
                    {"role": "user", "content": critique_prompt}
                ],
                **api_params
            )
            critique = response.choices[0].message.content.strip()
            log_and_emit(f"Critique generated: {critique[:100]}...", '✅')
        except Exception as e:
            log_and_emit(f"Error generating critique: {e}", '❌', 'error')
            return

        # LLM-based satisfaction scoring
        log_and_emit("🧠 Evaluating satisfaction with LLM scoring...", '🎯')
        
        satisfaction_result = llm_score_refinement(
            current_prompt, critique, history, attachments, model, client
        )
        
        # Store LLM scores in history
        history_item = {
            'prompt': current_prompt,
            'review_prompt': review_prompt,
            'critique': critique,
            'llm_scores': satisfaction_result['scores'],
            'iteration': iteration_num
        }
        history.append(history_item)

        # Display scoring results
        scores = satisfaction_result['scores']
        log_and_emit(f"📊 Overall Readiness: {satisfaction_result['overall_readiness']:.1f}/10 | Confidence: {satisfaction_result['confidence']:.1f}/10", '�')
        log_and_emit(f"📈 Content: {scores.get('content_accuracy', 0):.1f} | Complete: {scores.get('completeness', 0):.1f} | Audio: {scores.get('audio_clarity', 0):.1f} | Pharma: {scores.get('pharma_compliance', 0):.1f}", '📊')
        
        # Check for approach change recommendation
        if satisfaction_result['change_approach']:
            log_and_emit("🔄 LLM recommends changing refinement approach", '⚠️')
            log_and_emit(f"💡 Reasoning: {satisfaction_result['reasoning']}", '🧠')
            break
            
        # Satisfaction decision
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

        # 3. Refine prompt based on critique
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
            api_params = get_model_params(model, temperature, max_tokens)
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert at refining prompt instructions. You improve the INSTRUCTIONS themselves, you do not execute the instructions to create content. Always return improved instructions/prompts, never the content those instructions would generate."},
                    {"role": "user", "content": refinement_prompt}
                ],
                **api_params
            )
            current_prompt = response.choices[0].message.content.strip()
            log_and_emit(f"Iteration {iteration_num} refinement complete", '✨')
        except Exception as e:
            log_and_emit(f"Error refining prompt: {e}", '❌', 'error')
            return

    # Final result processing
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
        'used_default_review_prompt': not bool(review_prompt_override)
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
