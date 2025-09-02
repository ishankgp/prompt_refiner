import os
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from openai import OpenAI
from dotenv import load_dotenv
import json
import time
from datetime import datetime

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'prompt_refiner_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def save_refined_prompt(original_prompt, refined_prompt, history, metadata):
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
            "refinement_history": history
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved refined prompt to: {filename}")
        return filename
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('🔌 Client connected')

@socketio.on('disconnect')  
def handle_disconnect():
    print('🔌 Client disconnected')

@socketio.on('start_refinement')
def handle_refinement(data):
    """WebSocket handler for real-time refinement streaming"""
    print(f"🚀 Starting refinement process via WebSocket...")
    
    if not os.environ.get("OPENAI_API_KEY"):
        emit('error', {'message': 'Missing OPENAI_API_KEY. Set it in your .env file.'})
        return
        
    initial_prompt = data.get('prompt')
    attachments = data.get('attachments') or ""
    model = data.get('model', 'gpt-4o')
    
    emit('progress', {'message': f'🚀 Starting refinement with {model}...', 'emoji': '🚀'})
    print(f"📝 Initial prompt: {initial_prompt[:100]}{'...' if len(initial_prompt) > 100 else ''}")
    
    try:
        temperature = float(data.get('temperature', 1.0))
        max_tokens = int(data.get('max_tokens', 6000))
        max_iterations = int(data.get('max_iterations', 5))
    except (TypeError, ValueError):
        temperature, max_tokens, max_iterations = 1.0, 6000, 5
        
    # Clamp ranges
    temperature = max(0.0, min(2.0, temperature))
    max_iterations = max(1, min(10, max_iterations))
    
    review_prompt_override = (data.get('review_prompt') or '').strip()
    
    if not initial_prompt:
        emit('error', {'message': 'Prompt is required'})
        return
        
    current_prompt = initial_prompt
    history = []
    iterations_done = 0
    satisfied = False

    for i in range(max_iterations):
        iteration_num = i + 1
        emit('progress', {'message': f'🔄 Starting iteration {iteration_num}/{max_iterations}', 'emoji': '🔄'})
        print(f"🔄 Iteration {iteration_num}/{max_iterations}")
        
        # 1. Get review prompt
        if review_prompt_override:
            emit('progress', {'message': '📋 Using your custom review prompt', 'emoji': '📋'})
            print("📋 Using custom review prompt")
            review_prompt = review_prompt_override
        else:
            emit('progress', {'message': '🎯 Generating review criteria...', 'emoji': '🎯'})
            print("🎯 Generating review criteria...")
            
            review_prompt_generation = f"""
            Given the following user prompt, generate a concise "review prompt" to critique and identify gaps in the prompt.
            Focus on clarity, completeness, missing constraints, target audience, format/structure, and edge cases.

            Initial Prompt:
            "{current_prompt}"

            Attachments:
            "{attachments}"

            Output only the review prompt text, no preface or explanation.
            """

            try:
                emit('progress', {'message': '💬 Calling OpenAI API for review criteria...', 'emoji': '💬'})
                print("💬 Calling OpenAI API for review criteria...")
                response = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": "You are a precise assistant that crafts prompt review checklists."},
                        {"role": "user", "content": review_prompt_generation}
                    ]
                )
                review_prompt = response.choices[0].message.content.strip()
                emit('progress', {'message': '✅ Review criteria generated', 'emoji': '✅'})
                print(f"✅ Generated review prompt: {review_prompt[:100]}...")
            except Exception as e:
                emit('error', {'message': f'Error generating review prompt: {e}'})
                print(f"❌ Error generating review prompt: {e}")
                return

        # 2. Critique
        emit('progress', {'message': '🔍 Analyzing current prompt...', 'emoji': '🔍'})
        print("🔍 Analyzing current prompt...")
        
        critique_prompt = f"""
        Using the following review prompt, critique the user's prompt and list concrete improvements.
        If no improvements are necessary, respond with a single line containing exactly: SATISFIED

        Review Prompt:
        "{review_prompt}"

        Initial Prompt:
        "{current_prompt}"

        Attachments:
        "{attachments}"

        Provide the critique:
        """

        try:
            emit('progress', {'message': '💬 Calling OpenAI API for analysis...', 'emoji': '💬'})
            print("💬 Calling OpenAI API for analysis...")
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
            emit('progress', {'message': '✅ Analysis complete', 'emoji': '✅'})
            print(f"✅ Analysis complete: {critique[:100]}...")
        except Exception as e:
            emit('error', {'message': f'Error generating critique: {e}'})
            print(f"❌ Error generating critique: {e}")
            return

        history.append({
            'prompt': current_prompt,
            'review_prompt': review_prompt,
            'critique': critique
        })

        iterations_done = i + 1
        if 'SATISFIED' in critique:
            satisfied = True
            emit('progress', {'message': '🎉 Prompt is now satisfactory!', 'emoji': '🎉'})
            print("🎉 Prompt is now satisfactory!")
            break

        # 3. Refine
        emit('progress', {'message': '🔧 Refining prompt based on feedback...', 'emoji': '🔧'})
        print("🔧 Refining prompt based on feedback...")
        
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
            emit('progress', {'message': '💬 Calling OpenAI API for refinement...', 'emoji': '💬'})
            print("💬 Calling OpenAI API for refinement...")
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
            emit('progress', {'message': f'✨ Iteration {iteration_num} complete', 'emoji': '✨'})
            print(f"✨ Iteration {iteration_num} complete")
        except Exception as e:
            emit('error', {'message': f'Error refining prompt: {e}'})
            print(f"❌ Error refining prompt: {e}")
            return

    if not satisfied:
        emit('progress', {'message': f'⏹️ Reached max iterations ({max_iterations})', 'emoji': '⏹️'})
        print(f"⏹️ Reached max iterations ({max_iterations})")

    # Save to file
    metadata = {
        'model': model,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'iterations': iterations_done,
        'satisfied': satisfied
    }
    
    filename = save_refined_prompt(initial_prompt, current_prompt, history, metadata)
    
    if filename:
        emit('progress', {'message': f'💾 Saved to file: {filename}', 'emoji': '💾'})
    
    # Send final result
    result = {
        'refined_prompt': current_prompt,
        'history': history,
        'iterations': iterations_done,
        'satisfied': satisfied,
        'model': model,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'saved_file': filename
    }
    
    emit('complete', result)
    print(f"🏁 Refinement complete: {iterations_done} iterations, satisfied={satisfied}")
    print(f"📄 FINAL REFINED PROMPT:")
    print("=" * 50)
    print(current_prompt)
    print("=" * 50)

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5001, host='127.0.0.1')
