import os
from flask import Flask, render_template, request, jsonify, Response
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

@socketio.on('start_refinement')
def handle_refinement(data):
    """WebSocket handler for real-time refinement streaming"""
    if not os.environ.get("OPENAI_API_KEY"):
        emit('error', {'message': 'Missing OPENAI_API_KEY. Set it in your .env file.'})
        return
        
    initial_prompt = data.get('prompt')
    attachments = data.get('attachments') or ""
    model = data.get('model', 'gpt-4o')
    
    emit('progress', {'message': f'Starting refinement with {model}...', 'emoji': '🚀'})
    
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
        
    emit('progress', {'message': f'Initial prompt: "{initial_prompt[:100]}{"..." if len(initial_prompt) > 100 else ""}"', 'emoji': '📝'})
    
    current_prompt = initial_prompt
    history = []
    iterations_done = 0
    satisfied = False

    for i in range(max_iterations):
        iteration_num = i + 1
        emit('progress', {'message': f'Starting iteration {iteration_num}/{max_iterations}', 'emoji': '🔄'})
        
        # 1. Get review prompt
        if review_prompt_override:
            emit('progress', {'message': 'Using your custom review prompt', 'emoji': '📋'})
            review_prompt = review_prompt_override
        else:
            emit('progress', {'message': 'Generating review criteria...', 'emoji': '🎯'})
            
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
                emit('progress', {'message': 'Calling OpenAI API for review criteria...', 'emoji': '💬'})
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
                emit('progress', {'message': 'Review criteria generated ✅', 'emoji': '✅'})
            except Exception as e:
                emit('error', {'message': f'Error generating review prompt: {e}'})
                return

        # 2. Critique
        emit('progress', {'message': 'Analyzing current prompt...', 'emoji': '🔍'})
        
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
            emit('progress', {'message': 'Calling OpenAI API for analysis...', 'emoji': '💬'})
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
            emit('progress', {'message': 'Analysis complete ✅', 'emoji': '✅'})
        except Exception as e:
            emit('error', {'message': f'Error generating critique: {e}'})
            return

        history.append({
            'prompt': current_prompt,
            'review_prompt': review_prompt,
            'critique': critique
        })

        iterations_done = i + 1
        if 'SATISFIED' in critique:
            satisfied = True
            emit('progress', {'message': f'Prompt is now satisfactory! 🎉', 'emoji': '✅'})
            break

        # 3. Refine
        emit('progress', {'message': 'Refining prompt based on feedback...', 'emoji': '🔧'})
        
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
            emit('progress', {'message': 'Calling OpenAI API for refinement...', 'emoji': '💬'})
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
            emit('progress', {'message': f'Iteration {iteration_num} complete ✨', 'emoji': '✨'})
        except Exception as e:
            emit('error', {'message': f'Error refining prompt: {e}'})
            return

    if not satisfied:
        emit('progress', {'message': f'Reached max iterations ({max_iterations})', 'emoji': '⏹️'})

    # Save to file
    metadata = {
        'model': model,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'iterations': iterations_done,
        'satisfied': satisfied
    }
    
    filename = save_refined_prompt(initial_prompt, current_prompt, history, metadata)
    
    emit('progress', {'message': f'Saved to file: {filename}', 'emoji': '💾'})
    
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

@app.route('/')
def index():
    return render_template('index.html')

# REMOVED DUPLICATE HTTP STREAMING ENDPOINT - USING WEBSOCKET ONLY
            if review_prompt_override:
                yield f"data: {json.dumps({'type': 'progress', 'message': 'Using your custom review prompt', 'emoji': '📋'})}\n\n"
                review_prompt = review_prompt_override
            else:
                yield f"data: {json.dumps({'type': 'progress', 'message': 'Generating review criteria...', 'emoji': '🎯'})}\n\n"
                
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
                    yield f"data: {json.dumps({'type': 'progress', 'message': 'Review criteria generated ✅', 'emoji': '📝'})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Error generating review prompt: {e}'})}\n\n"
                    return

            # 2. Critique
            yield f"data: {json.dumps({'type': 'progress', 'message': 'Analyzing current prompt against criteria...', 'emoji': '🔍'})}\n\n"
            
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
                yield f"data: {json.dumps({'type': 'progress', 'message': 'Analysis complete ✅', 'emoji': '📋'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Error generating critique: {e}'})}\n\n"
                return

            history.append({
                'prompt': current_prompt,
                'review_prompt': review_prompt,
                'critique': critique
            })

            iterations_done = i + 1
            if 'SATISFIED' in critique:
                satisfied = True
                yield f"data: {json.dumps({'type': 'progress', 'message': f'Prompt is now satisfactory! 🎉', 'emoji': '✅'})}\n\n"
                break

            # 3. Refine
            yield f"data: {json.dumps({'type': 'progress', 'message': 'Refining prompt based on feedback...', 'emoji': '🔧'})}\n\n"
            
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
                yield f"data: {json.dumps({'type': 'progress', 'message': f'Iteration {iteration_num} complete ✨', 'emoji': '💫'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Error refining prompt: {e}'})}\n\n"
                return

        if not satisfied:
            yield f"data: {json.dumps({'type': 'progress', 'message': f'Reached max iterations ({max_iterations})', 'emoji': '⏹️'})}\n\n"

        # Send final result
        result = {
            'type': 'complete',
            'refined_prompt': current_prompt,
            'history': history,
            'iterations': iterations_done,
            'satisfied': satisfied,
            'model': model,
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        yield f"data: {json.dumps(result)}\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/refine', methods=['POST'])
def refine_prompt():
    """Main endpoint that returns complete results"""
    if not os.environ.get("OPENAI_API_KEY"):
        return jsonify({'error': 'Missing OPENAI_API_KEY. Set it in your .env file.'}), 500
    
    initial_prompt = request.json.get('prompt')
    attachments = request.json.get('attachments') or ""
    model = request.json.get('model', 'gpt-4o')
    
    print(f"🚀 Starting prompt refinement process...")
    print(f"📝 Initial prompt: {initial_prompt[:100]}{'...' if len(initial_prompt) > 100 else ''}")
    print(f"🤖 Using model: {model}")
    
    # Sanitize numeric params
    try:
        temperature = float(request.json.get('temperature', 1.0))
    except (TypeError, ValueError):
        temperature = 1.0
    try:
        max_tokens = int(request.json.get('max_tokens', 6000))
    except (TypeError, ValueError):
        max_tokens = 6000
    try:
        max_iterations = int(request.json.get('max_iterations', 5))
    except (TypeError, ValueError):
        max_iterations = 5
    # Clamp ranges to reasonable bounds
    if temperature < 0:
        temperature = 0.0
    if temperature > 2:
        temperature = 2.0
    if max_iterations < 1:
        max_iterations = 1
    if max_iterations > 10:
        max_iterations = 10

    review_prompt_override = (request.json.get('review_prompt') or '').strip()

    if not initial_prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    current_prompt = initial_prompt
    history = []
    iterations_done = 0
    satisfied = False

    for i in range(max_iterations):
        iteration_num = i + 1
        print(f"\n🔄 Starting iteration {iteration_num}/{max_iterations}")
        
        # 1. Obtain a review prompt (use user-provided if present, otherwise generate)
        if review_prompt_override:
            print(f"📋 Using user-provided review prompt")
            review_prompt = review_prompt_override
        else:
            print(f"🎯 Generating review prompt for iteration {iteration_num}...")
            review_prompt_generation_prompt = f"""
            Given the following user prompt, generate a concise "review prompt" to critique and identify gaps in the prompt.
            Focus on clarity, completeness, missing constraints, target audience, format/structure, and edge cases.

            Initial Prompt:
            "{current_prompt}"

            Attachments:
            "{attachments}"

            Output only the review prompt text, no preface or explanation.
            """

            try:
                print(f"💬 Calling OpenAI API to generate review prompt...")
                response = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": "You are a precise assistant that crafts prompt review checklists."},
                        {"role": "user", "content": review_prompt_generation_prompt}
                    ]
                )
                review_prompt = response.choices[0].message.content.strip()
                print(f"✅ Generated review prompt: {review_prompt[:100]}{'...' if len(review_prompt) > 100 else ''}")
            except Exception as e:
                print(f"❌ Error generating review prompt: {e}")
                return jsonify({'error': f'Error generating review prompt: {e}', 'history': history}), 500

        # 2. Use the review prompt to critique the current prompt
        print(f"🔍 Critiquing current prompt using review criteria...")
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
            print(f"💬 Calling OpenAI API to critique prompt...")
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
            print(f"📝 Critique received: {critique[:150]}{'...' if len(critique) > 150 else ''}")
        except Exception as e:
            print(f"❌ Error generating critique: {e}")
            return jsonify({'error': f'Error generating critique: {e}', 'history': history}), 500

        history.append({
            'prompt': current_prompt,
            'review_prompt': review_prompt,
            'critique': critique
        })

        # Stop when the review is satisfied
        iterations_done = i + 1
        if 'SATISFIED' in critique:
            satisfied = True
            print(f"✅ Review satisfied after {iterations_done} iteration(s)! Stopping refinement.")
            break

        # 3. Refine the current prompt based on the critique
        print(f"🔧 Refining prompt based on critique...")
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
            print(f"💬 Calling OpenAI API to refine prompt...")
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
            print(f"✨ Refined prompt: {current_prompt[:100]}{'...' if len(current_prompt) > 100 else ''}")
        except Exception as e:
            print(f"❌ Error refining prompt: {e}")
            return jsonify({'error': f'Error refining prompt: {e}', 'history': history}), 500

    # If loop ended without early break, set iterations_done accordingly
    if iterations_done == 0:
        iterations_done = min(max_iterations, len(history)) or 0

    if not satisfied:
        print(f"⏹️ Reached maximum iterations ({max_iterations}), stopping refinement.")
    
    print(f"🎯 Final result: {iterations_done} iteration(s), satisfied={satisfied}")
    print(f"📄 FINAL REFINED PROMPT:")
    print(f"=" * 50)
    print(current_prompt)
    print(f"=" * 50)
    
    # Save to file
    metadata = {
        'model': model,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'iterations': iterations_done,
        'satisfied': satisfied
    }
    
    filename = save_refined_prompt(initial_prompt, current_prompt, history, metadata)
    
    return jsonify({
        'refined_prompt': current_prompt,
        'history': history,
        'iterations': iterations_done,
        'satisfied': satisfied,
        'model': model,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'saved_file': filename
    })

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5001, host='127.0.0.1')
