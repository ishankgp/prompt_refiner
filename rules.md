# Prompt Refiner - Development Rules & Working State Documentation

## 🚀 Current Working State (DO NOT BREAK)

### ✅ What's Currently Working
1. **Flask Application** - Runs on port 5001 with SocketIO
2. **Real-time WebSocket Communication** - Live progress updates
3. **OpenAI API Integration** - Supports multiple models (GPT-5, GPT-4o, etc.)
4. **Virtual Environment** - `.venv` directory with all dependencies
5. **File Structure** - Templates, logs, outputs directories
6. **LLM Scoring System** - Advanced evaluation with multiple metrics
7. **Session Management** - Saves and loads previous refinement sessions
8. **Default Content Loading** - Automatic loading of webpage.txt and prompt_refinement.md
9. **Responsive UI** - Tabbed interface with comparison views

### 🔧 Working Dependencies
```
flask
openai
python-dotenv
flask-socketio
```

### 🏗️ Critical File Structure
```
prompt_refiner/
├── .venv/                    # Virtual environment (ESSENTIAL)
├── app.py                    # Main Flask application (CORE)
├── templates/
│   └── index.html           # Frontend UI (CORE)
├── logs/                    # Runtime logs (auto-created)
├── outputs/                 # Saved refinements (auto-created)  
├── requirements.txt         # Dependencies (CORE)
├── webpage.txt             # Default attachment content
├── prompt_refinement.md    # Default evaluation criteria
└── rules.md               # This file
```

## 🛡️ CRITICAL RULES - NEVER VIOLATE

### 1. Environment & Dependencies
- **NEVER** change the virtual environment path from `.venv/`
- **NEVER** remove or modify `requirements.txt` without testing
- **ALWAYS** use the full Python path when running commands: `D:/Repo/prompt_refiner/.venv/Scripts/python.exe`
- **NEVER** assume system Python will work - always use the venv interpreter

### 2. Core Application Files
- **NEVER** rename or move `app.py`
- **NEVER** change the Flask port from 5001
- **NEVER** modify the SocketIO configuration without testing
- **NEVER** remove the WebSocket event handlers (`@socketio.on`)

### 3. API Integration
- **NEVER** hardcode API keys in the code
- **ALWAYS** use environment variables for sensitive data
- **NEVER** remove the OpenAI client initialization check
- **NEVER** change the model parameter handling without testing all models

### 4. File I/O Operations
- **NEVER** assume files exist - always use try/except blocks
- **ALWAYS** use UTF-8 encoding for file operations
- **NEVER** remove the auto-directory creation for `logs/` and `outputs/`
- **NEVER** change the file naming conventions for saved outputs

### 5. Frontend Integrity
- **NEVER** remove the SocketIO JavaScript client
- **NEVER** change the tab system structure without testing
- **NEVER** modify the WebSocket event listeners without testing
- **ALWAYS** maintain backward compatibility for saved session data

### 6. Logging System
- **NEVER** remove the logging setup
- **NEVER** change the log file naming pattern
- **ALWAYS** use the logger for debugging and tracking
- **NEVER** remove the session ID generation

## 🎯 Safe Modification Guidelines

### ✅ Safe to Modify
- UI styling and colors
- Text content and labels
- Additional model options (following existing pattern)
- New evaluation criteria
- Log message content
- Temperature and token limits (within bounds)

### ⚠️ Modify with Caution
- API parameter handling (test all models)
- File path handling (test on different systems)
- WebSocket message structure (ensure frontend compatibility)
- Session data structure (maintain backward compatibility)

### ❌ High-Risk Areas
- Virtual environment configuration
- OpenAI API integration
- WebSocket connection handling
- File I/O error handling
- Model parameter mapping

## 🧪 Testing Requirements

### Before Any Changes
1. Ensure virtual environment is activated
2. Verify OpenAI API key is set
3. Test with at least 2 different models
4. Verify WebSocket connection works
5. Test file saving/loading functionality

### After Modifications
1. Run the application: `D:/Repo/prompt_refiner/.venv/Scripts/python.exe app.py`
2. Test a complete refinement cycle
3. Verify logs are created
4. Check output files are saved
5. Test session loading functionality

## 🔍 Debugging Checklist

### If Application Won't Start
1. Check virtual environment activation
2. Verify all dependencies installed: `pip list`
3. Check OpenAI API key environment variable
4. Verify port 5001 is available
5. Check file permissions

### If Refinement Fails
1. Check OpenAI API key and credits
2. Verify model availability
3. Check network connectivity
4. Review logs for errors
5. Verify input data format

### If UI Issues
1. Clear browser cache
2. Check console for JavaScript errors
3. Verify SocketIO connection
4. Test WebSocket functionality
5. Check template file integrity

## 📊 Performance Considerations

### Token Management
- GPT-5: Use max_completion_tokens instead of max_tokens
- Implement smart content limiting based on model
- Monitor token usage in logs

### Memory Management
- Limit log retention
- Clean old output files periodically
- Monitor WebSocket connections

### Error Handling
- Always provide fallbacks for API failures
- Graceful degradation for missing files
- User-friendly error messages

## 🚨 Emergency Recovery

### If Environment Breaks
```powershell
# Navigate to project directory
cd D:\Repo\prompt_refiner

# Remove broken environment
Remove-Item -Recurse -Force .venv

# Create new environment
python -m venv .venv

# Activate environment
.\.venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt
```

### If Application Won't Run
1. Check this rules.md file first
2. Verify virtual environment integrity
3. Check all critical files are present
4. Review recent changes against these rules
5. Test with minimal configuration

## 📈 Version Control Best Practices

### Always Include in Commits
- All Python source files
- requirements.txt
- Template files
- This rules.md file

### Never Commit
- .venv/ directory
- .env files with API keys
- logs/ directory contents
- outputs/ directory contents
- __pycache__/ directories

## 🎉 Success Indicators

### Application is Working When
- Flask starts without errors on port 5001
- WebSocket connections establish successfully
- OpenAI API calls complete successfully
- Files save to outputs/ directory
- Session data persists correctly
- UI tabs and navigation work smoothly

---

**Remember**: When in doubt, refer to this rules.md file FIRST before making any changes. The current codebase is working - preserve its functionality while making improvements!
