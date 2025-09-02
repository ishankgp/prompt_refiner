# Prompt Refiner Tool

A sophisticated AI-powered tool that iteratively refines prompts using OpenAI models with comprehensive logging and real-time progress tracking.

## Features

### 🔄 **Iterative Prompt Refinement**
- Automatically generates review criteria for prompt evaluation
- Iteratively improves prompts based on AI feedback
- Supports custom review prompts for specific use cases
- Stops when prompt quality is satisfactory or max iterations reached

### 📊 **Real-time Progress Tracking**
- WebSocket-based streaming for live updates
- Visual progress indicators with emojis
- Step-by-step process visibility in both terminal and browser

### 📝 **Comprehensive Logging System**
- **Session-based logging**: Each refinement session gets a unique ID
- **File-based logs**: Detailed logs stored in `logs/` directory with timestamps
- **Session logs**: Embedded in output JSON for post-run analysis
- **Multi-level logging**: Console and file logging with different detail levels

### 💾 **Data Persistence**
- Auto-saves refined prompts to `outputs/` directory in JSON format
- Includes original prompt, final result, refinement history, and session logs
- Timestamp-based file naming for easy organization

### 🎛️ **Configurable Parameters**
- Multiple OpenAI model support (GPT-4, GPT-4o, etc.)
- Adjustable temperature, max tokens, and iteration limits
- Custom review prompts for domain-specific refinement

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/ishankgp/prompt_refiner.git
cd prompt_refiner
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
```bash
# Create .env file
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

## Usage

### Starting the Server
```bash
python app.py
```
The server will start on http://127.0.0.1:5001

### Using the Web Interface
1. Navigate to http://127.0.0.1:5001
2. Enter your initial prompt
3. Optionally add attachments or custom review criteria  
4. Adjust parameters (model, temperature, iterations)
5. Click "Refine Prompt" to start the process

### Monitoring Progress
- **Browser**: Real-time progress log with emoji indicators
- **Terminal**: Detailed console output with session tracking
- **Log Files**: Comprehensive logs saved to `logs/prompt_refiner_YYYYMMDD_HHMMSS.log`

## Output Structure

### JSON Output Format
```json
{
  "timestamp": "2025-09-02T12:08:09.123456",
  "original_prompt": "Your original prompt...",
  "final_refined_prompt": "Refined version...",
  "metadata": {
    "model": "gpt-4o",
    "temperature": 1.0,
    "max_tokens": 6000,
    "iterations": 3,
    "satisfied": true,
    "session_id": "20250902_120809_123456"
  },
  "refinement_history": [
    {
      "prompt": "Current iteration prompt",
      "review_prompt": "Generated review criteria",
      "critique": "AI analysis and suggestions"
    }
  ],
  "session_logs": [
    {
      "timestamp": "2025-09-02T12:08:09.123456",
      "message": "Starting refinement process",
      "emoji": "🚀",
      "level": "info"
    }
  ]
}
```

### Log Analysis
Each session creates detailed logs for post-run analysis:

**Session Tracking:**
- Unique session IDs for correlation
- API call timing and responses
- Error tracking and debugging info

**Performance Metrics:**
- Token usage per iteration
- API response times
- Success/failure rates

## File Organization

```
prompt_refiner/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Web interface
├── outputs/              # Generated refined prompts
│   └── refined_prompt_*.json
├── logs/                 # Session logs
│   └── prompt_refiner_*.log
├── requirements.txt      # Python dependencies
├── .env                 # Environment variables (not tracked)
└── .gitignore          # Git ignore rules
```

## API Endpoints

### WebSocket Events
- `connect`: Client connection established
- `start_refinement`: Begin refinement process
- `progress`: Real-time progress updates
- `error`: Error notifications
- `complete`: Final results

### HTTP Routes
- `GET /`: Main web interface
- WebSocket endpoint at `/socket.io/`

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Logging Configuration
Logs are automatically configured with:
- File rotation by session
- Timestamp-based naming
- Dual output (console + file)
- Structured log format for analysis

## Error Handling

The system includes comprehensive error handling:
- API failure retry logic
- Session state recovery
- Detailed error logging
- User-friendly error messages

## Post-Run Analysis

Use the generated logs to analyze:
- **Performance**: API call efficiency and timing
- **Quality**: Refinement effectiveness over iterations  
- **Usage**: Token consumption and cost tracking
- **Debugging**: Detailed error traces and session flow

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add/update tests if needed
5. Submit a pull request

## License

This project is open source. Please check the repository for license details.