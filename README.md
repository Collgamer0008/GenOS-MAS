# Multi-Agent Linux Command System

A sophisticated AI-powered system that converts natural language requests into safe and executable Linux commands using a multi-agent architecture.

##  Features

- **Natural Language Processing**: Convert plain English requests into Linux commands  
- **Multi-Agent Architecture**: Four specialized agents working together for optimal results  
- **Safety First**: Built-in security analysis and risk assessment  
- **Knowledge Base**: Learns from successful executions to improve future performance  
- **Web Search Integration**: Searches for current information when needed  
- **Error Recovery**: Automatic replanning when commands fail  

##  Architecture
Here’s an overview of the system architecture showing how different components and agents interact with each other:

![System Model](genos_system_model.png)

The system consists of four specialized agents:

### 1.  Comprehend Agent
- Analyzes natural language input  
- Determines task complexity and requirements  
- Searches for additional context when needed  

### 2.  Planner Agent
- Creates step-by-step execution plans  
- Leverages knowledge base for similar requests  
- Generates Linux commands optimized for the task  

### 3.  Examiner Agent
- Performs security risk assessment  
- Categorizes commands by danger level  
- Requests user permission for risky operations  

### 4.  Execution Agent
- Safely executes approved commands  
- Provides real-time feedback  
- Handles errors and timeouts gracefully  

## 📦 Installation

### Prerequisites
- Python 3.8 or higher  
- Linux operating system  
- Internet connection for web search functionality  

### Required API Keys

1. **Groq API Key** – For AI language model access  
   - Sign up at [Groq Console](https://console.groq.com)  
   - Generate an API key  

2. **Tavily API Key** – For web search functionality  
   - Sign up at [Tavily](https://app.tavily.com)  
   - Get your API key  

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Collgamer0008/GenOS-MAS.git
   cd multi-agent-linux-commands
2. **Install dependencies**:
```bash
pip install -r requirements.txt
```
3. **Set environment variables**:
```bash
export GROQ_API_KEY="your_groq_api_key_here"
export TAVILY_API_KEY="your_tavily_api_key_here"
```
  or create a ```.env``` file
```bash
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```
4. **Run the system**:
```bash
python3 genos_mas.py
```

## Usage
Run the system and choose your input method:

```bash
python genos_mas.py
```

You'll be prompted to choose:

- **Single line input**: For simple requests  
- **Multi-line input**: For complex, detailed requests (end with `END`)  
---

##  Example Requests

### Simple Commands
- ```Create a new folder called projects```  
- ```Install Docker on Ubuntu```  
- ```Set up a Python virtual environment```  
- ```bash
   create folder structure like this :
    project-name/
  │
  ├── src/              
  │   ├── main/       
  │   ├── utils/       
  │   └── __init__.py   
  │
  ├── tests/            
  │
  ├── docs/             
  │
  ├── config/          
  │
  ├── scripts/         
  │
  ├── .gitignore       
  ├── README.md         
  ├── requirements.txt  
  └── LICENSE ```

### Dynamic Content
- ```Create a file with today's weather information```  
- ```Generate a report with current cryptocurrency prices```  
- ```Make a list of trending GitHub repositories```  

---

##  Security
The Examiner Agent checks the command plan generated for possible damageable commands or commands that install modules, etc, if present, asks for user permission to execute the plan, which adds a layer of human in the loop security, and categorizes the command into risk levels 
### Risk Assessment Levels
- **Low Risk**: Simple file operations, directory navigation  
- **Medium Risk**: Package installations, non-system modifications  
- **High Risk**: System file changes, service modifications, network configuration  

### Safety Mechanisms
- Command analysis before execution  
- User permission requests for dangerous operations  
- Timeout protection (5 minutes per command)  
- Error handling and recovery  
- No automatic system-level changes without approval  

---

##  Knowledge Base

The system maintains a local knowledge base (`command_action-kb.json`) that:  
- Stores successfully executed plans  
- Enables faster responses for similar requests  
- Improves accuracy over time  
- Uses fuzzy matching to find relevant past solutions


## Configuration
### Customizing the LLM Model
Edit the model configuration in the code:
```bash
llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="deepseek-r1-distill-llama-70b",  # Change this
    temperature=0.3
)

```
### Adjusting Risk Thresholds
Modify the security classification in ExaminerAgent.examine_plan() to adjust what commands require permission.


## Requirements
```bash
langchain>=0.1.0
langchain-groq>=0.1.0
langchain-community>=0.0.20
tavily-python>=0.3.0
```
##  Contributing

1. **Fork the repository**  
2. **Create a feature branch**  
   ```bash
   git checkout -b feature/amazing-feature
   ```
3.**Commit your changes**
  ```bash
  git commit -m "Add amazing feature"
  ```
4.**Push to the branch**
  ```bash
  git push origin feature/amazing-feature
  ```
5.**Open a Pull Request**

## Common Troubleshooting
### API key error:
```bash
 Error: GROQ_API_KEY environment variable not set
```
- Check your internet connection
- Verify Tavily API key is valid
### Search Failures:
```bash
 Search failed: Connection timeout
```
- Check your internet connection
- Verify Tavily API key is valid
