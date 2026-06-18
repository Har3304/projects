# Multi-Agent Legislative Problem Solver

This project uses LangChain and Meta Llama 3.1 (via Hugging Face) to create an automated, recursive AI loop. It turns a complex corporate business problem into an in-depth system prompt, assigns an expert role to a secondary AI agent, and evaluates solutions until an explicit exit condition is met.

## Features
* **Recursive AI Feedback:** The bot evaluates its own outputs in a `while` loop until it solves the prompt or passes it along.
* **Memory Management:** Uses LangChain's `RunnableWithMessageHistory` and `InMemoryChatMessageHistory` to keep track of the conversation context.
* **Dual Data Logging:** Automatically saves structured logs to a CSV file and appends full, un-truncated text outputs to a plain text file (`full_ai_responses.txt`).

## Setup and Installation

### 1. Install Dependencies
Make sure you have Python installed. Then, install the required packages using pip:
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
This script is designed to run inside Google Colab and looks for a Hugging Face User Access Token stored in your Colab Secrets.
1. Open your Google Colab notebook.
2. Click on the **Secrets** () icon on the left sidebar.
3. Add a new secret named `HF_TOKEN_READ`.
4. Paste your Hugging Face read-access token into the value field.

## How to Run

Run the Python script in your environment:
```python
# Run the script execution block
python main.py
```

### Understanding the Exit Logic
* The bot reads the business problem regarding government legislation and green energy bills.
* It wraps the problem into an expert prompt and begins a continuous loop.
* The loop will safely stop **only** when the AI outputs the word `exit` (case-insensitive) on a new line, indicating total work satisfaction.

## Project Outputs

After running the script, two files will be generated in your working directory:
1. `chat_history.csv`: A structured spreadsheet tracking session IDs and metadata. *(Note: Spreadsheet software like Excel may visually truncate long cell text).*
2. `full_ai_responses.txt`: A clean text backup file containing every single response in full detail with zero truncation limits.
