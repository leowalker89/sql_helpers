# SQL Question Answering and Grading App

This Streamlit application allows users to input SQL questions and receive answers from multiple AI models. It then grades these answers and provides a comparative analysis.

## Features

- Support for multiple AI models (OpenAI, Anthropic, Google, Groq, Fireworks)
- Asynchronous processing of AI model responses
- Grading of AI-generated SQL answers
- Interactive UI with Streamlit
- Detailed results display with expandable sections

## Prerequisites

- Python 3.7+
- pip

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/sql-qa-grading-app.git
   cd sql-qa-grading-app
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   - Copy the `.env.example` file to `.env`
   - Fill in your API keys in the `.env` file

## Usage

1. Run the Streamlit app:
   ```
   streamlit run sql_llm_app.py
   ```

2. Open your web browser and go to the URL displayed in the terminal (usually `http://localhost:8501`)

3. Enter your SQL question in the text area

4. Select the AI models you want to use from the sidebar

5. Click "Generate Answers" to get responses and grades

## Configuration

You can modify the `models` and `grading_models` dictionaries in `sql_llm_app.py` to add or remove AI models as needed.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project uses various AI models through the LangChain library
- Streamlit for the web interface
- Pandas for data manipulation
