---
title: ResumeParser
emoji: 🔥
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.29.0
app_file: app.py
pinned: false
---
# Resume Parser

This Streamlit app allows you to compare the capabilities of different language models (LLMs) in parsing resumes into structured Pydantic objects. It provides insights into the accuracy, inference time, and cost of using various LLMs for resume parsing.

## Features

- Upload a PDF resume and extract structured information
- Compare the performance of two selected LLMs side-by-side
- Evaluate LLMs based on accuracy, inference time, and cost
- Supports a range of LLMs, including Groq's lightweight models (Gemma 7B, Llama 3 8B, Llama 3 70B) and others like GPT-3.5-turbo, Anthropic Claude, and Google Generative AI
- Displays the extracted resume fields in a user-friendly format
- Provides timing information for each LLM's extraction process

## Getting Started

1. Clone the repository:
```
git clone https://github.com/yourusername/resume-parser.git
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Set up the necessary environment variables:
- Create a `.env` file in the project root directory
- Add the required API keys and credentials for the LLMs you want to use

4. Run the Streamlit app:
```
streamlit run app.py
```


5. Access the app in your web browser at `http://localhost:8501`

## Usage

1. Upload a PDF resume using the file uploader
2. Select two LLMs from the dropdown menus to compare their performance
3. Click the "Extract Resume Fields" button to start the parsing process
4. View the extracted resume fields and timing information for each LLM
5. Compare the accuracy, inference time, and cost of the selected LLMs

## Resume Template

The app uses a predefined resume template defined in `resume_template.py`. The template includes various sections such as personal details, education, work experience, projects, skills, certifications, publications, awards, and additional sections.

## LLM Configuration

The app supports multiple LLMs, which can be configured in the `llm_dict` dictionary in `app.py`. Each LLM is associated with its corresponding class and initialization parameters.

## Customization

- Modify the resume template in `resume_template.py` to match your specific requirements
- Add or remove LLMs in the `llm_dict` dictionary based on your needs
- Customize the Streamlit app's appearance and layout in `app.py`

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

