from dotenv import load_dotenv
# import io
import streamlit as st
# import streamlit.components.v1 as components
# import base64

# from langchain.prompts import PromptTemplate
# from langchain_core.output_parsers import PydanticOutputParser
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.exceptions import OutputParserException
# from pydantic import ValidationError
# from langchain_core.pydantic_v1 import BaseModel, Field
# from resume_template import Resume
# from json import JSONDecodeError
# import PyPDF2
# import json
import time
import os

import resume_helpers


# Set the LANGCHAIN_TRACING_V2 environment variable to 'true'
os.environ['LANGCHAIN_TRACING_V2'] = 'true'

# Set the LANGCHAIN_PROJECT environment variable to the desired project name
os.environ['LANGCHAIN_PROJECT'] = 'Resume_Project'

load_dotenv()

st.set_page_config(layout="wide")

st.title("Resume Parser")
col1, col2 = st.columns([1,6])

with col1:
    st.image("llamallms.png", use_column_width=True)

with col2:
    st.write("""
    ## 📝 Unlocking the Power of LLMs 🔓

    Welcome to the Resume Parser, a powerful tool designed to extract structured information from resumes using the magic of Language Models (LLMs)! 🪄📄 As a data scientist and military veteran, I understand the importance of efficiency and accuracy when it comes to processing information. That's why I've created this app to showcase how different LLMs can help us parse resumes with ease. 💪

    Resumes come in all shapes and sizes, and standardization is often a distant dream. 😴 But with the right LLM by your side, you can extract key information like personal details, education, work experience, and more, all with just a few clicks! 🖱️ Plus, by comparing the performance of various models, you can find the perfect balance of speed, accuracy, and cost for your specific use case. 💰

    So, whether you're a recruiter looking to streamline your hiring process, or a data enthusiast curious about the capabilities of LLMs, the Resume Parser has got you covered! 🙌 Upload a resume, select your models, and watch the magic happen. 🎩✨ Let's unlock the full potential of LLMs together and make resume parsing a breeze! 😎
    """)

llm_dict = {
    "GPT 3.5 turbo": ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125"),
    "GPT 4o": ChatOpenAI(temperature=0, model_name="gpt-4o"),
    "Anthropic 3.5 Sonnet": ChatAnthropic(model="claude-3-5-sonnet-20240620"),
    "Llama 3 8b": ChatGroq(model_name="llama3-8b-8192"),
    "Llama 3 70b": ChatGroq(model_name="llama3-70b-8192"),
    "Gemma 7b": ChatGroq(model_name="gemma-7b-it"),
    "Mixtral 8x7b": ChatGroq(model_name="mixtral-8x7b-32768"),
    "Gemini 1.5 Pro": ChatGoogleGenerativeAI(model="gemini-1.5-pro"),
    "Gemini 1.5 Flash": ChatGoogleGenerativeAI(model="gemini-1.5-flash"),
}

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
col1, col2 = st.columns(2)

with col1:
    selected_model1 = st.selectbox("Select Model 1", list(llm_dict.keys()), index=list(llm_dict.keys()).index("Llama 3 70b"))

with col2:
    selected_model2 = st.selectbox("Select Model 2", list(llm_dict.keys()), index=list(llm_dict.keys()).index("Mixtral 8x7b"))

if uploaded_file is not None:
    text = resume_helpers.pdf_to_string(uploaded_file)

    if st.button("Extract Resume Fields"):
        col1, col2 = st.columns(2)

        with col1:
            start_time = time.time()
            extracted_fields1 = resume_helpers.extract_resume_fields(text, selected_model1)
            end_time = time.time()
            elapsed_time = end_time - start_time
            st.write(f"Extraction completed in {elapsed_time:.2f} seconds")
            resume_helpers.display_extracted_fields(extracted_fields1, f"{selected_model1} Extracted Fields ")

        with col2:
            start_time = time.time()
            extracted_fields2 = resume_helpers.extract_resume_fields(text, selected_model2)
            end_time = time.time()
            elapsed_time = end_time - start_time
            st.write(f"Extraction completed in {elapsed_time:.2f} seconds")
            resume_helpers.display_extracted_fields(extracted_fields2, f"{selected_model2} Extracted Fields ")