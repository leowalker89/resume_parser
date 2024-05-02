from dotenv import load_dotenv
import io
import streamlit as st
import streamlit.components.v1 as components
import base64

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError
from langchain_core.pydantic_v1 import BaseModel, Field
from resume_template import Resume
from json import JSONDecodeError
import PyPDF2
import json
import time
import os


# Set the LANGCHAIN_TRACING_V2 environment variable to 'true'
os.environ['LANGCHAIN_TRACING_V2'] = 'true'

# Set the LANGCHAIN_PROJECT environment variable to the desired project name
os.environ['LANGCHAIN_PROJECT'] = 'Resume_Project'

load_dotenv()

def pdf_to_string(file):
    """
    Convert a PDF file to a string.

    Parameters:
    file (io.BytesIO): A file-like object representing the PDF file.

    Returns:
    str: The extracted text from the PDF.
    """
    pdf_reader = PyPDF2.PdfReader(file)
    num_pages = len(pdf_reader.pages)
    text = ''
    for i in range(num_pages):
        page = pdf_reader.pages[i]
        text += page.extract_text()
    file.close()
    return text

class CustomOutputParserException(Exception):
    pass

def extract_resume_fields(full_text, model):
    """
    Analyze a resume text and extract structured information using a specified language model.
    Parameters:
    full_text (str): The text content of the resume.
    model (str): The language model object to use for processing the text.
    Returns:
    dict: A dictionary containing structured information extracted from the resume.
    """
    # The Resume object is imported from the local resume_template file

    with open("prompts/resume_extraction.prompt", "r") as f:
        template = f.read()

    parser = PydanticOutputParser(pydantic_object=Resume)

    prompt_template = PromptTemplate(
        template=template,
        input_variables=["resume"],
        partial_variables={"response_template": parser.get_format_instructions()},
    )
    llm = llm_dict.get(model, ChatOpenAI(temperature=0, model=model))

    chain = prompt_template | llm | parser
    max_attempts = 2
    attempt = 1

    while attempt <= max_attempts:
        try:
            output = chain.invoke(full_text)
            print(output)
            return output
        except (CustomOutputParserException, ValidationError) as e:
            if attempt == max_attempts:
                raise e
            else:
                print(f"Parsing error occurred. Retrying (attempt {attempt + 1}/{max_attempts})...")
                attempt += 1

    return None

def display_extracted_fields(obj, section_title=None, indent=0):
    if section_title:
        st.subheader(section_title)
    for field_name, field_value in obj:
        if field_name in ["personal_details", "education", "work_experience", "projects", "skills", "certifications", "publications", "awards", "additional_sections"]:
            st.write(" " * indent + f"**{field_name.replace('_', ' ').title()}**:")
            if isinstance(field_value, BaseModel):
                display_extracted_fields(field_value, None, indent + 1)
            elif isinstance(field_value, list):
                for item in field_value:
                    if isinstance(item, BaseModel):
                        display_extracted_fields(item, None, indent + 1)
                    else:
                        st.write(" " * (indent + 1) + "- " + str(item))
            else:
                st.write(" " * (indent + 1) + str(field_value))
        else:
            st.write(" " * indent + f"{field_name.replace('_', ' ').title()}: " + str(field_value))

def get_json_download_link(json_str, download_name):
    # Convert the JSON string back to a dictionary
    data = json.loads(json_str)
    
    # Convert the dictionary back to a JSON string with 4 spaces indentation
    json_str_formatted = json.dumps(data, indent=4)
    
    b64 = base64.b64encode(json_str_formatted.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="{download_name}.json">Click here to download the JSON file</a>'
    return href

st.set_page_config(layout="wide")

st.title("Resume Parser")

llm_dict = {
    "GPT 3.5 turbo": ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
    "Anthropic Sonnet": ChatAnthropic(model_name="claude-3-sonnet-20240229"),
    "Llama 3 8b": ChatGroq(model_name="llama3-8b-8192"),
    "Llama 3 70b": ChatGroq(model_name="llama3-70b-8192"),
    "Gemma 7b": ChatGroq(model_name="gemma-7b-it"),
    "Mistral": ChatGroq(model_name="mixtral-8x7b-32768"),
    # "Gemini 1.5 Pro": ChatGoogleGenerativeAI(model_name="gemini-1.5-pro-latest"),
}



uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
col1, col2 = st.columns(2)

with col1:
    selected_model1 = st.selectbox("Select Model 1", list(llm_dict.keys()), index=list(llm_dict.keys()).index("Llama 3 70b"))

with col2:
    selected_model2 = st.selectbox("Select Model 2", list(llm_dict.keys()), index=list(llm_dict.keys()).index("Mistral"))

if uploaded_file is not None:
    text = pdf_to_string(uploaded_file)

    if st.button("Extract Resume Fields"):
        col1, col2 = st.columns(2)

        with col1:
            start_time = time.time()
            extracted_fields1 = extract_resume_fields(text, selected_model1)
            end_time = time.time()
            elapsed_time = end_time - start_time
            st.write(f"Extraction completed in {elapsed_time:.2f} seconds")
            display_extracted_fields(extracted_fields1, "Extracted Resume Fields (Model 1)")

        with col2:
            start_time = time.time()
            extracted_fields2 = extract_resume_fields(text, selected_model2)
            end_time = time.time()
            elapsed_time = end_time - start_time
            st.write(f"Extraction completed in {elapsed_time:.2f} seconds")
            display_extracted_fields(extracted_fields2, "Extracted Resume Fields (Model 2)")