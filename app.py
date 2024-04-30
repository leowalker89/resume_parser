from dotenv import load_dotenv
import io
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
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
        except ValidationError as e:
            if attempt == max_attempts:
                raise e
            else:
                print(f"Validation error occurred. Retrying (attempt {attempt + 1}/{max_attempts})...")
                attempt += 1

    return None

    # try:
    #     parsed_output = parser.parse(output.content)
    #     json_output = parsed_output.json()
    #     print(json_output)
    #     return json_output
    
    # except ValidationError as e:
    #     print(f"Validation error: {e}")
    #     print(output)
    #     return output.content
    
    # except JSONDecodeError as e:
    #     print(f"JSONDecodeError error: {e}")
    #     print(output)
    #     return output.content

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


st.title("Resume Parser")

llm_dict = {
    "GPT 3.5 turbo": ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
    "Anthropic Sonnet": ChatAnthropic(model_name="claude-3-sonnet-20240229"),
}

selected_model = st.selectbox("Select a model", list(llm_dict.keys()))

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    if st.button("Convert PDF to Text"):
        start_time = time.time()
        
        text = pdf_to_string(uploaded_file)
        
        extracted_fields = extract_resume_fields(text, selected_model)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        st.write(f"Extraction completed in {elapsed_time:.2f} seconds")
        
        display_extracted_fields(extracted_fields, "Extracted Resume Fields")

        # for key, value in extracted_fields.items():
        #     st.write(f"{key}: {value}")
        
