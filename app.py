from dotenv import load_dotenv
import io
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_community.chat_models import ChatAnthropic
from langchain_openai import ChatOpenAI
from pydantic import ValidationError
from resume_template import Resume
from json import JSONDecodeError
import PyPDF2
import json
<<<<<<< HEAD
import time
=======
>>>>>>> 726975d5ca7f0a98a5047fbda8870a0f03f55283

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
    # Invoke the language model and process the resume
    formatted_input = prompt_template.format_prompt(resume=full_text)
    llm = llm_dict.get(model, ChatOpenAI(temperature=0, model=model))
    # print("llm", llm)
    output = llm.invoke(formatted_input.to_string())
    
    # print(output)  # Print the output object for debugging
    
    try:
        parsed_output = parser.parse(output.content)
        json_output = parsed_output.json()
        print(json_output)
        return json_output
    
    except ValidationError as e:
        print(f"Validation error: {e}")
        print(output)
        return output.content
    
    except JSONDecodeError as e:
        print(f"JSONDecodeError error: {e}")
        print(output)
        return output.content

st.title("Resume Parser")

# Set up the LLM dictionary
llm_dict = {
<<<<<<< HEAD
    # "gpt-4-1106-preview": ChatOpenAI(temperature=0, model="gpt-4-1106-preview"),
    # "gpt-4": ChatOpenAI(temperature=0, model="gpt-4"),
    "gpt-3.5-turbo-1106": ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106"),
    # "claude-2": ChatAnthropic(model="claude-2", max_tokens=20_000),
=======
    "gpt-4-1106-preview": ChatOpenAI(temperature=0, model="gpt-4-1106-preview"),
    "gpt-4": ChatOpenAI(temperature=0, model="gpt-4"),
    "gpt-3.5-turbo-1106": ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106"),
    "claude-2": ChatAnthropic(model="claude-2", max_tokens=20_000),
>>>>>>> 726975d5ca7f0a98a5047fbda8870a0f03f55283
    "claude-instant-1": ChatAnthropic(model="claude-instant-1", max_tokens=20_000)
}

# Add a Streamlit dropdown menu for model selection
selected_model = st.selectbox("Select a model", list(llm_dict.keys()))

# Add a file uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Check if a file is uploaded
if uploaded_file is not None:
    # Add a button to trigger the conversion
    if st.button("Convert PDF to Text"):
<<<<<<< HEAD
        start_time = time.time()  # Start the timer
        
=======
>>>>>>> 726975d5ca7f0a98a5047fbda8870a0f03f55283
        # Convert the uploaded file to a string
        text = pdf_to_string(uploaded_file)
        
        # Extract resume fields using the selected model
        extracted_fields = extract_resume_fields(text, selected_model)
        
<<<<<<< HEAD
        end_time = time.time()  # Stop the timer
        elapsed_time = end_time - start_time  # Calculate the elapsed time
        
        # Display the elapsed time
        st.write(f"Extraction completed in {elapsed_time:.2f} seconds")

        # # Display the extracted fields on the Streamlit app
        # st.json(extracted_fields)
        
        # If extracted_fields is a JSON string, convert it to a dictionary
        if isinstance(extracted_fields, str):
            extracted_fields = json.loads(extracted_fields)

        for key, value in extracted_fields.items():
            st.write(f"{key}: {value}")
        
=======
        # Display the extracted fields on the Streamlit app
        st.json(extracted_fields)
>>>>>>> 726975d5ca7f0a98a5047fbda8870a0f03f55283
