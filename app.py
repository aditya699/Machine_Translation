import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
import base64

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["AZURE_OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

# Initialize the language models
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, safety_settings=None)
model_gpt4o = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
)
output_parser = StrOutputParser()

def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, chunk_size=5000):
    # Split text into chunks of specified size
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def translate_text_chunk(text_chunk, target_language, model):
    prompt_template = f"Translate the following text to {target_language}:\n\n{{text_chunk}}"
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | model | output_parser
    
    result = chain.invoke({"text_chunk": text_chunk})
    return result

def save_text_to_file(text, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(text)

def download_link(text, filename, text_type):
    b64 = base64.b64encode(text.encode()).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:text/{text_type};base64,{b64}" download="{filename}">Download {filename}</a>'

# Streamlit UI
st.title("PDF Translator")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
target_language = st.selectbox("Select the target language", ["French", "Spanish"])  # French, Spanish

if uploaded_file is not None:
    st.write("Extracting text from PDF...")
    pdf_text = get_pdf_text(uploaded_file)
    st.write("Text extracted successfully!")
    
    if st.button("Translate"):
        st.write("Translating text...")

        # Chunk the text for translation
        text_chunks = chunk_text(pdf_text)

        translated_chunks_gemini = []
        translated_chunks_gpt4o = []

        for i, chunk in enumerate(text_chunks):
            st.write(f"Chunk {i+1}:")
            st.write(chunk)

            st.write(f"Translating Chunk {i+1} with Gemini...")
            translated_chunk_gemini = translate_text_chunk(chunk, target_language, model)
            translated_chunks_gemini.append(translated_chunk_gemini)
            st.write(f"Translated Chunk {i+1} with Gemini:")
            st.write(translated_chunk_gemini)

            # st.write(f"Translating Chunk {i+1} with GPT-4...")
            # translated_chunk_gpt4o = translate_text_chunk(chunk, target_language, model_gpt4o)
            # translated_chunks_gpt4o.append(translated_chunk_gpt4o)
            # st.write(f"Translated Chunk {i+1} with GPT-4:")
            # st.write(translated_chunk_gpt4o)

        # Combine the translated chunks
        translated_text_gemini = '\n'.join(translated_chunks_gemini)
        # translated_text_gpt4o = '\n'.join(translated_chunks_gpt4o)

        st.write("Complete Translation with Gemini:")
        st.write(translated_text_gemini)
        st.write("Translation complete with Gemini!")

        # st.write("Complete Translation with GPT-4:")
        # st.write(translated_text_gpt4o)
        # st.write("Translation complete with GPT-4!")

        # Provide download links for the translated texts
        st.markdown(download_link(translated_text_gemini, 'translated_text_gemini.txt', 'plain'), unsafe_allow_html=True)
        #st.markdown(download_link(translated_text_gpt4o, 'translated_text_gpt4o.txt', 'plain'), unsafe_allow_html=True)
