import streamlit as st
from PyPDF2 import PdfReader
from openai import OpenAI
import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
import json

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    # Initialize ChromaDB client
    chroma_client = chromadb.Client()
except ValueError as e:
    st.error(f"Error connecting to ChromaDB: {e}")
    st.stop()

# Initialize embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name='text-embedding-ada-002')

original_schema =[
    {
      "templateSchemaId": 4,
      "templateSchemaName": "Transport",
      "templateSchemaProp": [
        {
          "category": "normal",
          "ctrlName": "Name",
          "ctrlDisplayName": "Name",
          "ctrlType": "textbox",
          "ctrlData": [
            {
              "id": "",
              "name": ""
            }
          ],
          "ctrlApi": ""
        },
        {
          "category": "normal",
          "ctrlName": "Email",
          "ctrlDisplayName": "Email",
          "ctrlType": "textbox",
          "ctrlData": [
            {
              "id": "",
              "name": ""
            }
          ],
          "ctrlApi": ""
        },
        {
          "category": "stage",
          "ctrlName": "stagetype",
          "ctrlDisplayName": "Stage Type",
          "ctrlType": "textbox",
          "ctrlData": [
            {
              "id": "Transport",
              "name": "Transport"
            }
          ],
          "ctrlApi": ""
        },
        {
          "category": "stage",
          "ctrlName": "stagename",
          "ctrlDisplayName": "Stage Name",
          "ctrlType": "textbox",
          "ctrlData": [
            {
              "id": "Transport",
              "name": "Transport"
            }
          ],
          "ctrlApi": ""
        },
        {
          "category": "confidence",
          "ctrlName": "Source",
          "ctrlDisplayName": "Data Source",
          "ctrlType": "dropdown",
          "ctrlData": [
            {
                "id": "106",
                "name": "Third_party"
              }
          ],          
          "ctrlApi": "/emissionfactor?emissionFactorType=Confidence_Weight&emissionFactorName=Confidence_Weight&emissionFactorSubType=Source"
        },
        {
          "category": "confidence",
          "ctrlName": "Frequency",
          "ctrlDisplayName": "Data Frequency",
          "ctrlType": "dropdown",
          "ctrlData": [
            {
                "id": "110",
                "name": "Yearly"
              }
          ],
        
          "ctrlApi": "/emissionfactor?emissionFactorType=Confidence_Weight&emissionFactorName=Confidence_Weight&emissionFactorSubType=Frequency"
        },
        {
          "category": "confidence",
          "ctrlName": "Completeness",
          "ctrlDisplayName": "Data Completeness",
          "ctrlType": "dropdown",
          "ctrlData": [
            {
              "id": 111,
              "name": "High"
            }
          ],
        
          "ctrlApi": "/emissionfactor?emissionFactorType=Confidence_Weight&emissionFactorName=Confidence_Weight&emissionFactorSubType=Completeness"
        },
        {
          "category": "confidence",
          "ctrlName": "Audited",
          "ctrlDisplayName": "Data Audited",
          "ctrlType": "dropdown",
          "ctrlData": [
            {
              "id": "115",
              "name": "Yes"
            }
          ],
          "ctrlApi": "/emissionfactor?emissionFactorType=Confidence_Weight&emissionFactorName=Confidence_Weight&emissionFactorSubType=Audited"
        },
        {
          "category": "factor",
          "ctrlName": "Transportation_distance",
          "ctrlDisplayName": "Distance Travelled (in km)",
          "ctrlType": "textbox",
          "ctrlData": [
            {
              "id": "",
              "name": ""
            }
          ],
                    "ctrlApi": ""
        },
        {
          "category": "factor",
          "ctrlName": "Fuel",
          "ctrlDisplayName": "Fuel Type",
          "ctrlType": "dropdown",
          "ctrlData": [
            {
              "id": "",
              "name": ""
            }
          ],
          "ctrlApi": "/emissionfactor?emissionFactorType=Transport&emissionFactorSubType=Transport"
        },
        {
          "category": "factor",
          "ctrlName": "Fuel_Value",
          "ctrlDisplayName": "Amount of Fuel Used (in liters)",
          "ctrlType": "textbox",
          "ctrlData": [
            {
              "id": "",
              "name": ""
            }
          ],
          "ctrlDropdown": "",
          "ctrlApi": ""
        }
      ]
    }
]

def modify_schema(original_schema, result_json):
    for prop in original_schema[0]["templateSchemaProp"]:
        ctrl_display_name = prop["ctrlDisplayName"]
        if ctrl_display_name in result_json:
            prop["ctrlData"][0]["name"] = result_json[ctrl_display_name]
    return original_schema

# Create Streamlit app
st.title("Smart Data Extraction")

# Upload PDF file
uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

if uploaded_file is not None:
    # Read PDF file and extract text
    reader = PdfReader(uploaded_file)
    page_texts = [p.extract_text() for p in reader.pages]

    # Create unique collection name based on timestamp
    collection_name = f'invoice_collection_{datetime.now().strftime("%Y%m%d%H%M%S")}'

    # Create embedding search index
    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=openai_ef)
    collection.add(documents=page_texts, ids=[str(i) for i in range(len(page_texts))])

    # Retrieve relevant page
    results = collection.query(
        query_texts=['Carbon emission terms','Supply chain terms','Emission factor terms'],
        n_results=1)
    if results['documents']:
        page = results['documents'][0][0]

        # Create prompt
        q = f'''
        You are a text processing agent working with an invoice document.

        Identify the supply chain stage(transportation or packaging)?
        
        if the stage is transportation return the following:
        if the following fields are not there in the pdf give null value for that field
        Return the answer as a proper JSON format seperated by commas with the following fields:
        - "Fuel Type" <string>
        - "Distance Travelled (in km)" <number>
        - "Amount of Fuel Used (in liters)" <number>
    
        
        else if the stage is packaging return the following:
        if the following fields are not there in the pdf give null value for that field
        Return the answer as a proper JSON format seperated by commas with the following fields:
        - "fuel type" <string>
        - "amount of fuel used" <number>
        - "electricity consumption" <number>
        - "material distributed" <number>
        - "energy used <number>
        - "transport type <string>
        - "number of  transport <number>
        - "distance travelled <number>
        - "packaging assembly done <string> yes or no
        - "distribution done <string> yes or no
        
        
        Do not infer any data based on previous training, strictly use only source text given below as input.
        =========
        {page}
        =========
        '''

        # OpenAI call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[{"role": "user", "content": q}])
        result_text = response.choices[0].message.content

        try:
            result_json = json.loads(result_text)
            st.text(type(result_json))
            st.text(json.dumps(result_json, indent=None, separators=(',', ':')))
            modified_schema = modify_schema(original_schema, result_json)
            st.text(json.dumps(modified_schema, indent=4))
         
        except json.JSONDecodeError:
            st.error("Failed to parse the response as JSON.")
