import streamlit as st
from dotenv import load_dotenv
from agents.agent_builder import build_agent
from agents.vector_store import VectorStoreManager
from agents.ocr_handler import OCRHandler
import os
import tempfile

load_dotenv()

st.set_page_config(page_title="WonderWeiss")
st.title("WonderWeiss")

# Initialize components
if "agent" not in st.session_state:
    st.session_state.agent = build_agent()

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStoreManager()

if "ocr_handler" not in st.session_state:
    st.session_state.ocr_handler = OCRHandler()

if "session_id" not in st.session_state:
    st.session_state.session_id = "default_session"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for file uploads
with st.sidebar:
    st.header("Upload Documents")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'txt', 'png', 'jpg', 'jpeg'],
        help="Upload PDF, text, or image files"
    )
    
    if uploaded_file is not None:
        if st.button("Process File"):
            with st.spinner("Processing file..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    
                    if file_extension == 'pdf':
                        chunks = st.session_state.vector_store.load_pdf(tmp_path)
                        st.success(f" PDF processed! Added {chunks} chunks to knowledge base.")
                    
                    elif file_extension == 'txt':
                        chunks = st.session_state.vector_store.load_text(tmp_path)
                        st.success(f" Text file processed! Added {chunks} chunks to knowledge base.")
                    
                    elif file_extension in ['png', 'jpg', 'jpeg']:
                        # Extract text using OCR
                        extracted_text = st.session_state.ocr_handler.extract_text_from_image(tmp_path)
                        
                        if extracted_text and not extracted_text.startswith("Error"):
                            chunks = st.session_state.vector_store.load_text_content(
                                extracted_text, 
                                source_name=uploaded_file.name
                            )
                            st.success(f" Image processed! Extracted text and added {chunks} chunks.")
                            with st.expander("View extracted text"):
                                st.text(extracted_text)
                        else:
                            st.error(f" {extracted_text}")
                    
                except Exception as e:
                    st.error(f" Error processing file: {str(e)}")
                finally:
                    # Clean up temp file
                    os.unlink(tmp_path)

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Type your message...")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        response = st.session_state.agent.invoke(
               {"input": user_input},
            config={"configurable": {"session_id": st.session_state.session_id}},
        )

        output = response["messages"][-1].content

    st.chat_message("assistant").write(output)
    st.session_state.messages.append({"role": "assistant", "content": output})
