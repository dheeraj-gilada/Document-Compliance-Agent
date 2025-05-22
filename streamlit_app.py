import streamlit as st
import os
import tempfile
import shutil
import asyncio
import json

# This MUST be the first Streamlit command.
st.set_page_config(layout="wide", page_title="Intelligent Document Compliance Agent")

# Attempt to import the processing function from main.py
processing_function_imported = False
run_document_processing_func = None
try:
    from main import run_document_processing
    run_document_processing_func = run_document_processing
    processing_function_imported = True
except ImportError as e:
    import_error_message = f"Critical Error: Could not import 'run_document_processing' from main.py: {e}. Ensure main.py is in the correct path and all its dependencies are installed. The application cannot proceed."
    # We will display this error inside main_app

# Define a dummy function for fallback if needed, though the app should ideally not run fully without the real one
async def dummy_run_document_processing(docs_dir, rules_content):
    await asyncio.sleep(1)
    return {
        "processed_documents": [{"filename": "dummy.pdf", "error": "Backend function not loaded"}],
        "aggregated_compliance_findings": [{"rule": "Dummy Rule", "error": "Backend function not loaded"}]
    }

if not processing_function_imported:
    run_document_processing_func = dummy_run_document_processing # Assign dummy if import failed

def main_app():
    st.title("Intelligent Document Compliance & Process Automation Agent")

    if not processing_function_imported:
        st.error(import_error_message)
        st.stop() # Stop further execution of the Streamlit app

    st.markdown("""
    Upload your documents and provide compliance rules to check against.
    The agent will process the documents, extract relevant data, and evaluate compliance.
    """)

    # --- Sidebar for Inputs ---
    with st.sidebar:
        st.header("Inputs")
        
        uploaded_files = st.file_uploader(
            "Upload Documents", 
            type=['pdf', 'docx', 'txt', 'jpeg', 'jpg', 'png', 'webp'], 
            accept_multiple_files=True,
            help="Upload one or more documents in supported formats."
        )

        rules_text = st.text_area(
            "Compliance Rules", 
            height=300, 
            placeholder="Enter each compliance rule on a new line or as a consolidated text block.",
            help="Define the compliance rules to check against the documents."
        )

        run_button = st.button("Run Compliance Check", type="primary", use_container_width=True)

    # --- Main Area for Outputs ---
    st.header("Results")

    if run_button:
        if not uploaded_files:
            st.warning("Please upload at least one document.")
        elif not rules_text.strip():
            st.warning("Please enter compliance rules.")
        else:
            with st.spinner("Processing documents and checking compliance..."):
                temp_docs_dir = None
                try:
                    # 1. Create a temporary directory for uploaded files
                    temp_docs_dir = tempfile.mkdtemp()
                    st.info(f"Temporary directory created: {temp_docs_dir}")

                    # 2. Save uploaded files to the temporary directory
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(temp_docs_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        st.write(f"Saved {uploaded_file.name} to temp dir.")
                    
                    # 3. Get rules_text (already available as 'rules_text')
                    
                    # 4. Call the main processing logic
                    st.info("Invoking backend processing workflow...")
                    # Run the async function using asyncio.run()
                    # This might need nest_asyncio if Streamlit itself uses an event loop
                    # For now, let's try the direct approach.
                    try:
                        # Ensure a new event loop for this asyncio.run call if needed
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        results = loop.run_until_complete(run_document_processing_func(temp_docs_dir, rules_text.strip()))
                    except RuntimeError as e:
                        if "cannot be called from a running event loop" in str(e):
                            st.warning("Asyncio runtime error. Trying with nest_asyncio. Please install it if not present (`pip install nest_asyncio`) and rerun.")
                            import nest_asyncio
                            nest_asyncio.apply()
                            # Rerun with the applied patch
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            results = loop.run_until_complete(run_document_processing_func(temp_docs_dir, rules_text.strip()))
                        else:
                            raise e # Re-raise other runtime errors

                    st.success("Processing workflow completed!")

                    # 5. Display results
                    processed_docs = results.get("processed_documents", [])
                    compliance_findings = results.get("aggregated_compliance_findings", [])
                    errors = results.get("error_messages", [])

                    if errors:
                        st.subheader("Processing Errors")
                        for error_msg in errors:
                            st.error(error_msg)

                    st.subheader("Extracted Document Data")
                    if processed_docs:
                        st.json(processed_docs)
                    else:
                        st.info("No document data was extracted or returned.")

                    st.subheader("Compliance Findings")
                    if compliance_findings:
                        st.json(compliance_findings)
                    else:
                        st.info("No compliance findings were generated or returned.")

                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
                    import traceback
                    st.text(traceback.format_exc())
                finally:
                    # 6. Clean up the temporary directory
                    if temp_docs_dir and os.path.exists(temp_docs_dir):
                        shutil.rmtree(temp_docs_dir)
                        st.info(f"Temporary directory {temp_docs_dir} cleaned up.")
    else:
        st.info("Upload documents and enter rules, then click 'Run Compliance Check' to start.")

if __name__ == "__main__":
    # Ensure main.py's asyncio parts don't conflict if it's also run directly
    # For Streamlit, it's better to run with `streamlit run streamlit_app.py`
    main_app()
