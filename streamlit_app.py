import streamlit as st
import os
import tempfile
import shutil
import asyncio
import json
import uuid # Added for unique rule IDs

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
    # Initialize session state for rules if not already present
    if 'rules_list_data' not in st.session_state:
        st.session_state.rules_list_data = []  # List of dicts: {'id': str, 'text': str}
    if 'current_new_rule_text' not in st.session_state: # For the 'Add Rule' input field
        st.session_state.current_new_rule_text = ""

    st.title("Intelligent Document Compliance & Process Automation Agent")

    if not processing_function_imported:
        st.error(import_error_message)
        st.stop() # Stop further execution of the Streamlit app

    # --- CALLBACK FUNCTIONS FOR RULE MANAGEMENT ---
    def add_new_rule_callback():
        rule_text = st.session_state.get("input_new_rule_text_key", "").strip()
        if rule_text:
            st.session_state.rules_list_data.append({"id": str(uuid.uuid4()), "text": rule_text})
            st.session_state.input_new_rule_text_key = "" # Clear the input field after adding

    def delete_rule_callback(rule_id_to_delete):
        st.session_state.rules_list_data = [
            rule for rule in st.session_state.rules_list_data if rule["id"] != rule_id_to_delete
        ]
    # --- END CALLBACK FUNCTIONS ---

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

        st.subheader("Compliance Rules")

        # Display existing rules with edit and delete options
        for i, rule_item in enumerate(st.session_state.rules_list_data):
            col1, col2 = st.columns([0.85, 0.15]) # Column for text input and delete button
            with col1:
                # Use a unique key for each text_input to allow editing
                # The edited value will be directly updated in session_state.rules_list_data[i]['text']
                # by Streamlit if we structure the key correctly or use on_change.
                # For simplicity, we'll allow direct editing and rely on Streamlit's reruns.
                # A more robust way is to use an on_change callback for each text_input.
                edited_text = st.text_input(
                    label=f"Rule {i+1}", 
                    value=rule_item["text"], 
                    key=f"rule_edit_{rule_item['id']}",
                    label_visibility="visible" # Show label like "Rule 1", "Rule 2"
                )
                # Update the rule text in session state if it has changed
                if edited_text != rule_item["text"]:
                    st.session_state.rules_list_data[i]["text"] = edited_text
            with col2:
                st.button("🗑️", key=f"delete_{rule_item['id']}", on_click=delete_rule_callback, args=(rule_item['id'],), help="Delete this rule")

        # Input for adding a new rule
        st.text_input(
            "Enter new rule text:", 
            key="input_new_rule_text_key", # This key is used in add_new_rule_callback
            placeholder="Type your rule here and click 'Add Rule'"
        )
        st.button("Add Rule", on_click=add_new_rule_callback, use_container_width=True)
        
        st.markdown("---") # Separator

        run_button = st.button("Run Compliance Check", type="primary", use_container_width=True)

    # --- Main Area for Outputs ---
    st.header("Results")

    if run_button:
        if not uploaded_files:
            st.warning("Please upload at least one document.")
        elif not st.session_state.rules_list_data: # Check if the list of rules is empty
            st.warning("Please add at least one compliance rule.")
        else:
            with st.spinner("Processing documents and checking compliance..."):
                temp_docs_dir = None
                try:
                    # 1. Create a temporary directory for uploaded files
                    temp_docs_dir = tempfile.mkdtemp()
                    # st.info(f"Temporary directory created: {temp_docs_dir}") # Can be verbose

                    # 2. Save uploaded files to the temporary directory
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(temp_docs_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        # st.write(f"Saved {uploaded_file.name} to temp dir.") # Can be verbose
                    
                    # 3. Format rules for the backend from session_state.rules_list_data
                    rules_for_backend_list = []
                    for i, rule_item in enumerate(st.session_state.rules_list_data):
                        # Ensure rule_item['text'] is up-to-date from its input field
                        # The direct edit in the loop above should handle this for text_input
                        # If using st.text_area per rule, would need explicit key access
                        rule_text_from_state = rule_item.get('text', '').strip()
                        if rule_text_from_state: # Only add non-empty rules
                           rules_for_backend_list.append(f"{i+1}. {rule_text_from_state}")
                    rules_for_backend = "\n".join(rules_for_backend_list)
                    
                    # DEBUG: Print rules being sent to backend
                    # print(f"DEBUG: Rules sent to backend:\n{rules_for_backend}")
                    # st.subheader("DEBUG: Formatted Rules for Backend")
                    # st.text(rules_for_backend if rules_for_backend else "No rules formatted.")

                    st.info("Invoking backend processing workflow...")
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        results = loop.run_until_complete(run_document_processing_func(temp_docs_dir, rules_for_backend))
                    except RuntimeError as e:
                        if "cannot be called from a running event loop" in str(e):
                            st.warning("Asyncio runtime error. Trying with nest_asyncio. Please install it if not present (`pip install nest_asyncio`) and rerun.")
                            import nest_asyncio
                            nest_asyncio.apply()
                            loop = asyncio.new_event_loop() # Get a new loop after applying nest_asyncio
                            asyncio.set_event_loop(loop)
                            results = loop.run_until_complete(run_document_processing_func(temp_docs_dir, rules_for_backend))
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
                    if temp_docs_dir and os.path.exists(temp_docs_dir):
                        shutil.rmtree(temp_docs_dir)
                        # st.info(f"Temporary directory {temp_docs_dir} cleaned up.") # Can be verbose
    else:
        st.info("Upload documents and enter rules, then click 'Run Compliance Check' to start.")

if __name__ == "__main__":
    main_app()
