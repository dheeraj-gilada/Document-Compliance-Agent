import streamlit as st
import os
import tempfile
import shutil
import asyncio
import json
import uuid # Added for unique rule IDs

# Workaround for PyTorch and Streamlit watcher conflict
import sys
if 'torch' in sys.modules:
    import torch
    # Monkey patch torch._classes to avoid the error with Streamlit's watcher
    if hasattr(torch, '_classes'):
        torch._classes.__getattr__ = lambda self, attr: None

# This MUST be the first Streamlit command.
st.set_page_config(layout="wide", page_title="Intelligent Document Compliance Agent")

# Attempt to import the processing functions from main.py
processing_function_imported = False
try:
    from main import run_document_processing, extract_document_data, run_compliance_check
    processing_function_imported = True
except ImportError as e:
    import_error_message = f"Critical Error: Could not import processing functions from main.py: {e}. Ensure main.py is in the correct path and all its dependencies are installed. The application cannot proceed."
    # We will display this error inside main_app

# Define dummy functions for fallback if needed
async def dummy_extract_document_data(docs_dir):
    await asyncio.sleep(1)
    return {
        "processed_documents": [{"filename": "dummy.pdf", "error": "Backend function not loaded"}],
        "error_messages": ["Document extraction function not loaded"]
    }

async def dummy_run_compliance_check(extracted_documents, rules_content):
    await asyncio.sleep(1)
    return {
        "processed_documents": extracted_documents,
        "aggregated_compliance_findings": [{"rule_id": "1", "rule_checked": "Dummy Rule", "status": "error", "details": "Backend function not loaded"}],
        "error_messages": ["Compliance check function not loaded"]
    }

async def dummy_run_document_processing(docs_dir, rules_content):
    await asyncio.sleep(1)
    return {
        "processed_documents": [{"filename": "dummy.pdf", "error": "Backend function not loaded"}],
        "aggregated_compliance_findings": [{"rule": "Dummy Rule", "error": "Backend function not loaded"}]
    }

# Assign dummy functions if import failed
if not processing_function_imported:
    extract_document_data = dummy_extract_document_data
    run_compliance_check = dummy_run_compliance_check
    run_document_processing = dummy_run_document_processing

def main_app():
    # Initialize session state variables if they don't exist
    if 'rules_list_data' not in st.session_state:
        st.session_state.rules_list_data = []  # List of dicts: {'id': str, 'text': str}
    if 'current_new_rule_text' not in st.session_state:
        st.session_state.current_new_rule_text = ""
    if 'compliance_results' not in st.session_state:
        st.session_state.compliance_results = None
    if 'csv_data' not in st.session_state:
        st.session_state.csv_data = None
    if 'has_new_results' not in st.session_state:
        st.session_state.has_new_results = False
    if 'extracted_data' not in st.session_state:
        st.session_state.extracted_data = None
    if 'processed_docs' not in st.session_state:
        st.session_state.processed_docs = []  # List to track processed document filenames
    if 'temp_docs_dir' not in st.session_state:
        st.session_state.temp_docs_dir = None

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

        # Two buttons: one for processing documents, one for compliance check
        col1, col2 = st.columns(2)
        with col1:
            process_docs_button = st.button("1️⃣ Process Documents", type="secondary", use_container_width=True, 
                                         help="Extract data from documents without running compliance checks")
        with col2:
            run_compliance_button = st.button("2️⃣ Run Compliance Check", type="primary", use_container_width=True,
                                           help="Run compliance checks on processed documents using the rules")

    # --- Main Area for Outputs ---
    st.header("Results")

    # Process Documents Button Logic
    if process_docs_button:
        if not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            with st.spinner("Processing documents..."):
                # Check if we need to create a new temp directory or use existing one
                if not st.session_state.temp_docs_dir or not os.path.exists(st.session_state.temp_docs_dir):
                    st.session_state.temp_docs_dir = tempfile.mkdtemp()
                    st.session_state.processed_docs = []  # Reset processed docs list if creating new directory
                
                # Track which files are new and need processing
                new_files = []
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(st.session_state.temp_docs_dir, uploaded_file.name)
                    # Check if this file is already processed
                    if uploaded_file.name not in st.session_state.processed_docs:
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        new_files.append(uploaded_file.name)
                        st.session_state.processed_docs.append(uploaded_file.name)
                
                if not new_files:
                    st.info("All documents have already been processed. Upload new documents or run compliance checks.")
                else:
                    st.info(f"Processing {len(new_files)} new document(s): {', '.join(new_files)}")
                    
                    try:
                        # Use the optimized extract_document_data function
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        extracted_data = loop.run_until_complete(extract_document_data(st.session_state.temp_docs_dir))
                    except RuntimeError as e:
                        if "cannot be called from a running event loop" in str(e):
                            st.warning("Asyncio runtime error. Trying with nest_asyncio.")
                            import nest_asyncio
                            nest_asyncio.apply()
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            extracted_data = loop.run_until_complete(extract_document_data(st.session_state.temp_docs_dir))
                        else:
                            raise e
                    
                    # Store extracted data in session state
                    st.session_state.extracted_data = extracted_data
                    
                    st.success(f"Successfully processed {len(new_files)} document(s)!")
                    st.info("You can now run compliance checks using the 'Run Compliance Check' button.")

    # Run Compliance Check Button Logic
    if run_compliance_button:
        if not st.session_state.processed_docs:
            st.warning("Please process at least one document first using the 'Process Documents' button.")
        elif not st.session_state.rules_list_data:
            st.warning("Please add at least one compliance rule.")
        elif not st.session_state.extracted_data:
            st.warning("No extracted data found. Please process documents first.")
        else:
            with st.spinner("Running compliance checks..."):
                try:
                    # Format rules for the backend
                    rules_for_backend_list = []
                    for i, rule_item in enumerate(st.session_state.rules_list_data):
                        rule_text_from_state = rule_item.get('text', '').strip()
                        if rule_text_from_state: # Only add non-empty rules
                           rules_for_backend_list.append(f"{i+1}. {rule_text_from_state}")
                    rules_for_backend = "\n".join(rules_for_backend_list)
                    
                    st.info(f"Running compliance checks with {len(rules_for_backend_list)} rules...")
                    
                    # Use the optimized run_compliance_check function
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # Get the extracted documents from session state
                    extracted_docs = st.session_state.extracted_data.get("processed_documents", [])
                    
                    # Ensure documents are in the correct format for the compliance agent
                    formatted_docs = []
                    for doc in extracted_docs:
                        # Create a properly formatted document with required fields
                        formatted_doc = {
                            "filename": doc.get("filename", "unknown_file"),
                            "doc_type": doc.get("doc_type", doc.get("document_type", "Unknown Type")),
                            "extracted_data": doc.get("extracted_data", {})
                        }
                        formatted_docs.append(formatted_doc)
                    
                    # Run compliance checks using the optimized function
                    results = loop.run_until_complete(run_compliance_check(formatted_docs, rules_for_backend))
                except RuntimeError as e:
                    if "cannot be called from a running event loop" in str(e):
                        st.warning("Asyncio runtime error. Trying with nest_asyncio.")
                        import nest_asyncio
                        nest_asyncio.apply()
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        # Reuse the same document formatting code
                        extracted_docs = st.session_state.extracted_data.get("processed_documents", [])
                        formatted_docs = []
                        for doc in extracted_docs:
                            formatted_doc = {
                                "filename": doc.get("filename", "unknown_file"),
                                "doc_type": doc.get("doc_type", doc.get("document_type", "Unknown Type")),
                                "extracted_data": doc.get("extracted_data", {})
                            }
                            formatted_docs.append(formatted_doc)
                        
                        # Run compliance checks
                        results = loop.run_until_complete(run_compliance_check(formatted_docs, rules_for_backend))
                    else:
                        raise e
                
                # Store compliance results in session state
                st.session_state.compliance_results = results
                st.session_state.has_new_results = True
                
                # Check for error messages
                error_messages = results.get("error_messages", [])
                if error_messages:
                    for error in error_messages:
                        st.error(f"Error during compliance check: {error}")
                else:
                    st.success("Compliance checks completed!")
                    
                # Store results for display
                findings = results.get("aggregated_compliance_findings", [])

    # Cleanup temp directory on app exit (if needed)
    if st.session_state.temp_docs_dir and os.path.exists(st.session_state.temp_docs_dir):
        # We don't actually want to delete the temp directory here since we need it for future runs
        # This is handled by the atexit module in Python, which will clean up temp directories
        pass

    # Display results section - outside the run button logic so it persists across reruns
    if st.session_state.compliance_results:
        st.subheader("Processing Results")
        # Display Compliance Findings
        st.header("📊 Compliance Findings")
        results = st.session_state.compliance_results
        compliance_findings = results.get("aggregated_compliance_findings", [])
        if not compliance_findings:
            st.info("No compliance findings were generated.")
        else:
            # Create a CSV for download
            import pandas as pd
            import io
            
            # Prepare data for CSV
            data = []
            for finding in compliance_findings:
                # Get the involved documents, ensuring it's a list
                involved_docs = finding.get('involved_documents', [])
                if not isinstance(involved_docs, list):
                    involved_docs = [str(involved_docs)] if involved_docs else []
                
                # Only include documents that are actually relevant to this rule
                # This prevents listing all documents for every rule
                data.append({
                    "Rule ID": finding.get('rule_id', 'N/A'),
                    "Rule Checked": finding.get('rule_checked', 'N/A'),
                    "Status": finding.get('status', 'N/A'),
                    "Details": finding.get('details', 'No details provided.'),
                    "Involved Document(s)": ", ".join(involved_docs) if involved_docs else 'None'
                })
            
            # Create CSV for download button
            df = pd.DataFrame(data)
            csv = df.to_csv(index=False)
            
            # Store the CSV in session state
            st.session_state.csv_data = csv
            
            # Add download button at the top
            col1, col2 = st.columns([3, 1])
            with col2:
                # Use a key for the download button to prevent rerun issues
                st.download_button(
                    label="📥 Download as CSV",
                    data=st.session_state.csv_data,
                    file_name="compliance_findings.csv",
                    mime="text/csv",
                    key="download_csv_button"
                )
            
            # Display findings as cards
            for i, finding in enumerate(compliance_findings):
                rule_id = finding.get('rule_id', 'N/A')
                rule_checked = finding.get('rule_checked', 'N/A')
                status = finding.get('status', 'N/A')
                details = finding.get('details', 'No details provided.')
                involved_docs = finding.get('involved_documents', [])
                
                # Determine card styling based on status
                if status.lower() == 'compliant':
                    border_color = "#28a745"  # Green
                    status_color = "#28a745"   # Green
                    status_icon = "✅"         # Checkmark
                elif status.lower() == 'non-compliant' or status.lower() == 'non_compliant':
                    border_color = "#dc3545"  # Red
                    status_color = "#dc3545"   # Red
                    status_icon = "❌"         # X mark
                else:
                    border_color = "#ffc107"  # Yellow/amber
                    status_color = "#ffc107"   # Yellow/amber
                    status_icon = "⚠️"         # Warning
                
                # Create a card-like container with custom styling
                st.markdown(f"""
                <div style="
                    background-color: transparent;
                    border: 2px solid {border_color};
                    border-radius: 10px;
                    padding: 15px;
                    margin-bottom: 15px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                ">
                    <h3 style="margin-top: 0;">Rule {rule_id}: {rule_checked}</h3>
                    <p style="
                        font-weight: bold;
                        color: {status_color};
                        font-size: 1.1em;
                    ">{status_icon} Status: {status.title()}</p>
                    <p><strong>Details:</strong> {details}</p>
                    <p><strong>Involved Document(s):</strong></p>
                    {"<ul>" + "".join([f"<li>{doc}</li>" for doc in involved_docs]) + "</ul>" if involved_docs else "<p>No specific documents involved</p>"}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Upload documents and enter rules, then click 'Run Compliance Check' to start.")

if __name__ == "__main__":
    main_app()
