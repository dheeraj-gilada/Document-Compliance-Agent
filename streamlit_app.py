import streamlit as st
import os
import tempfile
import shutil
import asyncio
import json
import uuid
import logging

logger = logging.getLogger(__name__)

import sys
if 'torch' in sys.modules:
    import torch
    if hasattr(torch, '_classes'):
        torch._classes.__getattr__ = lambda self, attr: None

st.set_page_config(layout="wide", page_title="Intelligent Document Compliance Agent")

processing_function_imported = False
import_error_message = ""
try:
    from main import extract_document_data, run_compliance_check
    processing_function_imported = True
except ImportError as e:
    import_error_message = f"Critical Error: Could not import processing functions from main.py: {e}. Ensure main.py is in the correct path and all its dependencies are installed. The application cannot proceed."
    logger.error(import_error_message)

async def dummy_extract_document_data(docs_dir):
    logger.warning("Using dummy_extract_document_data")
    await asyncio.sleep(1)
    return {"processed_documents": [], "error_messages": ["Document extraction function (dummy) not fully loaded"]}

async def dummy_run_compliance_check(extracted_documents, rules_content):
    logger.warning("Using dummy_run_compliance_check")
    await asyncio.sleep(1)
    return {"processed_documents": extracted_documents, "aggregated_compliance_findings": [], "error_messages": ["Compliance check function (dummy) not fully loaded"]}

if not processing_function_imported:
    extract_document_data = dummy_extract_document_data
    run_compliance_check = dummy_run_compliance_check

def get_rules_representation(rules_list_data):
    if not rules_list_data: return "[]"
    return json.dumps(sorted([{"id": r['id'], "text": r['text']} for r in rules_list_data], key=lambda x: x['id']))

def main_app():
    if 'rules_list_data' not in st.session_state: st.session_state.rules_list_data = []
    if 'compliance_results' not in st.session_state: st.session_state.compliance_results = None
    if 'csv_data' not in st.session_state: st.session_state.csv_data = None
    if 'uploaded_file_names_cache' not in st.session_state: st.session_state.uploaded_file_names_cache = []
    if 'temp_docs_dir' not in st.session_state: st.session_state.temp_docs_dir = tempfile.mkdtemp(prefix="compliance_docs_")
    if 'error_message' not in st.session_state: st.session_state.error_message = None
    
    if 'is_extracting' not in st.session_state: st.session_state.is_extracting = False
    if 'is_checking_compliance' not in st.session_state: st.session_state.is_checking_compliance = False
    if 'extracted_data_cache' not in st.session_state: st.session_state.extracted_data_cache = None
    if 'rules_cache' not in st.session_state: st.session_state.rules_cache = get_rules_representation([])

    st.title("Intelligent Document Compliance & Process Automation Agent")

    if not processing_function_imported and import_error_message:
        st.error(import_error_message)
        st.stop()

    ui_disabled = st.session_state.is_extracting or st.session_state.is_checking_compliance

    def add_new_rule_callback():
        rule_text = st.session_state.get("input_new_rule_text_key", "").strip()
        if rule_text:
            st.session_state.rules_list_data.append({"id": str(uuid.uuid4()), "text": rule_text})
            st.session_state.input_new_rule_text_key = ""

    def delete_rule_callback(rule_id_to_delete):
        st.session_state.rules_list_data = [rule for rule in st.session_state.rules_list_data if rule["id"] != rule_id_to_delete]

    with st.sidebar:
        st.header("Inputs")
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'docx', 'txt', 'jpeg', 'jpg', 'png', 'webp'],
            accept_multiple_files=True,
            help="Upload documents. Extraction starts automatically.",
            disabled=ui_disabled
        )

        st.subheader("Compliance Rules")
        for i, rule_item in enumerate(st.session_state.rules_list_data):
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                edited_text = st.text_input(label=f"Rule {i+1}", value=rule_item["text"], key=f"rule_edit_{rule_item['id']}", label_visibility="visible", disabled=ui_disabled)
                if edited_text != rule_item["text"]:
                    st.session_state.rules_list_data[i]["text"] = edited_text
            with col2:
                st.button("🗑️", key=f"delete_{rule_item['id']}", on_click=delete_rule_callback, args=(rule_item['id'],), help="Delete this rule", disabled=ui_disabled)

        st.text_input("Enter new rule text:", key="input_new_rule_text_key", placeholder="Type your rule here and click 'Add Rule'", disabled=ui_disabled)
        st.button("Add Rule", on_click=add_new_rule_callback, use_container_width=True, disabled=ui_disabled)
        st.markdown("---")

    st.header("Results")

    current_uploaded_file_names = sorted([f.name for f in uploaded_files]) if uploaded_files else []
    current_rules_representation = get_rules_representation(st.session_state.rules_list_data)

    if uploaded_files and current_uploaded_file_names != st.session_state.uploaded_file_names_cache:
        if not st.session_state.is_extracting and not st.session_state.is_checking_compliance:
            logger.info("New files detected, triggering document extraction.")
            st.session_state.uploaded_file_names_cache = current_uploaded_file_names
            st.session_state.is_extracting = True
            st.session_state.extracted_data_cache = None
            st.session_state.compliance_results = None
            st.session_state.error_message = None
            st.rerun()
    
    if not st.session_state.is_extracting and not st.session_state.is_checking_compliance:
        rules_are_present = any(rule.get('text', '').strip() for rule in st.session_state.rules_list_data)
        extracted_data_is_available = bool(st.session_state.extracted_data_cache and st.session_state.extracted_data_cache.get("processed_documents") is not None)
        rules_changed_since_last_check = (current_rules_representation != st.session_state.rules_cache)
        
        needs_compliance_run = extracted_data_is_available and rules_are_present and \
                               (not st.session_state.compliance_results or rules_changed_since_last_check)

        if needs_compliance_run:
            logger.info("Conditions met for compliance check, triggering.")
            st.session_state.is_checking_compliance = True
            st.session_state.error_message = None
            st.rerun()

    if st.session_state.is_extracting:
        if not uploaded_files:
            st.warning("No files to extract. Upload documents.")
            st.session_state.is_extracting = False
            st.rerun()
        else:
            with st.spinner("Extracting data from documents..."):
                try:
                    logger.info("Document extraction started.")
                    if not os.path.exists(st.session_state.temp_docs_dir):
                        st.session_state.temp_docs_dir = tempfile.mkdtemp(prefix="compliance_docs_")
                    
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(st.session_state.temp_docs_dir, uploaded_file.name)
                        with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
                    logger.info(f"Saved {len(uploaded_files)} files to {st.session_state.temp_docs_dir}")

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    extracted_results = loop.run_until_complete(extract_document_data(st.session_state.temp_docs_dir))
                    st.session_state.extracted_data_cache = extracted_results
                    logger.info("Document extraction completed.")

                    extraction_errors = extracted_results.get("error_messages", [])
                    if extraction_errors:
                        st.session_state.error_message = f"Error during document extraction: {'; '.join(extraction_errors)}"
                        logger.error(st.session_state.error_message)
                    else:
                        st.info("Document data extracted successfully. Add rules to run compliance checks.")

                except Exception as e:
                    logger.error(f"An unexpected error occurred during extraction: {e}", exc_info=True)
                    st.session_state.error_message = f"Critical error during extraction: {e}"
                finally:
                    st.session_state.is_extracting = False
                    st.rerun()

    if st.session_state.is_checking_compliance:
        if not st.session_state.extracted_data_cache or not st.session_state.extracted_data_cache.get("processed_documents"):
            st.warning("No extracted document data available for compliance check. Please upload and process documents first.")
            st.session_state.is_checking_compliance = False
            st.rerun()
        elif not any(rule.get('text', '').strip() for rule in st.session_state.rules_list_data):
            st.warning("No compliance rules defined. Please add rules in the sidebar.")
            st.session_state.is_checking_compliance = False
            st.rerun()
        else:
            with st.spinner("Running compliance checks..."):
                try:
                    logger.info("Compliance check started.")
                    processed_docs_for_check = st.session_state.extracted_data_cache.get("processed_documents", [])
                    rules_texts = [rule.get('text', '').strip() for rule in st.session_state.rules_list_data if rule.get('text', '').strip()]
                    rules_content = "\n".join([f"{i+1}. {text}" for i, text in enumerate(rules_texts)])
                    
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    compliance_run_results = loop.run_until_complete(run_compliance_check(processed_docs_for_check, rules_content))
                    st.session_state.compliance_results = compliance_run_results
                    st.session_state.rules_cache = current_rules_representation
                    logger.info("Compliance check completed.")

                    compliance_errors = compliance_run_results.get("error_messages", [])
                    if compliance_errors:
                        prev_error = st.session_state.get("error_message", "")
                        additional_error = f"Error during compliance check: {'; '.join(compliance_errors)}"
                        st.session_state.error_message = f"{prev_error}; {additional_error}".strip('; ').strip()
                        logger.error(additional_error)
                    
                    if not st.session_state.error_message:
                         st.success("Compliance checks completed successfully!")

                except Exception as e:
                    logger.error(f"An unexpected error occurred during compliance check: {e}", exc_info=True)
                    prev_error = st.session_state.get("error_message", "")
                    st.session_state.error_message = f"{prev_error}; Critical error during compliance check: {e}".strip('; ').strip()
                finally:
                    st.session_state.is_checking_compliance = False
                    st.rerun()

    if st.session_state.error_message and not (st.session_state.is_extracting or st.session_state.is_checking_compliance):
        st.error(st.session_state.error_message)

    if st.session_state.compliance_results and not (st.session_state.is_extracting or st.session_state.is_checking_compliance):
        st.header("📊 Compliance Findings")
        results_to_display = st.session_state.compliance_results
        compliance_findings = results_to_display.get("aggregated_compliance_findings", [])

        if not compliance_findings and not st.session_state.error_message:
            st.info("No compliance findings to display. All rules may have passed or no specific findings were generated.")
        elif compliance_findings:
            import pandas as pd
            csv_data_list = []
            for finding in compliance_findings:
                involved_docs_list = finding.get('involved_documents', [])
                if not isinstance(involved_docs_list, list): involved_docs_list = [str(involved_docs_list)] if involved_docs_list else []
                csv_data_list.append({
                    "Rule ID": finding.get('rule_id', 'N/A'),
                    "Rule Text": finding.get('rule_text', finding.get('rule_checked', 'N/A')),
                    "Is Compliant": "Yes" if finding.get('is_compliant', False) else "No",
                    "Reason": finding.get('reason', finding.get('details', 'No details provided.')),
                    "Involved Document(s)": ", ".join(involved_docs_list) if involved_docs_list else 'None'
                })
            if csv_data_list:
                df = pd.DataFrame(csv_data_list)
                csv_output = df.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Download Findings as CSV", data=csv_output, file_name="compliance_findings.csv", mime="text/csv", key="download_csv_button_findings")
            
            sorted_findings = sorted(compliance_findings, key=lambda x: x.get('is_compliant', True))
            for finding in sorted_findings:
                rule_id = finding.get('rule_id', 'N/A')
                rule_text = finding.get('rule_text', finding.get('rule_checked', 'N/A'))
                is_compliant_bool = finding.get('is_compliant', False)
                is_compliant_text = "Yes" if is_compliant_bool else "No"
                reason = finding.get('reason', finding.get('details', 'No details provided.'))
                involved_docs = finding.get('involved_documents', [])
                if not isinstance(involved_docs, list): involved_docs = [str(involved_docs)] if involved_docs else []
                status_emoji = "✅" if is_compliant_bool else "❌"
                expander_title = f"{status_emoji} Rule: {rule_text[:100]}{'...' if len(rule_text) > 100 else ''}"
                with st.expander(expander_title, expanded=not is_compliant_bool):
                    st.markdown(f"**Rule ID:** `{rule_id}`")
                    st.markdown(f"**Full Rule:** {rule_text}")
                    st.markdown(f"**Compliant:** {is_compliant_text}")
                    st.markdown(f"**Reason:** {reason}")
                    if involved_docs: st.markdown("**Involved Documents:**"); [st.markdown(f"- `{doc_name}`") for doc_name in involved_docs]
                    else: st.markdown("**Involved Documents:** None")
    elif not (st.session_state.is_extracting or st.session_state.is_checking_compliance or st.session_state.error_message or st.session_state.extracted_data_cache):
        st.info("Upload documents in the sidebar to begin extraction. Then, add compliance rules to run checks.")

import atexit
def cleanup_session_temp_dir():
    if 'temp_docs_dir' in st.session_state and st.session_state.temp_docs_dir and os.path.exists(st.session_state.temp_docs_dir):
        try:
            shutil.rmtree(st.session_state.temp_docs_dir)
            logger.info(f"Cleaned up session temp directory: {st.session_state.temp_docs_dir}")
        except Exception as e: logger.error(f"Error cleaning up temp directory {st.session_state.temp_docs_dir}: {e}")
if processing_function_imported: atexit.register(cleanup_session_temp_dir)

if __name__ == "__main__":
    main_app()
