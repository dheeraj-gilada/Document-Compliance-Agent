import json
import time
from typing import Any, Dict, List
from openai import AsyncOpenAI
import os
import logging

logger = logging.getLogger(__name__)

class UniversalComplianceAgent:
    """
    Agent responsible for checking all compliance rules (both single-document and cross-document)
    against a batch of processed documents.
    """

    SYSTEM_PROMPT = """
    You are an AI assistant specialized in comprehensive document compliance verification.
    You will be given a list of documents, each with its filename, document type, and extracted structured data.
    You will also be given a single, consolidated list of compliance rules.

    Your task is to evaluate EACH rule from the provided list against the ENTIRE set of documents.
    For each rule:
    1.  Determine the scope of the rule: Does it apply to a specific document type? Does it require comparing data within a single document? Or does it require comparing data across multiple documents that seem related (e.g., part of the same transaction)?
    2.  Identify all relevant document(s) and specific fields from the provided data that are needed to check this rule.
    3.  Perform the necessary comparisons or checks based on the rule's logic.
    4.  Determine if the rule passes, fails, or results in an error (e.g., required data missing, or rule not applicable to any documents in the batch).
    5.  Provide clear details for your finding. Explain your reasoning, mentioning the specific document filenames and values used for evaluation. If a rule is deemed 'not applicable' to the current set of documents (e.g., a rule about Purchase Orders when no POs are present), state that clearly.

    Output ALL your findings as a single JSON list. Each item in the list represents one checked rule and must include:
    -   "rule_id": The number of the rule from the input list (e.g., "1", "2", "11").
    -   "rule_checked": The exact text of the rule.
    -   "status": "pass", "fail", "error", or "not_applicable".
    -   "details": A clear explanation of your finding. If 'fail', explain why. If 'pass', briefly confirm. If 'error', describe the issue. If 'not_applicable', explain why; if this is because the rule targets a specific document type (e.g., 'Purchase Order') and no such documents were provided, explicitly state this connection (e.g., 'This rule is specific to Purchase Order documents. As no Purchase Orders were found in the batch, this rule is not applicable.').
    -   "involved_documents": A list of filenames of ONLY the documents that are DIRECTLY relevant to this specific rule. For example, if a rule is about VAT on invoices, only include invoice documents. If a rule is about matching PO numbers, include only the documents that contain the relevant PO numbers. Do NOT include documents that were checked but found to be irrelevant to the rule.

    Example of a finding for a single-document rule:
    {
      "rule_id": "3",
      "rule_checked": "Invoice Number must be present and unique.",
      "status": "pass",
      "details": "Invoice 'invoice_123.pdf' has Invoice Number 'INV001', which is present and unique in this batch.",
      "involved_documents": ["invoice_123.pdf"]
    }

    Example of a finding for a cross-document rule:
    {
      "rule_id": "11",
      "rule_checked": "If an Invoice and a Purchase Order are present for the same transaction, the Invoice total_amount_including_vat must match the Purchase Order total_amount.",
      "status": "fail",
      "details": "Invoice 'invoice_123.pdf' total_amount_including_vat (105.00) does not match Purchase Order 'po_abc.pdf' total_amount (100.00). These documents appear related by PO Number 'PO123'.",
      "involved_documents": ["invoice_123.pdf", "po_abc.pdf"]
    }
    
    Example of a 'not_applicable' finding:
    {
      "rule_id": "13",
      "rule_checked": "If a Goods Receipt Note and an Invoice are present for the same transaction, all line items (by description and quantity) on the Goods Receipt Note must be present on the Invoice.",
      "status": "not_applicable",
      "details": "This rule requires the presence of at least one Goods Receipt Note to be evaluated. No Goods Receipt Notes were found in the provided documents, therefore this rule is not applicable.",
      "involved_documents": []
    }

    Be precise and base your evaluation strictly on the data provided for all documents. If a rule implies looking for related documents, use common identifiers (like PO numbers, invoice numbers, or very similar amounts and dates if direct links are missing) to infer relationships for the purpose of that rule.
    Ensure you evaluate and report on EVERY rule in the input list.
    The final output must be a single JSON array of these findings.
    """

    SINGLE_RULE_SYSTEM_PROMPT = """
You are a meticulous Universal Compliance Agent. Your task is to evaluate a SINGLE compliance rule against a provided set of documents.

**Input:**
1.  **Rule ID:** The identifier of the rule (e.g., "1a", "7").
2.  **Rule Text:** The specific compliance rule text to evaluate.
3.  **Documents Data:** A list of dictionaries, where each dictionary represents a document. Each document dictionary contains:
    *   `"filename"`: The name of the document file.
    *   `"doc_type"`: The type of the document (e.g., "Invoice", "Purchase Order", "Shipping Manifest").
    *   `"extracted_data"`: A dictionary of key-value pairs extracted from the document.

**Your Task:**
Carefully analyze the provided **Rule Text** and determine its compliance status based on the **Documents Data**.
Consider ALL provided documents when evaluating the rule, as some rules may require information from multiple documents or depend on the presence/absence of specific document types.

**Output Format:**
You MUST output a SINGLE JSON object representing the compliance finding for the given rule. The JSON object should have the following structure:
{
    "rule_id": "<The Rule ID provided to you>",
    "rule_checked": "<The Rule Text provided to you>",
    "status": "<compliant | non-compliant | not_applicable>",
    "details": "<A clear, concise explanation of your finding. 
                 - If 'compliant', briefly state why.
                 - If 'non-compliant', clearly explain the violation and what is missing or incorrect.
                 - If 'not_applicable', explain why the rule does not apply to the given set of documents (e.g., required document type not present, or conditions for the rule are not met by any document). If this is because the rule targets a specific document type (e.g., 'Purchase Order') and no such documents were provided, explicitly state this connection.
                 Provide specific examples or data points from the documents if they support your finding.>",
    "involved_documents": ["<filename1.pdf>", "<filename2.txt>"] // List of filenames of documents that were relevant to this rule's evaluation (even if the rule was 'not_applicable' because a certain document type was missing).
}

**Important Considerations:**
*   **Scope:** Evaluate ONLY the single rule provided.
*   **Accuracy:** Be precise. Base your findings strictly on the data within the provided documents.
*   **Clarity:** Ensure your 'details' field is easy to understand.
*   **Involved Documents:** List ONLY the documents that are DIRECTLY relevant to this specific rule. For example, if a rule is about VAT on invoices, only include invoice documents. If a rule is about matching PO numbers, include only the documents that contain the relevant PO numbers. Do NOT include documents that were checked but found to be irrelevant to the rule.
*   **JSON Format:** Ensure your output is a valid JSON object as specified.
"""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        # Using temperature 0.0 for more deterministic compliance checks
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if not self.client.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.model_name = model_name
        self.temperature = temperature



    def _build_prompt_for_llm(
        self,
        all_documents_data: List[Dict[str, Any]], # Each dict: {"filename": str, "doc_type": str, "extracted_data": Dict}
        compliance_rules: str # A single string with all rules, numbered.
    ) -> str:
        # Use full data for maximum accuracy (no compression)
        prompt_parts = ["Here is the data extracted from all documents in this transaction set:"]
        if not all_documents_data:
            prompt_parts.append("No documents were provided or no data was successfully extracted.")
        else:
            for i, doc_data in enumerate(all_documents_data):
                prompt_parts.append(f"\n--- Document {i+1} ---")
                prompt_parts.append(f"Filename: {doc_data.get('filename', 'Unknown Filename')}")
                prompt_parts.append(f"Type: {doc_data.get('doc_type', 'Unknown Type')}")
                prompt_parts.append("Extracted Data:")
                # Use readable JSON formatting for maximum clarity
                prompt_parts.append(json.dumps(doc_data.get('extracted_data', {}), indent=2))
                prompt_parts.append("--- End of Document ---")

        prompt_parts.append("\nHere is the consolidated list of compliance rules you need to check against ALL the documents above:")
        prompt_parts.append(compliance_rules)
        prompt_parts.append("\nPlease evaluate EACH rule and provide your findings as a single JSON list, following the specified format in the system prompt.")
        return "\n".join(prompt_parts)

    async def check_all_compliance(
        self,
        all_documents_data: List[Dict[str, Any]],
        consolidated_rules: str
    ) -> List[Dict[str, Any]]:
        """
        Checks all compliance rules (single and cross-document) against the batch of documents.

        Args:
            all_documents_data: A list of dictionaries, where each dictionary contains
                                'filename', 'doc_type', and 'extracted_data' for a document.
            consolidated_rules: A string containing all compliance rules, typically numbered.

        Returns:
            A list of dictionaries, where each dictionary represents a compliance finding for a rule.
        """
        start_time = time.time()
        
        if not consolidated_rules.strip():
            logger.info("No compliance rules provided. Skipping compliance check.")
            return []
        
        # Parse rule count for logging
        rule_count = len([line for line in consolidated_rules.strip().split('\n') if line.strip() and line.strip()[0].isdigit()])
        
        user_prompt = self._build_prompt_for_llm(all_documents_data, consolidated_rules)
        
        # Token estimation (rough)
        estimated_input_tokens = len(user_prompt.split()) + len(self.SYSTEM_PROMPT.split())
        logger.info(f"🚀 BATCH COMPLIANCE: Processing {rule_count} rules against {len(all_documents_data)} documents")
        logger.info(f"📊 Full data mode (maximum accuracy): ~{estimated_input_tokens} estimated tokens")

        try:
            api_start = time.time()
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}, 
            )
            api_duration = time.time() - api_start
            
            response_content = response.choices[0].message.content
            if response_content is None:
                logger.error("LLM response content is None for universal compliance.")
                return [{"rule_id": "N/A", "rule_checked": "LLM Interaction Error", "status": "error", "details": "LLM returned no content.", "involved_documents": []}]

            logger.debug(f"Raw LLM response for universal compliance: {response_content}")
            
            # Expecting the response to be a JSON object with a single key (e.g., "findings") containing the list,
            # or the list directly if the model is prompted well.
            # The system prompt asks for "Output ALL your findings as a single JSON list."
            # Let's assume the model returns a JSON object like: {"compliance_findings": [...list...]}
            # and we will parse that key.
            parsed_response_outer = json.loads(response_content)
            
            findings = []
            if isinstance(parsed_response_outer, list):
                findings = parsed_response_outer # Model directly returned a list
            elif isinstance(parsed_response_outer, dict):
                # Try to find a key that holds the list of findings
                # Common keys could be 'findings', 'compliance_findings', 'results'
                potential_keys = ['findings', 'compliance_findings', 'results', 'all_compliance_findings']
                found_key = False
                for key in potential_keys:
                    if key in parsed_response_outer and isinstance(parsed_response_outer[key], list):
                        findings = parsed_response_outer[key]
                        found_key = True
                        break
                if not found_key:
                    # If it's a dict but no known key contains a list, or only one key and its value is a list
                    if len(parsed_response_outer) == 1 and isinstance(list(parsed_response_outer.values())[0], list):
                        findings = list(parsed_response_outer.values())[0]
                    else:
                        logger.error(f"Unexpected JSON structure from LLM. Expected a list of findings or a dict with a key containing a list. Got: {parsed_response_outer}")
                        raise ValueError("Unexpected JSON structure from LLM for compliance findings.")
            else:
                logger.error(f"LLM response was not a list or a dictionary: {type(parsed_response_outer)}")
                raise ValueError("LLM response was not a list or a dictionary.")

            # Performance logging
            total_duration = time.time() - start_time
            actual_input_tokens = response.usage.prompt_tokens if response.usage else estimated_input_tokens
            actual_output_tokens = response.usage.completion_tokens if response.usage else len(response_content.split())
            
            logger.info(f"✅ BATCH COMPLIANCE COMPLETED:")
            logger.info(f"   📈 Performance: {total_duration:.2f}s total ({api_duration:.2f}s API)")
            logger.info(f"   🎯 Rules processed: {len(findings)}/{rule_count}")
            logger.info(f"   💰 Tokens: {actual_input_tokens} in, {actual_output_tokens} out")
            logger.info(f"   ⚡ Speed: {len(findings)/total_duration:.1f} rules/second")
            
            return findings

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from LLM for universal compliance: {e}")
            logger.error(f"LLM Raw Response (check for malformed JSON or non-JSON output): {response_content if 'response_content' in locals() else 'response_content not captured'}")
            return [{"rule_id": "N/A", "rule_checked": "JSON Parsing Error", "status": "error", "details": f"Could not parse LLM response: {e}. Response: {response_content if 'response_content' in locals() else 'not captured'}", "involved_documents": []}]
        except Exception as e:
            logger.error(f"Error during universal compliance check with LLM: {e}", exc_info=True)
            return [{"rule_id": "N/A", "rule_checked": "LLM Interaction Error", "status": "error", "details": f"An unexpected error occurred: {e}", "involved_documents": []}]

    async def evaluate_single_rule(self, rule_id: str, rule_text: str, all_documents_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluates a single compliance rule against all provided document data."""
        logger.info(f"Evaluating single rule ID: {rule_id} - '{rule_text}' against {len(all_documents_data)} documents.")

        # Prepare the content for the LLM
        # We need to present the documents in a way the LLM can understand within the prompt context.
        # Let's serialize the document data to a compact JSON string for inclusion.
        documents_json_str = json.dumps(all_documents_data, indent=2)

        prompt_messages = [
            {"role": "system", "content": self.SINGLE_RULE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Here is the rule and the documents data:\n\n**Rule ID:** {rule_id}\n\n**Rule Text:** {rule_text}\n\n**Documents Data (JSON format):**\n```json\n{documents_json_str}\n```\n\nPlease evaluate this rule and provide the compliance finding in the specified JSON format."}
        ]

        default_error_finding = {
            "rule_id": rule_id,
            "rule_checked": rule_text,
            "status": "error",
            "details": "LLM processing failed or produced invalid JSON for this rule.",
            "involved_documents": [doc.get('filename', 'unknown_file') for doc in all_documents_data]
        }

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt_messages,
                temperature=self.temperature,
                response_format={"type": "json_object"}, 
            )
            
            response_content = response.choices[0].message.content
            if response_content is None:
                logger.error(f"LLM returned empty response for rule ID: {rule_id}.")
                return default_error_finding

            # Attempt to parse the JSON response
            # The LLM should return a single JSON object as per the prompt.
            # Remove potential markdown backticks if present
            if response_content.startswith("```json"): 
                response_content = response_content[7:] 
                if response_content.endswith("```"): 
                    response_content = response_content[:-3] 
            
            response_content = response_content.strip() 
            
            # Ensure the response is not empty before trying to parse
            if not response_content:
                logger.error(f"LLM returned empty response for rule ID: {rule_id}.")
                return default_error_finding

            parsed_finding = json.loads(response_content) 
            
            # Basic validation of the parsed structure (can be more detailed)
            if not all(key in parsed_finding for key in ["rule_id", "status", "details", "involved_documents"]):
                logger.error(f"LLM response for rule ID {rule_id} is missing required keys. Response: {parsed_finding}")
                # Augment with what was expected vs received if possible
                parsed_finding['details'] = f"Error: LLM response structure incorrect. Original details: {parsed_finding.get('details', '')}"
                parsed_finding['status'] = 'error'
                # Ensure all keys exist even if some are defaulted
                parsed_finding = {**default_error_finding, **parsed_finding, "rule_id": rule_id, "rule_checked": rule_text} 

            logger.info(f"Successfully evaluated rule ID: {rule_id}. Status: {parsed_finding.get('status')}")
            return parsed_finding
        
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError for rule ID {rule_id}: {e}. LLM Response: {response_content[:500]}") 
            # Return a structured error, but try to include the problematic response snippet if it's useful
            error_details = f"LLM response was not valid JSON. Error: {e}. Response snippet: {response_content[:200]}" 
            return {
                "rule_id": rule_id,
                "rule_checked": rule_text,
                "status": "error",
                "details": error_details,
                "involved_documents": [doc.get('filename', 'unknown_file') for doc in all_documents_data]
            }
        except Exception as e:
            logger.error(f"Unexpected error evaluating rule ID {rule_id}: {e}", exc_info=True)
            return default_error_finding
