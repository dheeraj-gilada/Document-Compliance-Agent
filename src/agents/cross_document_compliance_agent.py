import json
from typing import Any, Dict, List, Tuple
from openai import OpenAI
import os

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class CrossDocumentComplianceAgent:
    """
    Agent responsible for checking compliance rules that span across multiple documents
    within a transaction set.
    """

    SYSTEM_PROMPT = """
    You are an AI assistant specialized in cross-document compliance verification.
    You will be given a list of documents, each with its type and extracted structured data.
    You will also be given a set of cross-document compliance rules.
    Your task is to evaluate these rules against the provided documents.

    For each rule:
    1. Identify the documents and specific fields relevant to the rule.
    2. Perform the necessary comparisons or checks.
    3. Determine if the rule passes, fails, or results in an error (e.g., required data missing from one or more documents).
    4. Provide clear details for your finding, explaining your reasoning and mentioning the specific values from the documents you used.

    Output your findings as a JSON list, where each item represents a checked rule and includes:
    - "rule_checked": The exact text of the rule.
    - "status": "pass", "fail", or "error".
    - "details": A clear explanation of your finding, including document filenames and values used for evaluation.
    - "involved_documents": A list of filenames of the documents involved in checking this specific rule.

    Example of a finding:
    {
      "rule_checked": "Invoice total amount must match the PO total amount.",
      "status": "fail",
      "details": "Invoice 'invoice_123.pdf' total_amount (105.00) does not match PO 'po_abc.pdf' total_amount (100.00).",
      "involved_documents": ["invoice_123.pdf", "po_abc.pdf"]
    }

    If a document needed for a rule is not present in the provided list, or if key data is missing, mark the rule status as 'error'.
    Be precise and base your evaluation strictly on the data provided.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        self.temperature = temperature

    def _build_cross_document_prompt(
        self,
        all_documents_data: List[Dict[str, Any]], # Each dict: {"filename": str, "doc_type": str, "extracted_data": Dict}
        cross_document_rules: str
    ) -> str:
        prompt_parts = ["Here is the data extracted from the documents in this transaction set:"]
        for doc_data in all_documents_data:
            prompt_parts.append(f"\n--- Document: {doc_data.get('filename', 'Unknown Filename')} ---")
            prompt_parts.append(f"Type: {doc_data.get('doc_type', 'Unknown Type')}")
            prompt_parts.append("Extracted Data:")
            prompt_parts.append(json.dumps(doc_data.get('extracted_data', {}), indent=2))
            prompt_parts.append("--- End of Document ---")

        prompt_parts.append("\nHere are the cross-document compliance rules you need to check:")
        prompt_parts.append(cross_document_rules)
        prompt_parts.append("\nPlease evaluate these rules and provide your findings in the specified JSON format.")
        return "\n".join(prompt_parts)

    async def check_cross_document_compliance(
        self,
        all_documents_data: List[Dict[str, Any]],
        cross_document_rules: str
    ) -> List[Dict[str, Any]]:
        """
        Checks compliance rules that span across multiple documents.

        Args:
            all_documents_data: A list of dictionaries, where each dictionary contains
                                'filename', 'doc_type', and 'extracted_data' for a document.
            cross_document_rules: A string containing all cross-document compliance rules, one per line.

        Returns:
            A list of dictionaries, where each dictionary represents a compliance finding for a rule.
        """
        if not all_documents_data:
            logger.warning("No documents provided for cross-document compliance check.")
            return []
        if not cross_document_rules.strip():
            logger.info("No cross-document rules provided to check.")
            return []

        user_prompt = self._build_cross_document_prompt(all_documents_data, cross_document_rules)
        logger.debug(f"Cross-document compliance user prompt:\n{user_prompt}")

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}, 
            )
            
            response_content = response.choices[0].message.content
            if response_content is None:
                logger.error("LLM response content is None for cross-document compliance.")
                return [{"rule_checked": "LLM Interaction Error", "status": "error", "details": "LLM returned no content.", "involved_documents": []}]

            logger.debug(f"Raw LLM response for cross-doc compliance: {response_content}")
            
            parsed_response = json.loads(response_content)

            if isinstance(parsed_response, list):
                findings = parsed_response
            elif isinstance(parsed_response, dict) and len(parsed_response) == 1:
                findings = list(parsed_response.values())[0]
                if not isinstance(findings, list): 
                    raise ValueError("Expected a list of findings within the JSON object.")
            elif isinstance(parsed_response, dict) and "findings" in parsed_response: 
                findings = parsed_response["findings"]
                if not isinstance(findings, list):
                     raise ValueError("Expected 'findings' to be a list.")
            else: 
                logger.error(f"Unexpected JSON structure from LLM for cross-document compliance: {parsed_response}")
                raise ValueError("Unexpected JSON structure from LLM.")


            logger.info(f"Cross-document compliance check successful using {self.model_name}. {len(findings)} findings.")
            return findings

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from LLM for cross-document compliance: {e}")
            logger.error(f"LLM Raw Response causing JSONDecodeError: {response_content}")
            return [{"rule_checked": "JSON Parsing Error", "status": "error", "details": f"Could not parse LLM response: {e}", "involved_documents": []}]
        except Exception as e:
            logger.error(f"Error during cross-document compliance check with LLM: {e}")
            return [{"rule_checked": "LLM Interaction Error", "status": "error", "details": f"An unexpected error occurred: {e}", "involved_documents": []}]
