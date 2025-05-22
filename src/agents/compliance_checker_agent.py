import os
import json
import logging
from typing import Dict, Any, List
from openai import AsyncOpenAI

# Configure logging
logger = logging.getLogger(__name__)
# Ensure logging is configured. If main.py configures it, this might be redundant
# or could be set to a more specific level for this module if needed.
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ComplianceCheckerAgent:
    """Agent responsible for checking document compliance against a set of rules."""

    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        self.client = AsyncOpenAI(api_key=self.openai_api_key)
        self.model_name = "gpt-4o-mini"
        self.system_prompt = """
You are an AI assistant specialized in checking extracted document data against a set of compliance rules.
Your goal is to determine if the document is compliant, non-compliant, or if there are errors in checking compliance (e.g., missing data).

Key Instructions:

1.  **Input:** You will receive:
        `extracted_data`: A JSON object with data extracted from a document.
        `compliance_rules`: A list of rules, each as a natural language string.

2.  **Field Name Matching (IMPORTANT):**
        The field names mentioned in the `compliance_rules` (e.g., "Invoice ID", "Vendor Name") are conceptual.
        When evaluating a rule, you MUST intelligently look for corresponding fields in the `extracted_data` JSON. This means considering common variations in naming. For example:
            If a rule refers to "Invoice ID", you should check for keys like `invoice_id`, `invoice_number`, `InvoiceNo`, `DocumentID`, `ID`, etc., within the relevant part of the `extracted_data`.
            If a rule refers to "Vendor Name", you should check for keys like `vendor_name`, `supplier_name`, `seller_name`, `customer_name` (if the context implies the customer is the vendor from the document issuer's perspective), etc.
        Prioritize exact matches if available, but be flexible.
        If you match a rule's field name to a differently named key in the `extracted_data`, clearly state this mapping in your `details` for that finding (e.g., "Rule for 'Invoice ID' checked against extracted field 'invoice_number'.").
        If, after trying common variations, you cannot confidently find a field in extracted_data that corresponds to a field mentioned in a rule, the status for that rule check should be 'error', with 'details' explaining which field was not found.
        When a rule requires a field to possess multiple characteristics simultaneously (e.g., 'must contain both letters and numbers'), ensure your evaluation strictly verifies that ALL specified characteristics are present. If any characteristic is missing, the rule should fail.
        Interpret rules LITERALLY. For example, if a rule says 'quantity must be positive', '0' is not positive. If it says 'alphanumeric', it means it can contain letters OR numbers OR both; it does not imply it MUST contain both unless explicitly stated.
        If a specific field named in a rule (e.g., 'Vendor Name') is entirely missing from the `extracted_data` JSON, the status for that rule check should be 'error', and the 'details' should state that the required field was not found in the extracted data.

3.  **Rule Evaluation:**
        Interpret each rule LITERALLY. Do not make assumptions beyond what is explicitly stated. For example, "positive quantity" means > 0, not necessarily a whole number, unless "whole number" is also specified.
        For each rule, determine a status:
            `pass`: The data meets the rule's criteria.
            `fail`: The data does not meet the rule's criteria.
            `error`: The data required to check the rule is missing from `extracted_data` (even after attempting flexible field matching), or the rule cannot be confidently evaluated for other reasons (e.g., ambiguous data). Clearly state why it's an error.
        If a specific field named in a rule (e.g., "Vendor Name") is entirely absent from the `extracted_data` (and no semantic equivalent can be confidently found), the status for that rule check should be `error`, and the details should explain what was missing.

4.  **Output Format:**
        Return a single JSON object with two top-level keys: `compliance_status` and `findings`.
       `compliance_status`: Overall status. Can be:
            `compliant`: All rules passed.
            `non-compliant`: At least one rule failed.
            `error_in_checking`: At least one rule resulted in an "error" status, and no rules failed. (If any rule fails, the overall status is `non-compliant` regardless of errors in other rules).
        `findings`: A list of objects, one for each rule checked. Each object must have:
            `rule_checked`: The original compliance rule string.
            `status`: "pass", "fail", or "error".
            `details`: A concise explanation of why the rule passed, failed, or resulted in an error. If you used flexible field matching, mention it here.

Example Finding with Flexible Matching:
```json
{
  "rule_checked": "Invoice ID must be present and in the format INV-YYYY-NNN.",
  "status": "pass", 
  "details": "Rule for 'Invoice ID' checked against extracted field 'invoice_number'. Value '626867-ADS1-1' meets the criteria."
}
```

Example Finding for Missing Data (after flexible search):
```json
{
  "rule_checked": "A valid shipping date must be provided.",
  "status": "error",
  "details": "Could not find a field corresponding to 'shipping date' (e.g., 'shipping_date', 'ship_date', 'dispatch_date') in the extracted data."
}
```
"""

    async def check_compliance(
        self, 
        extracted_data: Dict[str, Any], 
        document_type: str, 
        compliance_instructions: str
    ) -> Dict[str, Any]:
        """
        Checks the extracted data from a document against a set of compliance instructions.

        Args:
            extracted_data: The structured data extracted from the document.
            document_type: The inferred type of the document.
            compliance_instructions: A string containing the compliance rules.

        Returns:
            A dictionary containing compliance status and findings.
            Example:
            {
                "compliance_status": "non-compliant", // "compliant", "non-compliant", "warning"
                "findings": [
                    {"rule_checked": "Invoice date must be within 30 days of delivery.", "status": "fail", "details": "Invoice date is 45 days after delivery."},
                    {"rule_checked": "Total amount must match sum of line items.", "status": "pass", "details": "Amounts match."}
                ]
            }
        """
        if not extracted_data or not isinstance(extracted_data, dict) or not extracted_data.get('extracted_data'):
            logger.warning(f"No actual extracted data found within the provided structure for document type '{document_type}'. Cannot perform compliance check.")
            return {
                "compliance_status": "error",
                "findings": [{
                    "rule_checked": "Data Availability", 
                    "status": "error", 
                    "details": "No valid extracted data (missing 'extracted_data' field or empty) available for compliance check."
                }]
            }

        prompt = self._build_compliance_prompt(
            extracted_data.get('extracted_data'), # Pass the actual data part
            document_type,
            compliance_instructions
        )

        raw_response_content = ""
        try:
            logger.info(f"Checking compliance for document type '{document_type}' using {self.model_name}...")
            response = await self.client.chat.completions.create(
                model=self.model_name,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1, # Lower temperature for more deterministic, factual output
            )
            
            raw_response_content = response.choices[0].message.content
            logger.debug(f"Raw compliance check LLM response for {document_type}: {raw_response_content}")
            compliance_result = json.loads(raw_response_content)
            
            # Validate the LLM's JSON output structure
            if not isinstance(compliance_result, dict) or \
               'compliance_status' not in compliance_result or \
               not isinstance(compliance_result['compliance_status'], str) or \
               'findings' not in compliance_result or \
               not isinstance(compliance_result['findings'], list):
                logger.error(f"LLM returned malformed JSON for compliance check of {document_type}. Raw: {raw_response_content}")
                raise ValueError("LLM returned malformed JSON structure for compliance check.")
            
            for finding in compliance_result['findings']:
                if not all(k in finding for k in ('rule_checked', 'status', 'details')):
                    logger.error(f"LLM returned malformed finding in JSON for {document_type}. Finding: {finding}. Raw: {raw_response_content}")
                    raise ValueError("LLM returned malformed finding structure in compliance check.")

            logger.info(f"Compliance check successful for document type '{document_type}'. Status: {compliance_result.get('compliance_status')}")
            return compliance_result

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from LLM response for compliance check ({document_type}): {e}. Raw response: {raw_response_content}")
            return {
                "compliance_status": "error_parsing_llm_response",
                "findings": [{
                    "rule_checked": "LLM Output Parsing", 
                    "status": "error", 
                    "details": f"Failed to parse LLM JSON response: {str(e)}. Raw: {raw_response_content[:500]}..."
                }]
            }
        except ValueError as e: # Catch our custom validation errors
            logger.error(f"Validation error in LLM response for compliance check ({document_type}): {e}. Raw response: {raw_response_content}")
            return {
                "compliance_status": "error_invalid_llm_response_structure",
                "findings": [{
                    "rule_checked": "LLM Output Structure Validation", 
                    "status": "error", 
                    "details": f"LLM response failed validation: {str(e)}. Raw: {raw_response_content[:500]}..."
                }]
            }
        except Exception as e:
            logger.error(f"Error during compliance check with LLM ({document_type}): {e}", exc_info=True)
            return {
                "compliance_status": "error_llm_api_call",
                "findings": [{
                    "rule_checked": "LLM Interaction", 
                    "status": "error", 
                    "details": f"LLM API call failed: {str(e)}"
                }]
            }

    def _build_compliance_prompt(
        self, 
        actual_extracted_data: Dict[str, Any], # This is the nested 'extracted_data' field
        document_type: str, 
        compliance_instructions: str
    ) -> str:
        """Builds the prompt for the LLM to perform compliance checks."""
        
        prompt = f"Please meticulously evaluate the following document data for compliance. "
        prompt += f"The document is of type: '{document_type}'.\n\n"
        prompt += "Extracted Document Data (this is the data to check):\n"
        prompt += "```json\n"
        prompt += json.dumps(actual_extracted_data, indent=2)
        prompt += "\n```\n\n"
        prompt += "Compliance Instructions to check against:\n"
        prompt += "-------------------------------------\n"
        prompt += compliance_instructions
        prompt += "\n-------------------------------------\n\n"
        return prompt

# Example Usage (for testing purposes)
async def _test_compliance_checker():
    checker = ComplianceCheckerAgent()
    sample_extracted_data = {
        "filename": "invoice_123.pdf",
        "doc_type": "invoice",
        "extracted_data": {
            "invoice_id": "INV-2024-001",
            "vendor_name": "TestCorp Ltd.",
            "invoice_date": "2024-01-15",
            "due_date": "2024-02-14",
            "total_amount": 1500.75,
            "line_items": [
                {"description": "Product A", "quantity": 2, "unit_price": 500.00, "total": 1000.00},
                {"description": "Service B", "quantity": 1, "unit_price": 500.75, "total": 500.75}
            ]
        }
    }
    sample_instructions = """
    1. Invoice ID must be present and in the format INV-YYYY-NNN.
    2. Vendor name must not be 'ForbiddenCorp'.
    3. Invoice date must be before the due_date.
    4. Total amount must be greater than 0.
    5. All line items must have a positive quantity and unit_price.
    """
    
    result = await checker.check_compliance(
        extracted_data=sample_extracted_data, 
        document_type="invoice", 
        compliance_instructions=sample_instructions
    )
    print("Compliance Check Result:")
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    # This allows testing the agent directly if OPENAI_API_KEY is set
    # python -m src.agents.compliance_checker_agent
    asyncio.run(_test_compliance_checker())
