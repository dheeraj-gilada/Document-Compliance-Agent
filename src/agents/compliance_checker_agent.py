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
                    {"role": "system", "content": """You are a meticulous compliance checking assistant. Your task is to analyze the provided 'Extracted Document Data' against the 'Compliance Instructions' and return your findings in a structured JSON format.

The JSON object MUST have two top-level keys:
1.  `compliance_status`: A string indicating the overall compliance. This can be 'compliant', 'non-compliant', 'error', or 'warning'.
    *   'non-compliant' if any rule explicitly fails.
    *   'error' if critical data is missing for multiple rules, preventing a proper assessment, or if the LLM encounters an internal error processing.
    *   'warning' if some non-critical data is missing (leading to 'error' status for those specific findings) or minor issues are found that don't constitute a hard 'fail'.
    *   'compliant' if all rules pass.
2.  `findings`: A list of objects. Each object represents a check for a single rule and MUST contain:
    *   `rule_checked`: (string) The exact compliance rule that was checked.
    *   `status`: (string) The outcome of the check for this rule. Must be one of 'pass', 'fail', or 'error'.
        *   'pass': The data complies with the rule.
        *   'fail': The data explicitly violates the rule.
        *   'error': The data required to check this rule is missing from the 'Extracted Document Data', OR the rule cannot be confidently evaluated due to ambiguity or insufficient information.
    *   `details`: (string) A clear, concise explanation for the status, especially for 'fail' or 'error' statuses. If 'error' due to missing data, specify which data was missing.

Interpret rules LITERALLY. Do not infer constraints not explicitly stated in the rule.
If a specific field mentioned in a rule (e.g., 'Invoice ID', 'Vendor Name') is NOT present in the 'Extracted Document Data', the status for that rule check should be 'error', and the details should state that the required data was missing.
Ensure the entire response is a single, valid JSON object."""},
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
        prompt += """Based on the instructions provided in the System Prompt:
Carefully evaluate each rule from the 'Compliance Instructions' against the 'Extracted Document Data'.
For each rule, determine if its status is 'pass', 'fail', or 'error'.
- Use 'pass' if the data meets the rule.
- Use 'fail' if the data explicitly violates the rule.
- Use 'error' if the data needed to check the rule is missing from the 'Extracted Document Data', or if you cannot confidently assess the rule. Provide specific details for 'fail' or 'error' statuses, including what data was missing if applicable.

Interpret rules LITERALLY. For example, if a rule says 'quantity must be positive', a value like 4.82 is compliant. Do not infer additional constraints like 'must be a whole number' unless the rule explicitly states it.

Return your complete response as a single JSON object adhering to the structure defined in the System Prompt ('compliance_status' and 'findings')."""
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
