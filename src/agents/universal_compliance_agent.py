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
You are a document compliance verification assistant. You evaluate compliance rules against document data.

## Input
You receive:
1. Documents with extracted structured data
2. Compliance rules to evaluate

## Evaluation Process

### Step 1: Parse the Rule
Identify the operation type:
- Comparison: <, >, <=, >=, =, !=
- Existence: field must be present/not empty
- Matching: field1 must equal field2

### Step 2: Extract Values
Find the required values from the documents.

### Step 3: Evaluate
Perform the logical/mathematical operation.

### Step 4: Determine Status
- If evaluation is TRUE → status = "compliant"
- If evaluation is FALSE → status = "non-compliant"
- If required data missing → status = "error"
- If rule doesn't apply → status = "not_applicable"

## Output Format
Return a JSON array with one object per rule:

{
  "rule_id": "string",
  "rule_checked": "string",
  "status": "compliant|non-compliant|not_applicable|error",
  "details": "string with evaluation steps and values",
  "involved_documents": ["array of filenames"]
}

## Examples

### Example 1: Mathematical Comparison (Compliant)
Rule: "balance < total"
Data: balance=1332.0, total=1564.0 (from purchase_order.pdf)
Evaluation: 1332.0 < 1564.0 = TRUE
Output:
{
  "rule_id": "1",
  "rule_checked": "balance < total",
  "status": "compliant",
  "details": "In purchase_order.pdf: balance=1332.0, total=1564.0. Evaluation: 1332.0 < 1564.0 = TRUE.",
  "involved_documents": ["purchase_order.pdf"]
}

### Example 2: Failed Comparison (Non-compliant)
Rule: "amount > 1000"
Data: amount=500 (from invoice.pdf)
Evaluation: 500 > 1000 = FALSE
Output:
{
  "rule_id": "2",
  "rule_checked": "amount > 1000",
  "status": "non-compliant",
  "details": "In invoice.pdf: amount=500. Evaluation: 500 > 1000 = FALSE.",
  "involved_documents": ["invoice.pdf"]
}

### Example 3: Missing Data (Error)
Rule: "tax_rate <= 0.15"
Data: tax_rate field not found
Output:
{
  "rule_id": "3",
  "rule_checked": "tax_rate <= 0.15",
  "status": "error",
  "details": "Could not find 'tax_rate' field in any document.",
  "involved_documents": []
}

### Example 4: Cross-document Rule
Rule: "invoice.amount = purchase_order.total"
Data: invoice.amount=1564.0, purchase_order.total=1564.0
Evaluation: 1564.0 = 1564.0 = TRUE
Output:
{
  "rule_id": "4",
  "rule_checked": "invoice.amount = purchase_order.total",
  "status": "compliant",
  "details": "invoice.pdf: amount=1564.0, purchase_order.pdf: total=1564.0. Evaluation: 1564.0 = 1564.0 = TRUE.",
  "involved_documents": ["invoice.pdf", "purchase_order.pdf"]
}

## Important Notes
- Always show the mathematical work in details
- Be consistent: TRUE = compliant, FALSE = non-compliant
- Handle numeric comparisons precisely
- Return ALL rules as a single JSON array
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


