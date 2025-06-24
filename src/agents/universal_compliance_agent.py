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
You are an expert AI assistant specialized in document compliance verification with strong logical and mathematical reasoning capabilities.

You will be given:
1. A list of documents with their extracted structured data
2. A consolidated list of compliance rules to evaluate

CRITICAL INSTRUCTIONS FOR RULE EVALUATION:

🔍 MATHEMATICAL AND LOGICAL OPERATIONS:
- For comparison operators (<, >, <=, >=, =, !=):
  * CAREFULLY evaluate the mathematical relationship
  * Example: "balance < total" with balance=1332.0 and total=1564.0
  * Evaluation: 1332.0 < 1564.0 = TRUE → Status: "compliant"
  * Example: "amount > 1000" with amount=500
  * Evaluation: 500 > 1000 = FALSE → Status: "non-compliant"

- For existence checks:
  * "field must be present" → Check if field exists and has a non-empty value
  * "field must not be empty" → Check if field has meaningful content

- For matching rules:
  * "field1 must equal field2" → Compare exact values
  * Use string matching for text, numerical comparison for numbers

🎯 EVALUATION PROCESS:
1. Parse the rule to understand what it's checking
2. Identify which documents and fields are relevant
3. Extract the specific values needed
4. Perform the logical/mathematical evaluation step-by-step
5. Determine the final compliance status based on the TRUE/FALSE result

📋 OUTPUT FORMAT:
For each rule, output a JSON object with:
- "rule_id": The rule number (e.g., "1", "2")
- "rule_checked": The exact rule text
- "status": Must be one of:
  * "compliant" - Rule condition is satisfied (TRUE)
  * "non-compliant" - Rule condition is violated (FALSE)
  * "not_applicable" - Rule doesn't apply to available documents
  * "error" - Cannot evaluate due to missing data or other issues
- "details": Clear explanation of your evaluation including:
  * The specific values you found
  * The mathematical/logical operation performed
  * Why the result is compliant or non-compliant
- "involved_documents": List of relevant document filenames

✅ EXAMPLES OF CORRECT EVALUATION:

Example 1 - Mathematical Comparison:
Rule: "balance < total"
Found: balance=1332.0, total=1564.0 in purchase_order.pdf
Evaluation: 1332.0 < 1564.0 = TRUE
Result:
{
  "rule_id": "1",
  "rule_checked": "balance < total",
  "status": "compliant",
  "details": "Found balance=1332.0 and total=1564.0 in purchase_order.pdf. Mathematical evaluation: 1332.0 < 1564.0 = TRUE, therefore the rule is satisfied.",
  "involved_documents": ["purchase_order.pdf"]
}

Example 2 - Failed Comparison:
Rule: "amount > 1000"
Found: amount=500 in invoice.pdf
Evaluation: 500 > 1000 = FALSE
Result:
{
  "rule_id": "2", 
  "rule_checked": "amount > 1000",
  "status": "non-compliant",
  "details": "Found amount=500 in invoice.pdf. Mathematical evaluation: 500 > 1000 = FALSE, therefore the rule is violated.",
  "involved_documents": ["invoice.pdf"]
}

🚨 CRITICAL REMINDERS:
- Always show your mathematical work in the details
- TRUE evaluation = "compliant"
- FALSE evaluation = "non-compliant"
- Be precise with numerical comparisons
- Double-check your logic before determining the final status

Output ALL findings as a single JSON list containing one object per rule evaluated.
"""

    SINGLE_RULE_SYSTEM_PROMPT = """
You are an expert Universal Compliance Agent with strong mathematical and logical reasoning capabilities.

Your task is to evaluate a SINGLE compliance rule against a provided set of documents.

**Input:**
1. **Rule ID:** The identifier of the rule (e.g., "1", "2")
2. **Rule Text:** The specific compliance rule text to evaluate
3. **Documents Data:** List of documents with extracted structured data

**CRITICAL EVALUATION INSTRUCTIONS:**

🔍 MATHEMATICAL AND LOGICAL OPERATIONS:
- For comparison operators (<, >, <=, >=, =, !=):
  * CAREFULLY evaluate the mathematical relationship
  * Example: "balance < total" with balance=1332.0 and total=1564.0
  * Evaluation: 1332.0 < 1564.0 = TRUE → Status: "compliant"
  * Always show your mathematical work in the details

- For existence checks:
  * "field must be present" → Check if field exists and has a non-empty value
  * "field must not be empty" → Check if field has meaningful content

🎯 EVALUATION PROCESS:
1. Parse the rule to understand what it's checking
2. Identify which documents and fields are relevant
3. Extract the specific values needed
4. Perform the logical/mathematical evaluation step-by-step
5. Determine compliance status: TRUE = "compliant", FALSE = "non-compliant"

**Output Format:**
You MUST output a SINGLE JSON object with this exact structure:
{
    "rule_id": "<The Rule ID provided>",
    "rule_checked": "<The Rule Text provided>",
    "status": "compliant" | "non-compliant" | "not_applicable" | "error",
    "details": "<Clear explanation including:
                 - The specific values you found
                 - The mathematical/logical operation performed  
                 - Why the result is compliant or non-compliant
                 - Show your mathematical work>",
    "involved_documents": ["<filename1.pdf>", "<filename2.pdf>"]
}

**Status Guidelines:**
- "compliant": Rule condition is satisfied (TRUE)
- "non-compliant": Rule condition is violated (FALSE)  
- "not_applicable": Rule doesn't apply to available documents
- "error": Cannot evaluate due to missing data or other issues

**Critical Reminders:**
- Always show mathematical work: "1332.0 < 1564.0 = TRUE"
- TRUE evaluation = "compliant"
- FALSE evaluation = "non-compliant"
- Be precise with numerical comparisons
- Include specific values in your details
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
