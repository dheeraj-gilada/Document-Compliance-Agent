# src/agents/document_classifier_agent.py
import os
import re
import logging
from openai import AsyncOpenAI
from typing import Literal, Tuple

logger = logging.getLogger(__name__)

# Define the specific, expected document types for classification
DocumentType = Literal[
    "Invoice", 
    "Purchase Order", 
    "Delivery Note", 
    "Receipt", 
    "Shipping Manifest", 
    "Bill of Lading", 
    "Other"
]

class DocumentTypeClassifierAgent:
    """
    Agent responsible for inferring the type of a document using a two-tier approach:
    1. Fast filename-based classification using keywords and patterns
    2. LLM-based classification as fallback for uncertain cases
    """

    # Filename patterns and keywords for each document type
    FILENAME_PATTERNS = {
        "Invoice": [
            # Direct keywords - more flexible
            r'\binvoice\b', r'\binv\b', r'\bbill\b', r'\bfacture\b',
            # Pattern-based with flexible separators
            r'inv[_\-\s]?\d+', r'invoice[_\-\s]?\d+', r'bill[_\-\s]?\d+',
            # Common invoice prefixes - more flexible
            r'\bINV[_\-\s]?\d+', r'\bBILL[_\-\s]?\d+', r'\bFACT[_\-\s]?\d+',
            # Handle filenames with "invoice" anywhere
            r'.*invoice.*', r'.*bill.*'
        ],
        "Purchase Order": [
            # Direct keywords - more flexible  
            r'\bpurchase[_\-\s]?order\b', r'\bpo\b', r'\border\b',
            # Pattern-based with flexible separators
            r'po[_\-\s]?\d+', r'purchase[_\-\s]?order[_\-\s]?\d+', r'order[_\-\s]?\d+',
            # Common PO prefixes
            r'\bPO[_\-\s]?\d+', r'\bORDER[_\-\s]?\d+', r'\bPURCH[_\-\s]?\d+',
            # Handle compound words and variations
            r'.*purchase.*order.*', r'.*purchaseorder.*', r'.*purchase_order.*'
        ],
        "Delivery Note": [
            # Direct keywords - more flexible
            r'\bdelivery[_\-\s]?note\b', r'\bdelivery\b', r'\bshipping[_\-\s]?note\b',
            r'\bpacking[_\-\s]?slip\b', r'\bconsignment\b', r'\bdn\b',
            # Pattern-based
            r'dn[_\-\s]?\d+', r'delivery[_\-\s]?\d+', r'ship[_\-\s]?\d+',
            # Common delivery note patterns
            r'\bDN[_\-\s]?\d+', r'\bDELIV[_\-\s]?\d+', r'\bSHIP[_\-\s]?\d+',
            # Handle compound words and variations
            r'.*delivery.*note.*', r'.*deliverynote.*', r'.*delivery_note.*',
            r'.*consignment.*', r'.*packing.*slip.*'
        ],
        "Receipt": [
            # Direct keywords
            r'\breceipt\b', r'\brecu\b', r'\brcpt\b',
            # Pattern-based
            r'receipt[_\-\s]?\d+', r'rcpt[_\-\s]?\d+',
            # Common receipt patterns
            r'\bRCPT[_\-\s]?\d+', r'\bRECEIPT[_\-\s]?\d+',
            # Handle variations
            r'.*receipt.*'
        ],
        "Shipping Manifest": [
            # Direct keywords
            r'\bmanifest\b', r'\bshipping[_\-\s]?manifest\b', r'\bcargo\b',
            # Pattern-based
            r'manifest[_\-\s]?\d+', r'cargo[_\-\s]?\d+',
            # Common manifest patterns
            r'\bMANI[_\-\s]?\d+', r'\bCARGO[_\-\s]?\d+',
            # Handle variations
            r'.*manifest.*', r'.*cargo.*'
        ],
        "Bill of Lading": [
            # Direct keywords
            r'\bbill[_\-\s]?of[_\-\s]?lading\b', r'\blading\b', r'\bbol\b', r'\bb/l\b',
            # Pattern-based
            r'bol[_\-\s]?\d+', r'lading[_\-\s]?\d+', r'bl[_\-\s]?\d+',
            # Common BOL patterns
            r'\bBOL[_\-\s]?\d+', r'\bBL[_\-\s]?\d+', r'\bLADING[_\-\s]?\d+',
            # Handle variations
            r'.*bill.*lading.*', r'.*lading.*'
        ]
    }

    LLM_SYSTEM_PROMPT = """
You are an expert document classifier. Your task is to determine the type of a document based on a sample of its text content.

Based on the input, you MUST respond with ONLY ONE of the following document type categories:
- "Invoice"
- "Purchase Order"
- "Delivery Note"
- "Receipt"
- "Shipping Manifest"
- "Bill of Lading"
- "Other"

Do NOT provide any explanation, additional text, or punctuation. Your entire response must be just one of the phrases from the list above. If you are unsure, classify the document as "Other".
"""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0, confidence_threshold: float = 0.5):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if not self.client.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set for DocumentTypeClassifierAgent.")
        self.model_name = model_name
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold

    def _classify_by_filename(self, filename: str) -> Tuple[str, float]:
        """
        Classify document based on filename patterns and keywords.
        
        Args:
            filename: The filename to analyze
            
        Returns:
            Tuple of (document_type, confidence_score)
            confidence_score is between 0.0 and 1.0
        """
        filename_lower = filename.lower()
        filename_clean = re.sub(r'[^a-zA-Z0-9\s\-_]', '', filename_lower)
        
        # Track matches for each document type
        type_scores = {}
        
        for doc_type, patterns in self.FILENAME_PATTERNS.items():
            score = 0.0
            matches = 0
            
            for pattern in patterns:
                if re.search(pattern, filename_lower, re.IGNORECASE):
                    matches += 1
                    # Weight different types of matches - more generous scoring
                    if r'\b' in pattern and r'\d+' not in pattern:  # Exact word matches
                        score += 1.5
                    elif r'\b' in pattern and r'\d+' in pattern:  # Word + number patterns
                        score += 1.2
                    elif r'\d+' in pattern:  # Number patterns (ID-based matches)
                        score += 1.0
                    elif r'.*' in pattern:  # Flexible "contains" patterns
                        score += 0.8
                    else:  # General patterns
                        score += 0.6
            
            if matches > 0:
                # More generous normalization - don't divide by total patterns
                normalized_score = min(score / 2.0, 1.0)  # Divide by 2 instead of pattern count
                if matches > 1:
                    normalized_score = min(normalized_score * 1.3, 1.0)  # Bigger bonus for multiple matches
                type_scores[doc_type] = normalized_score
        
        if not type_scores:
            return "Other", 0.0
        
        # Get the best match
        best_type = max(type_scores.keys(), key=lambda k: type_scores[k])
        best_score = type_scores[best_type]
        
        # Less harsh confidence adjustments
        if len(type_scores) > 1:
            # If multiple types match, only reduce confidence if they're very close
            sorted_scores = sorted(type_scores.values(), reverse=True)
            if len(sorted_scores) > 1 and sorted_scores[0] - sorted_scores[1] < 0.1:
                best_score *= 0.9  # Smaller penalty for close matches
        
        # Check for random/generated filenames (but less harsh penalty)
        if self._is_likely_random_filename(filename):
            best_score *= 0.7  # Less harsh penalty
        
        return best_type, best_score

    def _is_likely_random_filename(self, filename: str) -> bool:
        """Check if filename appears to be randomly generated."""
        filename_base = os.path.splitext(filename)[0].lower()
        
        # Remove common separators
        clean_name = re.sub(r'[-_\s]', '', filename_base)
        
        # Indicators of random/generated filenames
        random_indicators = [
            len(clean_name) > 15 and re.match(r'^[a-f0-9]+$', clean_name),  # Long hex strings
            len(clean_name) > 10 and re.match(r'^[0-9]+$', clean_name),     # Long numeric strings
            re.match(r'^[a-zA-Z0-9]{8,}$', clean_name) and not re.search(r'[aeiou]', clean_name),  # No vowels (UUID-like)
            re.search(r'[0-9]{8,}', clean_name),  # Contains long number sequences
        ]
        
        return any(random_indicators)

    async def _classify_by_llm(self, filename: str, text_sample: str) -> str:
        """
        Classify document using LLM as fallback.
        
        Args:
            filename: The filename (for context)
            text_sample: Sample of document text content
            
        Returns:
            The classified document type
        """
        user_prompt = f"""Filename: {filename}

Text Sample (first 500 characters):
---
{text_sample[:500]}
---

What is the document type?"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
            )
            
            doc_type = response.choices[0].message.content.strip()
            
            # Validate the response against our defined types
            valid_types = list(DocumentType.__args__)
            if doc_type in valid_types:
                return doc_type
            else:
                logger.warning(f"LLM returned unexpected document type ('{doc_type}') for '{filename}'. Defaulting to 'Other'.")
                return "Other"

        except Exception as e:
            logger.error(f"Error in LLM classification for '{filename}': {e}", exc_info=True)
            return "Other"

    async def infer_document_type(self, filename: str, text_sample: str) -> DocumentType:
        """
        Main classification method using two-tier approach.
        
        Args:
            filename: The filename of the document
            text_sample: A sample of the text content from the document
            
        Returns:
            The inferred document type
        """
        # Tier 1: Fast filename-based classification
        filename_type, confidence = self._classify_by_filename(filename)
        
        logger.info(f"Filename classification for '{filename}': '{filename_type}' (confidence: {confidence:.2f})")
        
        # If confidence is high enough, use filename classification
        if confidence >= self.confidence_threshold and filename_type != "Other":
            logger.info(f"✅ Using filename classification: '{filename}' → '{filename_type}' (fast path)")
            return filename_type
        
        # Tier 2: LLM-based classification for uncertain cases
        logger.info(f"🔍 Filename confidence too low ({confidence:.2f} < {self.confidence_threshold}), using LLM classification for '{filename}'")
        
        llm_type = await self._classify_by_llm(filename, text_sample)
        logger.info(f"✅ LLM classification: '{filename}' → '{llm_type}' (LLM path)")
        
        return llm_type 