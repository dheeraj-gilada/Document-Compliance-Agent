# Intelligent Document Compliance & Process Automation Agent

## 1. Overview

This project implements an intelligent agent capable of processing various document types, extracting structured data, and evaluating them against a user-defined set of compliance rules. It leverages Large Language Models (LLMs) for its core intelligence and LangGraph for orchestrating the processing workflow. A Streamlit application provides a user-friendly interface for document uploads and rule management.

## 2. Architecture and Components

The system is built upon a modular architecture orchestrated by LangGraph.

**Core Components:**
*   **Streamlit UI (`streamlit_app.py`):** Provides the user interface for:
    *   Uploading documents.
    *   Dynamically adding, editing, and deleting compliance rules.
    *   Initiating the compliance check process.
    *   Displaying extracted data and compliance findings.
*   **Main Orchestrator (`main.py`):**
    *   Entry point for the backend processing logic.
    *   Initializes and invokes the LangGraph workflow.
    *   Supports CLI operations for 'extract', 'compliance', and 'full' modes.
*   **LangGraph Workflow (`src/workflows/document_processing_workflow.py`):**
    *   Defines the state machine for document processing.
    *   Manages the flow of operations:
        1.  Document Listing & Queuing
        2.  Document Loading & Classification (per document)
        3.  Structured Data Extraction (per document)
        4.  Rule Initialization & Queuing (after all documents processed)
        5.  Single Rule Evaluation (per rule, against all processed documents)
        6.  Aggregation of Findings
*   **Agents (`src/agents/`):**
    *   `DocumentLoader` (utility in `src/utils/document_loader.py`): Uses the `unstructured` library to load various file types (PDF, DOCX, TXT, images) and perform initial content extraction (text, tables).
    *   `DocumentTypeClassifierAgent`: Uses an LLM (OpenAI `gpt-4o-mini`) to determine the type of a document (e.g., invoice, purchase order).
    *   `StructuredDataExtractorAgent`: Uses an LLM (OpenAI `gpt-4o-mini`) to extract structured key-value data from the document's content based on its type.
    *   `UniversalComplianceAgent`: Uses an LLM (OpenAI `gpt-4o-mini`) to evaluate a single compliance rule against the aggregated data from all processed documents.
*   **Configuration:**
    *   Compliance rules are provided by the user.
    *   LLM prompts are embedded within the respective agents.

**Data Flow (Simplified):**
User Uploads (Docs, Rules via Streamlit) -> `streamlit_app.py` -> `main.py` (invokes LangGraph)
LangGraph Workflow:
  Docs -> Load/Classify -> Extract Data (repeated for each doc)
  Rules -> Parse -> Evaluate Rule (repeated for each rule against all extracted data)
Results -> `main.py` -> `streamlit_app.py` -> Display to User

## 3. Tools, Models, and Reasoning Approach

*   **Primary Language:** Python (3.10)
*   **Workflow Orchestration:** `LangGraph` - Chosen for its flexibility in defining complex, stateful agentic workflows.
*   **Document Parsing:** `unstructured` library - Handles the complexities of extracting text and table data from diverse file formats.
*   **User Interface:** `Streamlit` - Enables rapid development of an interactive web application.
*   **LLM Provider:** OpenAI
*   **LLM Model:** `gpt-4o-mini` (default) - Selected for its balance of capability and cost for tasks like classification, data extraction, and rule-based reasoning.
*   **Reasoning Approach:**
    *   **Modular LLM Calls:** Each agent (classification, extraction, compliance) makes targeted calls to the LLM with specific prompts tailored to its task.
    *   **Data-Driven Extraction:** The `StructuredDataExtractorAgent` uses the document type (inferred by the `DocumentTypeClassifierAgent`) to guide the LLM in extracting relevant fields.
    *   **Rule-Level Compliance:** The `UniversalComplianceAgent` evaluates each user-defined rule individually against the full context of extracted data from all documents. This allows for granular feedback and better debuggability. The system prompt for this agent is designed to focus the LLM on the specific rule and the provided data.
    *   **State Management:** LangGraph maintains the state of the processing, including document queues, extracted data, rule queues, and aggregated findings.

## 4. Installation and Setup

**Prerequisites:**
*   Python 3.10 or higher.
*   Conda (recommended for environment management).
*   Access to an OpenAI API key.
*   (macOS) Homebrew for installing system dependencies. Other OS users will need to use their respective package managers.

**Steps:**

1.  **Clone the Repository (if applicable):**
    ```bash
    # git clone <repository_url>
    # cd <repository_directory>
    ```
    (If you downloaded the code as a ZIP, extract it and navigate to the project directory.)

2.  **Create and Activate Conda Environment:**
    ```bash
    conda create -n doc_agent_env python=3.10 -y
    conda activate doc_agent_env
    ```

3.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install System-Level Dependencies for `unstructured`:**
    The `unstructured` library requires certain system packages for processing different file types.

    *   **On macOS (using Homebrew):**
        ```bash
        brew install libmagic poppler tesseract libreoffice pandoc
        ```
    *   **On Linux (Debian/Ubuntu example):**
        ```bash
        sudo apt-get update
        sudo apt-get install -y libmagic1 poppler-utils tesseract-ocr libreoffice pandoc
        ```
    *   **On Windows:** Installation can be more complex. Refer to the individual documentation for `libmagic`, `poppler`, `tesseract`, `LibreOffice`, and `Pandoc` for Windows installation instructions. Ensure their binaries are added to your system's PATH.

5.  **Set OpenAI API Key:**
    The application requires an OpenAI API key to function. Set it as an environment variable:
    ```bash
    export OPENAI_API_KEY="your_openai_api_key_here"
    ```
    For persistent storage, you can add this line to your shell's configuration file (e.g., `~/.bashrc`, `~/.zshrc`) or use a `.env` file management approach (though current setup reads directly from env).

## 5. How to Run

**A. Streamlit Web Application (Recommended for interactive use):**

1.  Ensure your Conda environment (`doc_agent_env`) is active and `OPENAI_API_KEY` is set.
2.  Navigate to the project's root directory in your terminal.
3.  Run the Streamlit application:
    ```bash
    streamlit run streamlit_app.py
    ```
4.  The application will open in your web browser. You can then:
    *   Upload documents.
    *   Add, edit, or delete compliance rules using the sidebar.
    *   Click "Run Compliance Check" to process.

## 6. Assumptions, Limitations, and Future Improvements

**Assumptions:**
*   Documents are in one of the supported formats (PDF, DOCX, TXT, JPEG, JPG, PNG, WEBP).
*   Compliance rules are clearly defined and can be reasonably evaluated by an LLM against extracted text/data.
*   The OpenAI API key provided is valid and has sufficient quota.
*   System dependencies for `unstructured` are correctly installed.

**Limitations:**
*   **LLM Dependency:** The accuracy of document classification, data extraction, and compliance evaluation is inherently tied to the capabilities and potential biases of the `gpt-4o-mini` model. Complex or ambiguous cases may yield suboptimal results.
*   **Complex Document Structures:** While `unstructured` is powerful, extremely complex layouts, dense tables, or poor quality scans might not be parsed perfectly.
*   **Error Handling:** While basic error handling is in place, the system could be made more resilient to unexpected issues during processing.
*   **Scalability:** The current implementation processes documents and rules sequentially in parts of the workflow. Performance might degrade with a very large number of documents or an extensive rule set in a single run.
*   **No Data Persistence (Streamlit):** Rules added in the Streamlit app are session-based and not saved persistently.
*   **Image Modality:** For images, text extraction relies on OCR (e.g., Tesseract via `unstructured`). The quality of OCR can vary.

**Future Improvements:**
*   **Advanced Rule Engine:** Implement a more sophisticated rule definition and management system, potentially with support for logical operators, conditions, and different rule types beyond LLM evaluation.
*   **Enhanced Document Preprocessing:** Integrate more advanced OCR tools or image preprocessing steps for better text extraction from challenging images/PDFs.
*   **Fine-tuning LLMs:** For specific, high-volume document types or compliance regimes, fine-tuning smaller LLMs could improve accuracy and reduce costs.
*   **Vector Database for Semantic Rule Matching:** Store rules and document chunks in a vector DB to find relevant rules or document sections semantically.
*   **Batch Processing & Asynchronous Operations:** Optimize for handling large batches of documents more efficiently.
*   **Persistent Storage for Streamlit:** Allow users to save and load rule sets in the Streamlit application.
*   **Support for More File Types:** Extend `DocumentLoader` as needed.
*   **Confidence Scoring:** Provide confidence scores for extractions and compliance findings.
*   **Interactive Feedback Loop:** Allow users to correct extractions or validate compliance findings, potentially feeding this back to improve the system.
