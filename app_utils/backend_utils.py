import shutil
import os
from typing import List, Dict, Any, Optional
import streamlit as st

import haystack
from haystack.schema import Document
from haystack.document_stores.faiss import FAISSDocumentStore
from haystack.nodes.retriever.dense import EmbeddingRetriever
from haystack.pipelines import Pipeline
from haystack_entailment_checker import EntailmentChecker

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from app_utils.config import (
    STATEMENTS_PATH,
    INDEX_DIR,
    RETRIEVER_MODEL,
    RETRIEVER_MODEL_FORMAT,
    NLI_MODEL,
    PROMPT_MODEL,
)


@st.cache_data
def load_statements():
    """Load statements from file"""
    with open(STATEMENTS_PATH) as fin:
        statements = [
            line.strip() for line in fin.readlines() if not line.startswith("#")
        ]
    return statements


@st.cache_resource
def start_haystack():
    """
    load document store, retriever, entailment checker and create pipeline
    """
    # Copy database file to current directory as required by Haystack
    shutil.copy(f"{INDEX_DIR}/faiss_document_store.db", ".")

    # Initialize official Haystack FAISSDocumentStore
    document_store = FAISSDocumentStore(
        faiss_index_path=f"{INDEX_DIR}/my_faiss_index.faiss",
        faiss_config_path=f"{INDEX_DIR}/my_faiss_index.json",
    )
    print(f"Index size: {document_store.get_document_count()}")

    # Initialize official Haystack EmbeddingRetriever
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model=RETRIEVER_MODEL,
        model_format=RETRIEVER_MODEL_FORMAT,
    )

    # Initialize official EntailmentChecker from haystack-entailment-checker
    entailment_checker = EntailmentChecker(
        model_name_or_path=NLI_MODEL,
        use_gpu=True,  # Use GPU if available
        entailment_contradiction_threshold=0.5,
    )

    # Create Haystack Pipeline
    pipe = Pipeline()
    pipe.add_node(component=retriever, name="retriever", inputs=["Query"])
    pipe.add_node(component=entailment_checker, name="ec", inputs=["retriever"])

    class TransformersPromptNode:
        """
        A prompt node implementation using Transformers directly instead of Haystack's PromptNode
        """

        def __init__(self, model_name_or_path, use_gpu=True, max_length=150):
            """Initialize the model and tokenizer"""
            self.model_name = model_name_or_path
            self.max_length = max_length
            self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, token=os.environ.get("HUGGINGFACE_TOKEN")
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path, token=os.environ.get("HUGGINGFACE_TOKEN")
            ).to(self.device)

            print(f"Loaded {model_name_or_path} on {self.device}")

        def __call__(self, prompt):
            """Generate a response for the given prompt"""
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=self.max_length,
                num_beams=4,
                early_stopping=True,
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return [response]

    prompt_node = TransformersPromptNode(
        model_name_or_path=PROMPT_MODEL, use_gpu=True, max_length=150
    )

    print(f"✅ Pipeline initialized with official Haystack components:")
    print(f"  📚 Documents: {document_store.get_document_count()}")
    print(f"  🔍 Retriever: {RETRIEVER_MODEL}")
    print(f"  🧠 NLI Model: {NLI_MODEL}")
    print(f"  💬 LLM Model: {PROMPT_MODEL}")

    return pipe, prompt_node


pipe = None
prompt_node = None


def _ensure_initialized():
    """Ensure pipeline is initialized"""
    global pipe, prompt_node
    if pipe is None or prompt_node is None:
        pipe, prompt_node = start_haystack()


@st.cache_resource
def check_statement(statement: str, retriever_top_k: int = 10):
    """Run fact-checking pipeline on statement"""
    _ensure_initialized()

    params = {"retriever": {"top_k": retriever_top_k}}
    return pipe.run(statement, params=params)


@st.cache_data
def explain_using_llm(
    statement: str, documents: List[Document], entailment_or_contradiction: str
) -> str:
    """Generate explanation using LLM

    Args:
        statement: The claim to explain
        documents: Retrieved documents
        entailment_or_contradiction: The relationship type (entailment/contradiction/neutral)
    """

    if not documents:
        return f"No supporting documents found for: '{statement}'"

    _ensure_initialized()

    premise = " \n".join(
        [doc.content[:1000].replace("\n", ". ") for doc in documents[:2]]
    )

    # Use appropriate verb for the prompt based on entailment relationship
    if entailment_or_contradiction == "entailment":
        verb = "entails (supports)"
    elif entailment_or_contradiction == "contradiction":
        verb = "contradicts"
    else:
        verb = "is neutral about"

    prompt = f"Explain why this statement: '{statement}' is {verb} by the following evidence: {premise}"

    try:
        # Add a timeout to prevent hanging on large models
        with torch.no_grad():
            return prompt_node(prompt)[0]
    except Exception as e:
        print(f"Error in LLM explanation: {e}")
        import traceback

        error_details = traceback.format_exc()
        print(error_details)

        action = {
            "entailment": "supports",
            "contradiction": "contradicts",
            "neutral": "is neutral about",
        }
        return (
            f"Based on the retrieved documents, the evidence {action.get(entailment_or_contradiction, 'relates to')} "
            f"the claim: '{statement}'\n\n"
            f"Error generating detailed explanation: {str(e)}"
        )
