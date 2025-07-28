import json
import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import torch
from transformers import pipeline
from PIL import Image
import PyPDF2
from byaldi import RAGMultiModalModel

from model_manager import ModelManager


class WorkflowPipeline:
    """
    Main pipeline for processing documents with AI models
    """
    
    def __init__(self, 
                 output_dir: str = "workflow_output",
                 models_dir: str = "models"):
        """
        Initialize workflow pipeline
        
        Args:
            output_dir: Directory for storing workflow outputs
            models_dir: Directory for storing downloaded models
        """
        self.output_dir = Path(output_dir)
        self.models_dir = Path(models_dir)
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize model manager
        self.model_manager = ModelManager(
            base_model_dir=str(self.models_dir),
            cache_dir=str(self.output_dir / "cache")
        )
        
        # Initialize models
        self.models = {}
        self.pipelines = {}
        
        # Workflow configuration
        self.config = {
            "model_variant": "4b-it",  # MedGemma variant ["4b-it", "27b-text-it"]
            "default_models": {
                "medical_llm": "google/medgemma-4b-it",  # Updated to 4b-it
                "multimodal": "vidore/colpali-v1.3", 
                "embedding": "sentence-transformers/all-MiniLM-L6-v2",
                "classification": "distilbert-base-uncased"
            },
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "max_length": 512,
            "temperature": 0.1  # Lower temperature for medical accuracy
        }
        
        # Initialize RAG model for ColPali
        self.rag_model = None
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"workflow_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger = logging.getLogger(__name__)
        logger.info(f"Workflow started - Log file: {log_file}")
        return logger
    
    def setup_models(self, force_download: bool = False):
        """
        Setup and download required models
        
        Args:
            force_download: Force re-download of models
        """
        self.logger.info("Setting up medical models...")
        
        # Authenticate with HuggingFace if token available
        self.model_manager.authenticate_huggingface()
        
        try:
            # Setup MedGemma 4B model specifically
            model_variant = self.config["model_variant"]
            model_id = f"google/medgemma-{model_variant}"
            
            self.logger.info(f"Loading MedGemma {model_variant} model...")
            medgemma_model = self.model_manager.load_medgemma_model(model_variant)
            self.models["medical_llm"] = medgemma_model
            
            # Setup ColPali with Byaldi RAG
            self.logger.info("Loading ColPali with Byaldi RAG...")
            try:
                self.rag_model = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.3")
                self.logger.info("Byaldi ColPali RAG model loaded successfully")
                
                # Also load ColPali model components for direct access
                colpali_model = self.model_manager.load_colpali_model("v1.3")
                self.models["multimodal"] = colpali_model
                
            except Exception as e:
                self.logger.warning(f"Failed to load Byaldi RAG model: {e}")
                # Fallback to regular ColPali loading
                colpali_model = self.model_manager.load_colpali_model("v1.3")
                self.models["multimodal"] = colpali_model
            
            # Load embedding model
            self.logger.info("Loading embedding model...")
            embedding_model = self.config["default_models"]["embedding"]
            if force_download or not self.model_manager.get_model_info(embedding_model):
                self.model_manager.download_model(
                    model_name=embedding_model,
                    model_type="embedding"
                )
            self.models["embedding"] = self.model_manager.load_model(embedding_model)
            
            # Load classification model separately
            self.logger.info("Loading classification model...")
            classification_model = self.config["default_models"]["classification"]
            
            if force_download or not self.model_manager.get_model_info(classification_model):
                self.model_manager.download_model(
                    model_name=classification_model,
                    model_type="classification"
                )
            
            self.models["classification"] = self.model_manager.load_model(classification_model)
            
            self.logger.info("All models setup complete")
            self.logger.info(f"Loaded models: {list(self.models.keys())}")
            if self.rag_model:
                self.logger.info("Byaldi RAG model is available for document indexing")
            
        except Exception as e:
            self.logger.error(f"Failed to setup models: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            self.logger.info(f"Extracting text from PDF: {pdf_path}")
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            
            self.logger.info(f"Successfully extracted {len(text)} characters from PDF")
            return text.strip()
            
        except Exception as e:
            self.logger.error(f"Failed to extract text from PDF: {e}")
            raise
    
    def process_document_with_colpali(self, 
                                    pdf_path: str, 
                                    query: str, 
                                    k: int = 5) -> Dict[str, Any]:
        """
        Process document using ColPali with Byaldi RAG
        
        Args:
            pdf_path: Path to PDF document
            query: Query for retrieval
            k: Number of results to retrieve
            
        Returns:
            Processing results with retrieved content
        """
        try:
            if not self.rag_model:
                self.logger.warning("Byaldi RAG model not available, using fallback")
                return {
                    "success": False,
                    "error": "RAG model not loaded",
                    "fallback_used": True
                }
            
            self.logger.info(f"Processing document with ColPali RAG: {pdf_path}")
            
            # Index the document
            self.logger.info("Indexing document with ColPali...")
            doc_name = Path(pdf_path).stem
            
            self.rag_model.index(
                input_path=pdf_path,
                index_name=f"doc_{doc_name}",
                store_collection_with_index=True,
                overwrite=True
            )
            
            # Search with query
            self.logger.info(f"Searching with query: {query}")
            results = self.rag_model.search(query, k=k)
            
            # Process results
            retrieved_content = []
            for i, result in enumerate(results):
                retrieved_content.append({
                    "rank": i + 1,
                    "content": getattr(result, 'text', str(result)),
                    "score": getattr(result, 'score', 0.0),
                    "page": getattr(result, 'page_num', None)
                })
            
            processing_result = {
                "success": True,
                "document_path": pdf_path,
                "query": query,
                "num_results": len(results),
                "retrieved_content": retrieved_content,
                "index_name": f"doc_{doc_name}",
                "processing_timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"ColPali processing completed - retrieved {len(results)} results")
            return processing_result
            
        except Exception as e:
            self.logger.error(f"Failed to process document with ColPali: {e}")
            return {
                "success": False,
                "error": str(e),
                "document_path": pdf_path,
                "query": query
            }
    
    def process_text_with_llm(self, 
                             text: str, 
                             query: str, 
                             max_length: Optional[int] = None) -> Dict[str, Any]:
        """
        Process text with MedGemma language model
        
        Args:
            text: Input text to process
            query: Query or prompt for processing
            max_length: Maximum response length
            
        Returns:
            Processing results
        """
        try:
            self.logger.info("Processing text with MedGemma...")
            
            max_length = max_length or self.config["max_length"]
            
            # Prepare medical analysis prompt for MedGemma
            medical_prompt = f"""Please analyze the following medical document and answer the query.

Document content: {text[:2000]}...

Query: {query}

Please provide a detailed medical analysis including:
1. Key findings
2. Clinical significance  
3. Recommendations
4. Assessment

Analysis:"""
            
            # Use MedGemma model
            if "medical_llm" not in self.models:
                raise ValueError("MedGemma model not loaded")
            
            model = self.models["medical_llm"]["model"]
            tokenizer = self.models["medical_llm"]["tokenizer"]
            
            # Tokenize input
            inputs = tokenizer.encode(medical_prompt, return_tensors="pt", truncation=True, max_length=1024)
            
            # Generate response with medical-appropriate parameters
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=max_length,
                    temperature=self.config["temperature"],  # Lower temp for medical accuracy
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract generated part (remove prompt)
            generated_text = response[len(tokenizer.decode(inputs[0], skip_special_tokens=True)):].strip()
            
            result = {
                "query": query,
                "response": generated_text,
                "input_length": len(text),
                "response_length": len(generated_text),
                "model_used": self.config["default_models"]["medical_llm"],
                "model_type": "medgemma",
                "is_medical_analysis": True
            }
            
            self.logger.info("Medical text processing completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process text with MedGemma: {e}")
            raise
    
    def classify_text(self, text: str) -> Dict[str, Any]:
        """
        Classify text using classification model
        
        Args:
            text: Text to classify
            
        Returns:
            Classification results
        """
        try:
            self.logger.info("Classifying text...")
            
            if "classification" not in self.models:
                raise ValueError("Classification model not loaded")
            
            # Create classification pipeline if not exists
            if "classification" not in self.pipelines:
                self.pipelines["classification"] = self.model_manager.create_pipeline(
                    model_name=self.config["default_models"]["classification"],
                    task="text-classification",
                    device=self.config["device"]
                )
            
            # Perform classification
            results = self.pipelines["classification"](text[:512])  # Limit text length
            
            classification_result = {
                "predictions": results,
                "text_length": len(text),
                "model_used": self.config["default_models"]["classification"]
            }
            
            self.logger.info("Text classification completed")
            return classification_result
            
        except Exception as e:
            self.logger.error(f"Failed to classify text: {e}")
            raise
    
    def run_complete_pipeline(self, 
                             document_path: str, 
                             query: str,
                             output_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Run complete processing pipeline with ColPali RAG and MedGemma 4B
        
        Args:
            document_path: Path to document to process
            query: Query for processing
            output_filename: Optional output filename
            
        Returns:
            Complete pipeline results
        """
        try:
            self.logger.info(f"Starting complete pipeline for: {document_path}")
            
            # Generate unique pipeline ID
            pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Step 1: Extract text from document
            if document_path.lower().endswith('.pdf'):
                text_content = self.extract_text_from_pdf(document_path)
            else:
                # For other file types, read as text
                with open(document_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
            
            # Step 2: Process with ColPali RAG for document understanding
            colpali_results = self.process_document_with_colpali(document_path, query)
            
            # Step 3: Process with MedGemma 4B language model
            # Use retrieved content if available, otherwise use full text
            if colpali_results.get("success") and colpali_results.get("retrieved_content"):
                # Combine retrieved content for context
                retrieved_text = "\n".join([
                    item["content"] for item in colpali_results["retrieved_content"][:3]
                ])
                context_text = f"Retrieved relevant content:\n{retrieved_text}\n\nFull document preview:\n{text_content[:1000]}..."
            else:
                context_text = text_content
            
            llm_results = self.process_text_with_llm(context_text, query)
            
            # Step 4: Classify the content
            classification_results = self.classify_text(text_content)
            
            # Step 5: Compile results
            pipeline_results = {
                "pipeline_id": pipeline_id,
                "timestamp": datetime.now().isoformat(),
                "input_document": document_path,
                "query": query,
                "model_info": {
                    "medgemma_variant": self.config["model_variant"],
                    "colpali_version": "v1.3",
                    "rag_available": self.rag_model is not None
                },
                "text_extraction": {
                    "success": True,
                    "text_length": len(text_content),
                    "preview": text_content[:200] + "..." if len(text_content) > 200 else text_content
                },
                "colpali_processing": colpali_results,
                "llm_processing": llm_results,
                "classification": classification_results,
                "config_used": self.config.copy()
            }
            
            # Step 6: Save results
            output_filename = output_filename or f"{pipeline_id}_results.json"
            output_path = self.output_dir / output_filename
            
            with open(output_path, 'w') as f:
                json.dump(pipeline_results, f, indent=2, default=str)
            
            self.logger.info(f"Pipeline completed successfully. Results saved to: {output_path}")
            
            return pipeline_results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return {
                "pipeline_id": f"failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def list_available_models(self) -> List[Dict]:
        """List all available/downloaded models"""
        return self.model_manager.list_downloaded_models()
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            "models_loaded": list(self.models.keys()),
            "pipelines_created": list(self.pipelines.keys()),
            "device": self.config["device"],
            "output_dir": str(self.output_dir),
            "models_dir": str(self.models_dir)
        }


def main():
    """Main execution function"""
    # Initialize workflow
    workflow = WorkflowPipeline()
    
    try:
        # Setup models
        workflow.setup_models()
        
        # Find documents to process
        data_dir = Path(__file__).parent.parent / "data"
        if not data_dir.exists():
            workflow.logger.error(f"Data directory not found: {data_dir}")
            return
        
        # Look for PDF files
        pdf_files = list(data_dir.glob("*.pdf"))
        if not pdf_files:
            workflow.logger.error(f"No PDF files found in {data_dir}")
            return
        
        # Process first PDF file found
        pdf_file = pdf_files[0]
        query = "What are the key medical findings and clinical recommendations in this document?"
        
        workflow.logger.info(f"Processing document: {pdf_file.name}")
        workflow.logger.info(f"Query: {query}")
        workflow.logger.info(f"Using MedGemma {workflow.config['model_variant']} and ColPali v1.3")
        
        # Run pipeline
        results = workflow.run_complete_pipeline(str(pdf_file), query)
        
        if "error" not in results:
            workflow.logger.info("=== PIPELINE RESULTS ===")
            workflow.logger.info(f"Pipeline ID: {results['pipeline_id']}")
            workflow.logger.info(f"Models used: MedGemma {results['model_info']['medgemma_variant']}, ColPali {results['model_info']['colpali_version']}")
            workflow.logger.info(f"RAG available: {results['model_info']['rag_available']}")
            workflow.logger.info(f"Text extracted: {results['text_extraction']['text_length']} characters")
            
            # ColPali results
            colpali = results.get('colpali_processing', {})
            if colpali.get('success'):
                workflow.logger.info(f"ColPali retrieved: {colpali.get('num_results', 0)} relevant passages")
            
            # MedGemma results  
            llm_response = results['llm_processing']['response']
            workflow.logger.info(f"MedGemma 4B Analysis: {llm_response[:300]}...")
            
            # Classification
            workflow.logger.info(f"Classification: {results['classification']['predictions']}")
        else:
            workflow.logger.error(f"Pipeline failed: {results['error']}")
    
    except Exception as e:
        workflow.logger.error(f"Main execution failed: {e}")


if __name__ == "__main__":
    main()
