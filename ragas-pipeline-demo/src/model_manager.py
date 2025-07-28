import os
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    pipeline
)
from huggingface_hub import hf_hub_download, login, snapshot_download
import requests
from tqdm import tqdm

class ModelManager:
    """
    Manages downloading, storing, and loading of AI models locally
    """
    
    def __init__(self, base_model_dir: str = "models", cache_dir: str = "cache"):
        """
        Initialize ModelManager
        
        Args:
            base_model_dir: Directory to store downloaded models
            cache_dir: Directory for temporary cache
        """
        self.base_model_dir = Path(base_model_dir)
        self.cache_dir = Path(cache_dir)
        self.models_config_file = self.base_model_dir / "models_config.json"
        
        # Create directories
        self.base_model_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize models config
        self.models_config = self._load_models_config()
        
        # If config is empty but models directory has models, rebuild config
        if not self.models_config["downloaded_models"] and self.base_model_dir.exists():
            self.rebuild_models_config()
        
        # Supported model types
        self.supported_models = {
            "llm": ["text-generation", "causal-lm"],
            "medical_llm": ["text-generation", "medical-analysis"],
            "embedding": ["sentence-transformers", "feature-extraction"],
            "classification": ["text-classification", "sequence-classification"],
            "vision": ["image-classification", "object-detection"],
            "multimodal": ["vision-language", "document-understanding", "colpali"]
        }
    
    def _load_models_config(self) -> Dict:
        """Load existing models configuration"""
        if self.models_config_file.exists():
            with open(self.models_config_file, 'r') as f:
                return json.load(f)
        return {
            "downloaded_models": {},
            "model_metadata": {},
            "last_updated": None
        }
    
    def _save_models_config(self):
        """Save models configuration to file"""
        with open(self.models_config_file, 'w') as f:
            json.dump(self.models_config, f, indent=2)
    
    def rebuild_models_config(self):
        """
        Rebuild models configuration by scanning the models directory
        Useful when models exist but config is missing or corrupted
        """
        self.logger.info("Rebuilding models configuration...")
        
        if not self.base_model_dir.exists():
            return
        
        for model_dir in self.base_model_dir.iterdir():
            if model_dir.is_dir() and model_dir.name != "models_config.json":
                # Check if it's a valid model directory
                if (model_dir / "config.json").exists() or (model_dir / "pytorch_model.bin").exists() or (model_dir / "model.safetensors").exists():
                    model_name = model_dir.name.replace("_", "/")
                    model_id = model_dir.name
                    
                    if model_id not in self.models_config["downloaded_models"]:
                        self.logger.info(f"Found untracked model: {model_name}")
                        self.models_config["downloaded_models"][model_id] = {
                            "name": model_name,
                            "type": "llm",  # Default type
                            "local_path": str(model_dir),
                            "quantization": None,
                            "download_date": datetime.now().isoformat()
                        }
        
        self._save_models_config()
        self.logger.info("Models configuration rebuilt successfully")
    
    def authenticate_huggingface(self, token_file: str = "HF_token"):
        """
        Authenticate with Hugging Face Hub
        
        Args:
            token_file: Path to file containing HF token
        """
        try:
            token_path = Path(token_file)
            if not token_path.is_absolute():
                token_path = Path(__file__).parent / token_path
                
            if token_path.exists():
                with open(token_path, 'r') as f:
                    token = f.read().strip()
                login(token=token)
                self.logger.info("Successfully authenticated with Hugging Face Hub")
                return True
            else:
                self.logger.warning(f"HF token file not found: {token_path}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to authenticate with HF: {e}")
            return False
    
    def download_model(self, 
                      model_name: str, 
                      model_type: str = "llm",
                      force_download: bool = False,
                      quantization: Optional[str] = None) -> str:
        """
        Download a model from Hugging Face Hub
        
        Args:
            model_name: Name of the model (e.g., "google/Medgenma-4b-it")
            model_type: Type of model (llm, embedding, classification, etc.)
            force_download: Force re-download even if model exists
            quantization: Quantization type (4bit, 8bit, None)
            
        Returns:
            Local path to downloaded model
        """
        model_id = model_name.replace("/", "_")
        local_model_path = self.base_model_dir / model_id
        
        # Check if model already exists
        if local_model_path.exists() and not force_download:
            # Verify the model is properly downloaded by checking for key files
            if (local_model_path / "config.json").exists() or (local_model_path / "pytorch_model.bin").exists() or (local_model_path / "model.safetensors").exists():
                self.logger.info(f"Model {model_name} already exists locally")
                # Ensure it's tracked in config
                if model_id not in self.models_config["downloaded_models"]:
                    self.models_config["downloaded_models"][model_id] = {
                        "name": model_name,
                        "type": model_type,
                        "local_path": str(local_model_path),
                        "quantization": quantization,
                        "download_date": datetime.now().isoformat()
                    }
                    self._save_models_config()
                return str(local_model_path)
            else:
                self.logger.warning(f"Model directory {local_model_path} exists but seems incomplete, re-downloading...")
                import shutil
                shutil.rmtree(local_model_path)
        
        try:
            self.logger.info(f"Downloading model: {model_name}")
            
            # Download model using snapshot_download for complete model
            downloaded_path = snapshot_download(
                repo_id=model_name,
                cache_dir=str(self.cache_dir),
                local_dir=str(local_model_path),
                local_dir_use_symlinks=False
            )
            
            # Update models config
            self.models_config["downloaded_models"][model_id] = {
                "name": model_name,
                "type": model_type,
                "local_path": str(local_model_path),
                "quantization": quantization,
                "download_date": datetime.now().isoformat()
            }
            
            self._save_models_config()
            self.logger.info(f"Successfully downloaded {model_name} to {local_model_path}")
            
            return str(local_model_path)
            
        except Exception as e:
            self.logger.error(f"Failed to download model {model_name}: {e}")
            raise
    
    def load_model(self, 
                   model_name: str, 
                   device: str = "auto",
                   quantization: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a model for inference
        
        Args:
            model_name: Name of the model
            device: Device to load model on (auto, cpu, cuda)
            quantization: Quantization configuration
            
        Returns:
            Dictionary containing model and tokenizer
        """
        model_id = model_name.replace("/", "_")
        local_model_path = self.base_model_dir / model_id
        
        # Check if model exists locally
        if model_id not in self.models_config["downloaded_models"]:
            # Check if model directory exists but not tracked in config
            if local_model_path.exists():
                self.logger.info(f"Model {model_name} found locally but not in config, updating config...")
                # Add to config
                self.models_config["downloaded_models"][model_id] = {
                    "name": model_name,
                    "type": "llm",  # Default type
                    "local_path": str(local_model_path),
                    "quantization": None,
                    "download_date": datetime.now().isoformat()
                }
                self._save_models_config()
            else:
                self.logger.info(f"Model {model_name} not found locally, downloading...")
                self.download_model(model_name)
        
        model_info = self.models_config["downloaded_models"][model_id]
        local_path = model_info["local_path"]
        
        try:
            self.logger.info(f"Loading model: {model_name}")
            
            # Configure quantization if requested
            quantization_config = None
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(local_path)
            
            # Load model based on type
            model_type = model_info.get("type", "llm")
            
            if model_type in ["llm", "text-generation", "medical_llm"]:
                model = AutoModelForCausalLM.from_pretrained(
                    local_path,
                    quantization_config=quantization_config,
                    device_map=device if device != "auto" else "auto",
                    torch_dtype=torch.float16 if quantization else "auto"
                )
            elif model_type == "classification":
                model = AutoModelForSequenceClassification.from_pretrained(
                    local_path,
                    quantization_config=quantization_config,
                    device_map=device if device != "auto" else "auto"
                )
            elif model_type == "multimodal":
                # For ColPali and other multimodal models
                try:
                    from transformers import AutoModel
                    model = AutoModel.from_pretrained(
                        local_path,
                        quantization_config=quantization_config,
                        device_map=device if device != "auto" else "auto",
                        torch_dtype=torch.float16 if quantization else "auto"
                    )
                except:
                    # Fallback to CausalLM if AutoModel fails
                    model = AutoModelForCausalLM.from_pretrained(
                        local_path,
                        quantization_config=quantization_config,
                        device_map=device if device != "auto" else "auto",
                        torch_dtype=torch.float16 if quantization else "auto"
                    )
            else:
                # Generic auto model loading
                model = AutoModelForCausalLM.from_pretrained(
                    local_path,
                    quantization_config=quantization_config,
                    device_map=device if device != "auto" else "auto"
                )
            
            self.logger.info(f"Successfully loaded {model_name}")
            
            return {
                "model": model,
                "tokenizer": tokenizer,
                "model_name": model_name,
                "local_path": local_path
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def create_pipeline(self, 
                       model_name: str, 
                       task: str = "text-generation",
                       device: str = "auto",
                       **kwargs) -> Any:
        """
        Create a pipeline for the model
        
        Args:
            model_name: Name of the model
            task: Task type (text-generation, text-classification, etc.)
            device: Device to run on
            **kwargs: Additional pipeline arguments
            
        Returns:
            Transformers pipeline object
        """
        model_components = self.load_model(model_name, device)
        
        pipe = pipeline(
            task=task,
            model=model_components["model"],
            tokenizer=model_components["tokenizer"],
            device=0 if device == "cuda" and torch.cuda.is_available() else -1,
            **kwargs
        )
        
        return pipe
    
    def list_downloaded_models(self) -> List[Dict]:
        """List all downloaded models"""
        return list(self.models_config["downloaded_models"].values())
    
    def remove_model(self, model_name: str) -> bool:
        """
        Remove a downloaded model
        
        Args:
            model_name: Name of the model to remove
            
        Returns:
            True if successful, False otherwise
        """
        model_id = model_name.replace("/", "_")
        
        if model_id not in self.models_config["downloaded_models"]:
            self.logger.warning(f"Model {model_name} not found locally")
            return False
        
        try:
            model_info = self.models_config["downloaded_models"][model_id]
            local_path = Path(model_info["local_path"])
            
            # Remove model directory
            if local_path.exists():
                import shutil
                shutil.rmtree(local_path)
            
            # Remove from config
            del self.models_config["downloaded_models"][model_id]
            self._save_models_config()
            
            self.logger.info(f"Successfully removed model {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove model {model_name}: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get information about a downloaded model"""
        model_id = model_name.replace("/", "_")
        return self.models_config["downloaded_models"].get(model_id)
    
    def load_medgemma_model(self, variant: str = "4b-it", use_quantization: bool = True) -> Dict[str, Any]:
        """
        Load MedGemma 4B model specifically for medical analysis
        
        Args:
            variant: Model variant (default: "4b-it")
            use_quantization: Whether to use 4-bit quantization
            
        Returns:
            Dictionary containing model components
        """
        model_name = f"google/medgemma-{variant}"
        
        try:
            # Download if not exists
            if not self.get_model_info(model_name):
                self.download_model(model_name, model_type="medical_llm")
            
            # Load with specific configuration for medical use
            quantization = "4bit" if use_quantization else None
            model_components = self.load_model(
                model_name=model_name,
                device="auto",
                quantization=quantization
            )
            
            # Add MedGemma-specific metadata
            model_components["is_medical"] = True
            model_components["variant"] = variant
            model_components["capabilities"] = ["medical_analysis", "clinical_reasoning", "medical_qa"]
            
            self.logger.info(f"MedGemma {variant} loaded successfully")
            return model_components
            
        except Exception as e:
            self.logger.error(f"Failed to load MedGemma {variant}: {e}")
            raise
    
    def load_colpali_model(self, version: str = "v1.3") -> Dict[str, Any]:
        """
        Load ColPali v1.3 model for document understanding
        
        Args:
            version: ColPali version (default: "v1.3")
            
        Returns:
            Dictionary containing model components
        """
        model_name = f"vidore/colpali-{version}"
        
        try:
            # Download if not exists
            if not self.get_model_info(model_name):
                self.download_model(model_name, model_type="multimodal")
            
            # Load ColPali model
            model_components = self.load_model(
                model_name=model_name,
                device="auto"
            )
            
            # Add ColPali-specific metadata
            model_components["is_multimodal"] = True
            model_components["version"] = version
            model_components["capabilities"] = ["document_understanding", "visual_qa", "retrieval"]
            
            self.logger.info(f"ColPali {version} loaded successfully")
            return model_components
            
        except Exception as e:
            self.logger.error(f"Failed to load ColPali {version}: {e}")
            raise
    
    def create_medgemma_pipeline(self, variant: str = "4b-it", **kwargs) -> Any:
        """
        Create a pipeline specifically for MedGemma 4B model
        
        Args:
            variant: MedGemma variant (default: "4b-it")
            **kwargs: Additional pipeline arguments
            
        Returns:
            Medical analysis pipeline
        """
        model_name = f"google/medgemma-{variant}"
        
        # Set medical-specific parameters
        medical_kwargs = {
            "max_new_tokens": 512,
            "temperature": 0.1,  # Lower temperature for medical accuracy
            "do_sample": True,
            "top_p": 0.95,
            **kwargs
        }
        
        return self.create_pipeline(
            model_name=model_name,
            task="text-generation",
            **medical_kwargs
        )
    
    def setup_medical_models(self, force_download: bool = False) -> Dict[str, Any]:
        """
        Setup MedGemma 4B and ColPali v1.3 models for medical workflow
        
        Args:
            force_download: Force re-download of models
            
        Returns:
            Dictionary containing loaded models
        """
        models = {}
        
        try:
            self.logger.info("Setting up medical models...")
            
            # Setup MedGemma 4B model
            self.logger.info("Loading MedGemma 4B model...")
            models["medgemma"] = self.load_medgemma_model("4b-it")
            
            # Setup ColPali v1.3
            self.logger.info("Loading ColPali v1.3 model...")
            models["colpali"] = self.load_colpali_model("v1.3")
            
            # Setup embedding model for general text processing
            self.logger.info("Loading embedding model...")
            if force_download or not self.get_model_info("sentence-transformers/all-MiniLM-L6-v2"):
                self.download_model("sentence-transformers/all-MiniLM-L6-v2", "embedding")
            models["embeddings"] = self.load_model("sentence-transformers/all-MiniLM-L6-v2")
            
            self.logger.info("All medical models loaded successfully")
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to setup medical models: {e}")
            raise
    
    def download_recommended_models(self):
        """Download a set of recommended models for medical/general use"""
        recommended_models = [
            {
                "name": "google/medgemma-4b-it",
                "type": "medical_llm",
                "description": "MedGemma 4B instruction-tuned model for medical analysis"
            },
            {
                "name": "vidore/colpali-v1.3",
                "type": "multimodal", 
                "description": "ColPali v1.3 for document understanding"
            },
            {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "type": "embedding", 
                "description": "Sentence embeddings model"
            },
            {
                "name": "distilbert-base-uncased",
                "type": "classification",
                "description": "Text classification model"
            }
        ]
        
        for model_info in recommended_models:
            try:
                self.logger.info(f"Downloading recommended model: {model_info['name']}")
                self.download_model(
                    model_name=model_info["name"],
                    model_type=model_info["type"]
                )
            except Exception as e:
                self.logger.error(f"Failed to download {model_info['name']}: {e}")


# Example usage and utility functions
def main():
    """Example usage of ModelManager with medical models"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize model manager
    manager = ModelManager()
    
    # Authenticate with HuggingFace (optional)
    manager.authenticate_huggingface()
    
    print("=== Medical Model Manager Demo ===")
    
    try:
        # Option 1: Setup all medical models at once
        print("\n1. Setting up medical models (MedGemma + ColPali)...")
        medical_models = manager.setup_medical_models()
        print(f"Loaded models: {list(medical_models.keys())}")
        
        # Option 2: Load specific models individually
        print("\n2. Loading MedGemma 4B model...")
        medgemma = manager.load_medgemma_model("4b-it")
        print(f"MedGemma capabilities: {medgemma.get('capabilities', [])}")
        
        print("\n3. Loading ColPali model...")
        colpali = manager.load_colpali_model("v1.3")
        print(f"ColPali capabilities: {colpali.get('capabilities', [])}")
        
        # Option 3: Create medical pipeline
        print("\n4. Creating MedGemma 4B pipeline...")
        med_pipeline = manager.create_medgemma_pipeline("4b-it")
        print("MedGemma 4B pipeline created successfully!")
        
        # Test the pipeline with a medical example
        medical_prompt = """
        Patient presents with chest pain, shortness of breath, and elevated troponin levels.
        What is the most likely diagnosis and recommended treatment approach?
        """
        
        print("\n5. Testing medical analysis...")
        try:
            result = med_pipeline(medical_prompt, max_new_tokens=200)
            print(f"Medical Analysis: {result[0]['generated_text'][:200]}...")
        except Exception as e:
            print(f"Pipeline test failed: {e}")
        
        # List all downloaded models
        print("\n6. Downloaded models:")
        models = manager.list_downloaded_models()
        for model in models:
            print(f"  - {model['name']} ({model['type']})")
        
    except Exception as e:
        print(f"Error: {e}")
        
        # Fallback: Download recommended models
        print("\nFalling back to downloading recommended models...")
        try:
            manager.download_recommended_models()
            print("Recommended models downloaded successfully!")
        except Exception as e2:
            print(f"Fallback failed: {e2}")

if __name__ == "__main__":
    main()