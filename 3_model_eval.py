"""
LLM Philosopher Response Generator - Unified Version

Supports multiple model providers and modes:
- OpenAI (GPT-4, GPT-4o, GPT-5.1, etc.) - sync or batch API
- Anthropic (Claude 3.5 Sonnet, Claude 4, etc.) - sync or batch API
- HuggingFace (any model, including fine-tuned LoRA adapters) - sync only

Modes:
- "sync": Real-time API calls (good for small batches, testing)
- "batch_openai": OpenAI Batch API (50% discount, 24h turnaround)
- "batch_anthropic": Anthropic Batch API (50% discount)

Features:
- Clean provider abstraction layer
- Environment-based API key configuration
- Progress tracking and resume capability
- Support for fine-tuned LoRA models
- Batch splitting for large datasets

Usage:
    # Set API keys as environment variables:
    export OPENAI_API_KEY="your-key-here"
    export ANTHROPIC_API_KEY="your-key-here"
    export HF_TOKEN="your-key-here"
    
    # Run the script:
    python 3_model_eval.py
    
    # Limit to 100 evaluations (for testing):
    python 3_model_eval.py --limit 100
    
    # Specify provider and mode:
    python 3_model_eval.py --provider huggingface --mode sync --limit 50
    
    # Use a specific model:
    python 3_model_eval.py --model "Qwen/Qwen2.5-3B-Instruct" --limit 20
    
    # Use fine-tuned model:
    python 3_model_eval.py --finetuned --provider huggingface
"""

import os
import json
import time
import re
import argparse
from datetime import datetime
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from tqdm.auto import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# Mode: "sync", "batch_openai", "batch_anthropic", "auto"
# "auto" will detect available API keys and choose the best option
MODE = "auto"

# Model Provider Configuration (for sync mode)
# Options: "openai", "anthropic", "huggingface", "auto"
# "auto" will detect available API keys and fallback to huggingface if none found
MODEL_PROVIDER = "auto"

# Model name for each provider
OPENAI_MODEL = "gpt-4o"  # or "gpt-4", "gpt-3.5-turbo", "gpt-4-turbo", "gpt-5.1-2025-11-13"
ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"  # or "claude-3-5-sonnet-20241022"
HUGGINGFACE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # or "meta-llama/Llama-3.1-8B-Instruct"

# Fine-tuned Model Configuration (HuggingFace only)
USE_FINETUNED_MODEL = False  # Set to True to use your fine-tuned DPO model
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # Base model used for fine-tuning (must match 7_finetune_dpo.py)
LORA_ADAPTER_PATH = "./qwen2.5_0.5b_philosopher_dpo/final_model"  # Path to fine-tuned LoRA adapter

# API Keys (read from environment variables for security)
# export OPENAI_API_KEY="your-key-here"
# export ANTHROPIC_API_KEY="your-key-here"
# export HF_TOKEN="your-key-here"


def detect_available_provider() -> str:
    """Auto-detect which provider to use based on available API keys"""
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    elif os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    else:
        return "huggingface"


def detect_available_mode() -> str:
    """Auto-detect which mode to use based on available API keys"""
    if os.getenv("OPENAI_API_KEY"):
        return "sync"  # Default to sync for OpenAI
    elif os.getenv("ANTHROPIC_API_KEY"):
        return "sync"  # Default to sync for Anthropic
    else:
        return "sync"  # HuggingFace only supports sync

# Generation Settings
MAX_TOKENS = 100
MAX_RETRIES = 5
TEMPERATURE = 0.0

# File Paths (relative to script directory or absolute)
DATA_DIR = "."
OUTPUT_DIR = f"llm_responses_{MODEL_PROVIDER}"
RESUME_FILE = f"llm_responses_progress_{MODEL_PROVIDER}.json"
PHILOSOPHERS_FILE = "philosophers_with_countries.json"
QUESTIONS_FILE = "question_answer_options.json"

# Batch API Settings
BATCH_REQUESTS_FILE = "batch_requests.jsonl"
BATCH_MAPPING_FILE = "batch_requests_mapping.json"
BATCH_RESULTS_FILE = "batch_results.jsonl"
REQUESTS_PER_BATCH = 100  # For OpenAI batch splitting
MAX_CONCURRENT_BATCHES = 5  # How many OpenAI batches to run concurrently
POLL_INTERVAL = 60  # Seconds between batch status checks

# Testing
TEST_LIMIT = None  # Set to a number for testing, None for full run

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_option(option: str) -> str:
    """Normalize an option for comparison"""
    if not option:
        return ""

    normalized = str(option).lower().strip()
    normalized = ' '.join(normalized.split())
    normalized = normalized.rstrip('.,!?;')

    # Normalize colons
    while '::' in normalized:
        normalized = normalized.replace('::', ':')
    normalized = normalized.replace(' :', ':').replace(':', ': ').replace(':  ', ': ')
    normalized = ' '.join(normalized.split())

    return normalized


def parse_response_list(response_text: str) -> List[str]:
    """Parse LLM response to extract list of options"""
    response_text = response_text.strip()
    
    # Remove markdown code blocks if present
    if response_text.startswith('```'):
        lines = response_text.split('\n')
        response_text = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])
        response_text = response_text.strip()

    # Try JSON parsing
    json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if item]
        except json.JSONDecodeError:
            pass

    # Try quoted strings
    quoted_items = re.findall(r'"([^"]+)"', response_text)
    if quoted_items:
        return quoted_items

    single_quoted = re.findall(r"'([^']+)'", response_text)
    if single_quoted:
        return single_quoted

    return [response_text]


def validate_response(parsed_response: List[str], valid_options: List[str]) -> tuple:
    """Validate that all response items are in the valid options list"""
    if not parsed_response:
        return False, "Empty response"

    if not isinstance(parsed_response, list):
        return False, f"Response is not a list: {type(parsed_response)}"

    normalized_valid = {normalize_option(opt): opt for opt in valid_options}
    normalized_no_colon = {normalize_option(opt).replace(':', ''): opt for opt in valid_options}

    invalid_items = []

    for item in parsed_response:
        normalized_item = normalize_option(item)
        normalized_item_no_colon = normalized_item.replace(':', '')

        if (item in valid_options or
            normalized_item in normalized_valid or
            normalized_item_no_colon in normalized_no_colon):
            continue

        invalid_items.append(item)

    if invalid_items:
        return False, f"Invalid options: {invalid_items[:3]}"

    return True, "Valid"


def build_persona_prompt(persona: Dict[str, Any]) -> str:
    """Build persona prompt from philosopher data"""
    specializations = persona.get('areas_of_specialization', [])
    interests = persona.get('areas_of_interest', [])
    phd = persona.get('phd_institution')
    phd_country = persona.get('phd_country')
    phd_year = persona.get('year_of_phd_degree')
    institution = persona.get('current_institution')
    institution_country = persona.get('current_institution_country')

    prompt = "You are a professional philosopher"

    if institution:
        prompt += f" at {institution}"
        if institution_country:
            prompt += f" ({institution_country})"

    prompt += ".\n\n"

    if phd or phd_country or phd_year:
        prompt += "Your Educational Background:\n"
        if phd:
            prompt += f"- PhD from {phd}"
            if phd_country:
                prompt += f" ({phd_country})"
            if phd_year:
                prompt += f" in {phd_year}"
            prompt += "\n"
        prompt += "\n"

    if specializations:
        prompt += "Your Areas of Specialization:\n"
        for spec in specializations[:5]:
            prompt += f"- {spec}\n"
        prompt += "\n"

    if interests:
        prompt += "Your Areas of Interest:\n"
        for interest in interests[:8]:
            prompt += f"- {interest}\n"
        prompt += "\n"

    return prompt


def build_question_prompt(question_key: str, answer_options: List[str]) -> str:
    """Build question prompt with options"""
    prompt = """You are answering a survey by the reputable website PhilPapers, which collects
responses across various philosophical domains based on your background. You may select one or multiple options if you deem it appropriate to accurately represent your position.

Please respond with your chosen option(s) as a JSON list. If selecting multiple options, ensure they are logically consistent.

Examples of valid responses:
- ["Accept: physicalism"]
- ["Accept: physicalism", "Reject: non-physicalism"]
- ["Lean towards: physicalism", "Lean against: non-physicalism"]

Given your philosophical profile above, please express your stance on the following question.

Question: """

    prompt += question_key + "\n\nAvailable response options:\n"

    for option in answer_options:
        prompt += f"- {option}\n"

    prompt += "\nRespond ONLY with the JSON list, no other text or explanation.\n"

    return prompt


# ============================================================================
# MODEL PROVIDER ABSTRACTION (SYNC MODE)
# ============================================================================

class ModelProvider(ABC):
    """Abstract base class for model providers"""

    @abstractmethod
    def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate response from the model"""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name"""
        pass


class OpenAIProvider(ModelProvider):
    """OpenAI API provider"""

    def __init__(self, model_name: str = OPENAI_MODEL):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    def get_model_name(self) -> str:
        return self.model_name


class AnthropicProvider(ModelProvider):
    """Anthropic Claude API provider"""

    def __init__(self, model_name: str = ANTHROPIC_MODEL):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self.client = Anthropic(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def get_model_name(self) -> str:
        return self.model_name


class HuggingFaceProvider(ModelProvider):
    """HuggingFace transformers provider with LoRA support"""

    def __init__(self, model_name: str = HUGGINGFACE_MODEL, 
                 use_finetuned: bool = USE_FINETUNED_MODEL,
                 base_model: str = BASE_MODEL_NAME,
                 lora_path: str = LORA_ADAPTER_PATH):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError("Transformers/torch not installed. Run: pip install transformers torch")

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        hf_token = os.getenv("HF_TOKEN")

        if use_finetuned:
            # Load fine-tuned LoRA model
            try:
                from peft import PeftModel
            except ImportError:
                raise ImportError("PEFT not installed. Run: pip install peft")

            print(f"Loading FINE-TUNED model")
            print(f"Base model: {base_model}")
            print(f"LoRA adapter: {lora_path}")

            self.model_name = f"{base_model}_finetuned"
            self.tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load base model
            if self.device == "cuda":
                base = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=torch.float16,
                    device_map=None,
                    low_cpu_mem_usage=True,
                    token=hf_token
                )
            else:
                base = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    device_map="cpu",
                    token=hf_token
                )

            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(base, lora_path)
            self.model.eval()

            if self.device == "cuda":
                self.model = self.model.cuda()

            print("✓ Fine-tuned model loaded successfully")
        else:
            # Load base model
            self.model_name = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    token=hf_token
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="cpu",
                    token=hf_token
                )

            print(f"✓ Base model loaded: {model_name}")

    def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        messages = [{"role": "user", "content": prompt}]

        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            text = prompt

        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        input_length = inputs['input_ids'].shape[1]

        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id
            )

        generated_tokens = outputs[0][input_length:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def get_model_name(self) -> str:
        return self.model_name


def get_provider(provider_type: str) -> ModelProvider:
    """Factory function to get the appropriate provider"""
    if provider_type == "openai":
        return OpenAIProvider(OPENAI_MODEL)
    elif provider_type == "anthropic":
        return AnthropicProvider(ANTHROPIC_MODEL)
    elif provider_type == "huggingface":
        return HuggingFaceProvider(
            model_name=HUGGINGFACE_MODEL,
            use_finetuned=USE_FINETUNED_MODEL,
            base_model=BASE_MODEL_NAME,
            lora_path=LORA_ADAPTER_PATH
        )
    else:
        raise ValueError(f"Unknown provider: {provider_type}")


# ============================================================================
# BATCH REQUEST GENERATOR
# ============================================================================

class BatchRequestGenerator:
    """Generate batch requests for OpenAI or Anthropic batch APIs"""

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.philosophers = None
        self.question_options = None

    def load_data(self):
        """Load philosopher and question data"""
        phil_path = os.path.join(self.data_dir, PHILOSOPHERS_FILE)
        quest_path = os.path.join(self.data_dir, QUESTIONS_FILE)

        with open(phil_path) as f:
            self.philosophers = json.load(f)

        with open(quest_path) as f:
            self.question_options = json.load(f)

        print(f"Loaded {len(self.philosophers)} philosophers and {len(self.question_options)} questions")

    def generate_openai_batch_requests(self, output_file: str = BATCH_REQUESTS_FILE,
                                       mapping_file: str = BATCH_MAPPING_FILE,
                                       test_limit: Optional[int] = TEST_LIMIT) -> int:
        """Generate JSONL file with OpenAI batch requests"""
        print(f"\nGenerating OpenAI batch requests...")

        all_questions = list(self.question_options.keys())
        request_count = 0
        id_mapping = {}

        with open(output_file, 'w') as f:
            for philosopher in tqdm(self.philosophers, desc="Processing philosophers"):
                phil_name = philosopher.get('name', 'Unknown')

                for question_key in all_questions:
                    if test_limit is not None and request_count >= test_limit:
                        break

                    # Build prompt
                    persona_prompt = build_persona_prompt(philosopher)
                    question_prompt = build_question_prompt(
                        question_key,
                        self.question_options[question_key]
                    )
                    full_prompt = persona_prompt + question_prompt

                    custom_id = f"req_{request_count}"

                    # Store mapping
                    id_mapping[custom_id] = {
                        'philosopher_name': phil_name,
                        'question': question_key,
                        'philosopher': {
                            'name': phil_name,
                            'areas_of_specialization': philosopher.get('areas_of_specialization', []),
                            'areas_of_interest': philosopher.get('areas_of_interest', []),
                            'phd_institution': philosopher.get('phd_institution'),
                            'phd_country': philosopher.get('phd_country'),
                            'year_of_phd_degree': philosopher.get('year_of_phd_degree'),
                            'current_institution': philosopher.get('current_institution'),
                            'current_institution_country': philosopher.get('current_institution_country')
                        }
                    }

                    # Create OpenAI batch request format
                    batch_request = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": OPENAI_MODEL,
                            "messages": [{"role": "user", "content": full_prompt}],
                            "max_tokens": MAX_TOKENS,
                            "temperature": TEMPERATURE
                        }
                    }

                    f.write(json.dumps(batch_request) + '\n')
                    request_count += 1

                if test_limit is not None and request_count >= test_limit:
                    break

        # Save ID mapping
        with open(mapping_file, 'w') as f:
            json.dump(id_mapping, f, indent=2)

        print(f"\n✓ Generated {request_count} batch requests")
        print(f"✓ Saved to: {output_file}")
        print(f"✓ ID mapping saved to: {mapping_file}")

        return request_count

    def generate_anthropic_batch_requests(self, output_file: str = BATCH_REQUESTS_FILE,
                                          mapping_file: str = BATCH_MAPPING_FILE,
                                          test_limit: Optional[int] = TEST_LIMIT) -> int:
        """Generate JSONL file with Anthropic batch requests"""
        print(f"\nGenerating Anthropic batch requests...")

        all_questions = list(self.question_options.keys())
        request_count = 0
        id_mapping = {}

        with open(output_file, 'w') as f:
            for philosopher in tqdm(self.philosophers, desc="Processing philosophers"):
                phil_name = philosopher.get('name', 'Unknown')

                for question_key in all_questions:
                    if test_limit is not None and request_count >= test_limit:
                        break

                    # Build prompt
                    persona_prompt = build_persona_prompt(philosopher)
                    question_prompt = build_question_prompt(
                        question_key,
                        self.question_options[question_key]
                    )
                    full_prompt = persona_prompt + question_prompt

                    custom_id = f"req_{request_count}"

                    # Store mapping
                    id_mapping[custom_id] = {
                        'philosopher_name': phil_name,
                        'question': question_key,
                        'philosopher': {
                            'name': phil_name,
                            'areas_of_specialization': philosopher.get('areas_of_specialization', []),
                            'areas_of_interest': philosopher.get('areas_of_interest', []),
                            'phd_institution': philosopher.get('phd_institution'),
                            'phd_country': philosopher.get('phd_country'),
                            'year_of_phd_degree': philosopher.get('year_of_phd_degree'),
                            'current_institution': philosopher.get('current_institution'),
                            'current_institution_country': philosopher.get('current_institution_country')
                        }
                    }

                    # Create Anthropic batch request format
                    batch_request = {
                        "custom_id": custom_id,
                        "params": {
                            "model": ANTHROPIC_MODEL,
                            "max_tokens": MAX_TOKENS,
                            "temperature": TEMPERATURE,
                            "messages": [{"role": "user", "content": full_prompt}]
                        }
                    }

                    f.write(json.dumps(batch_request) + '\n')
                    request_count += 1

                if test_limit is not None and request_count >= test_limit:
                    break

        # Save ID mapping
        with open(mapping_file, 'w') as f:
            json.dump(id_mapping, f, indent=2)

        print(f"\n✓ Generated {request_count} batch requests")
        print(f"✓ Saved to: {output_file}")
        print(f"✓ ID mapping saved to: {mapping_file}")

        return request_count


# ============================================================================
# OPENAI BATCH PROCESSOR (WITH SPLITTING)
# ============================================================================

class OpenAIBatchProcessor:
    """Process OpenAI batches with splitting support"""

    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)

    def split_into_batches(self, input_file: str, requests_per_batch: int = REQUESTS_PER_BATCH) -> List[dict]:
        """Split input file into multiple batch files"""
        print(f"\nSplitting requests into smaller batches...")

        requests = []
        with open(input_file, 'r') as f:
            for line in f:
                if line.strip():
                    requests.append(json.loads(line))

        total_requests = len(requests)
        num_batches = (total_requests + requests_per_batch - 1) // requests_per_batch

        print(f"Total requests: {total_requests:,}")
        print(f"Requests per batch: {requests_per_batch}")
        print(f"Number of batches: {num_batches}")

        batches = []
        for i in range(num_batches):
            start_idx = i * requests_per_batch
            end_idx = min((i + 1) * requests_per_batch, total_requests)
            batch_requests = requests[start_idx:end_idx]

            batch_file = f"openai_batch_{i:04d}.jsonl"
            with open(batch_file, 'w') as f:
                for req in batch_requests:
                    f.write(json.dumps(req) + '\n')

            batches.append({
                'batch_num': i,
                'file': batch_file,
                'num_requests': len(batch_requests),
                'status': 'pending',
                'batch_id': None
            })

        print(f"✓ Created {num_batches} batch files\n")
        return batches

    def submit_batch(self, batch_file: str) -> str:
        """Submit a single batch"""
        with open(batch_file, 'rb') as f:
            batch_input_file = self.client.files.create(file=f, purpose="batch")

        batch = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        return batch.id

    def check_batch_status(self, batch_id: str) -> dict:
        """Check status of a batch"""
        batch = self.client.batches.retrieve(batch_id)
        return {
            'status': batch.status,
            'request_counts': {
                'total': batch.request_counts.total,
                'completed': batch.request_counts.completed,
                'failed': batch.request_counts.failed
            },
            'output_file_id': batch.output_file_id if hasattr(batch, 'output_file_id') else None
        }

    def download_results(self, batch_id: str, output_file: str):
        """Download results from completed batch"""
        batch = self.client.batches.retrieve(batch_id)
        if not batch.output_file_id:
            return None

        content = self.client.files.content(batch.output_file_id)
        with open(output_file, 'wb') as f:
            f.write(content.content)
        return output_file

    def orchestrate(self, batches: List[dict], max_concurrent: int = MAX_CONCURRENT_BATCHES,
                   tracker_file: str = "openai_batch_tracker.json"):
        """Orchestrate multiple batch submissions"""
        print(f"\n{'='*80}")
        print("BATCH ORCHESTRATION - OpenAI")
        print(f"{'='*80}\n")
        print(f"Model: {OPENAI_MODEL}")
        print(f"Total batches: {len(batches)}")
        print(f"Max concurrent: {max_concurrent}")
        print(f"Polling interval: {POLL_INTERVAL}s\n")

        # Load existing tracker if available
        if os.path.exists(tracker_file):
            with open(tracker_file, 'r') as f:
                saved_batches = json.load(f)
                for saved in saved_batches:
                    for batch in batches:
                        if batch['batch_num'] == saved['batch_num']:
                            batch.update(saved)

        def save_tracker():
            with open(tracker_file, 'w') as f:
                json.dump(batches, f, indent=2)

        while True:
            active = [b for b in batches if b['status'] in ['validating', 'in_progress', 'finalizing']]
            pending = [b for b in batches if b['status'] == 'pending']
            completed = [b for b in batches if b['status'] == 'completed']
            failed = [b for b in batches if b['status'] == 'failed']

            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Completed: {len(completed)}/{len(batches)} | "
                  f"Active: {len(active)} | "
                  f"Pending: {len(pending)} | "
                  f"Failed: {len(failed)}", end='', flush=True)

            # Submit new batches if we have capacity
            if len(active) < max_concurrent and pending:
                batch = pending[0]
                try:
                    print(f"\n\n✓ Submitting batch {batch['batch_num']:04d} ({batch['num_requests']} requests)...")
                    batch_id = self.submit_batch(batch['file'])
                    batch['batch_id'] = batch_id
                    batch['status'] = 'validating'
                    batch['submitted_at'] = datetime.now().isoformat()
                    save_tracker()
                    print(f"  Batch ID: {batch_id}")
                except Exception as e:
                    print(f"\n✗ Failed to submit batch {batch['batch_num']:04d}: {e}")
                    batch['status'] = 'failed'
                    batch['error'] = str(e)
                    save_tracker()

            # Check status of active batches
            for batch in active:
                try:
                    status_info = self.check_batch_status(batch['batch_id'])
                    batch['status'] = status_info['status']
                    batch['request_counts'] = status_info['request_counts']

                    if status_info['status'] == 'completed':
                        print(f"\n✓ Batch {batch['batch_num']:04d} completed!")
                        output_file = f"openai_batch_output_{batch['batch_num']:04d}.jsonl"
                        self.download_results(batch['batch_id'], output_file)
                        batch['output_file'] = output_file

                    elif status_info['status'] == 'failed':
                        print(f"\n✗ Batch {batch['batch_num']:04d} failed")

                    save_tracker()
                except Exception as e:
                    print(f"\n✗ Error checking batch {batch['batch_num']:04d}: {e}")

            if not active and not pending:
                break

            time.sleep(POLL_INTERVAL)

        print(f"\n\n{'='*80}")
        print("ALL BATCHES COMPLETE")
        print(f"{'='*80}")
        print(f"Completed: {len(completed)}")
        print(f"Failed: {len(failed)}")
        print(f"{'='*80}\n")

        return batches

    def combine_results(self, batches: List[dict], mapping_file: str = BATCH_MAPPING_FILE,
                       output_dir: str = OUTPUT_DIR) -> List[dict]:
        """Combine results from multiple batches"""
        print(f"\n{'='*80}")
        print("COMBINING RESULTS")
        print(f"{'='*80}\n")

        os.makedirs(output_dir, exist_ok=True)

        with open(mapping_file, 'r') as f:
            id_mapping = json.load(f)

        all_results = []
        total_input_tokens = 0
        total_output_tokens = 0
        success_count = 0
        error_count = 0

        for batch in tqdm(sorted(batches, key=lambda b: b['batch_num']), desc="Processing batches"):
            if batch['status'] != 'completed' or 'output_file' not in batch:
                continue

            with open(batch['output_file'], 'r') as f:
                for line in f:
                    if not line.strip():
                        continue

                    result = json.loads(line)
                    custom_id = result['custom_id']

                    mapping = id_mapping.get(custom_id, {})
                    philosopher_data = mapping.get('philosopher', {})
                    question_key = mapping.get('question', 'Unknown')

                    if result['response']['status_code'] == 200:
                        body = result['response']['body']
                        content = body['choices'][0]['message']['content']

                        usage = body.get('usage', {})
                        total_input_tokens += usage.get('prompt_tokens', 0)
                        total_output_tokens += usage.get('completion_tokens', 0)

                        parsed = parse_response_list(content)

                        all_results.append({
                            'timestamp': datetime.now().isoformat(),
                            'model': OPENAI_MODEL,
                            'philosopher': philosopher_data,
                            'question': question_key,
                            'response': {
                                'parsed': parsed,
                                'raw': content,
                                'success': True,
                                'error': None,
                                'generation_time': 0,
                                'attempts': 1,
                                'all_attempts': [{
                                    'attempt': 1,
                                    'parsed': parsed,
                                    'raw': content,
                                    'valid': True,
                                    'validation_msg': 'Valid',
                                    'time': 0,
                                    'usage': {
                                        'input_tokens': usage.get('prompt_tokens', 0),
                                        'output_tokens': usage.get('completion_tokens', 0)
                                    }
                                }]
                            }
                        })
                        success_count += 1
                    else:
                        all_results.append({
                            'timestamp': datetime.now().isoformat(),
                            'model': OPENAI_MODEL,
                            'philosopher': philosopher_data,
                            'question': question_key,
                            'response': {
                                'parsed': [],
                                'raw': '',
                                'success': False,
                                'error': f"HTTP {result['response']['status_code']}",
                                'generation_time': 0,
                                'attempts': 1,
                                'all_attempts': []
                            }
                        })
                        error_count += 1

        # Save combined results
        output_file = os.path.join(output_dir, 'processed_results.json')
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        # Calculate cost (gpt-4o batch pricing: 50% off)
        input_cost = (total_input_tokens / 1_000_000) * 1.25
        output_cost = (total_output_tokens / 1_000_000) * 2.50
        total_cost = input_cost + output_cost

        print(f"\n{'='*80}")
        print("FINAL RESULTS")
        print(f"{'='*80}")
        print(f"Total requests: {len(all_results):,}")
        if all_results:
            print(f"Successful: {success_count:,} ({success_count/len(all_results)*100:.1f}%)")
        print(f"Errors: {error_count}")
        print(f"\nToken usage:")
        print(f"  Input tokens: {total_input_tokens:,}")
        print(f"  Output tokens: {total_output_tokens:,}")
        print(f"\nEstimated cost:")
        print(f"  Input: ${input_cost:.2f}")
        print(f"  Output: ${output_cost:.2f}")
        print(f"  Total: ${total_cost:.2f}")
        print(f"\nResults saved to: {output_file}")
        print(f"{'='*80}\n")

        return all_results


# ============================================================================
# ANTHROPIC BATCH PROCESSOR
# ============================================================================

class AnthropicBatchProcessor:
    """Process Anthropic batches"""

    def __init__(self):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self.client = Anthropic(api_key=api_key)

    def submit_batch(self, requests_file: str) -> str:
        """Submit batch requests to Claude API"""
        print(f"\nSubmitting batch to Claude API...")

        requests_list = []
        with open(requests_file, 'r') as f:
            for line in f:
                if line.strip():
                    requests_list.append(json.loads(line))

        batch = self.client.messages.batches.create(requests=requests_list)
        batch_id = batch.id

        print(f"✓ Batch submitted successfully!")
        print(f"  Batch ID: {batch_id}")
        print(f"  Status: {batch.processing_status}")

        # Save batch info
        batch_info = {
            'batch_id': batch_id,
            'submitted_at': datetime.now().isoformat(),
            'status': batch.processing_status
        }
        with open('anthropic_batch_info.json', 'w') as f:
            json.dump(batch_info, f, indent=2)

        return batch_id

    def poll_batch_status(self, batch_id: str, poll_interval: int = POLL_INTERVAL):
        """Poll batch status until completion"""
        print(f"\nPolling batch status (checking every {poll_interval}s)...")

        start_time = time.time()

        with tqdm(desc="Waiting for batch", unit="check") as pbar:
            while True:
                batch = self.client.messages.batches.retrieve(batch_id)
                status = batch.processing_status

                pbar.set_postfix({
                    'status': status,
                    'elapsed': f"{(time.time() - start_time)/60:.1f}m"
                })

                if status == "ended":
                    print(f"\n✓ Batch processing complete!")
                    print(f"  Total time: {(time.time() - start_time)/60:.1f} minutes")
                    return batch

                elif status in ["canceling", "canceled", "expired"]:
                    print(f"\n✗ Batch {status}")
                    return batch

                time.sleep(poll_interval)
                pbar.update(1)

    def download_results(self, batch_id: str, output_file: str = BATCH_RESULTS_FILE):
        """Download batch results"""
        print(f"\nDownloading batch results...")

        results = self.client.messages.batches.results(batch_id)

        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result.model_dump()) + '\n')

        print(f"✓ Results saved to: {output_file}")
        return output_file

    def process_results(self, results_file: str = BATCH_RESULTS_FILE,
                       mapping_file: str = BATCH_MAPPING_FILE,
                       output_dir: str = OUTPUT_DIR) -> List[dict]:
        """Process batch results and save in standard format"""
        print(f"\nProcessing batch results...")

        os.makedirs(output_dir, exist_ok=True)

        with open(mapping_file, 'r') as f:
            id_mapping = json.load(f)

        results = []
        success_count = 0
        error_count = 0
        total_input_tokens = 0
        total_output_tokens = 0

        with open(results_file, 'r') as f:
            for line in tqdm(f, desc="Processing results"):
                if not line.strip():
                    continue

                result = json.loads(line)
                custom_id = result['custom_id']

                mapping = id_mapping.get(custom_id, {})
                philosopher_data = mapping.get('philosopher', {})
                question_key = mapping.get('question', 'Unknown')

                if result['result']['type'] == 'succeeded':
                    message = result['result']['message']
                    raw_response = message['content'][0]['text']
                    parsed_response = parse_response_list(raw_response)

                    usage = message.get('usage', {})
                    input_tokens = usage.get('input_tokens', 0)
                    output_tokens = usage.get('output_tokens', 0)
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens

                    processed_result = {
                        'timestamp': datetime.now().isoformat(),
                        'model': message['model'],
                        'philosopher': philosopher_data,
                        'question': question_key,
                        'response': {
                            'parsed': parsed_response,
                            'raw': raw_response,
                            'success': True,
                            'error': None,
                            'generation_time': 0,
                            'attempts': 1,
                            'all_attempts': [{
                                'attempt': 1,
                                'parsed': parsed_response,
                                'raw': raw_response,
                                'valid': True,
                                'validation_msg': 'Valid',
                                'time': 0,
                                'usage': {
                                    'input_tokens': input_tokens,
                                    'output_tokens': output_tokens
                                }
                            }]
                        }
                    }
                    success_count += 1

                elif result['result']['type'] == 'errored':
                    error = result['result']['error']
                    processed_result = {
                        'timestamp': datetime.now().isoformat(),
                        'model': ANTHROPIC_MODEL,
                        'philosopher': philosopher_data,
                        'question': question_key,
                        'response': {
                            'parsed': [],
                            'raw': '',
                            'success': False,
                            'error': error.get('message', 'Unknown error'),
                            'generation_time': 0,
                            'attempts': 1,
                            'all_attempts': [{
                                'attempt': 1,
                                'error': error.get('message', 'Unknown error'),
                                'time': 0
                            }]
                        }
                    }
                    error_count += 1
                else:
                    continue

                results.append(processed_result)

        # Save processed results
        output_file = os.path.join(output_dir, 'processed_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Calculate cost (Claude batch pricing: 50% discount)
        input_cost = (total_input_tokens / 1_000_000) * 1.50  # Claude Sonnet input
        output_cost = (total_output_tokens / 1_000_000) * 7.50  # Claude Sonnet output
        total_cost = input_cost + output_cost

        print(f"\n{'='*80}")
        print("BATCH PROCESSING COMPLETE!")
        print(f"{'='*80}")
        print(f"Total requests: {len(results)}")
        if results:
            print(f"Successful: {success_count} ({success_count/len(results)*100:.1f}%)")
        print(f"Errors: {error_count}")
        print(f"\nToken usage:")
        print(f"  Input tokens: {total_input_tokens:,}")
        print(f"  Output tokens: {total_output_tokens:,}")
        print(f"\nEstimated cost:")
        print(f"  Input: ${input_cost:.4f}")
        print(f"  Output: ${output_cost:.4f}")
        print(f"  Total: ${total_cost:.4f}")
        print(f"\nResults saved to: {output_file}")
        print(f"{'='*80}\n")

        return results


# ============================================================================
# SYNC MODE GENERATOR CLASS
# ============================================================================

class PhilosopherResponseGenerator:
    """Main class for generating philosopher responses in sync mode"""

    def __init__(self, provider: ModelProvider, data_dir: str = DATA_DIR):
        self.provider = provider
        self.data_dir = data_dir
        self.philosophers = None
        self.question_options = None

    def load_data(self):
        """Load philosopher and question data"""
        phil_path = os.path.join(self.data_dir, PHILOSOPHERS_FILE)
        quest_path = os.path.join(self.data_dir, QUESTIONS_FILE)

        with open(phil_path) as f:
            self.philosophers = json.load(f)

        with open(quest_path) as f:
            self.question_options = json.load(f)

        print(f"Loaded {len(self.philosophers)} philosophers and {len(self.question_options)} questions")

    def generate_response(self, philosopher: Dict, question_key: str,
                         answer_options: List[str], max_retries: int = MAX_RETRIES) -> Dict:
        """Generate a single response with validation and retry"""
        persona_prompt = build_persona_prompt(philosopher)
        question_prompt = build_question_prompt(question_key, answer_options)
        full_prompt = persona_prompt + question_prompt

        attempts = []
        total_time = 0

        for attempt in range(max_retries):
            try:
                start = time.time()
                raw_response = self.provider.generate(full_prompt, TEMPERATURE, MAX_TOKENS)
                elapsed = time.time() - start
                total_time += elapsed

                parsed_response = parse_response_list(raw_response)
                is_valid, validation_msg = validate_response(parsed_response, answer_options)

                attempts.append({
                    'attempt': attempt + 1,
                    'parsed': parsed_response,
                    'raw': raw_response,
                    'valid': is_valid,
                    'validation_msg': validation_msg,
                    'time': elapsed
                })

                if is_valid:
                    return {
                        'success': True,
                        'parsed': parsed_response,
                        'raw': raw_response,
                        'generation_time': total_time,
                        'attempts': attempt + 1,
                        'all_attempts': attempts
                    }

                if attempt == max_retries - 1:
                    return {
                        'success': False,
                        'error': f"Max retries reached. Last validation: {validation_msg}",
                        'parsed': parsed_response,
                        'raw': raw_response,
                        'generation_time': total_time,
                        'attempts': max_retries,
                        'all_attempts': attempts
                    }

            except Exception as e:
                error_detail = f"{type(e).__name__}: {str(e)}"
                attempts.append({
                    'attempt': attempt + 1,
                    'error': error_detail,
                    'time': 0
                })

                if attempt == max_retries - 1:
                    return {
                        'success': False,
                        'error': f"Exception after {max_retries} attempts: {error_detail}",
                        'generation_time': total_time,
                        'attempts': max_retries,
                        'all_attempts': attempts
                    }

        return {'success': False, 'error': 'Unknown error'}

    def generate_all_responses(self, output_dir: str = OUTPUT_DIR,
                              resume_file: str = RESUME_FILE,
                              test_limit: Optional[int] = TEST_LIMIT,
                              batch_size: int = 10) -> Dict:
        """Generate responses for all philosopher-question combinations"""
        os.makedirs(output_dir, exist_ok=True)

        # Load progress
        completed = set()
        if os.path.exists(resume_file):
            with open(resume_file, 'r') as f:
                progress_data = json.load(f)
                completed = set(progress_data.get('completed', []))

        all_questions = list(self.question_options.keys())
        total_combinations = len(self.philosophers) * len(all_questions)

        print(f"\n{'='*80}")
        print("CONFIGURATION")
        print(f"{'='*80}")
        print(f"Provider: {MODEL_PROVIDER}")
        print(f"Model: {self.provider.get_model_name()}")
        print(f"Max retries: {MAX_RETRIES}")
        print(f"Output directory: {output_dir}")
        if test_limit:
            print(f"TEST MODE: Limited to {test_limit} items")
        print(f"Total combinations: {total_combinations}")
        print(f"Already completed: {len(completed)}")
        print(f"Remaining: {total_combinations - len(completed)}")
        print(f"{'='*80}\n")

        # Build task list
        tasks = []
        for philosopher in self.philosophers:
            phil_name = philosopher.get('name', 'Unknown')
            for question_key in all_questions:
                if test_limit is not None and len(tasks) >= test_limit:
                    break

                combo_id = f"{phil_name}||{question_key}"
                if combo_id in completed:
                    continue

                tasks.append({
                    'philosopher': philosopher,
                    'question_key': question_key,
                    'answer_options': self.question_options[question_key],
                    'combo_id': combo_id,
                    'phil_name': phil_name
                })

            if test_limit is not None and len(tasks) >= test_limit:
                break

        print(f"Processing {len(tasks)} tasks...\n")

        # Process tasks
        results_batch = []
        start_time = time.time()
        retry_count = 0
        failed_count = 0

        pbar = tqdm(tasks, desc="Processing", unit="item")

        for task in pbar:
            result = self.generate_response(
                task['philosopher'],
                task['question_key'],
                task['answer_options']
            )

            if result['success']:
                status = "✓"
                if result.get('attempts', 1) > 1:
                    retry_count += 1
            else:
                status = "✗"
                failed_count += 1

            pbar.set_postfix({'status': status, 'retries': retry_count, 'failed': failed_count})

            full_result = {
                'timestamp': datetime.now().isoformat(),
                'model': self.provider.get_model_name(),
                'philosopher': {
                    'name': task['phil_name'],
                    'areas_of_specialization': task['philosopher'].get('areas_of_specialization', []),
                    'areas_of_interest': task['philosopher'].get('areas_of_interest', []),
                    'phd_institution': task['philosopher'].get('phd_institution'),
                    'phd_country': task['philosopher'].get('phd_country'),
                    'year_of_phd_degree': task['philosopher'].get('year_of_phd_degree'),
                    'current_institution': task['philosopher'].get('current_institution'),
                    'current_institution_country': task['philosopher'].get('current_institution_country')
                },
                'question': task['question_key'],
                'response': {
                    'parsed': result.get('parsed', []),
                    'raw': result.get('raw', ''),
                    'success': result['success'],
                    'error': result.get('error'),
                    'generation_time': result['generation_time'],
                    'attempts': result.get('attempts', 1),
                    'all_attempts': result.get('all_attempts', [])
                }
            }

            results_batch.append(full_result)
            completed.add(task['combo_id'])

            # Save batch periodically
            if len(results_batch) >= batch_size:
                batch_filename = os.path.join(output_dir, f"batch_{int(time.time())}.json")
                with open(batch_filename, 'w') as f:
                    json.dump(results_batch, f, indent=2)
                results_batch = []

                with open(resume_file, 'w') as f:
                    json.dump({'completed': list(completed)}, f)

        # Save remaining results
        if results_batch:
            batch_filename = os.path.join(output_dir, f"batch_final_{int(time.time())}.json")
            with open(batch_filename, 'w') as f:
                json.dump(results_batch, f, indent=2)

        with open(resume_file, 'w') as f:
            json.dump({'completed': list(completed)}, f)

        # Summary
        total_time = time.time() - start_time
        processed = len(tasks)
        success_count = processed - failed_count

        print(f"\n{'='*80}")
        print("COMPLETE!")
        print(f"{'='*80}")
        print(f"Total processed: {processed}")
        if processed > 0:
            print(f"Successful: {success_count} ({success_count/processed*100:.1f}%)")
            print(f"Required retry: {retry_count} ({retry_count/processed*100:.1f}%)")
            print(f"Failed: {failed_count} ({failed_count/processed*100:.1f}%)")
            print(f"Total time: {total_time/60:.1f} minutes")
            print(f"Average: {total_time/processed:.2f} seconds per item")
        print(f"Results saved in: {output_dir}/")
        print(f"{'='*80}")

        return {
            'total': processed,
            'success': success_count,
            'failed': failed_count,
            'retries': retry_count,
            'time': total_time
        }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def run_sync_mode(provider_type: str = None):
    """Run in synchronous API mode"""
    # Auto-detect provider if not specified or set to 'auto'
    if provider_type is None or provider_type == "auto":
        provider_type = detect_available_provider()
        print(f"Auto-detected provider: {provider_type}")
    
    provider = get_provider(provider_type)
    
    # Update output directory based on actual provider and fine-tuning status
    if USE_FINETUNED_MODEL:
        output_dir = f"llm_responses_{provider_type}_finetuned"
        resume_file = f"llm_responses_progress_{provider_type}_finetuned.json"
        print(f"Using FINE-TUNED model, output: {output_dir}")
    else:
        output_dir = f"llm_responses_{provider_type}"
        resume_file = f"llm_responses_progress_{provider_type}.json"

    generator = PhilosopherResponseGenerator(
        provider=provider,
        data_dir=DATA_DIR
    )

    generator.load_data()
    results = generator.generate_all_responses(
        output_dir=output_dir,
        resume_file=resume_file,
        test_limit=TEST_LIMIT
    )

    return results


def run_openai_batch_mode():
    """Run in OpenAI Batch API mode"""
    print(f"{'='*80}")
    print("OPENAI BATCH API MODE")
    print(f"{'='*80}\n")

    # Step 1: Generate batch requests
    generator = BatchRequestGenerator(data_dir=DATA_DIR)
    generator.load_data()
    request_count = generator.generate_openai_batch_requests(
        output_file=BATCH_REQUESTS_FILE,
        mapping_file=BATCH_MAPPING_FILE,
        test_limit=TEST_LIMIT
    )

    # Confirm before submitting
    response = input("\nReady to submit batch to OpenAI API? (yes/no): ")
    if response.lower() != 'yes':
        print("Batch submission canceled.")
        print(f"Requests saved to {BATCH_REQUESTS_FILE} for later submission.")
        return

    # Step 2: Split, submit, and orchestrate
    processor = OpenAIBatchProcessor()
    batches = processor.split_into_batches(BATCH_REQUESTS_FILE, REQUESTS_PER_BATCH)
    batches = processor.orchestrate(batches, MAX_CONCURRENT_BATCHES)

    # Step 3: Combine results
    results = processor.combine_results(batches, BATCH_MAPPING_FILE, OUTPUT_DIR)

    print("All done! 🎉")
    return results


def run_anthropic_batch_mode():
    """Run in Anthropic Batch API mode"""
    print(f"{'='*80}")
    print("ANTHROPIC BATCH API MODE")
    print(f"{'='*80}\n")

    # Step 1: Generate batch requests
    generator = BatchRequestGenerator(data_dir=DATA_DIR)
    generator.load_data()
    request_count = generator.generate_anthropic_batch_requests(
        output_file=BATCH_REQUESTS_FILE,
        mapping_file=BATCH_MAPPING_FILE,
        test_limit=TEST_LIMIT
    )

    # Confirm before submitting
    response = input("\nReady to submit batch to Anthropic API? (yes/no): ")
    if response.lower() != 'yes':
        print("Batch submission canceled.")
        print(f"Requests saved to {BATCH_REQUESTS_FILE} for later submission.")
        return

    # Step 2: Submit batch
    processor = AnthropicBatchProcessor()
    batch_id = processor.submit_batch(BATCH_REQUESTS_FILE)

    # Step 3: Poll for completion
    batch = processor.poll_batch_status(batch_id)

    if batch.processing_status != "ended":
        print(f"Batch did not complete successfully. Status: {batch.processing_status}")
        return

    # Step 4: Download and process results
    processor.download_results(batch_id, BATCH_RESULTS_FILE)
    results = processor.process_results(BATCH_RESULTS_FILE, BATCH_MAPPING_FILE, OUTPUT_DIR)

    print("All done! 🎉")
    return results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='LLM Philosopher Response Generator - Evaluate LLMs on philosophical survey questions'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Limit number of philosopher-question pairs to evaluate (for testing)'
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['sync', 'batch_openai', 'batch_anthropic', 'auto'],
        default=None,
        help='Evaluation mode (default: auto)'
    )
    parser.add_argument(
        '--provider', '-p',
        choices=['openai', 'anthropic', 'huggingface', 'auto'],
        default=None,
        help='Model provider for sync mode (default: auto)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model name to use (overrides default for the provider)'
    )
    parser.add_argument(
        '--finetuned',
        action='store_true',
        help='Use fine-tuned LoRA model (HuggingFace only)'
    )
    return parser.parse_args()


def main():
    """Main entry point for script execution"""
    global TEST_LIMIT, HUGGINGFACE_MODEL, OPENAI_MODEL, ANTHROPIC_MODEL, USE_FINETUNED_MODEL
    
    # Parse command line arguments
    args = parse_args()
    
    # Apply command line overrides
    if args.limit is not None:
        TEST_LIMIT = args.limit
    
    if args.model:
        # Override model for all providers
        HUGGINGFACE_MODEL = args.model
        OPENAI_MODEL = args.model
        ANTHROPIC_MODEL = args.model
    
    if args.finetuned:
        USE_FINETUNED_MODEL = True
    
    # Resolve mode
    effective_mode = args.mode if args.mode else MODE
    if effective_mode == "auto":
        effective_mode = detect_available_mode()
    
    # Resolve provider
    effective_provider = args.provider if args.provider else MODEL_PROVIDER
    if effective_provider == "auto":
        effective_provider = detect_available_provider()
    
    print(f"\n{'='*80}")
    print("PHILOSOPHER RESPONSE GENERATOR")
    print(f"{'='*80}")
    print(f"Mode: {effective_mode}" + (f" (auto-detected)" if (args.mode is None or args.mode == "auto") and MODE == "auto" else ""))
    print(f"Provider: {effective_provider}" + (f" (auto-detected)" if (args.provider is None or args.provider == "auto") and MODEL_PROVIDER == "auto" else ""))
    
    if TEST_LIMIT:
        print(f"Limit: {TEST_LIMIT} items")
    
    # Show API key status
    openai_key = "✓ found" if os.getenv("OPENAI_API_KEY") else "✗ not set"
    anthropic_key = "✓ found" if os.getenv("ANTHROPIC_API_KEY") else "✗ not set"
    hf_token = "✓ found" if os.getenv("HF_TOKEN") else "○ optional"
    print(f"\nAPI Keys:")
    print(f"  OPENAI_API_KEY: {openai_key}")
    print(f"  ANTHROPIC_API_KEY: {anthropic_key}")
    print(f"  HF_TOKEN: {hf_token}")
    print(f"{'='*80}\n")

    if effective_mode == "sync":
        return run_sync_mode(effective_provider)
    elif effective_mode == "batch_openai":
        if not os.getenv("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY required for batch_openai mode")
            print("Falling back to sync mode with HuggingFace...")
            return run_sync_mode("huggingface")
        return run_openai_batch_mode()
    elif effective_mode == "batch_anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("ERROR: ANTHROPIC_API_KEY required for batch_anthropic mode")
            print("Falling back to sync mode with HuggingFace...")
            return run_sync_mode("huggingface")
        return run_anthropic_batch_mode()
    else:
        raise ValueError(f"Unknown mode: {effective_mode}. Use 'sync', 'batch_openai', 'batch_anthropic', or 'auto'")


if __name__ == "__main__":
    main()
