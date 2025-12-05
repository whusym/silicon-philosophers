#!/usr/bin/env python3
"""
Prepare DPO (Direct Preference Optimization) dataset directly from survey data.

This script creates DPO training data with anonymized philosopher names from:
- philosopher_details.json (profile data)
- survey_responses_all_reprocessed.json (survey responses)

Outputs:
- philosopher_dpo_train.jsonl (training data)
- philosopher_dpo_val.jsonl (validation data)

DPO format: {"prompt": str, "chosen": str, "rejected": str}
"""

import json
import random
from typing import List, Dict, Tuple
from collections import defaultdict

# Configuration
DETAILS_FILE = "philosopher_details.json"
RESPONSES_FILE = "survey_responses_all_reprocessed.json"
RANDOM_SEED = 42
TRAIN_SPLIT = 0.85  # 85% train, 15% validation
MIN_DEMOGRAPHIC_FIELDS = 1  # Minimum required profile fields

# Question mapping for response matching
QUESTION_KEYWORDS = {
    'free will': {
        'keywords': ['free will', 'compatibilism', 'libertarianism', 'determinism'],
        'question': 'Free will: compatibilism, libertarianism, or no free will?'
    },
    'god': {
        'keywords': ['god', 'theism', 'atheism', 'agnosticism'],
        'question': 'God: theism or atheism?'
    },
    'mind': {
        'keywords': ['physicalism', 'dualism', 'mind-body', 'non-physicalism'],
        'question': 'Mind: physicalism or non-physicalism?'
    },
    'trolley': {
        'keywords': ['trolley', 'footbridge', 'switch'],
        'question': 'Trolley problem: switch or don\'t switch?'
    },
    'meta-ethics': {
        'keywords': ['moral realism', 'moral anti-realism', 'meta-ethics'],
        'question': 'Meta-ethics: moral realism or moral anti-realism?'
    },
    'abstract objects': {
        'keywords': ['platonism', 'nominalism', 'abstract objects'],
        'question': 'Abstract objects: Platonism or nominalism?'
    },
    'a priori': {
        'keywords': ['a priori', 'apriori', 'a-priori'],
        'question': 'A priori knowledge: yes or no?'
    },
    'normative ethics': {
        'keywords': ['deontology', 'consequentialism', 'virtue ethics'],
        'question': 'Normative ethics: deontology, consequentialism, or virtue ethics?'
    },
    'personal identity': {
        'keywords': ['personal identity', 'biological view', 'psychological view'],
        'question': 'Personal identity: biological, psychological, or further-fact view?'
    },
    'external world': {
        'keywords': ['external world', 'skepticism', 'non-skeptical realism', 'idealism'],
        'question': 'External world: non-skeptical realism, skepticism, or idealism?'
    },
}


def create_anonymized_id(index: int) -> str:
    """Create an anonymized philosopher ID"""
    return f"Philosopher_{index:04d}"


def match_response_to_question(response_text: str) -> Dict:
    """Match a response to its question using keyword matching"""
    response_lower = response_text.lower()
    
    scores = {}
    for q_key, q_data in QUESTION_KEYWORDS.items():
        score = sum(1 for keyword in q_data['keywords'] if keyword in response_lower)
        if score > 0:
            scores[q_key] = score
    
    if not scores:
        return None
    
    best_match = max(scores.items(), key=lambda x: x[1])[0]
    return QUESTION_KEYWORDS[best_match]


def create_persona_prompt(detail: Dict) -> str:
    """Create anonymized persona prompt (no real names)"""
    specs = detail.get('areas_of_specialization', [])
    interests = detail.get('areas_of_interest', [])
    phd = detail.get('phd_institution')
    institution = detail.get('current_institution')
    
    prompt = "You are a professional philosopher"
    
    if institution:
        prompt += f" at a research university"
    
    prompt += ".\n\n"
    
    if phd:
        prompt += "Educational Background:\n"
        prompt += f"- PhD from a leading institution\n\n"
    
    if specs:
        prompt += "Areas of Specialization:\n"
        for spec in specs[:5]:
            prompt += f"- {spec}\n"
        prompt += "\n"
    
    if interests and interests != specs:
        prompt += "Areas of Interest:\n"
        for interest in interests[:5]:
            prompt += f"- {interest}\n"
        prompt += "\n"
    
    return prompt


def create_negative_examples(response: str) -> List[str]:
    """Create negative examples by choosing opposite stances"""
    opposite_map = {
        'accept': 'reject',
        'reject': 'accept',
        'lean towards': 'lean against',
        'lean against': 'lean towards',
        'neutral': 'reject'
    }
    
    other_options = {
        'accept': ['lean against', 'reject'],
        'reject': ['lean towards', 'accept'],
        'lean towards': ['lean against', 'reject'],
        'lean against': ['lean towards', 'accept'],
        'neutral': ['accept', 'reject']
    }
    
    response_lower = response.strip().lower()
    
    # Extract the stance (e.g., "accept" from "Accept: compatibilism")
    stance = None
    for key in opposite_map.keys():
        if response_lower.startswith(key):
            stance = key
            break
    
    if not stance:
        # Default to providing generic alternatives
        return ['reject', 'neutral']
    
    negatives = []
    
    # Get opposite stance - preserve the answer part
    if ':' in response_lower:
        answer_part = response_lower.split(':', 1)[1].strip()
        opposite_stance = opposite_map[stance]
        negatives.append(f"{opposite_stance}: {answer_part}")
    else:
        negatives.append(opposite_map[stance])
    
    # Get alternative stance
    if stance in other_options:
        different = random.choice(other_options[stance])
        if different != opposite_map[stance]:
            if ':' in response_lower:
                negatives.append(f"{different}: {answer_part}")
            else:
                negatives.append(different)
    
    return negatives[:2]


def load_and_process_data() -> Tuple[List[Dict], List[Dict]]:
    """Load data and create DPO examples"""
    print("Loading data files...")
    
    with open(DETAILS_FILE) as f:
        details = json.load(f)
    
    with open(RESPONSES_FILE) as f:
        responses = json.load(f)
    
    # Create lookup
    details_dict = {d['profile_url']: d for d in details}
    
    print(f"Loaded {len(details)} profile details")
    print(f"Loaded {len(responses)} survey responses")
    
    # Process all philosopher data
    philosopher_data = []
    
    for idx, response_data in enumerate(responses):
        if not response_data.get('has_survey_responses', False):
            continue
        
        profile_url = response_data['profile_url']
        detail = details_dict.get(profile_url, {})
        
        # Check demographic requirements
        demo_count = 0
        if detail.get('areas_of_specialization'):
            demo_count += 1
        if detail.get('areas_of_interest'):
            demo_count += 1
        if detail.get('phd_institution'):
            demo_count += 1
        if detail.get('current_institution'):
            demo_count += 1
        
        if demo_count < MIN_DEMOGRAPHIC_FIELDS:
            continue
        
        philosopher_data.append({
            'anon_id': create_anonymized_id(idx),
            'detail': detail,
            'responses': response_data['survey_responses']
        })
    
    print(f"Found {len(philosopher_data)} philosophers with sufficient data")
    
    # Shuffle and split
    random.seed(RANDOM_SEED)
    random.shuffle(philosopher_data)
    
    split_idx = int(len(philosopher_data) * TRAIN_SPLIT)
    train_philosophers = philosopher_data[:split_idx]
    val_philosophers = philosopher_data[split_idx:]
    
    print(f"Split: {len(train_philosophers)} train, {len(val_philosophers)} val")
    
    # Create DPO examples
    train_examples = []
    val_examples = []
    
    for split_name, phil_list, examples_list in [
        ("train", train_philosophers, train_examples),
        ("val", val_philosophers, val_examples)
    ]:
        print(f"\nCreating {split_name} examples...")
        
        for philosopher in phil_list:
            for response in philosopher['responses']:
                response_text = response.get('raw_text', '')
                
                if not response_text or len(response_text) < 5:
                    continue
                
                # Match to question
                question_data = match_response_to_question(response_text)
                if not question_data:
                    continue
                
                # Create persona prompt (anonymized)
                persona = create_persona_prompt(philosopher['detail'])
                
                # Create question prompt
                question_prompt = f"""You are answering a philosophical survey. Based on your background, express your stance on:

Question: {question_data['question']}

Respond with your position (e.g., "Accept: X", "Lean towards: Y", "Reject: Z", or "Neutral").
"""
                
                # Full prompt
                full_prompt = f"System: {persona}\n\nUser: {question_prompt}"
                
                # Chosen response (actual philosopher response)
                chosen = response_text
                
                # Generate negative examples
                negative_responses = create_negative_examples(response_text)
                
                # Create DPO examples for each negative
                for neg_response in negative_responses:
                    examples_list.append({
                        "prompt": full_prompt,
                        "chosen": chosen,
                        "rejected": neg_response
                    })
        
        print(f"Created {len(examples_list)} {split_name} DPO examples")
    
    return train_examples, val_examples


def save_jsonl(data: List[Dict], filepath: str):
    """Save data as JSONL"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for example in data:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')


def main():
    print("=" * 80)
    print("DPO DATASET PREPARATION (Anonymized)")
    print("=" * 80)
    
    # Load and process data
    train_examples, val_examples = load_and_process_data()
    
    # Save datasets
    print("\n" + "=" * 80)
    print("SAVING DATASETS")
    print("=" * 80)
    
    save_jsonl(train_examples, 'philosopher_dpo_train.jsonl')
    save_jsonl(val_examples, 'philosopher_dpo_val.jsonl')
    
    print(f"\n✓ Saved: philosopher_dpo_train.jsonl ({len(train_examples)} examples)")
    print(f"✓ Saved: philosopher_dpo_val.jsonl ({len(val_examples)} examples)")
    
    # Show sample
    if train_examples:
        print("\n" + "=" * 80)
        print("SAMPLE DPO EXAMPLE")
        print("=" * 80)
        sample = train_examples[0]
        print("\nPrompt (truncated):")
        print(f"  {sample['prompt'][:300]}...")
        print("\nChosen (correct answer):")
        print(f"  {sample['chosen']}")
        print("\nRejected (negative example):")
        print(f"  {sample['rejected']}")
    
    print("\n" + "=" * 80)
    print("✓ DPO dataset ready!")
    print("\nKey features:")
    print("  • Philosopher names are anonymized")
    print("  • Each positive example generates ~2 negative examples")
    print("  • Ready for use with 9_finetune_dpo.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
