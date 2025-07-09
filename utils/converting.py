import json
import os

def load_cuad_dataset(file_path):
    """
    Load CUAD dataset from a local JSON file.
    Expected format: Dictionary with 'data' field containing list of contract documents.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # CUAD dataset has structure: {"version": "...", "data": [...]}
        if isinstance(data, dict) and 'data' in data:
            documents = data['data']
            print(f"Loaded {len(documents)} contract documents from {file_path}")
            return documents
        else:
            print(f"Unexpected format in {file_path}. Expected dict with 'data' field.")
            return None
            
    except FileNotFoundError:
        print(f"File {file_path} not found. Using mock data instead.")
        return None
    except json.JSONDecodeError:
        print(f"Error parsing JSON from {file_path}. Using mock data instead.")
        return None

def create_mock_data():
    """Create mock data for testing when real dataset is not available."""
    return [
        {
            "source": "CUAD-Croissant",
            "clause_type": "Non-Compete",
            "text": "Employee shall not engage in any business activity that competes with the Company for a period of 12 months following termination.",
            "full_context": "This agreement contains a non-compete clause that restricts the employee from working with competitors after leaving the company."
        },
        {
            "source": "CUAD-Croissant", 
            "clause_type": "Confidentiality",
            "text": "The parties agree to maintain the confidentiality of all proprietary information disclosed during the course of this agreement.",
            "full_context": "A confidentiality clause ensuring that sensitive business information remains protected between the parties."
        },
        {
            "source": "CUAD-Croissant",
            "clause_type": "Termination",
            "text": "Either party may terminate this agreement with 30 days written notice to the other party.",
            "full_context": "Termination clause specifying the conditions and notice period required to end the contractual relationship."
        },
        {
            "source": "CUAD-Croissant",
            "clause_type": "Force Majeure",
            "text": "Neither party shall be liable for any delay or failure to perform due to circumstances beyond their reasonable control.",
            "full_context": "Force majeure clause protecting parties from liability due to unforeseeable events or circumstances."
        },
        {
            "source": "CUAD-Croissant",
            "clause_type": "Governing Law",
            "text": "This agreement shall be governed by and construed in accordance with the laws of the State of California.",
            "full_context": "Governing law clause specifying which jurisdiction's laws will apply to the interpretation of this contract."
        }
    ]

def convert_to_jsonl_format(documents):
    """
    Convert CUAD dataset documents to the expected JSONL format.
    CUAD structure: documents with paragraphs containing QA pairs.
    """
    converted_records = []
    
    for doc in documents:
        title = doc.get('title', 'Unknown Contract')
        
        # Process each paragraph in the document
        for paragraph in doc.get('paragraphs', []):
            context = paragraph.get('context', '')
            
            # Process each QA pair in the paragraph
            for qa in paragraph.get('qas', []):
                question = qa.get('question', '')
                question_id = qa.get('id', '')
                
                # Process each answer in the QA pair
                for answer in qa.get('answers', []):
                    answer_text = answer.get('text', '')
                    answer_start = answer.get('answer_start', 0)
                    
                    if answer_text and question:
                        converted_record = {
                            "source": "CUAD-Dataset",
                            "clause_type": question,  # Use question as clause type
                            "text": answer_text,      # Use answer as clause text
                            "full_context": context,  # Use paragraph context
                            "contract_title": title,
                            "question_id": question_id,
                            "answer_start": answer_start
                        }
                        converted_records.append(converted_record)
    
    return converted_records

# Try to load the real CUAD dataset first
cuad_file_paths = [
    "CUAD_v1.json",
    "cuad_dataset.json",
    "data/cuad_dataset.json", 
    "cuad_data.json",
    "data/cuad_data.json"
]

records = None
for file_path in cuad_file_paths:
    if os.path.exists(file_path):
        records = load_cuad_dataset(file_path)
        if records:
            break

# If no real dataset found, use mock data
if not records:
    print("No CUAD dataset found. Using mock data for testing.")
    records = create_mock_data()

# Convert to JSONL format
converted_records = convert_to_jsonl_format(records)

# Define output file
output_path = "cuad_clause_library.jsonl"

# Write to JSONL file
with open(output_path, "w", encoding="utf-8") as f:
    for record in converted_records:
        f.write(json.dumps(record) + "\n")

print(f"Generated {len(converted_records)} records in {output_path}")
print(f"Dataset source: {'Real CUAD dataset' if records != create_mock_data() else 'Mock data'}")
