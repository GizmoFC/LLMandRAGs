import json

def view_sample_records(filename, num_samples=3):
    """View sample records from the JSONL file"""
    with open(filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            
            record = json.loads(line)
            print(f"\n=== Record {i+1} ===")
            print(f"Source: {record.get('source', 'N/A')}")
            print(f"Clause Type: {record.get('clause_type', 'N/A')[:100]}...")
            print(f"Text: {record.get('text', 'N/A')}")
            print(f"Contract Title: {record.get('contract_title', 'N/A')[:80]}...")
            print(f"Question ID: {record.get('question_id', 'N/A')}")
            print(f"Answer Start: {record.get('answer_start', 'N/A')}")
            print(f"Context Length: {len(record.get('full_context', ''))} characters")
            print("-" * 80)

if __name__ == "__main__":
    view_sample_records("cuad_clause_library.jsonl", 3) 