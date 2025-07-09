#!/usr/bin/env python3
"""
Test script to verify document structure and display
"""

import json

def test_document_structure():
    """Test the document structure from the clause library"""
    print("🔍 Testing document structure...")
    
    try:
        with open("cuad_clause_library.jsonl", "r", encoding="utf-8") as f:
            # Read first few documents
            for i, line in enumerate(f):
                if i >= 3:  # Only test first 3 documents
                    break
                    
                clause = json.loads(line.strip())
                print(f"\n📄 Document {i+1}:")
                print(f"  Contract Title: {clause.get('contract_title', 'N/A')}")
                print(f"  Clause Type: {clause.get('clause_type', 'N/A')}")
                print(f"  Text Length: {len(clause.get('text', ''))}")
                print(f"  Full Context Length: {len(clause.get('full_context', ''))}")
                print(f"  Source: {clause.get('source', 'N/A')}")
                print(f"  Question ID: {clause.get('question_id', 'N/A')}")
                
                # Show first 100 chars of text
                text_preview = clause.get('text', '')[:100]
                if text_preview:
                    print(f"  Text Preview: {text_preview}...")
                
    except Exception as e:
        print(f"❌ Error reading document: {e}")

if __name__ == "__main__":
    test_document_structure() 