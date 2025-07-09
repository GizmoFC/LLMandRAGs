import json

def inspect_cuad_file():
    """Inspect the structure of CUAD_v1.json file"""
    try:
        with open('CUAD_v1.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"File type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Dictionary keys: {list(data.keys())}")
            print(f"Number of top-level items: {len(data)}")
            
            # Show first few items
            for i, (key, value) in enumerate(data.items()):
                if i < 3:  # Show first 3 items
                    print(f"\nKey: {key}")
                    print(f"Value type: {type(value)}")
                    if isinstance(value, dict):
                        print(f"Value keys: {list(value.keys())}")
                    elif isinstance(value, list):
                        print(f"List length: {len(value)}")
                        if len(value) > 0:
                            print(f"First item type: {type(value[0])}")
                            if isinstance(value[0], dict):
                                print(f"First item keys: {list(value[0].keys())}")
                    print(f"Value preview: {str(value)[:200]}...")
                else:
                    break
                    
        elif isinstance(data, list):
            print(f"List length: {len(data)}")
            if len(data) > 0:
                print(f"First item type: {type(data[0])}")
                if isinstance(data[0], dict):
                    print(f"First item keys: {list(data[0].keys())}")
                print(f"First item preview: {str(data[0])[:200]}...")
        
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    inspect_cuad_file() 