import os
import re

def migrate_to_safe_float(directory):
    import_sentinel = "from app.utils.helpers import safe_float"
    
    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith(".py") or "helpers.py" in file:
                continue
            
            path = os.path.join(root, file)
            with open(path, "r") as f:
                content = f.read()
            
            # Check for raw float() calls (basic regex for simplicity but covering dict.get patterns)
            # We target float( ... )
            new_content = re.sub(r'(?<!safe_)float\(', 'safe_float(', content)
            
            if new_content != content:
                # Add import if not present
                if import_sentinel not in new_content:
                    # Inset after 'import logging' or at the top
                    if "import logging" in new_content:
                        new_content = new_content.replace("import logging", "import logging\n" + import_sentinel)
                    elif "import os" in new_content:
                        new_content = new_content.replace("import os", "import os\n" + import_sentinel)
                    else:
                        new_content = import_sentinel + "\n" + new_content
                
                with open(path, "w") as f:
                    f.write(new_content)
                print(f"Migrated: {path}")

if __name__ == "__main__":
    migrate_to_safe_float("app")
    migrate_to_safe_float("scripts")
