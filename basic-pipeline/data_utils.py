import os

# read lines from a file and offer path non-existance check
def load_lines_from_file(path=None):
    if path == None or os.path.exists(path) == False:
        raise Exception("None path or not exists")
    else:
        with open(path, "r", encoding="utf8") as f:
            content = f.readlines()        
    return content