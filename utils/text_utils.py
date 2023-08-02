import re

def clean_text(input_text):
    return re.sub(r"[^A-Z0-9- ]", "", input_text).strip("- ")
