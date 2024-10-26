"""
name: "Load File"
inputs:
    file_path:
        type: Str
        user_input: Text
outputs:
    file:
        type: Any
description: "Loads a file from the LocalStorage in the browser."
"""

import platform

def main(file_path):
    system = platform.system()

    if system == "Emscripten":
        import js
        import base64
        web_path = f'data:{file_path}'
        base64_encoded_file = js.localStorage.getItem(web_path)
        return base64.b64decode(base64_encoded_file)
    else:
        try:
            with open(file_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            print(f"The file {file_path} was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
