"""
name: "Debug Print"
inputs:
    input_string:
        type: Str
        user_input: True
description: "Prints input string to web console"
"""

import pyodide.code

def main(input_string):
    print("Python print")
    js_code = f'console.log({input_string});'
    pyodide.code.run_js(js_code)
