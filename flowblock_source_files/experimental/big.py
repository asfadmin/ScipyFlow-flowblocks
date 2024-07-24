"""
name: "Big"
requirements:
inputs:
    string1:
        type: Str
        default: "World"
        user_input: True

    string2:
        type: Str
        user_input: True
        
    string3:
        type: Str
        user_input: True
        
    string4:
        type: Str
        default: "is"
        user_input: True
        
    string5:
        type: Str
        user_input: True
        
    string6:
        type: Str
        default: "mine"
        user_input: True
        
    string7:
        type: Str
        user_input: True

outputs:
    concatenated_string:
        type: Str
description: "Does things"
"""

def main(string1, string2, string3, string4, string5, string6, string7):
    return string1 + string2 + string3 + string4 + string5 + string6 + string7
