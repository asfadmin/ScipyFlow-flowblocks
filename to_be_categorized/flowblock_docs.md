# Flowblock Documentation

## What is a flowblock?
Flowblocks are small snippets of python code that can be assembled into a larger workflow using ScipyFlow \
#### Example flowblock
```
"""
name: "Coolifier"
requirements:
    - numpy
inputs:
    input_string:
        type: Str
outputs:
    output_string:
        type: Str
description: "returns input string concatinated with \" is cool\""
"""

def main(input_string):
    import numpy
    x = numpy.abs(-5)
    return string + " is cool"
```

## How to create a flowblock?
Flowblocks are just python files with a few required components
1. A yaml docstring at the top of the file
2. `main()` function

### Docstring Creation
#### Example
```py
"""
name: "Coolifier"
requirements:
    - numpy
inputs:
    input_string:
        type: Str
outputs:
    output_string:
        type: Str
description: "returns input string concatinated with \" is cool\""
"""
```

#### Fields
These are the fields in the docstring, formatted in yaml
##### name
>**type**: string \
**purpose**: name of the flowblock
##### requirements
>**type**: array \
**purpose**: piodide python libraries that must be imported for flowblock to work
##### inputs
>**type**: map \
**purpose**: map of name of input variables to type of variable as defined in python types. Must know the type(s) that the flowblock will accept
##### outputs
>**type**: map \
**purpose**: map of name of output variables to type of variable as defined in python types. Must know the type(s) the flowblock will return
##### description
>**type**: string \
**purpose**: a short description of what the flowblock does


#### Python Types
**Bool**: for `bool` type \
**Binary**: for `bytes`, `bytearray`, and `memoryview` types \
**Set**: for `set` and `frozenset` types \
**Sequence**: for `list`, `tuple`, and `range` types \
**Str**: for `str` type \
**Map**: for `dict` type \
**Number**: for `int`, `float`, and `complex` types \
**UserInput(String)**: for a text box of user input, value will be static when running workflow

## Python code
The python code segment can work like any other python code except for the following
1. It can only use pyodide supported packages or any pure python wheel \
   **Exception**: requests library works thanks to pyodide_http library
2. The python source will start and return from the function `main()`

Examples of acceptable python code
```py
def main():
    return "Hello"
```
```py
def helper(x):
    return x + 10

def main():
    num = helper(5)
    return num
```
