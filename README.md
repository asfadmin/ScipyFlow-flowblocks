# ScipyFlow-flowblocks
ScipyFlow-flowblocks is the home repo for flowblocks

## How to use

### Importing Libraries

In order to import libraries into ScipyFlow, the `Flowblock URL` in settings must be set to the raw github content url of the `flowblock_source_files` in the form of
```
https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path_in_repo}/flowblock_source_files/
```
by default the url is set to `https://raw.githubusercontent.com/asfadmin/ScipyFlow-flowblocks/main/flowblock_source_files/`

## Contributing

Contributions should be done through a `pull request` from a feature branch to the `main` branch. To complete the pull request, at least 1 approving review is required

### Branching

For each new feature/minor change, a dedicated branch should be created under the format `{identifier}/{branch-name}` with the identifier being some set of characters to represent yourself such as your GitHub username. This is to help define ownership of features and branches.

### Conventional Commit

All commits made should follow [Conventional Commit](https://www.conventionalcommits.org/en/v1.0.0/). This allows for automatic changelog generation and acts as a standard for readability and commit size.

### Adding Flowblocks

#### What is a flowblock?
Flowblocks are small snippets of python code that can be assembled into a larger workflow using ScipyFlow

*Note: All `FlowBlock`s should be encoded in UTF-8*

#### How to create a flowblock?
Flowblocks are just python files with a few required components
1. A yaml docstring at the top of the file
    - The docstring requirements and options are described in the `Flowblock Specification` section
2. `main()` function
    - Helper functions can be used, but only the `main()` function will be called by the program

#### Flowblock Specification
```python
"""
name: "<< Enter Flowblock Name >>"
requirements:
    - add
    - requirements
    - here
inputs:
    name_of_input_1:
        type: << Any, Bool, Binary, Set, Sequence, Str, Map, None, Number, !CustomClass your.type.here >>
        default: << Value this variable should default to >>
        user_input: << None, Text, Dropdown, FileUpload >>
    name_of_input_N:
        type: << Any, Bool, Binary, Set, Sequence, Str, Map, None, Number, !CustomClass your.type.here >>
        default: << Value this variable should default to >>
        user_input: << None, Text, Dropdown, FileUpload >>
outputs:
    name_of_output:
        type: << Any, Bool, Binary, Set, Sequence, Str, Map, None, Number, !CustomClass your.type.here >>
description: "<< Describe what this flowblock does >>"
"""

def main(name_of_input_1, name_of_input_N):
    # python code 
    return <<your return>>

```

- `name`: The display name of your flowblock
- `requirements`: The packagages that need to be installed via micropip
- `inputs`: One or more variables that will be input into the flowblock
    - `type`: The type of the variable, options include
        - `Any`: Any of the following types
        - `Bool`: A Boolean value
        - `Binary`: Represents `bytes`, `bytearray`, and `memoryview` types
        - `Set`: Represents `set` and `frozenset` types
        - `Sequence`: Represents `list`, `tuple`, and `range` types
        - `Str`: Represents `str` type
        - `Map`: Represents `dict` type
        - `None`: Represents no type
        - `Number`: Represents `int`, `float`, or `complex` types
        - `CustomClass`: Allows for any arbitrary python class to be passed with the syntax `!CustomClass your.custom.class`
    - `default`: (OPTIONAL) The value that this input will default to
    - `user_input`: (OPTIONAL) Boolean value on if the user should be able to manually input a value, defaults to `False`
- `outputs`: One or more variables that will be returned from the flowblock
    - `type`: The type of the variable, options include
        - `Any`: Any of the following types
        - `Bool`: A Boolean value
        - `Binary`: Represents `bytes`, `bytearray`, and `memoryview` types
        - `Set`: Represents `set` and `frozenset` types
        - `Sequence`: Represents `list`, `tuple`, and `range` types
        - `Str`: Represents `str` type
        - `Map`: Represents `dict` type
        - `None`: Represents no type
        - `Number`: Represents `int`, `float`, or `complex` types
        - `CustomClass`: Allows for any arbitrary python class to be passed with the syntax `!CustomClass your.custom.class`
- `description`: A short text that describes what this flowblock does

#### Code Requirements

All python code must run in a WASM Emscripten environment. This means that all imported python packages must have a pure python wheel.
All python code should also be able to run in a Desktop environment. \
You can determine what system the code is currently running on using the platform library
```py
import platform
system = platform.system()

if system() == "Emscripten":
    # import web packages
    # Run web only code
else:
    # import desktop packages
    # Run desktop only code

# Run universal code

```

#### Location and Libraries

Flowblocks must be added to the `/flowblock_source_files` directory. Flowblocks should be in a "library", a `library` is a folder inside the flowblock_source_files directory so the path of a library is `/flowblock_source_files/{library name}`.

Libraries are meant to categorize the flowblocks. If there is an existing library relevant to your flowblock idea, put your new flowblock in there, otherwise create a new library with an appropriate name.
