import ast
import os
import sys
import inspect
import importlib.util

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def extract_functions_from_file(file_path: str):
    """
    Extract all function names and docstrings from a Python file.

    Args:
        file_path (str): Path to Python file.

    Returns:
        list: List of dictionaries with function metadata.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append({
                'type': 'function',
                'function':
                {"name": node.name,
                "description": ast.get_docstring(node),
                "parameters": [arg.arg for arg in node.args.args]}
            })

    return functions
def load_functions_from_file(filepath):
    module_name = os.path.basename(filepath).replace(".py", "")

    spec = importlib.util.spec_from_file_location(
        module_name,
        filepath
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    functions = []

    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj):
            functions.append(obj)

    return functions

def get_tools_dict(file_path: str) -> dict[str, callable]:
    """
    Returns a dict of {function_name: callable} for all functions
    defined in the given file, ready for agent tool usage.

    Args:
        file_path (str): Path to the Python file.

    Returns:
        dict: {name: callable} for each function defined in the file.
    """
    spec = importlib.util.spec_from_file_location("_tool_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Only include functions actually defined in this file (not imports)
    return {
        
        name: obj
        for name, obj in inspect.getmembers(module, inspect.isfunction)
        if obj.__module__ == module.__name__
    }