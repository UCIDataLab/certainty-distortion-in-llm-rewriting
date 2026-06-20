import os, json, yaml
import hashlib
import importlib
import inspect
import jsonlines

from typing import Any, Callable, Dict, List


def assign_uuid(example, cols: List[str] = None) -> str:
    """Compute unique identifier for example based on specified cols."""
    if cols is not None:
        example = {k: v for k, v in example.items() if k in cols and v is not None}

    content = json.dumps(example, sort_keys=True, indent=0).encode("utf-8")
    return hashlib.md5(content).hexdigest()


def load_config_from_path(input_str: str) -> dict:
    """
    If input_str is a path to an existing YAML file, load it.
    Otherwise, attempt to parse it as a JSON string.

    Returns:
        dict

    Raises:
        ValueError: if parsing fails or result is not a dict.
    """

    # Case 1: It's a file path and exists
    if os.path.isfile(input_str):
        if input_str.endswith(".yml") or input_str.endswith(".yaml"):
            with open(input_str, "r") as f:
                data = yaml.safe_load(f)
        elif input_str.endswith(".json"):
            with open(input_str, "r") as f:
                data = json.load(f)
        elif input_str.endswith(".txt"):
            with open(input_str, "r") as f:
                data = f.read().strip()
        return data

    try:
        # import pdb; pdb.set_trace()
        data = json.loads(input_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Input is neither a valid file path nor valid JSON.: {input_str}") from e
    return data


def load_method(fully_qualified_name: str) -> Callable:
    """Dynamically load a method based on its fully qualified name.

    Parameters
    ----------
    fully_qualified_name (str):
        Full path to the method, which includes the package, module and
        function name. Format: 'package.module.class.method' or
        'package.module.function'.

    Returns
    -------
    callable
        The loaded method/function.

    Raises:
        ImportError: If module cannot be imported.
        AttributeError: If method/class cannot be found.
        ValueError: If the fully qualified name is invalid.
    """
    try:
        # Split the full path into parts
        parts = fully_qualified_name.split(".")

        if len(parts) < 2:
            raise ValueError(
                "Fully qualified name must contain at least module and function/class"
            )

        # Get the module path and attribute chain
        module_path = ".".join(parts[:-1])  # Everything except the last part
        final_attribute = parts[-1]  # Last part

        # Import the module
        module = importlib.import_module(module_path)

        # Get the method/function
        return getattr(module, final_attribute)

    except ImportError as e:
        raise ImportError(f"Failed to import module {module_path}: {str(e)}")
    except AttributeError as e:
        raise AttributeError(
            f"Failed to find attribute {final_attribute} in module {module_path}: {str(e)}"
        )


def filter_kwargs(fn: callable, kwargs: Dict[str, Any]):
    # Get the parameter names that the function accepts
    signature = inspect.signature(fn)
    valid_params = signature.parameters.keys()

    # Filter the kwargs to only include valid parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return filtered_kwargs


def load_class_from_config(configs: Dict[str, Any], **kwargs) -> object:
    class_name = configs["_class_"]
    class_params = {k: v for k, v in configs.items() if k != "_class_"}

    clss = load_method(class_name)
    return clss(**class_params, **kwargs)



## IO Utils
def read_jsonlines(path: str) -> List[Dict[str, Any]]:
    data = []
    with jsonlines.open(path) as reader:
        for obj in reader.iter(type=dict, skip_invalid=True):
            data.append(obj)
    print(f"Read file with {len(data)}:\n - {path}")
    return data


def read_txt(path: str) -> str:
    with open(path) as f:
        return f.read()