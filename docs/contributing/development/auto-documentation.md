# 🔄 Automatic Documentation Generation

KDP includes tools to automatically generate documentation from code docstrings and visualize model architectures for different configurations. This ensures that the documentation remains up-to-date with the codebase.

## 🛠️ Documentation Tools

The following tools are included in the `scripts/` directory:

1. **`generate_docstring_docs.py`**: Extracts docstrings from classes and functions in the codebase and converts them to Markdown documentation.
2. **`generate_model_diagrams.py`**: Creates visualizations of model diagrams for different feature types and preprocessing configurations.

## 🚀 Generating Documentation

You can generate the documentation content using the Makefile target:

```bash
make generate_doc_content
```

This will:
- Extract API documentation from docstrings and save it to `docs/generated/api/`
- Generate model diagrams and save them to `docs/features/imgs/models/`
- Create an API index at `docs/generated/api_index.md`

## 🏗️ Model Diagram Types

The script generates visualizations for a variety of model configurations, including:

- Basic feature types (numerical, categorical, text, date, passthrough)
- Feature combinations and cross-features
- Different output modes (CONCAT and DICT)
- Advanced features like tabular attention and transformer blocks
- Distribution-aware encoding and numerical embeddings

Each diagram shows:
- The full TensorFlow model architecture
- Input and output shapes
- Layer connections and data flow

## 📚 API Documentation from Docstrings

The docstring extraction process:
1. Scans key modules in the codebase
2. Extracts docstrings from classes and their methods
3. Converts them to Markdown format
4. Organizes them into a structured API reference

## 🔄 Integration with MkDocs

The generated documentation is automatically included when building the MkDocs site:

```bash
make serve_doc      # Test documentation locally
```

or

```bash
make deploy_doc     # Deploy to GitHub Pages
```

## 📝 Documenting Your Code

To ensure your code is properly included in the automatic documentation:

1. **Use Google-style docstrings** for all classes and methods:

~~~python
def my_function(param1, param2):
    """
    One-line description of function.

    More detailed description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of the return value

    Examples:
        ```python
        result = my_function("example", 123)
        ```
    """
    # Function implementation
~~~

2. **Document parameters and return values** to provide clear usage instructions.

3. **Include examples** to demonstrate usage where appropriate.

## 🔄 Dynamic Preprocessing Pipeline

The `DynamicPreprocessingPipeline` class provides a flexible way to build preprocessing pipelines with optimized execution flow:

```python
class DynamicPreprocessingPipeline:
    """
    Dynamically initializes and manages a sequence of Keras preprocessing layers, with selective retention of outputs
    based on dependencies among layers, and supports streaming data through the pipeline.
    """
```

This class analyzes dependencies between layers and ensures that each layer receives the outputs it needs from previous layers.

## 🧹 Cleaning Up Old Diagrams

When updating the model diagram generation process, you may need to clean up old diagrams or directories. Use the Makefile target:

```bash
make clean_old_diagrams
```

This will:
- Remove obsolete diagram directories
- Clean up unused diagram files

This target is also automatically included when running `make clean`.
