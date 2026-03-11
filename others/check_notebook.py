import nbformat

# Try to read and identify the issue
with open(r'D:\project-2026\IISc-LLM\Assignments\Inference_profiling_and_optimization.ipynb', 'r') as f:
    try:
        nb = nbformat.read(f, as_version=4)
        print("Notebook is valid!")
    except Exception as e:
        print(f"Error: {e}")