from llm.llm_codegen import generate_python_code  # adjust path if needed
from typing import List, Dict 
from dotenv import load_dotenv
load_dotenv()


def generate_code_for_steps(steps: list[str]) -> dict[str, str]:
    """
    Given a list of step instructions from the planning agent,
    return a mapping of step_name → generated Python code.
    """
    code_map = {}
    for i, step in enumerate(steps):
        step_name = f"step_{i+1}"
        code = generate_python_code(step)  # this calls your LLM to generate code
        code_map[step_name] = code
    return code_map
