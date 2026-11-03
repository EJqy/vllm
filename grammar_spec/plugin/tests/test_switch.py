"""The plugin patches nothing unless GRAMMAR_DRAFT=1, and only patches a
vLLM module once that module is imported."""

import os
import subprocess
import sys

PROBE = """
import sys, vllm_grammar_draft
vllm_grammar_draft.register()
print("worker_imported", "vllm.v1.worker.gpu.model_runner" in sys.modules)
from vllm.v1.structured_output import StructuredOutputManager as M
print("trim", M.trim_spec_tokens.__module__)
print("accept", M.accept_tokens.__module__)
"""


def run_probe(env_value):
    env = {k: v for k, v in os.environ.items() if k != "GRAMMAR_DRAFT"}
    if env_value is not None:
        env["GRAMMAR_DRAFT"] = env_value
    out = subprocess.run([sys.executable, "-c", PROBE], env=env, check=True,
                         capture_output=True, text=True).stdout
    return dict(line.split(" ", 1) for line in out.strip().splitlines())


def test_off_by_default():
    res = run_probe(None)
    assert res["trim"] == "vllm.v1.structured_output"
    assert res["accept"] == "vllm.v1.structured_output"


def test_on_patches_lazily():
    res = run_probe("1")
    assert res["worker_imported"] == "False"
    assert res["trim"] == "vllm_grammar_draft.scheduler_hook"
    assert res["accept"] == "vllm_grammar_draft.scheduler_hook"
