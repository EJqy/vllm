"""Worker-side hook for the V2 model runner.

V2 reads draft token *values* from its own GPU buffer `req_states.draft_tokens`
(written by the speculator) and only takes the per-request draft *count* from
the scheduler. With synchronous scheduling the scheduler's draft is final when
the step starts, so we copy the scheduled values into that buffer before
`prepare_inputs` reads it. The V1 runner already uses the scheduler's values.
"""

import torch

from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def _write_scheduled_drafts(runner: GPUModelRunner, scheduler_output) -> None:
    buf = runner.req_states.draft_tokens
    k = buf.shape[1]
    index = runner.req_states.req_id_to_index
    rows, values = [], []
    for req_id, tokens in scheduler_output.scheduled_spec_decode_tokens.items():
        # -1 marks placeholder slots whose values live only on the GPU.
        if not tokens or min(tokens) < 0 or req_id not in index:
            continue
        rows.append(index[req_id])
        values.append(tokens + [0] * (k - len(tokens)))
    if not rows:
        return
    values_t = torch.tensor(values, dtype=buf.dtype, pin_memory=True)
    rows_t = torch.tensor(rows, dtype=torch.int64, pin_memory=True)
    buf.index_copy_(
        0,
        rows_t.to(buf.device, non_blocking=True),
        values_t.to(buf.device, non_blocking=True),
    )


def install() -> None:
    orig_update_requests = GPUModelRunner.update_requests

    def update_requests(self, scheduler_output):
        orig_update_requests(self, scheduler_output)
        if not self.vllm_config.scheduler_config.async_scheduling:
            _write_scheduled_drafts(self, scheduler_output)

    GPUModelRunner.update_requests = update_requests
