"""DeepSeek-V3 inference generation pipeline — primary entry point for autoregressive text generation.

This module serves as the top-level inference driver for the DeepSeek-V3 language model,
orchestrating distributed model initialization, token generation, and user interaction.
It connects the model architecture (``model.py``, which defines the ``Transformer`` and
``ModelArgs``) with the HuggingFace ``AutoTokenizer`` for prompt encoding/decoding and
``safetensors`` for efficient checkpoint loading.

Key functions:
    - ``sample(logits, temperature)``: Stochastic token selection via Gumbel-max trick
      using the exponential distribution reparameterization.
    - ``generate(model, prompt_tokens, ...)``: KV-cache-aware autoregressive generation
      loop supporting batched variable-length prompts with early EOS termination.
    - ``main(ckpt_path, config, ...)``: End-to-end inference pipeline with distributed
      process group initialization, model construction, checkpoint loading, and both
      interactive (multi-turn chat) and batch generation modes.

Distributed execution model:
    Designed to be launched via ``torchrun`` (PyTorch Elastic Launch) which sets the
    ``WORLD_SIZE``, ``RANK``, and ``LOCAL_RANK`` environment variables. Uses the NCCL
    backend for GPU-to-GPU collective communication required by tensor-parallel layers
    in ``model.py`` (``all_reduce`` in ``ParallelEmbedding``, ``RowParallelLinear``,
    ``MoE``; ``all_gather`` in ``Transformer.forward()`` for logits). In multi-rank mode,
    rank 0 handles user I/O and broadcasts prompts to all other ranks via
    ``dist.broadcast_object_list``.

Reference:
    DeepSeek-V3 Technical Report, arXiv:2412.19437
    - Section 2.1 (Multi-head Latent Attention) — implemented in model.py MLA class
    - Section 2.1.2 (DeepSeekMoE) — implemented in model.py Gate/MoE classes
    - Section 2.2 (Multi-Token Prediction) — MTP modules excluded during inference
    See also: model.py for the Transformer architecture, kernel.py for FP8 compute kernels.
"""

import os
import json
from argparse import ArgumentParser
from typing import List

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs


def sample(logits, temperature: float = 1.0):
    """Samples token indices from logits via the Gumbel-max trick using exponential distribution reparameterization.

    Implements stochastic token selection by dividing softmax probabilities by
    samples drawn from the standard exponential distribution, then taking the
    argmax.  This is mathematically equivalent to adding Gumbel(0, 1) noise to
    the log-probabilities and selecting the argmax (the *Gumbel-max trick*), but
    avoids explicitly computing Gumbel noise and the associated log-of-log
    numerical instabilities.

    Args:
        logits (torch.Tensor): Raw (unnormalized) prediction scores from the
            model's output head.  Shape: ``(batch_size, vocab_size)`` where
            ``vocab_size`` is 102400 for DeepSeek-V3 (see ``ModelArgs.vocab_size``).
        temperature (float, optional): Temperature scaling factor applied to
            logits before softmax.  Higher values increase randomness; lower
            values sharpen the distribution toward greedy selection.  Clamped
            internally to a minimum of 1e-5.  Defaults to 1.0.

    Returns:
        torch.Tensor: Selected token indices with shape ``(batch_size,)``,
            dtype ``torch.int64``.

    Notes:
        The Gumbel-max trick via exponential sampling: given probabilities
        ``p_i = softmax(logits_i / temperature)``, sampling ``e_i ~ Exp(1)``
        and computing ``argmax_i(p_i / e_i)`` produces samples from the
        categorical distribution defined by ``p_i``.  This avoids using
        ``torch.multinomial``, which incurs sequential CUDA kernel launches
        that become a performance bottleneck for large vocabulary sizes
        (>100K tokens such as DeepSeek-V3's 102400-token vocabulary).

        The temperature is clamped to a floor of 1e-5 (rather than testing
        for exact zero) to prevent division-by-zero while keeping the
        branch-free code path.  At temperature = 1e-5 the distribution is
        effectively deterministic, so the numerical impact is negligible.
    """
    # [Trade-offs] Clamp temperature floor to 1e-5 rather than checking for zero — avoids branch
    # while preventing division-by-zero; accepts negligible numerical impact at extremely low temperatures
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    # [Alternatives Considered] Using Gumbel-max trick via exponential distribution instead of
    # torch.multinomial — this formulation (probs / Exp(1)) is numerically equivalent to
    # Gumbel-softmax sampling but avoids torch.multinomial's sequential CUDA kernel launches
    # for large vocab sizes (>100K tokens)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
) -> List[List[int]]:
    """Generates new tokens autoregressively with KV-cache-aware prefill and decode phases.

    Implements batched autoregressive text generation using a two-phase approach:

    1. **Prefill phase**: The first ``model.forward()`` call processes all tokens
       from position 0 up to ``min(prompt_lens)`` in a single pass, populating
       the KV cache for the shared prefix across all sequences in the batch.
    2. **Decode phase**: Subsequent calls process a single token position at a
       time, leveraging the KV cache (via ``prev_pos``) to avoid recomputing
       attention over prior positions.

    Prompt tokens are preserved during generation: a ``prompt_mask`` prevents
    the model's predictions from overwriting original prompt positions when
    prompts in the batch have different lengths.  Generation terminates early
    when all sequences in the batch have produced an EOS token.

    Args:
        model (Transformer): The DeepSeek-V3 Transformer model instance
            (see ``model.py``).  Must have ``model.max_seq_len`` set and KV
            caches pre-allocated in each ``MLA`` layer.  The model's
            ``forward()`` method accepts token IDs of shape
            ``(batch_size, seq_len)`` and returns logits of shape
            ``(batch_size, vocab_size)``.
        prompt_tokens (List[List[int]]): A list of ``batch_size`` token ID
            sequences.  Each inner list contains integer token IDs produced
            by the tokenizer.  Sequences may have different lengths; the
            shortest prompt length determines the boundary between shared
            prefill and per-sequence decode.
        max_new_tokens (int): Maximum number of new tokens to generate per
            sequence (beyond the prompt).  Actual generation may be shorter
            if EOS is encountered or ``model.max_seq_len`` is reached.
        eos_id (int): End-of-sequence token ID used to detect completion.
            Set to -1 to disable early termination (e.g., during warmup).
        temperature (float, optional): Sampling temperature passed to
            ``sample()``.  When ``temperature <= 0``, deterministic greedy
            decoding (argmax) is used instead of stochastic sampling.
            Defaults to 1.0.

    Returns:
        List[List[int]]: A list of ``batch_size`` completion token lists.
            Each inner list contains only the newly generated tokens (prompt
            tokens excluded), truncated at the first EOS token if present.

    Raises:
        AssertionError: If the longest prompt in ``prompt_tokens`` exceeds
            ``model.max_seq_len``.

    Notes:
        Internal tensor shapes during generation:
            - ``tokens``: ``(batch_size, total_len)`` on CUDA, dtype ``torch.long``,
              initialized to -1 as a sentinel for unfilled positions.
            - ``logits`` from ``model.forward()``: ``(batch_size, vocab_size)``
              (vocab_size is 102400 for the 671B configuration).
            - ``next_token``: ``(batch_size,)`` — one token per sequence.
            - ``prompt_mask``: ``(batch_size, total_len)`` boolean — ``True`` where
              the position contains a prompt token.

        The ``prev_pos`` variable tracks the start of the KV cache window: after
        each step, ``prev_pos = cur_pos`` so the next call only processes one new
        token while the model reads cached keys/values for all prior positions.

        See ``model.py`` ``Transformer.forward()`` for details on how ``start_pos``
        controls KV cache indexing and causal mask construction.
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    # [Assumptions Made] Initialize with -1 as sentinel value — assumes -1 is never a valid token ID,
    # used to distinguish prompt vs. unfilled positions via prompt_mask
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda")
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    # [Trade-offs] Start with prev_pos=0 for initial prefill pass — the model's KV cache uses prev_pos
    # to determine cache insertion point, enabling efficient autoregressive generation without
    # recomputing prior attention
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    prompt_mask = tokens != -1
    # [Alternatives Considered] Starting from min(prompt_lens) rather than 0 — in batched generation
    # with variable-length prompts, all prompts share the initial prefix up to the shortest prompt
    # length, enabling a single prefill pass for the common prefix
    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        # [Trade-offs] When temperature=0, use deterministic argmax instead of sampling — greedy
        # decoding is faster and deterministic but sacrifices output diversity; sampling with
        # temperature>0 enables stochastic generation
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        # [Assumptions Made] Prompt tokens take precedence over generated tokens — during the prefill
        # phase, where prompt positions haven't been fully consumed yet, the original prompt token is
        # preserved rather than the model's prediction
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> None:
    """End-to-end inference pipeline: distributed init, model loading, and generation loop.

    Orchestrates the complete inference lifecycle:

    1. **Distributed initialization**: Reads ``WORLD_SIZE``, ``RANK``, and
       ``LOCAL_RANK`` environment variables (set by ``torchrun``) and initializes
       the NCCL process group for GPU-to-GPU collective communication.  Print
       output is suppressed on non-rank-0 processes to avoid duplicate logging.
    2. **Model construction**: Instantiates the ``Transformer`` from ``model.py``
       using configuration loaded from a JSON file (mapped to ``ModelArgs``).
       Global state (``world_size``, ``rank``, ``Linear.dtype``) is set inside
       ``Transformer.__init__()``.
    3. **Warmup pass**: Runs a short dummy generation (prompt ``"DeepSeek"``,
       ``max_new_tokens=2``, ``eos_id=-1``) to trigger Triton kernel JIT
       compilation and CUDA graph capture *before* loading the actual checkpoint
       weights.  This avoids a first-token latency spike during real inference.
    4. **Checkpoint loading**: Loads the rank-specific safetensors shard
       ``model{rank}-mp{world_size}.safetensors`` from ``ckpt_path`` using
       ``safetensors.torch.load_model``.
    5. **Generation mode dispatch**:
       - *Interactive mode*: Multi-turn chat loop where rank 0 reads user input
         and broadcasts it to all ranks via ``dist.broadcast_object_list``.
         Conversation history (``messages`` list) is maintained across turns and
         formatted using the tokenizer's chat template.  Commands ``/exit`` and
         ``/clear`` control the loop and history.
       - *Batch mode*: Reads prompts from ``input_file`` (one per line), encodes
         all prompts, and generates completions in a single ``generate()`` call
         bounded by ``ModelArgs.max_batch_size``.

    Args:
        ckpt_path (str): Path to the model checkpoint directory.  Must contain
            safetensors weight shards named ``model{rank}-mp{world_size}.safetensors``
            and a HuggingFace tokenizer (``tokenizer.json``, ``tokenizer_config.json``).
        config (str): Path to a JSON configuration file whose keys map to
            ``ModelArgs`` dataclass fields (e.g., ``inference/configs/config_671B.json``).
        input_file (str, optional): Path to a text file with one prompt per line
            for batch generation mode.  Defaults to ``""`` (unused in interactive
            mode).
        interactive (bool, optional): If ``True``, enters a multi-turn interactive
            chat loop; if ``False``, reads prompts from ``input_file``.
            Defaults to ``True``.
        max_new_tokens (int, optional): Maximum number of new tokens to generate
            per prompt.  Defaults to 100.
        temperature (float, optional): Sampling temperature passed to
            ``sample()`` via ``generate()``.  Defaults to 1.0.

    Returns:
        None

    Notes:
        Distributed communication operations in this function:
            - ``dist.init_process_group("nccl")``: Initializes the NCCL backend
              process group across all ranks.  NCCL is selected for its optimized
              GPU-to-GPU bandwidth on NVIDIA hardware.
            - ``dist.broadcast_object_list(objects, 0)``: Rank 0 broadcasts the
              user prompt string to all other ranks so every process generates
              from the same input.  Called twice per turn in interactive mode
              (once on rank 0 to send, once on other ranks to receive).
            - ``dist.destroy_process_group()``: Cleans up the process group on exit.

        The model's tensor-parallel layers (``ParallelEmbedding``,
        ``ColumnParallelLinear``, ``RowParallelLinear``, ``MoE``) perform their
        own ``all_reduce`` / ``all_gather`` calls internally during
        ``model.forward()`` — see ``model.py`` for details.

        Weight loading via ``load_model`` happens *after* the warmup pass so that
        Triton kernels are already compiled when real inference begins.

        See also: ``model.py`` ``Transformer`` class, ``kernel.py`` for FP8
        quantized compute kernels invoked by the model's ``Linear`` layers.
    """
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        # [Alternatives Considered] Using NCCL backend instead of Gloo or MPI — NCCL is optimized
        # for GPU-to-GPU collective communication and provides highest bandwidth for all_reduce/
        # all_gather operations required by tensor-parallel inference
        dist.init_process_group("nccl")
    global print
    # [Trade-offs] Suppress print on non-rank-0 processes — prevents duplicate output in distributed
    # mode while maintaining single-process output semantics; uses lambda replacement instead of
    # logging framework for simplicity
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    # [Assumptions Made] Fix CPU thread count to 8 — bounds OpenMP/MKL thread contention in
    # multi-process distributed inference; the value is a conservative default for typical GPU
    # server configurations with multiple inference processes per node
    torch.set_num_threads(8)
    # [Assumptions Made] Fixed seed 965 for reproducibility — ensures deterministic weight
    # initialization during model construction before actual checkpoint weights are loaded;
    # the specific value is arbitrary but fixed for reproducible inference behavior
    torch.manual_seed(965)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print(args)
    with torch.device("cuda"):
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    # [Trade-offs] Warmup generation with dummy prompt before loading weights — triggers Triton
    # kernel JIT compilation and CUDA graph capture with placeholder weights, avoiding first-token
    # latency spike during actual inference; uses short prompt "DeepSeek" with max_new_tokens=2
    # and eos_id=-1 (never matches) to minimize warmup cost
    tokenizer.decode(generate(model, [tokenizer.encode("DeepSeek")], 2, -1, 1.)[0])
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))

    if interactive:
        messages = []
        while True:
            if world_size == 1:
                prompt = input(">>> ")
            elif rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                # [Distributed] broadcast_object_list from rank 0 → all ranks — rank 0 sends
                # the user prompt string so every rank generates from identical input.
                # Participating ranks: all ranks in default process group. Data: list[str].
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                # [Distributed] broadcast_object_list receive on non-rank-0 — receives the
                # prompt string broadcast by rank 0. objects[0] is overwritten in-place.
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue
            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature)
            completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
            print(completion)
            messages.append({"role": "assistant", "content": completion})
    else:
        with open(input_file) as f:
            prompts = [line.strip() for line in f.readlines()]
        assert len(prompts) <= args.max_batch_size, f"Number of prompts exceeds maximum batch size ({args.max_batch_size})"
        prompt_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True) for prompt in prompts]
        completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)
        completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
        for prompt, completion in zip(prompts, completions):
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()

    if world_size > 1:
        # [Distributed] Cleanup — destroys the NCCL process group to release GPU
        # communication resources. All ranks must call this before process exit.
        dist.destroy_process_group()


if __name__ == "__main__":
    """
    Command-line interface for distributed text generation.

    Arguments:
        --ckpt-path (str): Path to the model checkpoint directory.
        --config (str): Path to the model configuration file.
        --input-file (str, optional): File containing prompts for batch processing.
        --interactive (bool, optional): Enable interactive mode for generating text.
        --max-new-tokens (int, optional): Maximum number of new tokens to generate. Defaults to 200.
        --temperature (float, optional): Temperature for sampling. Defaults to 0.2.

    Raises:
        AssertionError: If neither input-file nor interactive mode is specified.
    """
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()
    assert args.input_file or args.interactive, "Either input-file or interactive mode must be specified"
    main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature)
