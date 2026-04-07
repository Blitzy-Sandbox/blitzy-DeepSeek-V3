# Technical Specification

# 0. Agent Action Plan

## 0.1 Intent Clarification

### 0.1.1 Core Documentation Objective

Based on the provided requirements, the Blitzy platform understands that the documentation objective is to **transform the entirely undocumented DeepSeek-V3 open-source inference codebase into a self-explanatory "Developer's Log"** that achieves 100% coverage of docstrings and inline rationale comments across all Python modules, without modifying any executable code logic.

**Request Category:** Create new documentation + Improve documentation coverage

**Documentation Type:** Developer's Log — a hybrid format combining Google-style API docstrings with inline "why" rationale commentary, plus companion architectural markdown files.

**Target Audience:** ML engineers and researchers proficient in PyTorch distributed computing but unfamiliar with DeepSeek-V3's specific architectural innovations (MLA, auxiliary-loss-free MoE routing, MTP, block-wise FP8 quantization).

**Detailed Requirements:**

- Add comprehensive Google-style docstrings to every function, class, method, and module-level construct in 5 Python files under `inference/`
- Add inline "why" rationale comments to every non-trivial implementation decision, categorized by: Alternatives Considered, Refactoring Rationale, Assumptions Made, Trade-offs, and Future-proofing
- Document all tensor shapes at input/output boundaries for every tensor-accepting function
- Document all distributed communication operations (`all_reduce`, `all_gather`, `broadcast_object_list`) with participating ranks and data movement semantics
- Create companion markdown files: `inference/ARCHITECTURE.md` (pipeline overview) and `inference/configs/CONFIG_REFERENCE.md` (parameter reference)
- Assess root-level `README.md` and `README_WEIGHTS.md` for documentation gaps; add a "Code Documentation" cross-reference section if warranted
- Cross-reference all major components to the DeepSeek-V3 technical report (arXiv:2412.19437), specifically Sections 2.1 (MLA), 2.1.2 (DeepSeekMoE), and 2.2 (MTP)

**Implicit Documentation Needs Surfaced:**

- The `kernel.py` module was not explicitly listed in the user's primary "Files in Scope" enumeration but is imported by both `model.py` and `fp8_cast_bf16.py` and is present under `inference/` — it requires full documentation treatment as a critical dependency
- The `Gate` class implements the auxiliary-loss-free load balancing mechanism via a bias parameter (`self.bias`) conditioned on `self.dim == 7168`, which requires explicit rationale documentation explaining this hardcoded dimension check
- The `MLA` class implements two distinct attention strategies (`naive` vs. `absorb`) selected by the global `attn_impl` variable, each requiring separate documentation of KV cache semantics and computational trade-offs
- Four separate configuration JSON files (`config_16B.json`, `config_236B.json`, `config_671B.json`, `config_v3.1.json`) exist with differing parameter sets, requiring comprehensive documentation in `CONFIG_REFERENCE.md`

### 0.1.2 Special Instructions and Constraints

**Preservation Mandate (CRITICAL):**

- MUST NOT modify any executable code logic, control flow, mathematical operations, or function signatures
- MUST NOT alter import statements, variable names, class hierarchies, or module structure
- MUST NOT add type hints to function signatures (types documented in docstrings only)
- MUST NOT modify config JSON files structurally — create companion `.md` files instead
- MUST NOT touch `.github/`, `figures/`, `LICENSE-CODE`, `LICENSE-MODEL`, or external framework code

**Bug Reporting Protocol:** If code quality issues or bugs are discovered during documentation, note them in `# NOTE:` comments adjacent to the relevant code. Do not fix, refactor, or optimize.

**Docstring Standard:** Google-style Python docstrings with mandatory sections: Purpose, Args (with types and tensor shapes), Returns (with types and shapes), Raises, Notes (cross-references to technical report).

**Inline Comment Standard:** Comments explain developer reasoning ("why"), not code behavior ("what"). Each significant decision documented with ≥1 mandatory "why" category.

**Forbidden Patterns:**

- Comments restating code behavior (e.g., `# add bias to output` above `output += bias`)
- Docstrings without parameters, return values, or purpose
- Omitting rationale for non-obvious implementation choices
- Vague rationale without specific justification (e.g., "for efficiency" without stating the metric)
- Speculating about rationale when the technical report provides explicit justification — cite the report instead

**Execution Sequence (User-Specified):**

- Document `model.py` first (architectural core, establishes terminology)
- Document `generate.py` second (primary entry point)
- Document `convert.py` and `fp8_cast_bf16.py` (utility modules)
- Create `inference/ARCHITECTURE.md` (synthesized cross-module overview)
- Create `inference/configs/CONFIG_REFERENCE.md` (configuration parameter reference)
- Assess root-level READMEs for gaps
- Run validation gate against all documented constructs

### 0.1.3 Technical Interpretation

These documentation requirements translate to the following technical documentation strategy:

- To **document the Transformer core**, we will UPDATE `inference/model.py` by adding module-level docstrings, enriching existing class/method docstrings with tensor shapes and distributed semantics, and inserting inline "why" comments for all non-trivial implementation decisions (MLA latent compression, absorbed vs. naive attention, expert routing with bias-based load balancing, RoPE frequency correction, mscale computation)
- To **document the generation pipeline**, we will UPDATE `inference/generate.py` by adding module-level docstrings, documenting the Gumbel-softmax sampling strategy in `sample()`, the KV-cache-aware autoregressive loop in `generate()`, and the distributed prompt broadcast mechanism in `main()`
- To **document checkpoint conversion**, we will UPDATE `inference/convert.py` by adding module-level docstrings, documenting the HuggingFace-to-model-parallel weight mapping dictionary, expert sharding logic, and the layer-61 skip for MTP modules
- To **document FP8-to-BF16 conversion**, we will UPDATE `inference/fp8_cast_bf16.py` by adding module-level docstrings, documenting the block-wise dequantization scheme, scale_inv tensor handling, and memory management via 2-shard caching
- To **document Triton kernels**, we will UPDATE `inference/kernel.py` by adding module-level docstrings and enriching existing kernel docstrings with detailed block-level quantization semantics, FP8 E4M3 format specifics, and autotuning configuration rationale
- To **provide architectural overview**, we will CREATE `inference/ARCHITECTURE.md` with pipeline stages, module dependency graph, key architectural decisions, and a glossary of DeepSeek-V3-specific terms
- To **document config parameters**, we will CREATE `inference/configs/CONFIG_REFERENCE.md` with every JSON parameter documented: type, valid range, purpose, relationship to architecture dimensions, and rationale for default values
- To **ensure README completeness**, we will UPDATE `README.md` by adding a "Code Documentation" section linking to per-module and architectural documentation

### 0.1.4 Inferred Documentation Needs

Based on code analysis:

- `inference/model.py` contains 10 classes and 4 standalone functions, all of which are public-facing and require documentation. Existing docstrings are present but lack tensor shape annotations, distributed communication documentation, and "why" rationale
- The `Gate.forward()` method in `model.py` (line 564) contains a hardcoded condition `if self.dim == 7168` for bias initialization — this non-obvious design choice requires an inline rationale comment explaining its purpose (the 671B model dimension)
- `inference/kernel.py` contains 3 Triton kernel functions and 3 Python wrapper functions — all require documentation of block-level quantization semantics and FP8 precision characteristics
- The `MoE.forward()` method uses `dist.all_reduce(y)` for cross-rank expert output aggregation (line 692) — this distributed communication pattern requires documentation of participating ranks and data movement

Based on structure:

- The inference pipeline spans `generate.py` → `model.py` → `kernel.py` in a clear dependency chain that requires consolidated documentation in `ARCHITECTURE.md`
- Weight conversion involves two separate paths (`convert.py` for HuggingFace→MP, `fp8_cast_bf16.py` for FP8→BF16) that share the `kernel.py` dequantization pathway

Based on dependencies:

- The `model.py` ↔ `kernel.py` interface (through `act_quant`, `weight_dequant`, `fp8_gemm` imports) requires interface documentation
- The `Transformer.__init__()` method sets global state (`world_size`, `rank`, `Linear.dtype`, `Linear.scale_fmt`) affecting all downstream components — this initialization pattern requires documentation

Based on user journey:

- A new ML engineer needs: setup guide (covered by existing README), architecture overview (new `ARCHITECTURE.md`), per-module API reference (enhanced docstrings), configuration guide (new `CONFIG_REFERENCE.md`), and rationale trail ("why" comments)

## 0.2 Documentation Discovery and Analysis

### 0.2.1 Existing Documentation Infrastructure Assessment

Repository analysis reveals a **minimal documentation footprint** with no dedicated documentation framework, no auto-generation tooling, and no structured documentation directory. The repository is a lean research-oriented inference codebase with documentation limited to root-level README files and sparse inline docstrings.

**Documentation Files Found:**

| File | Type | Coverage Status |
|------|------|-----------------|
| `README.md` | Project overview, benchmarks, deployment instructions | Covers model capabilities, benchmark results, and framework integration instructions; does not reference per-module code documentation |
| `README_WEIGHTS.md` | Weight file structure, FP8 quantization config | Documents weight file naming, FP8 format (`e4m3`, 128×128 block), and MTP module structure; does not link to code-level documentation |

**Documentation Generator:** None detected — no `mkdocs.yml`, `docusaurus.config.js`, `sphinx/conf.py`, `.readthedocs.yml`, or equivalent configuration files exist in the repository.

**API Documentation Tools:** None detected — no JSDoc, Sphinx, pydoc, or pdoc configuration. Existing docstrings in Python files use Google-style format but are incomplete (missing tensor shapes, missing distributed semantics, missing "why" rationale).

**Diagram Tools:** None detected — no Mermaid, PlantUML, or diagram generation configurations. The `figures/` directory exists at the root level but is explicitly out of scope.

**Documentation Hosting:** None configured — no GitHub Pages, ReadTheDocs, or static site deployment present.

**Existing Docstring Inventory (Current State):**

| File | Classes with Docstrings | Functions with Docstrings | Completeness |
|------|------------------------|--------------------------|--------------|
| `inference/model.py` | 10/10 | 4/4 | Partial — present but lacking tensor shapes, distributed docs, "why" rationale |
| `inference/generate.py` | 0/0 | 3/3 | Partial — present but lacking tensor shapes, sampling rationale |
| `inference/convert.py` | 0/0 | 1/1 | Partial — present but lacking weight mapping rationale, expert sharding details |
| `inference/fp8_cast_bf16.py` | 0/0 | 2/2 | Partial — present but lacking block-wise dequantization semantics |
| `inference/kernel.py` | 0/0 | 6/6 | Partial — present but lacking FP8 format specifics, autotuning rationale |

**Key Finding:** All Python files have basic docstrings for their public constructs, but none meet the user's required standard for: (a) tensor shape annotations at input/output boundaries, (b) distributed communication documentation, (c) cross-references to the technical report, or (d) inline "why" rationale comments for implementation decisions.

### 0.2.2 Repository Code Analysis for Documentation

**Search Patterns Employed:**

- Public APIs: Scanned all `.py` files under `inference/` for `class` and `def` declarations
- Module interfaces: Examined all imports across files to map the dependency graph
- Configuration options: Read all JSON files under `inference/configs/`
- CLI commands: Analyzed argument parsers in `generate.py`, `convert.py`, and `fp8_cast_bf16.py`

**Key Directories Examined:**

| Directory | Contents | Documentation Relevance |
|-----------|----------|------------------------|
| `inference/` | 5 Python files + `requirements.txt` | Primary documentation target — all code modules |
| `inference/configs/` | 4 JSON configuration files | Requires companion `CONFIG_REFERENCE.md` |
| Root (`/`) | `README.md`, `README_WEIGHTS.md` | Assessment target for documentation gap analysis |

**Code Construct Inventory:**

| Module | Classes | Standalone Functions | Methods | Triton Kernels | Total Constructs |
|--------|---------|---------------------|---------|----------------|------------------|
| `model.py` | 10 (`ModelArgs`, `ParallelEmbedding`, `Linear`, `ColumnParallelLinear`, `RowParallelLinear`, `RMSNorm`, `MLA`, `MLP`, `Gate`, `Expert`, `MoE`, `Block`, `Transformer`) | 4 (`linear()`, `precompute_freqs_cis()`, `apply_rotary_emb()`, plus 3 nested helpers) | 15 (`__init__` + `forward` for each class) | 0 | ~29 |
| `generate.py` | 0 | 3 (`sample()`, `generate()`, `main()`) | 0 | 0 | 3 |
| `convert.py` | 0 | 1 (`main()`) | 0 | 0 | 1 |
| `fp8_cast_bf16.py` | 0 | 2 (`main()`, `get_tensor()` nested) | 0 | 0 | 2 |
| `kernel.py` | 0 | 3 wrappers (`act_quant()`, `weight_dequant()`, `fp8_gemm()`) | 0 | 3 (`act_quant_kernel`, `weight_dequant_kernel`, `fp8_gemm_kernel`) | 6 |
| **Total** | **10** | **13** | **15** | **3** | **~41** |

**Related Documentation Found:**

- The existing `README.md` provides high-level usage instructions for running inference via `torchrun` but does not document internal code architecture or module-level APIs
- `README_WEIGHTS.md` documents the FP8 quantization format (E4M3, 128×128 blocks) and weight file structure — this information should be cross-referenced in the new `CONFIG_REFERENCE.md` and inline kernel documentation

### 0.2.3 Web Search Research Conducted

**Technical Report Context (arXiv:2412.19437):**

Research confirmed the DeepSeek-V3 technical report structure relevant to documentation cross-references:

- **Section 2.1 / 2.1.1 — Multi-Head Latent Attention (MLA):** Documents the latent compression scheme for KV cache reduction, where keys and values are jointly compressed into a lower-dimensional latent space. MLA was "thoroughly validated in DeepSeek-V2" and reduces KV cache memory from O(n·d_head) to O(n·d_latent). The `MLA` class in `model.py` directly implements this mechanism.
- **Section 2.1.2 — DeepSeekMoE with Auxiliary-Loss-Free Load Balancing:** Documents the bias-based load balancing strategy that avoids adding auxiliary loss terms to the training objective. The `Gate` class implements this via `self.bias` parameter.
- **Section 2.2 — Multi-Token Prediction (MTP):** Documents the MTP training objective architecture. The `convert.py` skip of layer 61 relates to MTP module exclusion during inference.
- **Section 3 — FP8 Training:** Documents block-wise FP8 quantization with 128×128 tile granularity, directly implemented in `kernel.py`.

**Documentation Best Practices Confirmed:**

- Google-style docstrings are the standard for ML research codebases with PyTorch, consistent with the user's specification
- Tensor shape documentation using notation like `(batch_size, seq_len, hidden_dim)` is the established convention for ML codebases
- Inline rationale comments for architectural decisions are recognized as high-value documentation for research code

## 0.3 Documentation Scope Analysis

### 0.3.1 Code-to-Documentation Mapping

**Module: `inference/model.py` (Highest-Density Documentation Target)**

- Public APIs requiring documentation enrichment:
  - `ModelArgs` dataclass — 25 attributes, all requiring type/range/purpose documentation with cross-references to config JSON fields
  - `ParallelEmbedding` — `__init__()`, `forward()` — requires distributed sharding documentation (vocabulary partitioned by `world_size`, `all_reduce` aggregation)
  - `linear()` standalone function — FP8/BF16 dispatch logic requires rationale for three-path branching
  - `Linear` — class-level `dtype` and `scale_fmt` class attributes, scale tensor allocation for FP8 block quantization
  - `ColumnParallelLinear` — output feature partitioning across ranks
  - `RowParallelLinear` — input feature partitioning with `all_reduce` synchronization
  - `RMSNorm` — normalization choice rationale (RMSNorm vs. LayerNorm)
  - `precompute_freqs_cis()` + 3 nested helpers (`find_correction_dim`, `find_correction_range`, `linear_ramp_factor`) — YaRN RoPE extension rationale, frequency correction mathematics
  - `apply_rotary_emb()` — complex-number RoPE application, tensor shape transformations
  - `MLA` — latent compression scheme (Q LoRA decomposition, KV joint compression, decoupled RoPE), two attention implementations (`naive` vs. `absorb`), KV cache strategies, softmax scale adjustment for extended sequences
  - `MLP` — SwiGLU activation pattern (w1/w3 gated), parallel linear decomposition
  - `Gate` — scoring function dispatch (`softmax` vs. `sigmoid`), bias-based auxiliary-loss-free load balancing, group routing with top-k selection
  - `Expert` — individual expert MLP structure
  - `MoE` — expert partitioning across ranks, gated routing, shared experts mechanism, `all_reduce` for cross-rank aggregation
  - `Block` — dense-layer vs. MoE-layer selection based on `layer_id < n_dense_layers`
  - `Transformer` — global state initialization (`world_size`, `rank`, `Linear.dtype`), model construction, `all_gather` for logits synchronization
- Current documentation: Present but incomplete — missing tensor shapes, distributed semantics, "why" rationale
- Documentation needed: Enhanced docstrings with tensor shapes, distributed communication documentation, inline "why" comments for ~50+ non-trivial decisions

**Module: `inference/generate.py`**

- Public APIs requiring documentation enrichment:
  - `sample()` — Gumbel-softmax sampling via exponential distribution trick, temperature scaling
  - `generate()` — autoregressive token generation loop with KV-cache-aware prefill/decode split, prompt masking, EOS termination
  - `main()` — distributed process initialization (`dist.init_process_group("nccl")`), model loading, interactive vs. batch mode dispatch, prompt broadcasting via `dist.broadcast_object_list`
- Current documentation: Basic docstrings present, missing tensor shapes and distributed semantics
- Documentation needed: Enhanced docstrings, inline rationale for sampling strategy choice, distributed coordination documentation

**Module: `inference/convert.py`**

- Public APIs requiring documentation enrichment:
  - `mapping` dictionary — 17-entry HuggingFace-to-internal name mapping with shard dimension indicators
  - `main()` — checkpoint conversion logic: expert partitioning by `n_local_experts`, weight sharding by dimension, layer-61 MTP skip, tokenizer artifact copying
- Current documentation: Basic docstring present, no mapping rationale, no expert sharding explanation
- Documentation needed: Module-level docstring, mapping dictionary documentation, expert partitioning rationale, MTP layer skip explanation

**Module: `inference/fp8_cast_bf16.py`**

- Public APIs requiring documentation enrichment:
  - `main()` — FP8-to-BF16 conversion pipeline: safetensors iteration, block-wise dequantization via `weight_dequant()`, scale_inv tensor handling, model index rewriting
  - `get_tensor()` nested function — lazy file loading with caching
- Current documentation: Decent docstring present, missing dequantization block-size semantics and memory management rationale
- Documentation needed: Module-level docstring, block-wise dequantization documentation, 2-shard cache rationale, index.json rewrite documentation

**Module: `inference/kernel.py`**

- Public APIs requiring documentation enrichment:
  - `act_quant_kernel` (Triton) — per-block activation quantization, `amax` computation, scale factor derivation (448 = max FP8 E4M3 representable value), optional `ue8m0` scale format
  - `act_quant()` (Python wrapper) — block-size assertion, output tensor allocation
  - `weight_dequant_kernel` (Triton) — 2D block-tiled weight dequantization, scale factor lookup
  - `weight_dequant()` (Python wrapper) — contiguity/dimension assertions, grid launch
  - `fp8_gemm_kernel` (Triton) — blocked FP8 GEMM with per-block scale accumulation, autotuned configuration
  - `fp8_gemm()` (Python wrapper) — matrix shape extraction, output allocation
  - `fp8_gemm_configs` — autotuning configuration space (36 configs: 3 block_m × 3 block_n × 4 num_stages)
- Current documentation: Basic docstrings present, missing FP8 format specifics, autotuning rationale, block-size selection justification
- Documentation needed: Module-level docstring, E4M3 format documentation, autotuning space rationale, block-size 128 justification

**Configuration Files Requiring Companion Documentation:**

| Config File | Parameters | Documentation Status |
|-------------|-----------|---------------------|
| `inference/configs/config_671B.json` | 25 parameters | Undocumented — requires full parameter reference |
| `inference/configs/config_236B.json` | 25 parameters | Undocumented — requires variant documentation |
| `inference/configs/config_16B.json` | 25 parameters | Undocumented — requires variant documentation |
| `inference/configs/config_v3.1.json` | 26 parameters (includes `scale_fmt`) | Undocumented — requires v3.1 differences documentation |

### 0.3.2 Documentation Gap Analysis

Given the requirements and repository analysis, documentation gaps include:

**Undocumented Implementation Decisions (Critical Gaps):**

- MLA `absorb` mode computational trick: fusing KV projection into query space to avoid explicit key materialization — no rationale in current code
- `Gate` bias initialization gated by `self.dim == 7168` — hardcoded dimension check with no explanation
- `precompute_freqs_cis()` YaRN correction mathematics — beta_fast/beta_slow frequency ramp with no cross-reference to YaRN paper or DeepSeek-V3 report
- `Transformer.__init__()` sets global mutable state (`world_size`, `rank`, `Linear.dtype`) — non-standard initialization pattern with no rationale
- `sample()` uses Gumbel-softmax via `exponential_(1)` trick instead of standard `torch.multinomial` — no rationale
- `convert.py` mapping dictionary entries lack documentation of which dimension (0 or 1) corresponds to column vs. row parallelism
- `fp8_gemm_configs` autotuning space: BLOCK_SIZE_K fixed at 128, BLOCK_SIZE_M in {16, 32, 64}, BLOCK_SIZE_N in {32, 64, 128} — selection criteria undocumented

**Missing Tensor Shape Documentation:**

- `MLA.forward()` — input `(batch_size, seq_len, dim)`, intermediate projections through latent space `(batch_size, seq_len, kv_lora_rank)`, output `(batch_size, seq_len, dim)`
- `MoE.forward()` — input reshaping from `(batch_size, seq_len, dim)` to `(batch_size * seq_len, dim)` for expert dispatch
- `Transformer.forward()` — logits output shape differences in single-GPU vs. multi-GPU (`(batch_size, part_vocab_size)` vs. `(batch_size, vocab_size)` after `all_gather`)

**Missing Distributed Communication Documentation:**

| Operation | Location | Documentation Status |
|-----------|----------|---------------------|
| `dist.all_reduce(y)` | `ParallelEmbedding.forward()` line 127 | Missing: aggregates partial embeddings across ranks |
| `dist.all_reduce(y)` | `RowParallelLinear.forward()` line 264 | Missing: aggregates partial matrix products across ranks |
| `dist.all_reduce(y)` | `MoE.forward()` line 692 | Missing: aggregates expert outputs across ranks |
| `dist.all_gather(all_logits, logits)` | `Transformer.forward()` line 796 | Missing: gathers vocabulary-partitioned logits |
| `dist.broadcast_object_list(objects, 0)` | `main()` in `generate.py` lines 129, 132 | Missing: broadcasts user prompt from rank 0 to all ranks |
| `dist.init_process_group("nccl")` | `main()` in `generate.py` line 104 | Missing: NCCL backend selection rationale |

**Missing Cross-References to Technical Report:**

- MLA latent compression → arXiv:2412.19437 Section 2.1.1
- Auxiliary-loss-free load balancing → arXiv:2412.19437 Section 2.1.2
- Multi-Token Prediction (layer 61 skip) → arXiv:2412.19437 Section 2.2
- FP8 block-wise quantization → arXiv:2412.19437 Section 3.3
- Sigmoid scoring function for MoE routing → arXiv:2412.19437 Section 2.1.2

**Missing Companion Documentation:**

- `inference/ARCHITECTURE.md` — does not exist; needed for pipeline overview, module dependency graph, architectural glossary
- `inference/configs/CONFIG_REFERENCE.md` — does not exist; needed for parameter reference across all 4 config variants

## 0.4 Documentation Implementation Design

### 0.4.1 Documentation Structure Planning

The documentation output follows two parallel structures: (a) in-code documentation within existing Python files, and (b) companion markdown files for cross-module and configuration documentation.

**Documentation Hierarchy:**

```
inference/
├── model.py              (UPDATE: module docstring + enriched class/method docstrings + inline "why" comments)
├── generate.py           (UPDATE: module docstring + enriched function docstrings + inline "why" comments)
├── convert.py            (UPDATE: module docstring + enriched function docstrings + inline "why" comments)
├── fp8_cast_bf16.py      (UPDATE: module docstring + enriched function docstrings + inline "why" comments)
├── kernel.py             (UPDATE: module docstring + enriched kernel/wrapper docstrings + inline "why" comments)
├── ARCHITECTURE.md       (CREATE: pipeline overview, module dependency graph, glossary)
├── requirements.txt      (NO CHANGE)
└── configs/
    ├── config_671B.json   (NO CHANGE — documented via companion file)
    ├── config_236B.json   (NO CHANGE — documented via companion file)
    ├── config_16B.json    (NO CHANGE — documented via companion file)
    ├── config_v3.1.json   (NO CHANGE — documented via companion file)
    └── CONFIG_REFERENCE.md (CREATE: parameter reference for all config variants)

README.md                  (UPDATE: add "Code Documentation" section)
README_WEIGHTS.md          (ASSESS: note gaps if found; no mandatory changes)
```

### 0.4.2 Content Generation Strategy

**Information Extraction Approach:**

- Extract API signatures, class hierarchies, and function parameters directly from source code analysis of all 5 Python files under `inference/`
- Generate tensor shape annotations by tracing data flow through `model.py` forward methods: input shapes from `ModelArgs` dimensions, intermediate shapes from projection operations, output shapes from parallel gather operations
- Derive "why" rationale from three sources: (1) the DeepSeek-V3 technical report (arXiv:2412.19437), (2) code-structural analysis of alternative approaches, and (3) domain knowledge of MoE/MLA/FP8 systems
- Create configuration documentation by cross-referencing `ModelArgs` dataclass attributes with JSON config file values and technical report architecture descriptions

**Documentation Standards:**

- Markdown formatting with proper headers (`#`, `##`, `###`) for companion markdown files
- Mermaid diagram integration using fenced code blocks for `ARCHITECTURE.md` pipeline visualization
- Code examples using fenced blocks with `python` syntax highlighting for usage patterns
- Source citations as inline references: `Source: inference/model.py:LineNumber`
- Tables for parameter descriptions, config values, and distributed operation summaries
- Consistent terminology aligned with the DeepSeek-V3 technical report nomenclature

**Module-Level Docstring Template (Applied to Each `.py` File):**

```python
"""Module purpose and architectural role.
Key classes/functions contained.
Relationship to other modules.
Reference: arXiv:2412.19437, Section X.Y
"""
```

**Class Docstring Enhancement Pattern:**

```python
"""Purpose in DeepSeek-V3 architecture.
Architectural context and report reference.
Attributes: name (type): description, shape
"""
```

**Inline "Why" Comment Pattern:**

```python
# [Alternatives Considered] Using X instead of Y because...

#### [Trade-offs] Accepting A in exchange for B because...

```

### 0.4.3 Diagram and Visual Strategy

**Mermaid Diagrams for `ARCHITECTURE.md`:**

- **Pipeline Flow Diagram:** Weight loading → Model initialization → Token generation → Output decoding, showing data flow between `generate.py`, `model.py`, and `kernel.py`
- **Module Dependency Graph:** Directed graph showing import relationships: `generate.py` → `model.py` → `kernel.py`, `fp8_cast_bf16.py` → `kernel.py`
- **Transformer Block Diagram:** RMSNorm → MLA → Residual → RMSNorm → MoE/MLP → Residual, showing the architectural pattern within each `Block`
- **MLA Attention Flow:** Query LoRA decomposition → latent KV compression → decoupled RoPE → attention computation → output projection
- **MoE Routing Flow:** Gate scoring → group selection → top-k expert selection → expert computation → shared expert addition → all_reduce

**Text-Based Diagrams for Inline Documentation:**

- Tensor shape transformation chains documented as inline comments within complex methods (e.g., `MLA.forward()`)
- Configuration parameter relationship diagrams in `CONFIG_REFERENCE.md` using markdown tables rather than visual diagrams

**No External Image Generation Required:** All diagrams use Mermaid syntax within markdown files. The `figures/` directory is out of scope and will not be modified.

## 0.5 Documentation File Transformation Mapping

### 0.5.1 File-by-File Documentation Plan

**Documentation Transformation Modes:**
- **CREATE** — Create a new documentation file
- **UPDATE** — Update an existing file with documentation-only changes (docstrings, comments)
- **ASSESS** — Evaluate for documentation gaps without mandatory changes
- **REFERENCE** — Use as context for documentation content generation

| Target Documentation File | Transformation | Source Code/Docs | Content/Changes |
|---------------------------|----------------|------------------|-----------------|
| `inference/model.py` | UPDATE | `inference/model.py` | Add module-level docstring; enrich all 10 class docstrings with tensor shapes, distributed semantics, and report cross-references; enrich all method/function docstrings with tensor shape annotations; add ~50+ inline "why" rationale comments for MLA, MoE, RoPE, parallel linear, and global state decisions |
| `inference/generate.py` | UPDATE | `inference/generate.py` | Add module-level docstring; enrich `sample()` docstring with Gumbel-softmax rationale; enrich `generate()` docstring with KV-cache-aware loop semantics and tensor shapes; enrich `main()` docstring with distributed initialization and prompt broadcast documentation; add inline "why" comments for sampling strategy, seed selection, thread count |
| `inference/convert.py` | UPDATE | `inference/convert.py` | Add module-level docstring; document `mapping` dictionary with per-entry rationale for dimension assignment; enrich `main()` docstring with expert partitioning logic, layer-61 MTP skip rationale, tokenizer copying purpose; add inline "why" comments for sharding strategy and naming conventions |
| `inference/fp8_cast_bf16.py` | UPDATE | `inference/fp8_cast_bf16.py` | Add module-level docstring; enrich `main()` docstring with block-wise dequantization pipeline documentation; document `get_tensor()` caching strategy; add inline "why" comments for 2-shard cache limit, index.json rewrite logic, scale_inv handling |
| `inference/kernel.py` | UPDATE | `inference/kernel.py` | Add module-level docstring; enrich all 3 Triton kernel docstrings with FP8 E4M3 format specifics, block-level quantization semantics, scale factor derivation (448 = max E4M3); enrich all 3 wrapper docstrings with tensor shape annotations; document `fp8_gemm_configs` autotuning space rationale; add inline "why" comments for block size selection, scale format support, clamp threshold |
| `inference/ARCHITECTURE.md` | CREATE | `inference/model.py`, `inference/generate.py`, `inference/kernel.py`, `inference/convert.py`, `inference/fp8_cast_bf16.py` | High-level pipeline overview with Mermaid diagrams; module dependency graph; key architectural decisions and rationale; cross-references to technical report Sections 2.1, 2.1.2, 2.2, 3.3; glossary of DeepSeek-V3 terms (MLA, DeepSeekMoE, MTP, auxiliary-loss-free balancing, FP8 E4M3) |
| `inference/configs/CONFIG_REFERENCE.md` | CREATE | `inference/configs/config_671B.json`, `inference/configs/config_236B.json`, `inference/configs/config_16B.json`, `inference/configs/config_v3.1.json`, `inference/model.py` (`ModelArgs`) | Every JSON parameter documented with type, valid range, purpose, and relationship to architecture dimensions; comparison table across all 4 config variants; default value rationale; mapping between config keys and `ModelArgs` attributes |
| `README.md` | UPDATE | `README.md` | Add a "Code Documentation" section linking to `inference/ARCHITECTURE.md`, `inference/configs/CONFIG_REFERENCE.md`, and describing the per-module Developer's Log documentation approach |
| `README_WEIGHTS.md` | ASSESS | `README_WEIGHTS.md` | Evaluate for gaps between code behavior and documented weight structure; note any discrepancies in a documentation gap report; no mandatory changes unless gaps are found |

### 0.5.2 New Documentation Files Detail

**File: `inference/ARCHITECTURE.md`**

```
File: inference/ARCHITECTURE.md
Type: Architectural Overview / Developer's Log Introduction
Source Code: inference/model.py, inference/generate.py, inference/kernel.py, 
             inference/convert.py, inference/fp8_cast_bf16.py
Sections:
    - Overview (DeepSeek-V3 inference pipeline purpose and scope)
    - Pipeline Stages (weight loading → model init → token generation → output decode)
    - Module Dependency Graph (text-based + Mermaid diagram)
    - Key Architectural Decisions:
        - MLA over standard MHA/GQA (Section 2.1.1 of report)
        - DeepSeekMoE with auxiliary-loss-free balancing (Section 2.1.2 of report)
        - FP8 block-wise quantization with 128×128 tiles (Section 3.3 of report)
        - Two attention implementations (naive vs. absorb) and when to use each
        - Multi-Token Prediction module exclusion during inference (Section 2.2 of report)
    - Distributed Execution Model (tensor parallelism, expert parallelism)
    - Weight Conversion Paths (HuggingFace→MP, FP8→BF16)
    - Glossary (MLA, DeepSeekMoE, MTP, RoPE, YaRN, FP8 E4M3, auxiliary-loss-free)
Diagrams:
    - Mermaid flowchart: inference pipeline stages
    - Mermaid graph: module import dependencies
    - Mermaid flowchart: Transformer Block architecture (MLA → MoE/MLP)
Key Citations: arXiv:2412.19437 Sections 2.1, 2.1.2, 2.2, 3.3
```

**File: `inference/configs/CONFIG_REFERENCE.md`**

```
File: inference/configs/CONFIG_REFERENCE.md
Type: Configuration Parameter Reference
Source Code: inference/configs/config_671B.json, config_236B.json, config_16B.json, 
             config_v3.1.json, inference/model.py (ModelArgs dataclass)
Sections:
    - Overview (configuration system design, JSON-to-ModelArgs mapping)
    - Parameter Reference Table (all 26 parameters):
        - max_batch_size, max_seq_len, dtype, scale_fmt
        - vocab_size, dim, inter_dim, moe_inter_dim
        - n_layers, n_dense_layers, n_heads
        - n_routed_experts, n_shared_experts, n_activated_experts
        - n_expert_groups, n_limited_groups, score_func, route_scale
        - q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim
        - original_seq_len, rope_theta, rope_factor, beta_fast, beta_slow, mscale
    - Model Variant Comparison Table (16B vs 236B vs 671B vs v3.1)
    - Parameter Relationships (dim → n_heads × head_dim, etc.)
    - Default Values and Rationale
Key Citations: inference/model.py lines 20-86 (ModelArgs), arXiv:2412.19437 Table 1
```

### 0.5.3 Documentation Files to Update Detail

**`inference/model.py` — Comprehensive Documentation Enhancement**

- New content: Module-level docstring (purpose, architectural role, key classes, report references)
- Enhanced `ModelArgs` docstring: Add parameter grouping (general, MoE, MLA, YaRN), valid ranges, cross-references to config JSON
- Enhanced `MLA` class/method docstrings: Add tensor shapes at every projection boundary, document `naive` vs. `absorb` attention strategies with KV cache semantics, cross-reference Section 2.1.1
- Enhanced `Gate` class docstrings: Document auxiliary-loss-free load balancing mechanism, explain `self.dim == 7168` bias conditional, cross-reference Section 2.1.2
- Enhanced `MoE` class docstrings: Document expert partitioning, shared expert mechanism, `all_reduce` semantics
- Enhanced `Transformer` class docstrings: Document global state initialization pattern, `all_gather` logits synchronization
- Inline "why" comments: ~50+ comments covering MLA design choices, MoE routing alternatives, RoPE correction mathematics, parallel linear patterns, FP8 dispatch logic

**`inference/generate.py` — Generation Pipeline Documentation**

- New content: Module-level docstring (inference driver role, relationship to model.py and tokenizer)
- Enhanced `sample()`: Document Gumbel-softmax via exponential distribution trick rationale
- Enhanced `generate()`: Document KV-cache-aware autoregressive loop, prompt masking strategy, batch handling
- Enhanced `main()`: Document distributed initialization, interactive/batch mode, prompt broadcasting
- Inline "why" comments: Sampling strategy choice, temperature clamping, manual seed (965), thread count (8)

**`inference/convert.py` — Checkpoint Conversion Documentation**

- New content: Module-level docstring (HuggingFace-to-MP conversion pipeline)
- Enhanced `mapping` dict: Inline comments for each of 17 entries explaining dimension assignment (0=column, 1=row, None=replicated)
- Enhanced `main()`: Document expert partitioning by `n_local_experts`, layer-61 skip for MTP, tokenizer artifact copying
- Inline "why" comments: Sharding strategy, naming convention choices, MTP exclusion rationale

**`inference/fp8_cast_bf16.py` — FP8 Conversion Documentation**

- New content: Module-level docstring (FP8-to-BF16 dequantization utility)
- Enhanced `main()`: Document pipeline stages, scale_inv handling, model index rewriting
- Enhanced `get_tensor()`: Document lazy loading caching strategy
- Inline "why" comments: 2-shard cache limit memory management, CUDA empty_cache timing, index.json metadata reset

**`inference/kernel.py` — Triton Kernel Documentation**

- New content: Module-level docstring (FP8 quantization/dequantization kernel library, relationship to model.py and fp8_cast_bf16.py)
- Enhanced kernel docstrings: FP8 E4M3 format specifics (4-bit exponent, 3-bit mantissa, max value 448), block-level scale factor derivation, `ue8m0` scale format support
- Enhanced `fp8_gemm_configs`: Document autotuning space rationale (BLOCK_SIZE_K=128 matches quantization block, M/N varied for workload diversity)
- Inline "why" comments: Scale clamp to 1e-4, E4M3 max value 448, block size alignment with quantization granularity

**`README.md` — Documentation Cross-Reference Addition**

- Add "Code Documentation" section after existing content
- Link to `inference/ARCHITECTURE.md` for architectural overview
- Link to `inference/configs/CONFIG_REFERENCE.md` for configuration reference
- Describe the Developer's Log documentation approach

### 0.5.4 Documentation Configuration Updates

No documentation framework configuration files exist in the repository and none need to be created. The documentation output consists exclusively of:

- In-code Python docstrings and comments (no build system required)
- Standalone markdown files (`ARCHITECTURE.md`, `CONFIG_REFERENCE.md`) rendered natively by GitHub

No updates needed for: `mkdocs.yml` (does not exist), `docusaurus.config.js` (does not exist), `.readthedocs.yml` (does not exist), `sphinx/conf.py` (does not exist).

### 0.5.5 Cross-Documentation Dependencies

**Shared Terminology:**

- All documentation files must use consistent terminology aligned with the DeepSeek-V3 technical report and the `ARCHITECTURE.md` glossary
- Key terms: MLA (Multi-head Latent Attention), DeepSeekMoE, MTP (Multi-Token Prediction), auxiliary-loss-free, FP8 E4M3, RoPE, YaRN, latent compression, absorbed attention

**Navigation Links Between Documents:**

- `README.md` → `inference/ARCHITECTURE.md` (overview)
- `README.md` → `inference/configs/CONFIG_REFERENCE.md` (configuration)
- `ARCHITECTURE.md` → `README_WEIGHTS.md` (weight structure details)
- `ARCHITECTURE.md` → individual module references (e.g., "See `model.py` MLA class for implementation details")
- `CONFIG_REFERENCE.md` → `model.py` `ModelArgs` dataclass (source of truth for parameter definitions)
- `CONFIG_REFERENCE.md` → `README_WEIGHTS.md` (FP8 format details)

**Inline Cross-References Within Python Files:**

- `model.py` docstrings reference arXiv:2412.19437 sections by number
- `model.py` docstrings reference `kernel.py` functions when FP8 operations are invoked
- `generate.py` docstrings reference `model.py` Transformer class and ModelArgs
- `convert.py` docstrings reference `model.py` weight naming conventions
- `fp8_cast_bf16.py` docstrings reference `kernel.py` `weight_dequant()` function

## 0.6 Dependency Inventory

### 0.6.1 Documentation Dependencies

All key documentation tools and packages relevant to this documentation exercise:

| Registry | Package Name | Version | Purpose |
|----------|--------------|---------|---------|
| pip | torch | 2.4.1 | Core ML framework — documented functions use PyTorch tensor operations, distributed primitives, and FP8 dtype support (`torch.float8_e4m3fn`) |
| pip | triton | 3.0.0 | GPU kernel compilation — `kernel.py` Triton kernels require Triton JIT documentation with `@triton.jit` and `@triton.autotune` semantics |
| pip | transformers | 4.46.3 | Tokenizer integration — `generate.py` uses `AutoTokenizer` for chat template application and token encoding/decoding |
| pip | safetensors | 0.4.5 | Weight serialization — `convert.py` uses `safe_open`/`save_file`, `fp8_cast_bf16.py` uses `load_file`/`save_file`, `generate.py` uses `load_model` |
| pip | tqdm | (transitive) | Progress bars — used in `convert.py` and `fp8_cast_bf16.py` for checkpoint processing progress display |

**Version Source:** All versions are exact-pinned in `inference/requirements.txt` (lines 1-4):
- `torch==2.4.1`
- `triton==3.0.0`
- `transformers==4.46.3`
- `safetensors==0.4.5`

**Documentation-Only Tools (No Additional Installation Required):**

This documentation task does not require any documentation generation tools. All output consists of:
- Python docstrings and inline comments (no build system needed)
- Standalone markdown files (rendered natively by GitHub)
- Mermaid diagrams embedded in markdown (rendered natively by GitHub)

No additional packages such as `mkdocs`, `sphinx`, `pdoc`, `typedoc`, or `@mermaid-js/mermaid-cli` are required.

### 0.6.2 Documentation Reference Updates

**Internal Documentation Links to Establish:**

| Source File | Link Text | Target |
|-------------|-----------|--------|
| `README.md` | "Architecture Overview" | `inference/ARCHITECTURE.md` |
| `README.md` | "Configuration Reference" | `inference/configs/CONFIG_REFERENCE.md` |
| `inference/ARCHITECTURE.md` | "Weight Structure" | `README_WEIGHTS.md` |
| `inference/ARCHITECTURE.md` | Per-module references | `inference/model.py`, `inference/generate.py`, etc. |
| `inference/configs/CONFIG_REFERENCE.md` | "ModelArgs source" | `inference/model.py` line 20 |
| `inference/configs/CONFIG_REFERENCE.md` | "FP8 format details" | `README_WEIGHTS.md` |

**Link Transformation Rules:**

- All links use relative paths from the source file's directory
- From `README.md` (root): `[Architecture](inference/ARCHITECTURE.md)`
- From `inference/ARCHITECTURE.md`: `[Weight Structure](../README_WEIGHTS.md)`
- From `inference/configs/CONFIG_REFERENCE.md`: `[Model Source](../model.py)`

**External Documentation Links:**

| Reference | URL | Usage |
|-----------|-----|-------|
| DeepSeek-V3 Technical Report | `https://arxiv.org/abs/2412.19437` | Cross-referenced in all module docstrings and ARCHITECTURE.md |
| DeepSeek-V3 GitHub Repository | `https://github.com/deepseek-ai/DeepSeek-V3` | Referenced in ARCHITECTURE.md overview |
| HuggingFace Model Card | `https://huggingface.co/deepseek-ai/DeepSeek-V3` | Referenced in convert.py documentation for source checkpoint format |

## 0.7 Coverage and Quality Targets

### 0.7.1 Documentation Coverage Metrics

**Current Coverage Analysis:**

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Module-level docstrings | 0/5 (0%) | 5/5 (100%) | All 5 Python files need module-level docstrings |
| Class docstrings present | 10/10 (100%) | 10/10 (100%) | Present but incomplete |
| Class docstrings with tensor shapes | 0/10 (0%) | 10/10 (100%) | No tensor shape annotations exist |
| Function/method docstrings present | ~25/25 (100%) | 25/25 (100%) | Present but incomplete |
| Function docstrings with tensor shapes | 0/25 (0%) | 25/25 (100%) | No tensor shape annotations at boundaries |
| Distributed communication documented | 0/6 (0%) | 6/6 (100%) | None of the 6 distributed ops are documented |
| Inline "why" rationale comments | 0/~50 (0%) | ≥45/~50 (≥90%) | No rationale comments exist |
| Technical report cross-references | 0/~15 (0%) | 15/15 (100%) | No arXiv references exist in code |
| Companion markdown files | 0/2 (0%) | 2/2 (100%) | Neither ARCHITECTURE.md nor CONFIG_REFERENCE.md exists |
| README documentation links | 0/2 (0%) | 2/2 (100%) | No links to code-level documentation |

**Per-Module Coverage Targets:**

| Module | Docstring Target | "Why" Comment Target | Tensor Shape Target |
|--------|-----------------|---------------------|---------------------|
| `model.py` | 100% of 10 classes + 4 functions + all methods | ≥90% of ~35 non-trivial decisions | 100% of tensor-accepting functions |
| `generate.py` | 100% of 3 functions | ≥90% of ~8 non-trivial decisions | 100% of tensor-accepting functions |
| `convert.py` | 100% of 1 function + mapping dict | ≥90% of ~5 non-trivial decisions | N/A (no tensor operations) |
| `fp8_cast_bf16.py` | 100% of 2 functions | ≥90% of ~4 non-trivial decisions | 100% of tensor-accepting functions |
| `kernel.py` | 100% of 3 kernels + 3 wrappers | ≥90% of ~8 non-trivial decisions | 100% of tensor-accepting functions |

### 0.7.2 Documentation Quality Criteria

**Completeness Requirements (Dual-Criteria Review Gate):**

- **"What" Validation:** Every construct passes if it has: purpose statement, all parameters documented with types and descriptions, return value documented with type, tensor shapes specified for all tensor parameters and returns, exceptions documented where applicable
- **"Why" Validation:** Every non-trivial implementation decision passes if it has an adjacent inline comment documenting rationale using ≥1 mandatory "why" category

**Docstring Quality Checklist (Per Construct):**

- Purpose statement present and specific to DeepSeek-V3 context
- All `Args` documented with: name, type annotation, description
- Tensor parameters include shape notation: `(batch_size, seq_len, hidden_dim)`
- `Returns` documented with type and description (including shapes for tensors)
- `Raises` documented where exceptions are possible
- `Notes` include cross-references to arXiv:2412.19437 sections where applicable
- No boilerplate or template text that adds no information

**Inline Comment Quality Checklist (Per Decision):**

- Comment appears immediately adjacent to the code it references
- Comment explains "why" (developer reasoning), not "what" (code behavior)
- Comment uses ≥1 mandatory "why" category: Alternatives Considered, Refactoring Rationale, Assumptions Made, Trade-offs, Future-proofing
- Rationale is specific and verifiable (not vague like "for efficiency")
- Where the technical report provides justification, the report is cited instead of speculation

**Accuracy Validation:**

- All tensor shape annotations must match actual code dimensions (verified by tracing through `ModelArgs` default values and projection dimensions)
- All parameter types must match actual Python type annotations or runtime types
- All distributed communication documentation must accurately describe participating ranks and data movement
- All arXiv cross-references must point to correct report sections

**Clarity Standards:**

- Technical accuracy with accessible language for the target audience (ML engineers unfamiliar with DeepSeek-V3 specifics)
- Progressive disclosure: module-level overview → class-level context → method-level detail
- Consistent terminology throughout, aligned with `ARCHITECTURE.md` glossary

### 0.7.3 Example and Diagram Requirements

**Minimum Documentation per API Element:**

| Construct Type | Minimum Docstring Sections | "Why" Comments |
|---------------|---------------------------|----------------|
| Module | Purpose, key classes/functions, relationships, report reference | N/A |
| Class | Purpose, architectural context, all attributes with types/shapes | ≥1 per class for design choice rationale |
| Method (`forward`) | Purpose, all Args with types/shapes, Returns with shapes | ≥1 per non-trivial operation |
| Standalone function | Purpose, all Args with types, Returns with type | ≥1 per non-obvious choice |
| Triton kernel | Purpose, all Args with pointer types, block semantics | ≥1 per kernel for algorithm choice |

**Diagram Requirements for Companion Markdown Files:**

| Diagram | File | Type | Content |
|---------|------|------|---------|
| Inference Pipeline | `ARCHITECTURE.md` | Mermaid flowchart | `generate.py` → `model.py` → `kernel.py` data flow |
| Module Dependencies | `ARCHITECTURE.md` | Mermaid graph | Import relationships between all 5 modules |
| Transformer Block | `ARCHITECTURE.md` | Mermaid flowchart | RMSNorm → MLA → residual → RMSNorm → MoE/MLP → residual |
| MLA Attention Flow | `ARCHITECTURE.md` | Mermaid sequence/flowchart | Q-LoRA → KV compression → RoPE → attention → output |
| MoE Routing | `ARCHITECTURE.md` | Mermaid flowchart | Gate → group select → top-k → expert dispatch → all_reduce |
| Config Comparison | `CONFIG_REFERENCE.md` | Markdown table | Side-by-side 16B vs 236B vs 671B vs v3.1 parameters |

**Validation Method for Examples and Diagrams:**

- All Mermaid diagrams verified for correct syntax by ensuring proper node naming and connection formatting
- All configuration parameter values in `CONFIG_REFERENCE.md` cross-checked against actual JSON file contents
- All tensor shape annotations cross-checked against `ModelArgs` default values and projection dimension calculations

## 0.8 Scope Boundaries

### 0.8.1 Exhaustively In Scope

**Python Source Files (Documentation Enhancement via Docstrings and Comments):**

- `inference/model.py` — Module-level docstring, enriched class/method docstrings, inline "why" comments
- `inference/generate.py` — Module-level docstring, enriched function docstrings, inline "why" comments
- `inference/convert.py` — Module-level docstring, enriched function docstrings, mapping documentation, inline "why" comments
- `inference/fp8_cast_bf16.py` — Module-level docstring, enriched function docstrings, inline "why" comments
- `inference/kernel.py` — Module-level docstring, enriched kernel/wrapper docstrings, inline "why" comments

**New Companion Documentation Files:**

- `inference/ARCHITECTURE.md` — Architectural overview with Mermaid diagrams, module dependency graph, glossary
- `inference/configs/CONFIG_REFERENCE.md` — Configuration parameter reference for all 4 JSON config variants

**Root-Level Documentation Updates:**

- `README.md` — Add "Code Documentation" section with links to companion documentation files

**Root-Level Documentation Assessment:**

- `README_WEIGHTS.md` — Evaluate for gaps; document findings but no mandatory modifications unless gaps are identified

**Configuration Files (Read-Only, Documented via Companion File):**

- `inference/configs/config_671B.json` — Read for parameter values, documented in `CONFIG_REFERENCE.md`
- `inference/configs/config_236B.json` — Read for parameter values, documented in `CONFIG_REFERENCE.md`
- `inference/configs/config_16B.json` — Read for parameter values, documented in `CONFIG_REFERENCE.md`
- `inference/configs/config_v3.1.json` — Read for parameter values, documented in `CONFIG_REFERENCE.md`

**Documentation Content Types Produced:**

- Google-style Python docstrings (module, class, method, function level)
- Inline rationale comments using mandatory "why" categories
- `# NOTE:` comments for discovered code quality issues or bugs (documentation-only, not fixes)
- Markdown files with Mermaid diagrams, tables, and structured narrative
- Cross-reference links between files (relative paths, arXiv URLs)

### 0.8.2 Explicitly Out of Scope

**Source Code Modifications (MUST NOT):**

- Any executable code logic, control flow, or mathematical operations
- Function signatures, return types, or parameter defaults
- Import statements (no additions, removals, or reordering)
- Variable names, class hierarchies, or module structure
- Type hints added to function signatures (types documented in docstrings only)
- Bug fixes, refactoring, or optimization of any kind

**Files and Directories (MUST NOT Touch):**

- `.github/` directory — CI/CD configuration, workflows
- `figures/` directory — Images and visual assets
- `LICENSE-CODE` — MIT License file
- `LICENSE-MODEL` — Model License Agreement file
- External framework code (SGLang, vLLM, LMDeploy, TensorRT-LLM, LightLLM)
- `inference/requirements.txt` — Dependency specification (no changes)

**Configuration Files (MUST NOT Modify Structurally):**

- `inference/configs/config_671B.json` — No structural changes; documented via companion `.md` file
- `inference/configs/config_236B.json` — No structural changes
- `inference/configs/config_16B.json` — No structural changes
- `inference/configs/config_v3.1.json` — No structural changes

**Documentation Types NOT Produced:**

- Test documentation (no tests exist in repository)
- Deployment configuration documentation (deployment covered by existing README)
- API client documentation (no client library exists)
- User-facing GUI documentation (no GUI exists)
- Training documentation (repository is inference-only)
- Documentation for model variants not present in the repository configs
- Any documentation requiring new Python packages or build tools

## 0.9 Execution Parameters

### 0.9.1 Documentation-Specific Instructions

**Documentation Build Command:** Not applicable — no documentation build system exists or is required. All documentation is either embedded in Python source files (docstrings, comments) or standalone markdown files rendered natively by GitHub.

**Documentation Preview Command:** Standard markdown preview via any markdown renderer or GitHub's built-in rendering:
- Local preview: Any markdown editor or `python -m http.server` with a markdown renderer
- Remote preview: Push to GitHub; markdown files and Mermaid diagrams render automatically

**Diagram Generation Command:** Not applicable — all diagrams use Mermaid syntax embedded in markdown fenced code blocks (` ```mermaid ... ``` `), which GitHub renders natively without external tooling.

**Documentation Deployment Command:** Not applicable — documentation is committed directly to the repository alongside source code. No separate documentation hosting or deployment pipeline is required.

**Default Format:** Markdown with Mermaid diagrams for companion files; Google-style Python docstrings for in-code documentation.

**Citation Requirement:** Every module docstring must reference the DeepSeek-V3 technical report (arXiv:2412.19437) with specific section numbers. Every "why" rationale comment that draws from the report must cite the relevant section. Source code line references use the format `Source: inference/model.py:LineNumber`.

**Style Guide:** Documentation follows Google Python Style Guide for docstrings. Inline comments follow the user's mandatory "why" category system with bracket-prefixed labels: `[Alternatives Considered]`, `[Refactoring Rationale]`, `[Assumptions Made]`, `[Trade-offs]`, `[Future-proofing]`.

**Documentation Validation:**

- **Docstring Coverage Check:** Verify that every `class`, `def`, and module-level scope in all 5 Python files contains a docstring. Manual review of each file against the construct inventory (10 classes, ~13 standalone functions, ~15 methods, 3 Triton kernels = ~41 total constructs)
- **Tensor Shape Completeness:** Verify every function accepting or returning tensors documents shapes in the docstring. Cross-reference against `ModelArgs` default values to validate dimension accuracy
- **Distributed Communication Audit:** Verify all 6 identified distributed operations (`all_reduce` ×3, `all_gather` ×1, `broadcast_object_list` ×1, `init_process_group` ×1) are documented with participating ranks and data movement semantics
- **"Why" Comment Coverage:** Verify ≥90% of non-trivial implementation decisions have adjacent inline rationale comments. Non-trivial decisions are identified by the criterion: "a reasonable ML engineer could ask 'why this approach instead of [alternative]?'"
- **Link Validation:** Verify all cross-references between markdown files resolve correctly using relative paths. Verify all arXiv URLs are valid

**Execution Sequence:**

- Phase 1: Document `inference/model.py` (architectural core, establishes terminology)
- Phase 2: Document `inference/generate.py` (primary entry point, references model.py)
- Phase 3: Document `inference/kernel.py` (shared dependency of model.py and fp8_cast_bf16.py)
- Phase 4: Document `inference/convert.py` (utility module, self-contained)
- Phase 5: Document `inference/fp8_cast_bf16.py` (utility module, references kernel.py)
- Phase 6: Create `inference/ARCHITECTURE.md` (synthesized cross-module overview)
- Phase 7: Create `inference/configs/CONFIG_REFERENCE.md` (configuration parameter reference)
- Phase 8: Update `README.md` (add "Code Documentation" section)
- Phase 9: Assess `README_WEIGHTS.md` (note gaps if found)
- Phase 10: Run validation gate against all documented constructs

## 0.10 Rules for Documentation

The following rules are explicitly emphasized by the user and must be adhered to without exception throughout all documentation activities:

**Rule 1 — Zero Code Modification ("Minimal Change Clause"):**
Add only documentation-related content: docstrings, inline comments, module-level documentation strings, and companion markdown files. DO NOT modify any executable code logic, function signatures, imports, variable names, class hierarchies, or module structure. DO NOT add type hints to function signatures — document types in docstrings only to avoid modifying executable code.

**Rule 2 — Bug Documentation, Not Bug Fixing:**
If code quality issues or bugs are discovered during documentation, note them in a `# NOTE:` comment adjacent to the relevant code. DO NOT fix, refactor, or optimize. When documenting complex logic, explain the existing implementation — do not prescribe alternatives.

**Rule 3 — Google-Style Docstring Standard:**
All docstrings must follow Google Python Style Guide format with mandatory sections: Purpose, Args (name, type, description, tensor shapes), Returns (type, description, shapes), Raises (exceptions with conditions), Notes (cross-references, performance, distributed considerations).

**Rule 4 — "Why" Over "What" for Inline Comments:**
Inline comments MUST explain developer reasoning and decision-making. Comments MUST NOT describe what the code does — the docstring and code itself handle "what." Each significant implementation decision must document ≥1 rationale from the five mandatory "why" categories: Alternatives Considered, Refactoring Rationale, Assumptions Made, Trade-offs, Future-proofing.

**Rule 5 — Technical Report Citation Over Speculation:**
When the DeepSeek-V3 technical report (arXiv:2412.19437) provides explicit justification for an implementation decision, cite the report section instead of speculating about rationale. Use format: `Reference: arXiv:2412.19437, Section X.Y`.

**Rule 6 — Tensor Shape Documentation at Boundaries:**
Every function that accepts or returns tensors MUST document shapes at input and output boundaries using notation: `(batch_size, seq_len, hidden_dim)` or `(n_experts, expert_dim, intermediate_dim)`. Document shape transformations within functions where dimensionality changes.

**Rule 7 — Distributed Communication Documentation:**
Every `all_reduce`, `all_gather`, `reduce_scatter`, `send`, `recv`, `broadcast_object_list`, or equivalent distributed operation MUST document: which process group/ranks participate, what data is being communicated, why this communication pattern was chosen, and the expected tensor shapes before and after communication.

**Rule 8 — Forbidden Documentation Patterns:**
Never write comments that merely restate code behavior. Never add docstrings without documenting parameters, return values, or purpose. Never omit rationale for non-obvious implementation choices. Never use vague rationale without specific justification. Never insert boilerplate or template comments that add no information.

**Rule 9 — Companion File Constraints:**
Config JSON files MUST NOT be modified structurally — create companion `.md` documentation files instead. Only two new files are permitted: `inference/ARCHITECTURE.md` and `inference/configs/CONFIG_REFERENCE.md`.

**Rule 10 — Execution Sequence Discipline:**
Document `model.py` first (architectural core, establishes terminology), then `generate.py` (primary entry point), then utility modules (`convert.py`, `fp8_cast_bf16.py`, `kernel.py`), then create companion markdown files, then update README. This sequence ensures consistent terminology propagation.

**Rule 11 — Dual-Criteria Review Gate:**
Every function, class, and module MUST pass both "What" validation (complete Google-style docstring) and "Why" validation (inline rationale for non-trivial decisions) before the documentation task is considered complete. Target: 1.0 for docstrings, ≥0.9 for inline rationale.

**Rule 12 — Domain-Specific Documentation Requirements:**
MLA documentation must include latent compression scheme, KV cache comparison with standard MHA, and decoupled RoPE explanation. MoE documentation must include auxiliary-loss-free mechanism, gating function, and shared expert mechanism. MTP documentation must include module architecture and speculative decoding use case. FP8 documentation must include block-wise scheme, scale factor storage, and numerical stability considerations.

## 0.11 References

### 0.11.1 Repository Files and Folders Searched

The following files and folders were systematically searched and analyzed to derive the conclusions in this Agent Action Plan:

**Source Code Files (Full Content Retrieved):**

| File Path | Purpose | Lines |
|-----------|---------|-------|
| `inference/model.py` | Transformer architecture: ModelArgs, ParallelEmbedding, Linear variants, RMSNorm, MLA, MLP, Gate, Expert, MoE, Block, Transformer | ~800 |
| `inference/generate.py` | Token generation: sample(), generate(), main(), CLI argument parsing, distributed initialization | 186 |
| `inference/convert.py` | HuggingFace-to-MP checkpoint conversion: mapping dict, expert sharding, layer-61 skip | 97 |
| `inference/fp8_cast_bf16.py` | FP8-to-BF16 weight conversion: block-wise dequantization, scale_inv handling, index rewrite | 113 |
| `inference/kernel.py` | Triton kernels: act_quant_kernel, weight_dequant_kernel, fp8_gemm_kernel, Python wrappers | 197 |
| `inference/requirements.txt` | Dependency specifications: torch==2.4.1, triton==3.0.0, transformers==4.46.3, safetensors==0.4.5 | 4 |

**Configuration Files (Full Content Retrieved):**

| File Path | Model Variant | Key Parameters |
|-----------|--------------|----------------|
| `inference/configs/config_671B.json` | Full 671B model | 61 layers, 256 routed experts, 8 activated, sigmoid scoring, FP8 dtype |
| `inference/configs/config_236B.json` | 236B variant | 60 layers, 160 routed experts, 6 activated |
| `inference/configs/config_16B.json` | 16B variant | 27 layers, 64 routed experts, 6 activated |
| `inference/configs/config_v3.1.json` | v3.1 variant | Similar to 671B with scale_fmt: ue8m0 |

**Documentation Files (Full Content Retrieved):**

| File Path | Content Summary |
|-----------|----------------|
| `README.md` (lines 1-350) | Project introduction, model summary (671B params, 37B active), evaluation results, deployment instructions (SGLang, LMDeploy, TRT-LLM, vLLM), local inference guide via torchrun |
| `README_WEIGHTS.md` (full) | Weight file structure (Main Model 671B + MTP 14B), FP8 quantization config (e4m3 format, 128×128 block size, scale_inv tensor documentation) |

**Folder Structure Explored:**

| Folder Path | Children Found |
|-------------|---------------|
| Root (`/`) | `README.md`, `README_WEIGHTS.md`, `.github/`, `figures/`, `inference/`, `LICENSE-CODE`, `LICENSE-MODEL` |
| `inference/` | `model.py`, `generate.py`, `convert.py`, `fp8_cast_bf16.py`, `kernel.py`, `requirements.txt`, `configs/` |
| `inference/configs/` | `config_671B.json`, `config_236B.json`, `config_16B.json`, `config_v3.1.json` |

### 0.11.2 External References

| Reference | URL | Relevance |
|-----------|-----|-----------|
| DeepSeek-V3 Technical Report | https://arxiv.org/abs/2412.19437 | Primary architectural reference for all documentation cross-references; covers MLA (Section 2.1.1), DeepSeekMoE (Section 2.1.2), MTP (Section 2.2), FP8 training (Section 3.3) |
| DeepSeek-V3 GitHub Repository | https://github.com/deepseek-ai/DeepSeek-V3 | Source repository being documented |
| DeepSeek-V3 HuggingFace Model Card | https://huggingface.co/deepseek-ai/DeepSeek-V3 | Model card documenting 685B total size (671B main + 14B MTP), safetensors format, FP8/BF16/F32 tensor types |
| NVIDIA Megatron Bridge — DeepSeek V3 | https://docs.nvidia.com/nemo/megatron-bridge/0.2.0/models/llm/deepseek-v3.html | External reference documenting DeepSeekMoE architecture details: 256 routed experts, shared experts, sigmoid gating with expert bias, RoPE embeddings |

### 0.11.3 Tech Spec Sections Referenced

| Section | Content Leveraged |
|---------|-------------------|
| 1.1 Executive Summary | Project overview, core specifications (671B params, 37B active, 128K context, 2.788M GPU hours), stakeholder groups, business impact |
| 2.1 Feature Catalog | Feature inventory (F-001 through F-007), feature dependencies, technical context for each component |
| 9.4 File and Folder Reference | Consolidated file index, configuration files index, folder structure diagram |

### 0.11.4 Attachments

No attachments were provided for this project. All analysis was conducted directly against the repository contents and publicly available external references.