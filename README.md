> **FlagOS** is a fully open-source AI system software stack for heterogeneous AI chips, allowing AI models to be developed once and seamlessly ported to a wide range of AI hardware with minimal effort. This repository collects reusable **Skills** for FlagOS — injecting domain knowledge, workflow standards, and best practices into AI coding agents.
>
> [中文版](README_zh.md)

## What are Skills?

Skills are **folder-based capability packages**: each skill uses documentation, scripts, and resources to teach agents to reliably and reproducibly complete tasks in a specific domain. Each skill folder contains a `SKILL.md` file with YAML frontmatter (name + description) followed by detailed agent instructions. Skills can also include reference docs, scripts, and assets.

This repository follows the [Agent Skills open standard](https://agentskills.io/specification).

## Quick Start

FlagOS Skills are compatible with **Claude Code**, **Cursor**, **Codex**, and any agent supporting the [Agent Skills standard](https://agentskills.io/specification).

### npx (Recommended — works with all agents)

Use the [`skills`](https://www.npmjs.com/package/skills) CLI to install skills directly — no cloning needed:

```bash
# List available skills in this repository
npx skills add flagos-ai/skills --list

# Install a specific skill into your project
npx skills add flagos-ai/skills --skill model-migrate-flagos

# Install a specific skill globally (user-level)
npx skills add flagos-ai/skills --skill model-migrate-flagos --global

# Install all skills at once
npx skills add flagos-ai/skills --all

# Install for specific agents only
npx skills add flagos-ai/skills --agent claude-code cursor
```

Other useful commands:

```bash
npx skills list              # List installed skills
npx skills find              # Search for skills interactively
npx skills update            # Update all skills to latest versions
npx skills remove            # Interactive remove
```

> **Note:** No prior installation needed — `npx` downloads the [`skills`](https://skills.sh/) CLI automatically.

### Claude Code

1. Register the repository as a plugin marketplace (in Claude Code interactive mode):

```
/plugin marketplace add flagos-ai/skills
```

Or from the terminal:

```bash
claude plugin marketplace add flagos-ai/skills
```

2. Install skills:

```
/plugin install flagos-skills@flagos-skills
```

Or from the terminal:

```bash
claude plugin install flagos-skills@flagos-skills
```

After installation, mention the skill in your prompt — Claude automatically loads the corresponding `SKILL.md` instructions.

### Cursor

This repository includes Cursor plugin manifests (`.cursor-plugin/plugin.json` and `.cursor-plugin/marketplace.json`).

Install from the repository URL or local checkout via the Cursor plugin flow.

### Codex

Use the `$skill-installer` inside Codex:

```
$skill-installer install model-migrate-flagos from flagos-ai/skills
```

Or provide the GitHub directory URL:

```
$skill-installer install https://github.com/flagos-ai/skills/tree/main/skills/model-migrate-flagos
```

Alternatively, copy skill folders into Codex's standard `.agents/skills` location:

```bash
cp -r skills/model-migrate-flagos $REPO_ROOT/.agents/skills/
```

See the [Codex Skills guide](https://developers.openai.com/codex/skills/) for more details.

### Gemini CLI

```bash
gemini extensions install https://github.com/flagos-ai/skills.git --consent
```

This repo includes `gemini-extension.json` and `agents/AGENTS.md` for Gemini CLI integration. See [Gemini CLI extensions docs](https://geminicli.com/docs/extensions/) for more help.

### Manual / Other Agents

For any agent that supports the [Agent Skills standard](https://agentskills.io/specification), point it at the `skills/` directory in this repository. Each skill is self-contained with a `SKILL.md` entry point. The `agents/AGENTS.md` file can also be used as a fallback for agents that don't support skills natively.

## Skills Catalog

<!-- BEGIN_SKILLS_TABLE -->
| Category | Sub-category | Skill | Description |
|----------|-------------|-------|-------------|
| **Deployment & Release** | Release Pipeline | [PR #6 `flagrelease-entrance-flagos`](https://github.com/flagos-ai/skills/pull/6) | Orchestrate the end-to-end release pipeline: stack installation → environment verification → model verification → performance testing, producing a structured release report. |
| | Stack Installation | [PR #6 `install-stack-flagos`](https://github.com/flagos-ai/skills/pull/6) | Install the FlagOS software stack inside a GPU container in order (vLLM → FlagTree → FlagGems → FlagCX → vllm-plugin-FL) with mainland China mirror detection and per-package gate validation. |
| | Base Image Selection | [PR #5 `gpu-container-setup-flagos`](https://github.com/flagos-ai/skills/pull/5) | Auto-detect GPU vendor and find the optimal PyTorch base Docker image by priority (vendor hub → BAAI Harbor → web search → local images), with GPU availability verification. |
| | Model Migration | [`model-migrate-flagos`](skills/model-migrate-flagos/) | Migrate a model from upstream vLLM into vllm-plugin-FL. Automates the full 13-step copy-then-patch workflow with E2E verification. |
| **Benchmarking & Eval** | Deployment A/B Verification | [PR #6 `model-verify-flagos`](https://github.com/flagos-ai/skills/pull/6) | Dual-run comparison: Run A (native CUDA) vs Run B (FlagGems + FlagCX), diff analysis with per-component error attribution, outputting `recommended_stack` (full/base/none). |
| | Accuracy & Performance Test | [PR #6 `perf-test-flagos`](https://github.com/flagos-ai/skills/pull/6) | Two-part testing: Part A runs FlagEval accuracy benchmarks (currently placeholder), Part B runs `vllm bench serve` across 5 workload profiles collecting throughput, latency, TTFT, and TPOT metrics. |
| | Post-Deploy Auto Eval | *Planned* | Automatically trigger evaluation after model deployment, track evaluation status, report errors on failure, and push notifications with results upon completion. |
| | FlagPerf Case Creation | *Planned* | Generate FlagPerf-compliant directory structures, config files, run scripts, and expected metric baselines for new model/chip benchmark cases. |
| **Kernel & Operator Development** | TLE Lifecycle Dev | [PR #2 `tle-developer-flagos`](https://github.com/flagos-ai/skills/pull/2) | Full TLE (Triton Language Extensions) lifecycle: intake → kernel writing → API/verifier/lowering development → debugging → validation → artifact generation → merge decision. |
| | Kernel Generation | [`kernelgen-flagos`](skills/kernelgen-flagos/) | MCP-driven 9-step kernel generation workflow: requirement parsing → kernel writing → correctness verification → performance tuning → multi-chip adaptation. |
| | Kernel Gen for FlagGems | [`kernelgen-flagos`](skills/kernelgen-flagos/kernelgen-for-flaggems.md) | FlagGems-specific kernel generation with `@pointwise_dynamic` wrapper rewriting, `_FULL_CONFIG` registration, and operator signature alignment. |
| | Kernel Gen for vLLM | [`kernelgen-flagos`](skills/kernelgen-flagos/kernelgen-for-vllm.md) | vLLM-specific kernel generation with SPDX headers, `@triton.autotune`, custom op registration, and dispatch integration. |
| | Experimental Op Promotion | *Planned* | Scan FlagGems ~130 experimental ops, check test coverage, align signatures, complete `_FULL_CONFIG` registration, and generate migration PRs to promote them to main ops. |
| | Complex Operator Dev | *Planned* | Generate skeleton code for multi-step fused operators (fused attention, fused MoE, etc.), handling shared memory tiling strategies and multi-backend branching. |
| | Operator Diagnosis | *Planned* | Diagnose abnormal operators in the FlagOS stack — identify precision errors, performance regressions, and backend-specific failures across chips. |
| **Multi-Chip Backend Onboarding** | vLLM Vendor Backend | *Planned* | Scaffold a new vllm-plugin-FL vendor backend from the template: generate vendor directory, `Backend` subclass, `is_available` detection, `register_ops` framework, and test skeleton. |
| | Dispatch Op Extension | *Planned* | Query dispatchable ops from `base.py`, generate impl template files, add `OpImpl` registration to `register_ops.py`, and create unit test skeletons. |
| | FlagGems Chip Backend | *Planned* | Generate the full `_vendor/` scaffold: `__init__.py` (VendorInfoBase config) + `heuristics_config_utils.py` + `tune_configs.yaml` + `ops/` directory following the FlagGems backend contribution guide. |
| | FlagCX Comm Backend | *Planned* | Parse 20+ device and 15+ CCL function pointer signatures from header files, generate all stub implementations with trivial function fills, plus CMake build configuration. |
| | Heterogeneous Training Config | *Planned* | Generate valid FlagScale heterogeneous training configs from hardware topology descriptions, auto-compute `hetero_process_meshes` / `hetero_pipeline_layer_split`, and validate constraints (TP×DP×PP = device count). |
| **Developer Tooling** | Skill Development | [`skill-creator-flagos`](skills/skill-creator-flagos/) | Four modes (Create / Improve / Validate / Eval) for managing SKILL.md lifecycle — scaffolding, conventions check, and quality evaluation. |
| | Feedback Submission | [`kernelgen-flagos`](skills/kernelgen-flagos/kernelgen-submit-feedback.md) | Auto-collect environment info, construct structured GitHub issues, and submit to flagos-ai/skills with email fallback when GitHub CLI is unavailable. |
| | Local Dev Environment | *Planned* | Set up local development and debugging environments for FlagOS modules (FlagGems / FlagTree / FlagCX / etc.) — configure dependencies, environment variables, and debug toolchains. |
<!-- END_SKILLS_TABLE -->

### Using skills in your agent

Once a skill is installed, mention it directly in your prompt:

- "Use model-migrate-flagos to migrate the Qwen3-5 model from upstream vLLM"
- "/model-migrate-flagos qwen3_5"
- "Port the DeepSeek-V4 model to vllm-plugin-FL"

Your agent automatically loads the corresponding `SKILL.md` instructions and helper scripts.

## Repository Structure

```
├── .claude-plugin/          # Claude Code plugin manifest
│   └── marketplace.json
├── .cursor-plugin/          # Cursor plugin manifest
│   ├── marketplace.json
│   └── plugin.json
├── agents/                  # Codex / Gemini CLI fallback
│   └── AGENTS.md
├── assets/                  # Repository-level static resources
├── contributing.md          # Contribution guidelines
├── gemini-extension.json    # Gemini CLI extension manifest
├── scripts/                 # Repository-level utility scripts
│   └── validate_skills.py   # Batch validate all skills
├── skills/                  # Skill directories
│   ├── model-migrate-flagos/    # Model migration workflow
│   └── ...
├── spec/                    # Agent Skills standard & local conventions
│   ├── README.md
│   └── agent-skills-spec.md
└── template/                # Template for creating new skills
    └── SKILL.md
```

## Creating a New Skill

1. **Create directory & copy template**
   ```bash
   mkdir skills/<skill-name>
   cp template/SKILL.md skills/<skill-name>/SKILL.md
   ```

2. **Edit frontmatter** — `name` (lowercase + hyphens, must match directory name) and `description` (what it does + when to trigger)

3. **Write the body** — Overview, Prerequisites, Execution steps, Examples (2-3), Troubleshooting

4. **Add supporting files** (optional) — `references/`, `scripts/`, `assets/`, `LICENSE.txt`

5. **Validate**
   ```bash
   python scripts/validate_skills.py
   ```

See [contributing.md](contributing.md) for the full contribution guide.

## License

[Apache License 2.0](LICENSE)
