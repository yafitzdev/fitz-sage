---
generated_at: "2026-02-25T00:00:00.000000"
source: "claude-sonnet-4-6 with full project context"
note: "Reference plan — written with complete fitz-ai codebase knowledge"
---

# Project: Build an OpenAI YAML Chat Plugin for fitz-ai

## Context

**Description:**
fitz-ai has a declarative YAML plugin system for LLM providers. New providers can be added by dropping a `.yaml` file into `fitz_ai/llm/chat/` without writing Python code. An `openai.py` Python provider already exists in `fitz_ai/llm/providers/`. The task is to add the YAML declarative definition so the system discovers and loads it through the standard plugin pipeline, rather than relying on the Python fallback.

**What already exists:**
- `fitz_ai/llm/providers/openai.py` — Python provider using the official OpenAI SDK
- `fitz_ai/llm/config.py` — plugin loader that discovers YAML files and falls back to Python providers
- `fitz_ai/llm/chat/` — directory where YAML plugins live
- `fitz_ai/plugin_gen/validators.py` — schema validator for YAML plugins
- `fitz_ai/engines/fitz_krag/config/fitz_krag.yaml` — engine config that references plugins by name

**What does not yet exist:**
- `fitz_ai/llm/chat/openai.yaml` — the declarative plugin definition
- Tests specifically covering the YAML code path for OpenAI

**Requirements:**
- YAML plugin discovered and loaded by existing plugin loader in `fitz_ai/llm/config.py`
- Auth via `OPENAI_API_KEY` environment variable — consistent with every other plugin
- Three model tiers defined: `smart`, `fast`, `balanced`
- Response extraction paths must match OpenAI `/chat/completions` response schema exactly
- Must not break existing behavior for users already using the Python provider path

**Constraints:**
- No new Python code for the core provider logic — that is the entire point of the YAML system
- Must pass the existing schema validator in `fitz_ai/plugin_gen/validators.py`
- The Python provider `openai.py` must continue to exist — it handles streaming and SDK-level retry. The YAML plugin is for the standard non-streaming path.
- Azure OpenAI has a different base URL and auth header — treat as a separate plugin, not a config flag

**Stakeholders:**
- fitz-ai users configuring `fitz_krag.yaml` who need a working standard OpenAI integration
- Developers adding future providers who will use this as a reference YAML
- `plugin_gen/` validation pipeline which gates the plugin on schema correctness

---

## Architecture

### Explored Approaches

**1. YAML-only — replace Python provider as default for standard completions**
Add `openai.yaml`. The loader finds it and uses it for all non-streaming completions. Python provider remains available but is no longer the default.
- Pros: Aligns with system design intent. Reduces Python surface area. Single source of truth.
- Cons: Raw HTTP calls (YAML path) differ subtly from the `openai` SDK: no automatic retry, no connection pooling managed by the library. Failure messages differ.

**2. YAML as documentation only, Python provider remains active**
Write the YAML but configure the loader to prefer the Python provider.
- Pros: No behavior change.
- Cons: Solves nothing. The YAML is inert. Contradicts the system design.

**3. YAML for standard completions, Python provider for streaming (loader routes by call type)**
The loader uses YAML for non-streaming, Python for streaming.
- Pros: Clean separation of concerns.
- Cons: Only viable if `config.py` already supports routing by call type. If not, requires Python changes beyond scope.

### Recommended: Approach 1

Add the YAML file. The loader's fallback logic (YAML preferred, Python fallback) means adding the YAML activates it. No changes to `config.py` needed unless the loader inverts precedence — which must be verified in Phase 1 before any code is written.

**Honest scope statement:** This task is primarily writing one YAML file correctly, validating it, and writing two focused tests. The engineering judgment is in getting the response paths right, deciding on Azure, and verifying loader precedence behavior.

---

## Design

### Architectural Decision Records

**ADR-1: YAML plugin takes precedence over Python provider for standard completions**
- Decision: Once `openai.yaml` is present, it becomes the active code path for non-streaming OpenAI chat completions.
- Rationale: This is what the YAML system is designed for. Keeping Python as default makes the YAML system pointless for OpenAI and sets a bad precedent.
- Consequence: Any behavior difference between raw HTTP (YAML) and the `openai` SDK (Python) becomes observable. Retry behavior, connection reuse, and error message formatting may differ.

**ADR-2: Azure OpenAI is out of scope for this plugin**
- Decision: `openai.yaml` targets `api.openai.com` only. Azure requires a separate `azure_openai.yaml`.
- Rationale: Azure has a materially different URL structure, uses `api-key` header instead of `Authorization: Bearer`, and model names are deployment names (user-defined strings). Cramming this into one YAML via optional fields makes the schema conditionally valid, which breaks the validator.
- Consequence: Users on Azure must use a separate plugin (future work) or the Python provider.

**ADR-3: Streaming is not supported via YAML plugin**
- Decision: `stream: false` set as a static field. The YAML plugin system cannot handle SSE streaming.
- Rationale: Streaming requires consuming chunked response bodies and reassembling `delta.content` — procedural logic that cannot be expressed declaratively.
- Consequence: Applications relying on streaming continue to use the Python provider. Document this in comments in the YAML file.

**ADR-4: `o1`/`o3`/`o4-mini` excluded from model tier defaults**
- Decision: Model tiers only map to `gpt-4o` and `gpt-4o-mini`. Reasoning models are excluded.
- Rationale: `o1`/`o3` use `max_completion_tokens` instead of `max_tokens`, do not support `temperature`, and require different parameter handling. The static `param_map` cannot handle this without producing API errors.

### The Complete Plugin File

```yaml
# fitz_ai/llm/chat/openai.yaml
#
# Declarative OpenAI chat plugin for fitz-ai.
# Covers the synchronous chat completions path only.
#
# OUT OF SCOPE (use Python provider or separate plugin):
#   - Streaming completions (requires SSE, not expressible declaratively)
#   - Azure OpenAI (different URL structure, auth header, deployment model names)
#   - Reasoning models: o1, o3, o4-mini (different params: max_completion_tokens, no temperature)

plugin_name: "openai"
plugin_type: "chat"
version: "1.0"

provider:
  name: "openai"
  base_url: "https://api.openai.com/v1"

auth:
  type: "bearer"
  header_name: "Authorization"
  header_format: "Bearer {key}"
  env_vars:
    - "OPENAI_API_KEY"

endpoint:
  path: "/chat/completions"
  method: "POST"
  timeout: 120

defaults:
  models:
    smart: "gpt-4o"
    fast: "gpt-4o-mini"
    balanced: "gpt-4o-mini"
  temperature: 0.2
  max_tokens: null

request:
  messages_transform: "openai_chat"
  static_fields:
    stream: false
  param_map:
    model: "model"
    temperature: "temperature"
    max_tokens: "max_tokens"

response:
  content_path: "choices[0].message.content"
  is_array: false
  metadata_paths:
    finish_reason: "choices[0].finish_reason"
    tokens_input: "usage.prompt_tokens"
    tokens_output: "usage.completion_tokens"
```

### Data Model Notes

`choices[0].message.content` matches the OpenAI `/chat/completions` response exactly:
```json
{
  "choices": [{"message": {"content": "..."}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 10, "completion_tokens": 25}
}
```

Edge case: if the model returns a tool call, `message.content` is `null`. The YAML plugin returns null/empty silently. The Python provider handles this explicitly. Acceptable given ADR-3 scope.

---

## Roadmap

### Phase 1: Verify loader behavior before writing anything (~1 hour)

**This is the most important phase.** Before writing the YAML, verify three things by reading source:

1. Read `fitz_ai/llm/config.py` — does the loader prefer YAML or Python when both exist? Is there a hardcoded exclusion list for OpenAI?
2. Read one existing YAML plugin (e.g., `cohere.yaml`) — confirm exact schema the validator expects. Specifically: is `max_tokens: null` valid, or must the field be omitted?
3. Read `fitz_ai/plugin_gen/validators.py` — which fields are required vs. optional?

**Deliverable:** A list of schema differences between the YAML above and what the validator actually requires. Zero code written.

### Phase 2: Write and validate the YAML (~2 hours)

1. Write `fitz_ai/llm/chat/openai.yaml` using the content from Design, adjusted for Phase 1 findings.
2. Run the plugin validator directly:
   ```bash
   python -m fitz_ai.plugin_gen.validators fitz_ai/llm/chat/openai.yaml
   ```
3. Verify the loader picks it up:
   ```python
   from fitz_ai.llm.config import create_chat_provider
   provider = create_chat_provider("openai/gpt-4o", tier="smart")
   print(type(provider))  # should show YAML-backed provider, not Python provider
   ```

**Deliverable:** `fitz_ai/llm/chat/openai.yaml` passing schema validation.

### Phase 3: Tests (~2 hours)

Three focused tests in `tests/unit/llm/`:

```python
def test_openai_yaml_plugin_loads():
    """Loader discovers openai.yaml and returns a YAML-backed provider, not Python SDK."""
    provider = create_chat_provider("openai/gpt-4o", tier="smart")
    assert not isinstance(provider, OpenAIPythonProvider)
    assert provider.base_url == "https://api.openai.com/v1"

def test_openai_yaml_model_tiers():
    """smart → gpt-4o, fast → gpt-4o-mini."""
    assert create_chat_provider("openai", tier="smart").model == "gpt-4o"
    assert create_chat_provider("openai", tier="fast").model == "gpt-4o-mini"

def test_openai_response_content_extraction():
    """content_path correctly extracts from a realistic OpenAI response."""
    mock_response = {
        "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
    }
    result = yaml_plugin_runner.extract_response(mock_response)
    assert result.content == "Hello"
    assert result.metadata["tokens_input"] == 12
```

**Deliverable:** Three passing tests. No network calls needed.

### Phase 4: Config example update (~30 min)

Update `fitz_ai/engines/fitz_krag/config/fitz_krag.yaml` with a concrete example showing the plugin in use and the model tier override pattern.

### Critical Path
```
Phase 1 (verify loader + schema)
    ↓
Phase 2 (write + validate YAML)
    ↓              ↓
Phase 3 (tests)  Phase 4 (config docs)
```

**Total: ~5-6 hours.**

---

## Risk Analysis

### Risk 1: `max_tokens: null` fails schema validation
**Likelihood: Medium | Impact: Low**
YAML validators often enforce strict types. `null` for `max_tokens` might be rejected if the schema expects an integer or requires the field to be absent when no limit is intended. Sending `"max_tokens": null` in the JSON body may also cause a 422 from the OpenAI API.
**Mitigation:** Check in Phase 1. May need to omit `max_tokens` from `param_map` entirely and only include it when explicitly set by callers.

### Risk 2: Loader prefers Python provider when both exist
**Likelihood: Medium | Impact: High**
The spec says loader "falls back to Python provider if no YAML found" — but "falls back" could mean Python is primary. If `config.py` checks Python providers first by class name, adding `openai.yaml` has no effect.
**Mitigation:** Phase 1 reads `config.py` explicitly. If Python takes precedence, one targeted change to the loader inverts priority — but this must be done carefully since it affects all providers.

### Risk 3: OpenAI response schema change
**Likelihood: Low | Impact: Medium**
If OpenAI changes response field names, `content_path` silently returns `null` rather than raising. The failure is quiet.
**Mitigation:** The response extraction tests (Phase 3) use realistic mocked responses. Re-run after OpenAI API updates. Add non-null assertion on `result.content` in production call sites.

### Risk 4: Model deprecation
**Likelihood: High (inevitable, long-term) | Impact: Low**
`gpt-4o` and `gpt-4o-mini` will eventually be deprecated. OpenAI routes to successors for a period then returns errors.
**Mitigation:** The tier override pattern in `fitz_krag.yaml` lets users override the YAML default without editing the plugin. Document this explicitly. The YAML default is a baseline, not a mandate.

### Risk 5: Tool call responses return null content
**Likelihood: Low for RAG use cases | Impact: Low**
If the model returns a tool call response, `choices[0].message.content` is `null`. YAML plugin returns empty silently.
**Mitigation:** Accepted limitation per ADR-3. RAG pipelines should not be configuring tool use. Document in YAML comments.
