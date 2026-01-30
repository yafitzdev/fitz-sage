# Enterprise LLM Authentication

## What This Is

Adding enterprise authentication support to fitz-ai's LLM system so it can be deployed in corporate environments (BMW and similar). Enterprise users need M2M OAuth2 token refresh + LLM API keys + certificate support — all with minimal setup friction.

## Core Value

Enterprise users can connect fitz-ai to their internal LLM gateways as easily as casual users connect to public APIs.

## Requirements

### Validated

- API key auth for casual users — existing
- M2MAuth class with OAuth2 client credentials flow — existing
- Certificate path support in auth providers — existing
- Provider wrappers (OpenAI, Cohere, Anthropic, Ollama) — existing

### Active

- [ ] Fix token refresh: providers must use dynamic auth, not frozen tokens at init
- [ ] Two-layer enterprise auth: M2M bearer token + LLM API key combined
- [ ] Enterprise config schema: cert_path, token_url, client_id, client_secret, api_key
- [ ] Auto-refresh M2M tokens before expiry during long-running processes
- [ ] Low-friction enterprise setup (env vars or config file, minimal steps)
- [ ] Validation at BMW: deploy and confirm working

### Out of Scope

- mTLS (client certificates) — only CA certs for now, can add later if needed
- Custom enterprise providers — use OpenAI-compatible endpoints with base_url
- GUI/web setup wizard — CLI and config file only

## Context

**Current state:**
- `fitz_ai/llm/auth/` has `ApiKeyAuth` and `M2MAuth` classes
- `M2MAuth` has token refresh logic but providers call `get_headers()` once at init
- Providers extract API key and pass to SDK — token is frozen, never refreshes
- Enterprise flow at BMW: M2M token authenticates to gateway, API key authenticates to LLM

**Enterprise auth flow:**
1. User provides: cert_path, token_url, client_id, client_secret, llm_api_key
2. System fetches M2M bearer token (OAuth2 client credentials)
3. Requests include both M2M token and API key headers
4. M2M token auto-refreshes before expiry

**Validation target:** BMW internal LLM gateway deployment

## Constraints

- **Backwards compatible**: Casual users (API key only) must continue working unchanged
- **SDK usage**: Keep using official SDKs (OpenAI, Cohere, Anthropic) where possible
- **Existing structure**: Work within current provider architecture, don't rewrite everything

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use httpx custom client for dynamic auth | SDKs support custom http_client, allows per-request auth headers | — Pending |
| Two-layer auth (M2M + API key) | BMW's enterprise flow requires both | — Pending |
| Config-based setup (not interactive wizard) | Enterprise users often script deployments | — Pending |

---
*Last updated: 2025-01-30 after initialization*
