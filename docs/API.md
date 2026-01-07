# REST API Reference

Complete reference for the Fitz REST API.

---

## Quick Start

```bash
# Install with API support
pip install fitz-ai[api]

# Start the server
fitz serve

# Server runs at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

---

## Server Options

```bash
fitz serve [OPTIONS]

Options:
  -p, --port INTEGER    Port number (default: 8000)
  --host TEXT          Host to bind (default: 127.0.0.1)
  --reload             Enable auto-reload for development
```

**Examples:**

```bash
# Custom port
fitz serve -p 3000

# All interfaces (for Docker/remote access)
fitz serve --host 0.0.0.0

# Development mode
fitz serve --reload
```

---

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query` | Query knowledge base |
| POST | `/chat` | Multi-turn chat |
| POST | `/ingest` | Ingest documents |
| GET | `/collections` | List collections |
| GET | `/collections/{name}` | Get collection stats |
| DELETE | `/collections/{name}` | Delete collection |
| GET | `/health` | Health check |

---

## POST /query

Query the knowledge base with a single question.

### Request

```json
{
  "question": "What is the refund policy?",
  "collection": "default",
  "top_k": 5
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `question` | string | Yes | - | The question to ask |
| `collection` | string | No | `"default"` | Collection to query |
| `top_k` | integer | No | config value | Chunks to retrieve |

### Response

```json
{
  "text": "The refund policy allows returns within 30 days...",
  "mode": "confident",
  "sources": [
    {
      "source_id": "policies/refund.md",
      "excerpt": "Returns are accepted within 30 days of purchase...",
      "metadata": {
        "chunk_index": 2,
        "page": 1
      }
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | The answer text |
| `mode` | string | `confident`, `qualified`, `disputed`, or `abstain` |
| `sources` | array | Sources used in the answer |

### Example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'
```

---

## POST /chat

Multi-turn conversation with the knowledge base.

The server is **stateless** - the client must manage and send conversation history.

### Request

```json
{
  "message": "What about returns?",
  "history": [
    {"role": "user", "content": "What is the refund policy?"},
    {"role": "assistant", "content": "The refund policy allows returns within 30 days..."}
  ],
  "collection": "default",
  "top_k": 5
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `message` | string | Yes | - | Current user message |
| `history` | array | No | `[]` | Previous messages |
| `collection` | string | No | `"default"` | Collection to query |
| `top_k` | integer | No | config value | Chunks to retrieve |

**History message format:**

```json
{"role": "user" | "assistant", "content": "message text"}
```

### Response

Same as `/query`:

```json
{
  "text": "For returns, you need to...",
  "mode": "confident",
  "sources": [...]
}
```

### Example

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What about returns?",
    "history": [
      {"role": "user", "content": "What is the refund policy?"},
      {"role": "assistant", "content": "The refund policy allows..."}
    ]
  }'
```

---

## POST /ingest

Ingest documents into a collection.

### Request

```json
{
  "source": "./docs",
  "collection": "default",
  "clear_existing": false
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `source` | string | Yes | - | Path to file or directory |
| `collection` | string | No | `"default"` | Target collection |
| `clear_existing` | boolean | No | `false` | Clear collection first |

### Response

```json
{
  "documents": 15,
  "chunks": 234,
  "collection": "default"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `documents` | integer | Documents ingested |
| `chunks` | integer | Chunks created |
| `collection` | string | Target collection name |

### Example

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source": "./docs", "collection": "mydata"}'
```

---

## GET /collections

List all available collections.

### Response

```json
[
  {"name": "default", "chunk_count": 234},
  {"name": "physics", "chunk_count": 567}
]
```

### Example

```bash
curl http://localhost:8000/collections
```

---

## GET /collections/{name}

Get statistics for a specific collection.

### Response

```json
{
  "name": "default",
  "chunk_count": 234,
  "metadata": {
    "created_at": "2024-01-15T10:30:00",
    "last_updated": "2024-01-16T14:20:00"
  }
}
```

### Example

```bash
curl http://localhost:8000/collections/default
```

---

## DELETE /collections/{name}

Delete a collection and all its chunks.

### Response

```json
{
  "deleted": true,
  "collection": "default",
  "chunks_deleted": 234
}
```

### Example

```bash
curl -X DELETE http://localhost:8000/collections/old_data
```

---

## GET /health

Health check endpoint.

### Response

```json
{
  "status": "healthy",
  "version": "0.4.0",
  "config_exists": true
}
```

### Example

```bash
curl http://localhost:8000/health
```

---

## Error Responses

All endpoints return standard HTTP error codes:

| Code | Description |
|------|-------------|
| 400 | Bad request (invalid input) |
| 404 | Resource not found |
| 500 | Internal server error |
| 501 | Feature not supported by vector DB |

**Error response format:**

```json
{
  "detail": "Error message here"
}
```

---

## Answer Modes

The `mode` field in responses indicates answer confidence:

| Mode | Description | Typical Cause |
|------|-------------|---------------|
| `confident` | Strong evidence supports answer | Clear, unambiguous sources |
| `qualified` | Answer with limitations | Missing some context |
| `disputed` | Conflicting sources | Sources disagree |
| `abstain` | Cannot answer | Insufficient evidence |

---

## Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000"

# Query
response = requests.post(f"{BASE_URL}/query", json={
    "question": "What is the refund policy?",
    "collection": "default"
})
answer = response.json()
print(answer["text"])

# Chat with history
history = []
message = "What is the refund policy?"

response = requests.post(f"{BASE_URL}/chat", json={
    "message": message,
    "history": history
})
answer = response.json()

# Update history for next turn
history.append({"role": "user", "content": message})
history.append({"role": "assistant", "content": answer["text"]})

# Continue conversation
response = requests.post(f"{BASE_URL}/chat", json={
    "message": "What about returns?",
    "history": history
})
```

---

## JavaScript Client Example

```javascript
const BASE_URL = 'http://localhost:8000';

// Query
async function query(question) {
  const response = await fetch(`${BASE_URL}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question })
  });
  return response.json();
}

// Chat
async function chat(message, history = []) {
  const response = await fetch(`${BASE_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history })
  });
  return response.json();
}

// Usage
const answer = await query("What is the refund policy?");
console.log(answer.text);
```

---

## See Also

- [SDK.md](SDK.md) - Python SDK documentation
- [CLI.md](CLI.md) - CLI reference
- [CONFIG.md](CONFIG.md) - Configuration reference
