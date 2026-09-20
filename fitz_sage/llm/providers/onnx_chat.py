# fitz_sage/llm/providers/onnx_chat.py
"""
In-process ONNX GenAI provider for managed local Qwen work.

This is deliberately narrow: Fitz owns one tiny local generation backend for
semantic query terms and optional background enrichment, using a pre-built
Qwen3.5 0.8B Q4F16 ONNX model from Hugging Face on CPU. No external server, no
GGUF, no llama.cpp, no torch.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from fitz_sage.core.exceptions import ManagedModelError

DEFAULT_QWEN_MODEL_ALIAS = "qwen3.5-0.8b"
DEFAULT_QWEN_MODEL_ID = "onnx-community/Qwen3.5-0.8B-Text-ONNX"
DEFAULT_QWEN_MODEL_REVISION = "1e45daba048899e7f771657ada617ec49350aa91"
DEFAULT_QWEN_ONNX_SUBFOLDER = "onnx"
DEFAULT_QWEN_ONNX_FILE = "model_q4f16.onnx"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_MAX_CONTEXT_TOKENS = 8192

_MODEL_ALIASES: dict[str, tuple[str, str, str, str | None]] = {
    DEFAULT_QWEN_MODEL_ALIAS: (
        DEFAULT_QWEN_MODEL_ID,
        DEFAULT_QWEN_ONNX_SUBFOLDER,
        DEFAULT_QWEN_ONNX_FILE,
        DEFAULT_QWEN_MODEL_REVISION,
    )
}

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

logger = logging.getLogger(__name__)


class OnnxChatModelError(ManagedModelError):
    """Managed Qwen model download, validation, or runtime initialization failed."""


@dataclass(frozen=True)
class OnnxChatModelInfo:
    """Resolved managed Qwen model files and version metadata."""

    name: str
    repo_id: str
    revision: str
    snapshot_dir: str
    onnx_path: str
    onnx_subfolder: str
    onnx_file: str
    onnx_bytes: int
    external_data_paths: list[str]
    total_bytes: int
    tokenizer_path: str
    bundle_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-serializable model metadata."""
        return asdict(self)


@dataclass(frozen=True)
class ManagedQwenSpec:
    """Resolved managed Qwen model identity and ONNX artifact names."""

    name: str
    repo_id: str
    onnx_subfolder: str
    onnx_file: str
    revision: str | None = None


class ManagedQwenSnapshot:
    """Download and validate the Hugging Face snapshot for managed Qwen."""

    def __init__(self, spec: ManagedQwenSpec) -> None:
        self._spec = spec

    @property
    def spec(self) -> ManagedQwenSpec:
        """Return the resolved model spec."""
        return self._spec

    def download(self) -> Path:
        """Download the managed Qwen snapshot into the Hugging Face cache."""
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise OnnxChatModelError(
                "Managed Qwen enrichment requires `huggingface_hub` to download "
                "the ONNX model. Install fitz-sage with its runtime dependencies "
                "and retry."
            ) from e

        logger.info(
            "Ensuring managed Qwen %s is available from %s (%s/%s)",
            self._spec.name,
            self._spec.repo_id,
            self._spec.onnx_subfolder,
            self._spec.onnx_file,
        )
        try:
            return Path(
                snapshot_download(
                    repo_id=self._spec.repo_id,
                    revision=self._spec.revision,
                    allow_patterns=self.allow_patterns(),
                )
            )
        except Exception as e:
            raise OnnxChatModelError(
                f"Could not download Fitz's managed Qwen model `{self._spec.name}` "
                f"from Hugging Face repo `{self._spec.repo_id}`. Check network "
                "access or pre-populate the Hugging Face cache, then retry."
            ) from e

    def allow_patterns(self) -> list[str]:
        """Return the minimal HF snapshot file set for the managed runtime."""
        return [
            "added_tokens.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "genai_config.json",
            f"{self._spec.onnx_subfolder}/{self._spec.onnx_file}",
            f"{self._spec.onnx_subfolder}/{self._spec.onnx_file}_data*",
        ]

    def build_model_info(
        self,
        snapshot_dir: Path,
        *,
        include_checksum: bool,
    ) -> OnnxChatModelInfo:
        """Validate required files and construct inspectable snapshot metadata."""
        onnx_path = snapshot_dir / self._spec.onnx_subfolder / self._spec.onnx_file
        external_data_paths = sorted(onnx_path.parent.glob(f"{self._spec.onnx_file}_data*"))
        tokenizer_path = snapshot_dir / "tokenizer.json"
        required_paths = [onnx_path, tokenizer_path]
        if self._spec.name == DEFAULT_QWEN_MODEL_ALIAS:
            required_paths.append(snapshot_dir / "config.json")
        else:
            required_paths.append(snapshot_dir / "genai_config.json")
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise OnnxChatModelError(
                f"Fitz's managed Qwen snapshot is incomplete for `{self._spec.name}`. "
                f"Missing required file(s): {', '.join(missing)}"
            )

        model_paths = [onnx_path, *external_data_paths]
        checksum = _bundle_sha256(model_paths) if include_checksum else None
        total_bytes = sum(path.stat().st_size for path in model_paths)
        return OnnxChatModelInfo(
            name=self._spec.name,
            repo_id=self._spec.repo_id,
            revision=snapshot_dir.name,
            snapshot_dir=str(snapshot_dir),
            onnx_path=str(onnx_path),
            onnx_subfolder=self._spec.onnx_subfolder,
            onnx_file=self._spec.onnx_file,
            onnx_bytes=onnx_path.stat().st_size,
            external_data_paths=[str(path) for path in external_data_paths],
            total_bytes=total_bytes,
            tokenizer_path=str(tokenizer_path),
            bundle_sha256=checksum,
        )


class GenAiRuntimeBundle:
    """Prepare the Fitz-owned CPU ONNX GenAI runtime directory."""

    def __init__(self, spec: ManagedQwenSpec) -> None:
        self._spec = spec

    @staticmethod
    def require_genai() -> Any:
        """Import ONNX Runtime GenAI with an actionable product error."""
        try:
            import onnxruntime_genai as og

            return og
        except ImportError as e:
            raise OnnxChatModelError(
                "Managed Qwen enrichment requires `onnxruntime-genai`. "
                "Install fitz-sage with its runtime dependencies and retry."
            ) from e

    def prepare(self, info: OnnxChatModelInfo) -> Path:
        """Materialize Fitz's CPU GenAI config without mutating the HF snapshot."""
        from fitz_sage.core.paths import FitzPaths

        snapshot_dir = Path(info.snapshot_dir)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", info.name)
        runtime_dir = FitzPaths.user_home() / "models" / "qwen" / safe_name / info.revision
        runtime_dir.mkdir(parents=True, exist_ok=True)
        (runtime_dir / self._spec.onnx_subfolder).mkdir(parents=True, exist_ok=True)

        for name in (
            "added_tokens.json",
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ):
            source = snapshot_dir / name
            if source.exists():
                _link_or_copy(source, runtime_dir / name)

        genai_config_path = snapshot_dir / "genai_config.json"
        onnx_source = Path(info.onnx_path)
        onnx_target = runtime_dir / self._spec.onnx_subfolder / onnx_source.name
        if genai_config_path.exists():
            _link_or_copy(onnx_source, onnx_target)
        else:
            self._prepare_qwen35_onnx(onnx_source, onnx_target)
        for source in (Path(path) for path in info.external_data_paths):
            _link_or_copy(source, runtime_dir / self._spec.onnx_subfolder / source.name)

        if genai_config_path.exists():
            config = json.loads(genai_config_path.read_text(encoding="utf-8"))
        else:
            config = self._build_qwen35_genai_config(snapshot_dir)
        decoder = config.setdefault("model", {}).setdefault("decoder", {})
        decoder["filename"] = f"{self._spec.onnx_subfolder}/{self._spec.onnx_file}"
        decoder.setdefault("session_options", {})["provider_options"] = []
        search = config.setdefault("search", {})
        search["do_sample"] = False
        search["temperature"] = 0
        search["max_length"] = min(
            int(search.get("max_length", DEFAULT_MAX_CONTEXT_TOKENS) or DEFAULT_MAX_CONTEXT_TOKENS),
            DEFAULT_MAX_CONTEXT_TOKENS,
        )
        (runtime_dir / "genai_config.json").write_text(
            json.dumps(config, indent=2),
            encoding="utf-8",
        )
        return runtime_dir

    def _prepare_qwen35_onnx(self, source: Path, target: Path) -> None:
        """Rename Transformers.js cache tensors to the ORT GenAI convention."""
        if self._spec.name != DEFAULT_QWEN_MODEL_ALIAS:
            raise OnnxChatModelError(
                f"Custom managed Qwen model `{self._spec.name}` must provide genai_config.json."
            )
        try:
            import onnx
        except ImportError as e:
            raise OnnxChatModelError(
                "Managed Qwen3.5 enrichment requires `onnx` to prepare the "
                "ONNX Community graph for ONNX Runtime GenAI."
            ) from e

        model = onnx.load(str(source), load_external_data=False)
        renamed: dict[str, str] = {}
        for value in (*model.graph.input, *model.graph.output):
            new_name = _qwen35_genai_tensor_name(value.name)
            if new_name != value.name:
                renamed[value.name] = new_name
                value.name = new_name
        for node in model.graph.node:
            for index, name in enumerate(node.input):
                node.input[index] = renamed.get(name, name)
            for index, name in enumerate(node.output):
                node.output[index] = renamed.get(name, name)

        # ORT GenAI supplies num_logits_to_keep as a one-element tensor. The
        # Transformers.js export unsqueezes it once more, producing invalid
        # two-dimensional Slice starts. Reshape accepts either representation
        # and guarantees the one-dimensional input Slice requires.
        logits_shape_name = "fitz_sage_num_logits_to_keep_shape"
        model.graph.initializer.append(
            onnx.helper.make_tensor(
                logits_shape_name,
                onnx.TensorProto.INT64,
                dims=[1],
                vals=[-1],
            )
        )
        adapted_logits_input = False
        for node in model.graph.node:
            if node.name == "/lm_head/num_logits_to_keep/Unsqueeze":
                node.op_type = "Reshape"
                node.input[1] = logits_shape_name
                del node.attribute[:]
                adapted_logits_input = True
                break
        if not adapted_logits_input:
            raise OnnxChatModelError(
                "The managed Qwen3.5 ONNX graph does not match its pinned export contract."
            )

        if target.exists():
            target.unlink()
        onnx.save(model, str(target))

    def _build_qwen35_genai_config(self, snapshot_dir: Path) -> dict[str, Any]:
        """Build the ORT GenAI manifest omitted by the ONNX Community export."""
        if self._spec.name != DEFAULT_QWEN_MODEL_ALIAS:
            raise OnnxChatModelError(
                f"Custom managed Qwen model `{self._spec.name}` must provide genai_config.json."
            )

        hf_config = json.loads((snapshot_dir / "config.json").read_text(encoding="utf-8"))
        model_type = str(hf_config.get("model_type", ""))
        if model_type != "qwen3_5_text":
            raise OnnxChatModelError(
                "The managed Qwen3.5 snapshot has an unexpected model type: "
                f"`{model_type or 'missing'}`."
            )

        eos_token_id = int(hf_config["eos_token_id"])
        return {
            "model": {
                "bos_token_id": 1,
                "context_length": int(hf_config["max_position_embeddings"]),
                "decoder": {
                    "session_options": {
                        "log_id": "onnxruntime-genai",
                        "provider_options": [],
                    },
                    "filename": f"{self._spec.onnx_subfolder}/{self._spec.onnx_file}",
                    "head_size": int(hf_config["head_dim"]),
                    "hidden_size": int(hf_config["hidden_size"]),
                    "inputs": {
                        "input_ids": "input_ids",
                        "attention_mask": "attention_mask",
                        "past_key_names": "past_key_values.%d.key",
                        "past_value_names": "past_key_values.%d.value",
                    },
                    "outputs": {
                        "logits": "logits",
                        "present_key_names": "present.%d.key",
                        "present_value_names": "present.%d.value",
                    },
                    "num_attention_heads": int(hf_config["num_attention_heads"]),
                    "num_hidden_layers": int(hf_config["num_hidden_layers"]),
                    "num_key_value_heads": int(hf_config["num_key_value_heads"]),
                },
                "eos_token_id": [eos_token_id],
                "pad_token_id": eos_token_id,
                "type": model_type,
                "vocab_size": int(hf_config["vocab_size"]),
            },
            "search": {
                "do_sample": False,
                "early_stopping": True,
                "max_length": DEFAULT_MAX_CONTEXT_TOKENS,
                "num_beams": 1,
                "num_return_sequences": 1,
                "past_present_share_buffer": True,
                "temperature": 0,
                "top_k": 50,
                "top_p": 1.0,
            },
        }


class OnnxGenAiGenerator:
    """Loaded ONNX GenAI session plus Qwen tokenizer formatting."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        genai_model: Any,
        genai_tokenizer: Any,
    ) -> None:
        self._tokenizer = tokenizer
        self._genai_model = genai_model
        self._genai_tokenizer = genai_tokenizer

    def format_messages(self, messages: list[dict[str, Any]]) -> str:
        """Apply the model chat template, disabling Qwen thinking by default."""
        if getattr(self._tokenizer, "chat_template", None):
            return cast(
                str,
                self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                ),
            )
        rendered: list[str] = []
        for message in messages:
            rendered.append(f"{message.get('role', 'user')}: {message.get('content', '')}")
        rendered.append("assistant:")
        return "\n".join(rendered)

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """Run ONNX GenAI generation and return generated text."""
        input_tokens = self._genai_tokenizer.encode(prompt)
        params = self._make_generator_params(
            max_length=_bounded_generation_length(
                len(input_tokens),
                max_new_tokens,
            ),
            temperature=temperature,
            top_p=top_p,
        )
        og = GenAiRuntimeBundle.require_genai()
        token_stream = self._genai_tokenizer.create_stream()
        generated = og.Generator(self._genai_model, params)
        try:
            generated.append_tokens(input_tokens)

            chunks: list[str] = []
            while not generated.is_done():
                generated.generate_next_token()
                chunks.append(token_stream.decode(generated.get_next_tokens()[0]))
            return "".join(chunks)
        finally:
            # The native generator owns per-request KV-cache allocations.
            del generated

    def _make_generator_params(
        self,
        *,
        max_length: int,
        temperature: float,
        top_p: float,
    ) -> Any:
        """Build ORT GenAI search parameters for greedy default generation."""
        og = GenAiRuntimeBundle.require_genai()
        params = og.GeneratorParams(self._genai_model)
        options: dict[str, Any] = {
            "max_length": max_length,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            options["temperature"] = temperature
            options["top_p"] = top_p
        params.set_search_options(**options)
        return params


@dataclass(frozen=True)
class _LoadedOnnxChatRuntime:
    """Process-shared native runtime for one managed model artifact."""

    generator: OnnxGenAiGenerator
    run_lock: threading.Lock


_ONNX_CHAT_RUNTIME_CACHE: dict[tuple[str, str | None, str, str], _LoadedOnnxChatRuntime] = {}
_ONNX_CHAT_RUNTIME_CACHE_LOCK = threading.Lock()


def _strip_thinking(text: str) -> str:
    """Remove Qwen reasoning blocks from completed responses."""
    text = _THINK_RE.sub("", text)
    if "<think>" in text:
        text = (
            text.split("</think>")[-1].lstrip()
            if "</think>" in text
            else text.split("<think>")[0].rstrip()
        )
    return text


def _bounded_generation_length(input_tokens: int, requested_new_tokens: int) -> int:
    """Fit one generation inside the managed runtime's context window."""
    if input_tokens >= DEFAULT_MAX_CONTEXT_TOKENS:
        raise OnnxChatModelError(
            "Managed Qwen prompt exceeds its "
            f"{DEFAULT_MAX_CONTEXT_TOKENS}-token context limit "
            f"({input_tokens} input tokens supplied)."
        )
    available = DEFAULT_MAX_CONTEXT_TOKENS - input_tokens
    return input_tokens + min(max(1, requested_new_tokens), available)


def _is_native_allocation_error(error: RuntimeError) -> bool:
    """Identify the transient allocation failure emitted by ORT GenAI."""
    return "bad allocation" in str(error).casefold()


def _token_text(value: Any) -> str | None:
    """Extract a token string from tokenizer_config values."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        return str(content) if content is not None else None
    return None


def _qwen35_genai_tensor_name(name: str) -> str:
    """Map ONNX Community Qwen3.5 cache names to ORT GenAI templates."""
    replacements = (
        (r"past_conv\.(\d+)", r"past_key_values.\1.conv_state"),
        (r"past_recurrent\.(\d+)", r"past_key_values.\1.recurrent_state"),
        (r"present_conv\.(\d+)", r"present.\1.conv_state"),
        (r"present_recurrent\.(\d+)", r"present.\1.recurrent_state"),
    )
    for pattern, replacement in replacements:
        if re.fullmatch(pattern, name):
            return re.sub(pattern, replacement, name)
    return name


def _bundle_sha256(paths: list[Path]) -> str:
    """Compute a SHA256 checksum over model files in stable order."""
    digest = sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _resolve_qwen_spec(
    model_id: str,
    *,
    onnx_subfolder: str | None,
    onnx_file: str | None,
) -> ManagedQwenSpec:
    """Resolve a public provider spec into a managed Qwen model spec."""
    resolved = _MODEL_ALIASES.get(model_id)
    if resolved:
        repo_id, default_subfolder, default_file, revision = resolved
        name = model_id
    else:
        repo_id = model_id
        default_subfolder = DEFAULT_QWEN_ONNX_SUBFOLDER
        default_file = DEFAULT_QWEN_ONNX_FILE
        revision = None
        name = model_id

    return ManagedQwenSpec(
        name=name,
        repo_id=repo_id,
        onnx_subfolder=onnx_subfolder if onnx_subfolder is not None else default_subfolder,
        onnx_file=onnx_file if onnx_file is not None else default_file,
        revision=revision,
    )


class OnnxChat:
    """CPU generation over the Fitz-managed Qwen3.5 0.8B Q4F16 ONNX graph."""

    def __init__(
        self,
        model_id: str = DEFAULT_QWEN_MODEL_ALIAS,
        *,
        onnx_subfolder: str | None = None,
        onnx_file: str | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        spec = _resolve_qwen_spec(
            model_id,
            onnx_subfolder=onnx_subfolder,
            onnx_file=onnx_file,
        )
        self._snapshot = ManagedQwenSnapshot(spec)
        self._runtime = GenAiRuntimeBundle(spec)
        self._max_new_tokens = max_new_tokens
        self._load_lock = threading.RLock()
        self._model_info: OnnxChatModelInfo | None = None
        self._loaded_runtime: _LoadedOnnxChatRuntime | None = None

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """Generate a chat completion with deterministic local ONNX GenAI inference."""
        self._load()
        max_new_tokens = int(
            kwargs.pop("max_tokens", kwargs.pop("max_new_tokens", self._max_new_tokens))
        )
        temperature = float(kwargs.pop("temperature", 0.0) or 0.0)
        top_p = float(kwargs.pop("top_p", 1.0) or 1.0)
        stop = kwargs.pop("stop", None)

        runtime = self._loaded_runtime
        if runtime is None:
            raise OnnxChatModelError("Managed Qwen runtime was not loaded.")
        prompt = runtime.generator.format_messages(messages)

        with runtime.run_lock:
            try:
                text = runtime.generator.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            except RuntimeError as error:
                if not _is_native_allocation_error(error):
                    raise
                logger.warning(
                    "Managed Qwen hit a transient native allocation failure; "
                    "retrying once after releasing the request generator."
                )
                text = runtime.generator.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )

        text = _strip_thinking(text).strip()
        return self._apply_stop(text, stop)

    def ensure_available(self, *, include_checksum: bool = False) -> OnnxChatModelInfo:
        """Download and validate managed Qwen files, returning resolved metadata.

        This does not initialize ONNX Runtime GenAI. It exists so first-run and smoke
        checks can verify the exact managed model snapshot before inference.
        """
        with self._load_lock:
            GenAiRuntimeBundle.require_genai()
            if self._model_info is not None:
                if not include_checksum or self._model_info.bundle_sha256 is not None:
                    return self._model_info

            snapshot_dir = self._snapshot.download()
            self._model_info = self._snapshot.build_model_info(
                snapshot_dir,
                include_checksum=include_checksum,
            )
            logger.info(
                "Managed Qwen ONNX ready: repo=%s revision=%s path=%s bytes=%s",
                self._model_info.repo_id,
                self._model_info.revision,
                self._model_info.onnx_path,
                self._model_info.total_bytes,
            )
            return self._model_info

    def model_info(self, *, include_checksum: bool = False) -> OnnxChatModelInfo:
        """Return inspectable metadata for the managed Qwen model snapshot."""
        return self.ensure_available(include_checksum=include_checksum)

    def _load(self) -> None:
        """Load tokenizer and ONNX GenAI model once per process."""
        if self._loaded_runtime is not None:
            return
        spec = self._snapshot.spec
        cache_key = (spec.repo_id, spec.revision, spec.onnx_subfolder, spec.onnx_file)
        with _ONNX_CHAT_RUNTIME_CACHE_LOCK:
            runtime = _ONNX_CHAT_RUNTIME_CACHE.get(cache_key)
            if runtime is not None:
                self._loaded_runtime = runtime
                return
            og = GenAiRuntimeBundle.require_genai()

            logger.info(
                "Loading ONNX GenAI chat model %s (%s/%s)",
                spec.repo_id,
                spec.onnx_subfolder,
                spec.onnx_file,
            )
            info = self.ensure_available()
            snapshot_dir = Path(info.snapshot_dir)
            try:
                tokenizer = self._load_tokenizer(snapshot_dir)
            except Exception as e:
                raise OnnxChatModelError(
                    f"Could not load the tokenizer for Fitz's managed Qwen `{spec.name}` "
                    f"ONNX model from {snapshot_dir}."
                ) from e

            runtime_dir = self._runtime.prepare(info)
            try:
                genai_model = og.Model(str(runtime_dir))
                genai_tokenizer = og.Tokenizer(genai_model)
                generator = OnnxGenAiGenerator(
                    tokenizer=tokenizer,
                    genai_model=genai_model,
                    genai_tokenizer=genai_tokenizer,
                )
            except Exception as e:
                raise OnnxChatModelError(
                    "Could not initialize ONNX Runtime GenAI for Fitz's managed "
                    f"Qwen `{spec.name}` model at {runtime_dir}."
                ) from e
            runtime = _LoadedOnnxChatRuntime(
                generator=generator,
                run_lock=threading.Lock(),
            )
            _ONNX_CHAT_RUNTIME_CACHE[cache_key] = runtime
            self._loaded_runtime = runtime

    @staticmethod
    def _require_genai() -> Any:
        """Import ONNX Runtime GenAI with an actionable product error."""
        return GenAiRuntimeBundle.require_genai()

    def _download_snapshot(self) -> Path:
        """Download the managed Qwen snapshot into the Hugging Face cache."""
        return self._snapshot.download()

    def _build_model_info(
        self,
        snapshot_dir: Path,
        *,
        include_checksum: bool,
    ) -> OnnxChatModelInfo:
        """Validate required files and construct inspectable snapshot metadata."""
        return self._snapshot.build_model_info(
            snapshot_dir,
            include_checksum=include_checksum,
        )

    def _prepare_genai_runtime(self, info: OnnxChatModelInfo) -> Path:
        """Materialize Fitz's CPU GenAI config without mutating the HF snapshot."""
        return self._runtime.prepare(info)

    def _load_tokenizer(self, snapshot_dir: Path) -> Any:
        """Load Qwen tokenizer without requiring a model-specific Python class."""
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(str(snapshot_dir))
        except ValueError:
            from transformers import PreTrainedTokenizerFast

            config_path = snapshot_dir / "tokenizer_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(snapshot_dir / "tokenizer.json"),
                bos_token=_token_text(config.get("bos_token")),
                eos_token=_token_text(config.get("eos_token")),
                pad_token=_token_text(config.get("pad_token")),
                unk_token=_token_text(config.get("unk_token")),
            )
            tokenizer.chat_template = config.get("chat_template")
            return tokenizer

    @staticmethod
    def _apply_stop(text: str, stop: Any) -> str:
        """Apply OpenAI-style stop string truncation."""
        if stop is None:
            return text
        stops = [stop] if isinstance(stop, str) else list(stop)
        cut = len(text)
        for marker in stops:
            if not marker:
                continue
            index = text.find(str(marker))
            if index >= 0:
                cut = min(cut, index)
        return text[:cut].rstrip()


def _link_or_copy(source: Path, target: Path) -> None:
    """Hardlink model artifacts into the Fitz runtime cache; copy when unavailable."""
    if target.exists() and target.stat().st_size == source.stat().st_size:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


__all__ = [
    "DEFAULT_QWEN_MODEL_ALIAS",
    "DEFAULT_QWEN_MODEL_ID",
    "DEFAULT_QWEN_MODEL_REVISION",
    "DEFAULT_QWEN_ONNX_FILE",
    "GenAiRuntimeBundle",
    "ManagedQwenSnapshot",
    "ManagedQwenSpec",
    "OnnxChat",
    "OnnxGenAiGenerator",
    "OnnxChatModelError",
    "OnnxChatModelInfo",
]
