import hashlib
import json
import os
import random
import re
import socket
import threading
import time
from email.utils import parsedate_to_datetime
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def _normalize_usage(data):
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


class OpenAICompatClient:
    def __init__(
        self,
        base_url,
        model,
        api_key="",
        user_agent="",
        proxy_url="",
        timeout=60,
        max_retries=None,
        retry_backoff_seconds=1.0,
        retry_jitter_seconds=0.25,
        error_log_dir="",
        trace_log_dir="",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.user_agent = user_agent or os.getenv("TASKSVC_LLM_USER_AGENT", "")
        self.proxy_url = str(proxy_url or "").strip()
        self.timeout = timeout
        configured_retries = max_retries
        if configured_retries is None:
            configured_retries = os.getenv("TASKSVC_LLM_MAX_RETRIES", "4")
        self.max_retries = max(1, int(configured_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self.retry_jitter_seconds = max(0.0, float(retry_jitter_seconds))
        self.error_log_dir = error_log_dir or os.getenv("TASKSVC_LLM_ERROR_LOG_DIR", "")
        self.trace_log_dir = trace_log_dir or os.getenv("TASKSVC_LLM_TRACE_LOG_DIR", "")
        self._artifact_lock = threading.Lock()
        self._artifact_counter = 0
        # Keep transport independent from ambient env proxy/no_proxy behavior.
        # We apply explicit proxy per-request when needed.
        self._opener = urllib.request.build_opener()

    def _explicit_proxy_target(self):
        if not self.proxy_url:
            return ""
        raw = self.proxy_url.strip()
        if not raw:
            return ""
        if "://" not in raw:
            raw = f"http://{raw}"
        parsed = urllib.parse.urlparse(raw)
        target = (parsed.netloc or "").strip()
        if not target:
            return ""
        return target

    def _hard_timeout_seconds(self):
        base_timeout = max(float(self.timeout or 0), 1.0)
        return base_timeout + min(30.0, max(5.0, base_timeout * 0.1))

    def _headers(self):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            # Avoid half-dead keepalive sockets piling up inside long-running workers.
            "Connection": "close",
        }
        if self.user_agent:
            headers["User-Agent"] = self.user_agent
        return headers

    def _retry_delay(self, attempt_index):
        if attempt_index <= 0:
            return 0.0
        base_delay = self.retry_backoff_seconds * (2 ** (attempt_index - 1))
        if self.retry_jitter_seconds <= 0:
            return base_delay
        return base_delay + random.uniform(0.0, self.retry_jitter_seconds)

    def _retry_after_seconds(self, exc):
        headers = getattr(exc, "headers", {}) or {}
        raw = headers.get("Retry-After")
        if not raw:
            return None
        text = str(raw).strip()
        if not text:
            return None
        try:
            return max(0.0, float(text))
        except ValueError:
            pass
        try:
            dt = parsedate_to_datetime(text)
        except Exception:
            return None
        if dt is None:
            return None
        now = time.time()
        return max(0.0, dt.timestamp() - now)

    def _build_request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=self._headers(),
        )
        proxy_target = self._explicit_proxy_target()
        if proxy_target:
            # Force proxy usage even if environment has no_proxy='*'.
            req.set_proxy(proxy_target, "http")
            req.set_proxy(proxy_target, "https")
        return req

    def _read_http_error_body(self, exc):
        body = ""
        try:
            raw = exc.read()
            if isinstance(raw, bytes):
                body = raw.decode("utf-8", errors="replace")
            elif raw is not None:
                body = str(raw)
        except Exception:
            pass
        try:
            exc.close()
        except Exception:
            pass
        return body

    def _sanitize_fragment(self, value, default="unknown"):
        text = str(value or default).strip()
        text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
        if not text:
            text = default
        digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:8]
        head = text[:24] or default
        return f"{head}_{digest}"

    def _next_artifact_counter(self):
        with self._artifact_lock:
            self._artifact_counter += 1
            return self._artifact_counter

    def _artifact_path(self, target_dir, request_context, attempt_number, status_fragment):
        ctx = request_context or {}
        counter = self._next_artifact_counter()
        parts = [
            time.strftime("%Y%m%dT%H%M%S"),
            f"{counter:04d}",
            self._sanitize_fragment(ctx.get("stage"), "stage"),
            self._sanitize_fragment(ctx.get("task_id"), "task"),
            self._sanitize_fragment(ctx.get("tool_name"), "tool"),
            f"attempt{int(attempt_number)}",
            status_fragment,
        ]
        return Path(target_dir) / f"{'__'.join(parts)}.json"

    def _persist_http_error(self, path, payload, exc, body_text, request_context, attempt_number):
        ctx = request_context or {}
        target_dir = self.error_log_dir or (request_context or {}).get("error_log_dir")
        if not target_dir:
            trace_dir = self.trace_log_dir or (request_context or {}).get("trace_log_dir")
            if trace_dir:
                target_dir = str(Path(trace_dir) / "http_error")
        if not target_dir:
            return None
        out_dir = Path(target_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._artifact_path(
            out_dir,
            request_context,
            attempt_number,
            f"http{int(getattr(exc, 'code', 0) or 0)}",
        )
        payload_dump = {
            "path": path,
            "url": getattr(exc, "filename", None) or f"{self.base_url}{path}",
            "http_status": getattr(exc, "code", None),
            "reason": getattr(exc, "reason", None),
            "attempt_number": int(attempt_number),
            "request_context": ctx,
            "request_payload": payload,
            "response_headers": dict(getattr(exc, "headers", {}) or {}),
            "response_body": body_text,
        }
        out_path.write_text(json.dumps(payload_dump, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return str(out_path)

    def _persist_request_exception(self, path, payload, exc, request_context, attempt_number, status_fragment):
        target_dir = self.error_log_dir or (request_context or {}).get("error_log_dir")
        if not target_dir:
            trace_dir = self.trace_log_dir or (request_context or {}).get("trace_log_dir")
            if trace_dir:
                target_dir = str(Path(trace_dir) / "http_error")
        if not target_dir:
            return None
        out_dir = Path(target_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._artifact_path(out_dir, request_context, attempt_number, status_fragment)
        payload_dump = {
            "path": path,
            "url": f"{self.base_url}{path}",
            "attempt_number": int(attempt_number),
            "request_context": request_context or {},
            "request_payload": payload,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
        }
        out_path.write_text(json.dumps(payload_dump, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return str(out_path)

    def _persist_http_success(self, path, payload, response_body, response_payload, request_context, attempt_number, response_headers=None):
        target_dir = self.trace_log_dir or (request_context or {}).get("trace_log_dir")
        if not target_dir:
            return None
        out_dir = Path(target_dir) / "http_success"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._artifact_path(out_dir, request_context, attempt_number, "http200")
        payload_dump = {
            "path": path,
            "url": f"{self.base_url}{path}",
            "http_status": 200,
            "attempt_number": int(attempt_number),
            "request_context": request_context or {},
            "request_payload": payload,
            "response_headers": dict(response_headers or {}),
            "response_body": response_body,
            "response_payload": response_payload,
        }
        out_path.write_text(json.dumps(payload_dump, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return str(out_path)

    def _perform_request(self, req):
        response_box = {}
        error_box = {}

        def _worker():
            try:
                with self._opener.open(req, timeout=self.timeout) as resp:
                    response_box["body"] = resp.read().decode("utf-8")
                    response_box["headers"] = getattr(resp, "headers", {}) or {}
            except BaseException as exc:
                error_box["error"] = exc

        worker = threading.Thread(
            target=_worker,
            name=f"openai-compat-http-{self._next_artifact_counter()}",
            daemon=True,
        )
        worker.start()
        worker.join(self._hard_timeout_seconds())
        if worker.is_alive():
            raise socket.timeout(
                f"hard request deadline exceeded after {self._hard_timeout_seconds():.1f}s"
            )
        if "error" in error_box:
            raise error_box["error"]
        return response_box.get("body", ""), response_box.get("headers", {})

    def _format_http_error(self, exc, request_context=None, artifact_path=None, body_text=""):
        message = f"HTTP Error {exc.code}: {exc.reason}"
        if request_context:
            message += f" | context={json.dumps(request_context, ensure_ascii=False, sort_keys=True)}"
        if artifact_path:
            message += f" | artifact={artifact_path}"
        preview = (body_text or "").strip()
        if preview:
            message += f" | body_preview={preview[:400]}"
        return message

    def _format_request_exception(self, exc, request_context=None, artifact_path=None):
        message = f"{type(exc).__name__}: {exc}"
        if request_context:
            message += f" | context={json.dumps(request_context, ensure_ascii=False, sort_keys=True)}"
        if artifact_path:
            message += f" | artifact={artifact_path}"
        return message

    def _request_json(self, path, payload=None, request_context=None):
        last_error = None
        for attempt_index in range(self.max_retries):
            req = self._build_request(path, payload=payload)
            try:
                body, response_headers = self._perform_request(req)
                data = json.loads(body)
                self._persist_http_success(
                    path,
                    payload,
                    body,
                    data,
                    request_context,
                    attempt_index + 1,
                    response_headers=response_headers,
                )
                return data
            except urllib.error.HTTPError as exc:
                body_text = self._read_http_error_body(exc)
                artifact_path = self._persist_http_error(
                    path,
                    payload,
                    exc,
                    body_text,
                    request_context,
                    attempt_index + 1,
                )
                wrapped = RuntimeError(
                    self._format_http_error(
                        exc,
                        request_context=request_context,
                        artifact_path=artifact_path,
                        body_text=body_text,
                    )
                )
                last_error = wrapped
                retryable = exc.code in {429, 500, 502, 503, 504}
                if not retryable or attempt_index + 1 >= self.max_retries:
                    raise wrapped
                delay = self._retry_delay(attempt_index + 1)
                retry_after = self._retry_after_seconds(exc)
                if retry_after is not None:
                    delay = max(delay, retry_after)
                elif exc.code == 429:
                    delay = max(delay, 8.0 * (attempt_index + 1))
                if delay > 0:
                    time.sleep(delay)
                continue
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt_index + 1 >= self.max_retries:
                    raise
            except (TimeoutError, socket.timeout) as exc:
                artifact_path = self._persist_request_exception(
                    path,
                    payload,
                    exc,
                    request_context,
                    attempt_index + 1,
                    "timeout",
                )
                wrapped = RuntimeError(
                    self._format_request_exception(
                        exc,
                        request_context=request_context,
                        artifact_path=artifact_path,
                    )
                )
                last_error = wrapped
                if attempt_index + 1 >= self.max_retries:
                    raise wrapped
                delay = max(self._retry_delay(attempt_index + 1), 2.0 * (attempt_index + 1))
                if delay > 0:
                    time.sleep(delay)
                continue
            delay = self._retry_delay(attempt_index + 1)
            if delay > 0:
                time.sleep(delay)
        if last_error is not None:
            raise last_error
        raise RuntimeError("OpenAICompatClient._request_json exhausted retries without response.")

    def list_models(self):
        return self._request_json("/models")

    def chat_completion(
        self,
        messages,
        temperature=0.2,
        max_tokens=16000,
        tools=None,
        tool_choice=None,
        request_context=None,
    ):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        return self._request_json("/chat/completions", payload=payload, request_context=request_context)

    def complete_text(self, system_prompt, user_prompt, temperature=0.2, max_tokens=16000, request_context=None):
        data = self.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            request_context=request_context,
        )
        choice = data["choices"][0]["message"]
        return {
            "text": choice.get("content") or "",
            "reasoning": choice.get("reasoning"),
            "usage": _normalize_usage(data),
            "raw": data,
        }
