from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping

import httpx
import requests

import comfy.model_management as model_management


@dataclass
class CancellableResponse:
    status_code: int
    headers: httpx.Headers
    content: bytes
    url: str = ""
    reason_phrase: str = ""
    encoding: str | None = None

    @property
    def text(self) -> str:
        if self.encoding:
            return self.content.decode(self.encoding, errors="replace")
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        if self.status_code < 400:
            return
        message = f"{self.status_code} {self.reason_phrase}".strip()
        raise requests.HTTPError(message, response=self)


class CancellableHttpRequestError(requests.RequestException):
    pass


class _RequestState:
    def __init__(self):
        self.loop: asyncio.AbstractEventLoop | None = None
        self.task: asyncio.Task | None = None
        self.client: httpx.AsyncClient | None = None
        self.result: CancellableResponse | None = None
        self.exception: BaseException | None = None
        self.done = threading.Event()
        self.cancel_requested = threading.Event()


class CancellableHttpClient:
    def __init__(
        self,
        *,
        proxy_url: str | None = None,
        timeout: float | int | None = 0,
        check_interval: float = 0.25,
        transport: httpx.AsyncBaseTransport | None = None,
        follow_redirects: bool = True,
    ):
        self.proxy_url = proxy_url or None
        self.timeout = timeout
        self.check_interval = max(float(check_interval), 0.01)
        self.transport = transport
        self.follow_redirects = follow_redirects

    def get(self, url: str, **kwargs) -> CancellableResponse:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> CancellableResponse:
        return self.request("POST", url, **kwargs)

    def sleep(self, seconds: float, *, check_interval: float | None = None) -> None:
        deadline = time.monotonic() + max(float(seconds), 0.0)
        interval = self.check_interval if check_interval is None else max(float(check_interval), 0.01)
        while True:
            if self._processing_interrupted():
                model_management.throw_exception_if_processing_interrupted()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(interval, remaining))

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        data: Any = None,
        files: Any = None,
        content: Any = None,
        timeout: float | int | None = None,
        proxies: Mapping[str, str] | str | None = None,
        **kwargs,
    ) -> CancellableResponse:
        model_management.throw_exception_if_processing_interrupted()

        state = _RequestState()
        thread = threading.Thread(
            target=self._run_request_thread,
            args=(
                state,
                method,
                url,
                headers,
                params,
                json,
                data,
                files,
                content,
                self._normalize_timeout(timeout),
                self._normalize_proxy(proxies),
                kwargs,
            ),
            daemon=True,
        )
        thread.start()

        while not state.done.wait(self.check_interval):
            if self._processing_interrupted():
                self._cancel_state(state)
                thread.join(timeout=1.0)
                model_management.throw_exception_if_processing_interrupted()

        if state.exception is not None:
            if isinstance(state.exception, asyncio.CancelledError):
                model_management.throw_exception_if_processing_interrupted()
            if isinstance(state.exception, httpx.RequestError):
                raise CancellableHttpRequestError(str(state.exception)) from state.exception
            raise state.exception

        if state.result is None:
            raise CancellableHttpRequestError("HTTP request finished without a response")
        return state.result

    def _run_request_thread(
        self,
        state: _RequestState,
        method: str,
        url: str,
        headers: Mapping[str, str] | None,
        params: Mapping[str, Any] | None,
        json_data: Any,
        data: Any,
        files: Any,
        content: Any,
        timeout: float | int | None,
        proxy_url: str | None,
        extra_kwargs: dict[str, Any],
    ) -> None:
        loop = asyncio.new_event_loop()
        state.loop = loop
        try:
            asyncio.set_event_loop(loop)
            state.result = loop.run_until_complete(
                self._request_async(
                    state,
                    method,
                    url,
                    headers,
                    params,
                    json_data,
                    data,
                    files,
                    content,
                    timeout,
                    proxy_url,
                    extra_kwargs,
                )
            )
        except BaseException as exception:
            state.exception = exception
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()
                state.done.set()

    async def _request_async(
        self,
        state: _RequestState,
        method: str,
        url: str,
        headers: Mapping[str, str] | None,
        params: Mapping[str, Any] | None,
        json_data: Any,
        data: Any,
        files: Any,
        content: Any,
        timeout: float | int | None,
        proxy_url: str | None,
        extra_kwargs: dict[str, Any],
    ) -> CancellableResponse:
        client_kwargs: dict[str, Any] = {
            "timeout": timeout,
            "follow_redirects": self.follow_redirects,
        }
        if proxy_url:
            client_kwargs["proxy"] = proxy_url
        if self.transport is not None:
            client_kwargs["transport"] = self.transport

        async with httpx.AsyncClient(**client_kwargs) as client:
            state.client = client
            request_kwargs = dict(extra_kwargs)
            request_kwargs.update(
                {
                    "headers": headers,
                    "params": params,
                    "json": json_data,
                    "data": data,
                    "files": files,
                    "content": content,
                }
            )
            request_kwargs = {
                key: value for key, value in request_kwargs.items() if value is not None
            }
            task = asyncio.create_task(client.request(method, url, **request_kwargs))
            state.task = task
            if state.cancel_requested.is_set():
                task.cancel()

            response = await task
            body = await response.aread()
            return CancellableResponse(
                status_code=response.status_code,
                headers=response.headers,
                content=body,
                url=str(response.url),
                reason_phrase=response.reason_phrase,
                encoding=response.encoding,
            )

    def _cancel_state(self, state: _RequestState) -> None:
        state.cancel_requested.set()

        def cancel() -> None:
            if state.task is not None and not state.task.done():
                state.task.cancel()

        if state.loop is not None and not state.loop.is_closed():
            state.loop.call_soon_threadsafe(cancel)

    def _normalize_timeout(self, timeout: float | int | None) -> float | int | None:
        value = self.timeout if timeout is None else timeout
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return value
        return numeric if numeric > 0 else None

    def _normalize_proxy(self, proxies: Mapping[str, str] | str | None) -> str | None:
        if isinstance(proxies, str):
            return proxies or self.proxy_url
        if proxies:
            return proxies.get("https") or proxies.get("http") or self.proxy_url
        return self.proxy_url

    @staticmethod
    def _processing_interrupted() -> bool:
        checker = getattr(model_management, "processing_interrupted", None)
        if callable(checker):
            return bool(checker())
        return False
