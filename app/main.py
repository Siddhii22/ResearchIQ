"""FastAPI application entry point."""

import logging

from fastapi import FastAPI
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.limits import MAX_RESEARCH_BODY_BYTES
from app.api.routes import router
from app.api.research import router as research_router
from app.config import settings

logging.basicConfig(
	level=logging.WARNING,
	format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logging.getLogger("app").setLevel(logging.INFO)

class ResearchBodyLimitMiddleware:
	def __init__(self, app: ASGIApp, max_bytes: int = MAX_RESEARCH_BODY_BYTES) -> None:
		self.app = app
		self.max_bytes = max_bytes

	async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
		if scope["type"] != "http" or scope.get("path") != "/api/v1/research/query":
			await self.app(scope, receive, send)
			return

		content_length = next(
			(value for name, value in scope.get("headers", []) if name.lower() == b"content-length"),
			None,
		)
		if content_length is not None:
			try:
				if int(content_length) > self.max_bytes:
					await self._send_too_large(send)
					return
			except ValueError:
				pass

		body_messages: list[Message] = []
		body_size = 0
		while True:
			message = await receive()
			if message["type"] == "http.disconnect":
				return
			body_size += len(message.get("body", b""))
			if body_size > self.max_bytes:
				await self._send_too_large(send)
				return
			body_messages.append(message)
			if not message.get("more_body", False):
				break

		async def replay_receive() -> Message:
			if body_messages:
				return body_messages.pop(0)
			return await receive()

		await self.app(scope, replay_receive, send)

	async def _send_too_large(self, send: Send) -> None:
		body = b'{"detail":"Request body exceeds the allowed size."}'
		await send(
			{
				"type": "http.response.start",
				"status": 413,
				"headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
			}
		)
		await send({"type": "http.response.body", "body": body})


app = FastAPI(title=settings.app_name)
app.add_middleware(ResearchBodyLimitMiddleware)
app.include_router(router)
app.include_router(research_router)