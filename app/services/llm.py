"""Typed LLM interface and OpenAI chat adapter."""

import json
from typing import Protocol, TypeVar

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, ValidationError

from app.core.config import Settings, StructuredMode
from app.core.exceptions import DependencyError
from app.schemas.chat import ChatMessage, MessageRole

Output = TypeVar("Output", bound=BaseModel)


class LLMProvider(Protocol):
    """Generate text and schema-validated responses."""

    def complete(self, messages: list[ChatMessage]) -> str:
        """Generate a plain-text answer."""
        ...

    def structured(self, messages: list[ChatMessage], schema: type[Output]) -> Output:
        """Generate a response constrained to a Pydantic schema."""
        ...


def _openai_messages(messages: list[ChatMessage]) -> list[ChatCompletionMessageParam]:
    result: list[ChatCompletionMessageParam] = []
    for message in messages:
        if message.role == MessageRole.SYSTEM:
            result.append(ChatCompletionSystemMessageParam(role="system", content=message.content))
        elif message.role == MessageRole.USER:
            result.append(ChatCompletionUserMessageParam(role="user", content=message.content))
        else:
            result.append(
                ChatCompletionAssistantMessageParam(role="assistant", content=message.content)
            )
    return result


class OpenAILLMProvider:
    """Use OpenAI chat completions with JSON-schema structured output."""

    def __init__(self, client: OpenAI, settings: Settings) -> None:
        self._client: OpenAI = client
        self._settings: Settings = settings

    def _check_key(self) -> None:
        if not self._settings.openai_api_key.get_secret_value():
            raise DependencyError("OPENAI_API_KEY is not configured.")

    def complete(self, messages: list[ChatMessage]) -> str:
        """Generate an answer and reject empty output."""
        self._check_key()
        try:
            response = self._client.chat.completions.create(
                model=self._settings.openai_chat_model, messages=_openai_messages(messages)
            )
            answer = response.choices[0].message.content
            if not answer or not answer.strip():
                raise ValueError("Empty completion")
            return answer
        except Exception as exc:
            raise DependencyError("The language model is unavailable.") from exc

    def structured(self, messages: list[ChatMessage], schema: type[Output]) -> Output:
        """Parse the provider's JSON-schema-constrained response."""
        self._check_key()
        try:
            response = self._client.chat.completions.parse(
                model=self._settings.openai_chat_model,
                messages=_openai_messages(messages),
                response_format=schema,
            )
            parsed = response.choices[0].message.parsed
            if parsed is None:
                raise ValueError("The language model refused or returned no structured output")
            return parsed
        except Exception as exc:
            raise DependencyError(
                "The language model could not return valid structured output."
            ) from exc


class GroqLLMProvider:
    """Use Groq's OpenAI-compatible API with validated structured output."""

    def __init__(self, client: OpenAI, settings: Settings) -> None:
        self._client: OpenAI = client
        self._settings: Settings = settings

    def _check_key(self) -> None:
        if not self._settings.groq_api_key.get_secret_value():
            raise DependencyError("GROQ_API_KEY is not configured.")

    def complete(self, messages: list[ChatMessage]) -> str:
        """Generate text using the configured Groq model."""
        self._check_key()
        try:
            response = self._client.chat.completions.create(
                model=self._settings.groq_chat_model, messages=_openai_messages(messages)
            )
            answer = response.choices[0].message.content
            if not answer or not answer.strip():
                raise ValueError("Empty Groq response")
            return answer
        except Exception as exc:
            raise DependencyError("The Groq language model is unavailable.") from exc

    def structured(self, messages: list[ChatMessage], schema: type[Output]) -> Output:
        """Use strict schema mode when supported, otherwise validate JSON with one retry."""
        self._check_key()
        mode = self._settings.groq_structured_mode
        use_schema = mode == StructuredMode.JSON_SCHEMA or (
            mode == StructuredMode.AUTO
            and self._settings.groq_chat_model in self._settings.groq_schema_models
        )
        if use_schema:
            try:
                response = self._client.chat.completions.parse(
                    model=self._settings.groq_chat_model,
                    messages=_openai_messages(messages),
                    response_format=schema,
                )
                if response.choices[0].message.parsed is None:
                    raise ValueError("Groq returned no structured output")
                return response.choices[0].message.parsed
            except Exception as exc:
                raise DependencyError("Groq could not return valid schema output.") from exc
        return self._json_output(messages, schema)

    def _json_output(self, messages: list[ChatMessage], schema: type[Output]) -> Output:
        prompt = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "Return only a JSON object matching this JSON schema. "
                    "Include every required field, use null when permitted, and no markdown: "
                    + json.dumps(schema.model_json_schema())
                ),
            ),
            *messages,
        ]
        for attempt in range(self._settings.structured_output_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._settings.groq_chat_model,
                    messages=_openai_messages(prompt),
                    response_format={"type": "json_object"},
                )
                return schema.model_validate_json(response.choices[0].message.content or "")
            except ValidationError as exc:
                if attempt == self._settings.structured_output_retries:
                    raise DependencyError(
                        "Groq returned invalid JSON or schema fields after validation retries."
                    ) from exc
                prompt.append(
                    ChatMessage(
                        role=MessageRole.USER,
                        content=(
                            "The previous output failed JSON/schema validation. "
                            "Return a corrected JSON object satisfying the provided schema."
                        ),
                    )
                )
            except Exception as exc:
                raise DependencyError("Groq could not return JSON output.") from exc
        raise DependencyError("Groq structured output attempts were exhausted.")
