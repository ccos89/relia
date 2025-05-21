import os

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
)


class EliaChatModel(BaseModel):
    name: str
    """The name of the model e.g. `gpt-4.1`.
    This must match the name of the model specified by the provider.
    """
    id: str | None = None
    """If you have multiple versions of the same model (e.g. a personal
    OpenAI gpt-4o and a work OpenAI gpt-4o with different API keys/org keys),
    you need to be able to refer to them. For example, when setting the `default_model`
    key in the config, if you write `gpt-4o`, there's no way to know whether you
    mean your work or your personal `gpt-4o`. That's where `id` comes in."""
    display_name: str | None = None
    """The display name of the model in the UI."""
    provider: str | None = None
    """The provider of the model, e.g. openai, anthropic, etc"""
    api_key: SecretStr | None = None
    """If set, this will be used in place of the environment variable that
    would otherwise be used for this model (instead of OPENAI_API_KEY,
    ANTHROPIC_API_KEY, etc.)."""
    api_base: AnyHttpUrl | None = None
    """If set, this will be used as the base URL for making API calls.
    This can be useful if you're hosting models on a LocalAI server, for
    example."""
    organization: str | None = None
    """Some providers, such as OpenAI, allow you to specify an organization.
    In the case of """
    description: str | None = Field(default=None)
    """A description of the model which may appear inside the Elia UI."""
    product: str | None = Field(default=None)
    """For example `ChatGPT`, `Claude`, `Gemini`, etc."""
    temperature: float = Field(default=1.0)
    """The temperature to use. Low temperature means the same prompt is likely
    to produce similar results. High temperature means a flatter distribution
    when predicting the next token, and so the next token will be more random.
    Setting a very high temperature will likely produce junk output."""
    max_retries: int = Field(default=0)
    """The number of times to retry a request after it fails before giving up."""

    @property
    def lookup_key(self) -> str:
        return self.id or self.name


def get_builtin_openai_models() -> list[EliaChatModel]:
    return [
        EliaChatModel(
            id="elia-gpt-4.1",
            name="gpt-4.1",
            display_name="GPT-4.1",
            provider="OpenAI",
            product="ChatGPT",
            description="Flagship GPT model for complex tasks.",
            temperature=0.7,
        ),
        EliaChatModel(
            id="elia-gpt-4o",
            name="gpt-4o",
            display_name="GPT-4o",
            provider="OpenAI",
            product="ChatGPT",
            description="Fast, intelligent, flexible GPT model.",
            temperature=0.7,
        ),
        EliaChatModel(
            id="elia-o4-mini",
            name="o4-mini",
            display_name="o4-mini",
            provider="OpenAI",
            product="ChatGPT",
            description="Faster, more affordable reasoning model",
        ),
    ]


def get_builtin_anthropic_models() -> list[EliaChatModel]:
    return [
        EliaChatModel(
            id="elia-claude-3-7-sonnet-20250219",
            name="claude-3-7-sonnet-20250219",
            display_name="Claude 3.7 Sonnet",
            provider="Anthropic",
            product="Claude 3.7",
            description=("Anthropic's most intelligent model with extended thinking capabilities"),
        ),
        EliaChatModel(
            id="elia-claude-3-5-sonnet-20241022",
            name="claude-3-5-sonnet-20241022",
            display_name="Claude 3.5 Sonnet",
            provider="Anthropic",
            product="Claude 3.5 Sonnet",
            description=(
                "Anthropic's previous most intelligent model."
            ),
        ),
        EliaChatModel(
            id="elia-claude-3-5-haiku-20241022",
            name="claude-3-5-haiku-20241022",
            display_name="Claude 3.5 Haiku",
            provider="Anthropic",
            product="Claude 3.5 Haiku",
            description=(
                "Anthropic's fastest and most cost-effective model."
            ),
        ),
    ]


def get_builtin_google_models() -> list[EliaChatModel]:
    return [
        EliaChatModel(
            id="elia-gemini-2.5-pro-preview-05-06",
            name="gemini-2.5-pro-preview-05-06",
            display_name="Gemini 2.5 Pro Preview",
            provider="Google",
            product="Gemini",
            description="Google's most powerful thinking model.",
        ),
        EliaChatModel(
            id="elia-gemini-2.5-flash-preview-05-20",
            name="gemini-2.5-flash-preview-05-20",
            display_name="Gemini 2.5 Flash Preview",
            provider="Google",
            product="Gemini",
            description="Google's first hybrid reasoning model with thinking budgets.",
        ),
    ]


def get_builtin_models() -> list[EliaChatModel]:
    return (
        get_builtin_openai_models()
        + get_builtin_anthropic_models()
        + get_builtin_google_models()
    )


class LaunchConfig(BaseModel):
    """The config of the application at launch.

    Values may be sourced via command line options, env vars, config files.
    """

    model_config = ConfigDict(frozen=True)

    default_model: str = Field(default="elia-gpt-4.1")
    """The ID or name of the default model."""
    system_prompt: str = Field(
        default=os.getenv(
            "ELIA_SYSTEM_PROMPT", "You are a helpful assistant named Elia."
        )
    )
    message_code_theme: str = Field(default="monokai")
    """The default Pygments syntax highlighting theme to be used in chatboxes."""
    models: list[EliaChatModel] = Field(default_factory=list)
    builtin_models: list[EliaChatModel] = Field(
        default_factory=get_builtin_models, init=False
    )
    theme: str = Field(default="nebula")

    @field_validator("system_prompt")
    def non_empty(cls, value: str) -> str:
        if not value or value.strip() == "":
            raise ValueError("system prompt must not be empty")
        return value

    @property
    def all_models(self) -> list[EliaChatModel]:
        return self.models + self.builtin_models

    @property
    def default_model_object(self) -> EliaChatModel:
        from elia_chat.models import get_model

        return get_model(self.default_model, self)

    @classmethod
    def get_current(cls) -> "LaunchConfig":
        return cls()
