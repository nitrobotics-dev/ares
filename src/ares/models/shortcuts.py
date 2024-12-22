from ares.models.base import VLM, Embedder, SentenceTransformerEmbedder


def get_siglip_embedder() -> Embedder:
    return Embedder(provider="google", name="siglip-base-patch16-224")


def get_nomic_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(provider="nomic-ai", name="nomic-embed-text-v1")


def get_gemini_15_flash() -> VLM:
    return VLM(provider="gemini", name="gemini-1.5-flash")


def get_gemini_2_flash() -> VLM:
    return VLM(provider="gemini", name="gemini-2.0-flash-exp")


def get_4o_mini() -> VLM:
    return VLM(provider="openai", name="gpt-4o-mini")


def summarize(vlm: VLM, data: list[str], description: str) -> str:
    info = {"data": "\n".join(data), "description": description}
    messages, response = vlm.ask(
        info,
        prompt_filename="summarizing.jinja2",
    )
    return response.choices[0].message.content
