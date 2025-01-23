from ares.models.base import VLM, Embedder, SentenceTransformerEmbedder


def get_siglip_embedder() -> Embedder:
    return Embedder(provider="google", name="siglip-base-patch16-224")


def get_nomic_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(provider="nomic-ai", name="nomic-embed-text-v1")


def get_all_embedders() -> dict[str, Embedder]:
    return {
        "siglip": get_siglip_embedder(),
        "nomic": get_nomic_embedder(),
    }


def get_gemini_15_flash() -> VLM:
    return VLM(provider="gemini", name="gemini-1.5-flash")


def get_gemini_15_pro() -> VLM:
    return VLM(provider="gemini", name="gemini-1.5-pro")


def get_gemini_2_flash() -> VLM:
    return VLM(provider="gemini", name="gemini-2.0-flash-exp")


def get_gpt_4o_mini() -> VLM:
    return VLM(provider="openai", name="gpt-4o-mini")


def get_gpt_4o() -> VLM:
    return VLM(provider="openai", name="gpt-4o")


def get_gpt_o1_mini() -> VLM:
    return VLM(provider="openai", name="o1-preview")


def get_claude_3_5_sonnet() -> VLM:
    return VLM(provider="anthropic", name="claude-3-5-sonnet-20240620")


def get_claude_3_5_haiku() -> VLM:
    return VLM(provider="anthropic", name="claude-3-5-haiku-20241022")


name_to_vlm_fn_mapping = {
    "gemini-1.5-pro": get_gemini_15_pro,
    "gemini-2-flash": get_gemini_2_flash,
    "gemini-1.5-flash": get_gemini_15_flash,
    "gpt-4o-mini": get_gpt_4o_mini,
    "gpt-4o": get_gpt_4o,
    "gpt-o1-mini": get_gpt_o1_mini,
    "claude-3-5-sonnet": get_claude_3_5_sonnet,
}


def get_all_vlm_fns() -> dict[str, VLM]:
    return name_to_vlm_fn_mapping


def get_vlm(name: str) -> VLM:
    if name not in name_to_vlm_fn_mapping:
        raise ValueError(
            f"VLM {name} not found from name_to_vlm_fn_mapping: {name_to_vlm_fn_mapping.keys()}"
        )
    return name_to_vlm_fn_mapping[name]()


def summarize(vlm: VLM, data: list[str], description: str) -> str:
    info = {"data": "\n".join(data), "description": description}
    messages, response = vlm.ask(
        info,
        prompt_filename="summarizing.jinja2",
    )
    return response.choices[0].message.content
