from pydantic import BaseModel


class CapturedUsage(BaseModel):
    """browser-use's own accounting block, as save_history writes it.

    Not optexity's TokenUsage: that counts different things under different
    names. agentic_nodes is this layer's own addition when several are summed.
    """

    agentic_nodes: int = 0
    entry_count: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
