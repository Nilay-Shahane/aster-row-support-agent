from pydantic import BaseModel, Field
from typing import Optional

class ContextorOutput(BaseModel):
    query: str = Field(description="The standalone, rewritten user query based on conversation history.")
    order_id: Optional[str] = Field(default=None, description="The extracted order ID (e.g., ORD-1007) if mentioned.")