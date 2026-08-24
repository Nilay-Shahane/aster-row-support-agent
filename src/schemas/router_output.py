from pydantic import BaseModel
from typing import Literal, Optional

class RouteDecision(BaseModel):
    route: Literal["order", "rag" , "direct"]
    order_id: Optional[str] = None