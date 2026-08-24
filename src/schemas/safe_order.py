from typing import Optional, List
from pydantic import BaseModel


class SafeItem(BaseModel):
    sku: str
    name: str
    quantity: int
    final_sale: bool


class SafeOrder(BaseModel):
    order_id: str

    membership_tier: str

    items: List[SafeItem]

    placed_at: str
    status: str
    status_updated_at: str

    shipped_at: Optional[str] = None
    delivered_at: Optional[str] = None

    carrier: Optional[str] = None
    tracking_number: Optional[str] = None

    estimated_delivery: Optional[str] = None

    customer_safe_message: str


    class Config:
        extra = "ignore"