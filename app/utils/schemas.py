from typing import List, Literal
from pydantic import BaseModel

class StockData(BaseModel):
    symbol: str
    sector: str
    daily_change: float

class NewsItem(BaseModel):
    headline: str
    summary: str
    sentiment: Literal["positive", "negative", "neutral"]
    scope: Literal["market", "sector", "stock"]
    related_entities: List[str]

class PortfolioItem(BaseModel):
    symbol: str
    quantity: int
    avg_price: float
