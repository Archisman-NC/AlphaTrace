import os
from dotenv import load_dotenv
from langfuse import Langfuse

# Load environment variables early for module-level initialization
load_dotenv()

def normalize_sector(sector: str) -> str:
    """Cleans and normalizes a sector name for strict mapping comparisons."""
    if not sector:
        return ""
    return sector.strip().upper()

def build_stock_to_sector_map(sector_mapping: dict) -> dict:
    """Builds a flat stock -> sector mapping dictionary from the nested structural JSON payload."""
    stock_map = {}
    sectors = sector_mapping.get("sectors", {})
    
    for sector_name, details in sectors.items():
        stocks = details.get("stocks", [])
        for stock in stocks:
            stock_map[stock] = normalize_sector(sector_name)
            
    return stock_map

# Initialize Langfuse Observer (Singleton-ish)
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL", os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"))
)
