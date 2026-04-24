import json
import os
import logging
from typing import List, Any, Type, TypeVar
from app.utils.schemas import StockData, NewsItem, PortfolioItem
from pydantic import BaseModel, ValidationError
from app.utils.helpers import build_stock_to_sector_map

logger = logging.getLogger(__name__)

# Set up generic type bounds for validating Pydantic models
T = TypeVar('T', bound=BaseModel)

def extract_list(data: Any) -> list:
    """
    Safely extracts the first valid array from JSON data structure.
    Raises ValueError if none is found.
    """
    if isinstance(data, list):
        return data
        
    elif isinstance(data, dict):
        logger.warning("Dictionary payload detected. Attempting to extract list dynamically...")
        found_lists = [v for v in data.values() if isinstance(v, list)]
        
        if not found_lists:
            raise ValueError("No list found inside the dictionary.")
            
        if len(found_lists) > 1:
            logger.warning(f"Multiple lists ({len(found_lists)}) found inside dictionary. Extracting the first one only.")
            
        return found_lists[0]
        
    else:
        raise ValueError(f"Expected list or dictionary wrap, got {type(data).__name__}")

def load_json_data(file_path: str) -> list:
    """Loads JSON data from a file and runs list extraction safely. Always returns a list."""
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return []
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_json = json.load(f)
            return extract_list(raw_json)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON in {file_path}: {e}")
        return []
    except ValueError as e:
        logger.error(f"Extraction error in {file_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error loading {file_path}: {e}")
        return []

def validate_items(parsed_list: list, schema: Type[T]) -> List[T]:
    """Generically validates a list of dictionaries into a list of Pydantic models."""
    validated_objects: List[T] = []
    
    for item in parsed_list:
        if not isinstance(item, dict):
            logger.warning(f"Malformed array item encountered (expected dict for instantiation), skipping.")
            continue
            
        try:
            validated_objects.append(schema(**item))
        except ValidationError as e:
            logger.warning(f"Validation error for {schema.__name__} item: {e}")
            
    return validated_objects

def load_stocks(file_path: str) -> List[StockData]:
    """Loads and validates StockData from the mock JSON file."""
    parsed_list = load_json_data(file_path)
    return validate_items(parsed_list, StockData)

def load_news(file_path: str) -> List[NewsItem]:
    """Loads and validates NewsItems from the mock JSON file."""
    parsed_list = load_json_data(file_path)
    return validate_items(parsed_list, NewsItem)

def load_portfolio(file_path: str) -> List[PortfolioItem]:
    """Loads and validates PortfolioItems from the mock JSON file."""
    parsed_list = load_json_data(file_path)
    return validate_items(parsed_list, PortfolioItem)


class DataLoader:
    """Centralized Data Accessor caching disk memory efficiently."""
    
    def __init__(self, data_dir: str = "data/mock"):
        from app.utils.helpers import build_stock_to_sector_map
        import os
        import json
        
        # We perform internal raw loads here to permit exact dictionary key addressing natively
        # since `load_json_data` strictly enforces list-extrusion for schema validators.
        def _load_raw(filename: str) -> dict:
            path = os.path.join(data_dir, filename)
            if not os.path.exists(path):
                logger.warning(f"Dataset block missing: {path}")
                return {}
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Internal structure parse crash {path}: {e}")
                return {}
                
        self.market_data = _load_raw("market_data.json")
        self.news_data = _load_raw("news_data.json")
        self.portfolios = _load_raw("portfolios.json")
        self.mutual_funds = _load_raw("mutual_funds.json")
        self.sector_mapping = _load_raw("sector_mapping.json")
        
        # FIX 6: Harden registry with Omni-Gated Mapper
        self.stock_to_sector = build_stock_to_sector_map(self.sector_mapping or {})
        
    def get_market_data(self) -> dict:
        return {
            "indices": self.market_data.get("indices", {}),
            "sector_performance": self.market_data.get("sector_performance", {})
        }

    def get_news(self) -> list:
        return self.news_data.get("news", [])

    def get_portfolio(self, portfolio_id: str) -> dict:
        return self.portfolios.get("portfolios", {}).get(portfolio_id, {})

    def get_mutual_fund(self, mf_code: str) -> dict:
        return self.mutual_funds.get("mutual_funds", {}).get(mf_code, {})

    def get_sector_mapping(self) -> dict:
        return self.sector_mapping.get("sectors", {})
