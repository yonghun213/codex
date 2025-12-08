import os
import requests
import logging
import json
import configparser
from pathlib import Path
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetabaseClient:
    """
    A robust client for interacting with the Metabase API.
    Handles authentication and card (question) creation.
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize the Metabase client.
        
        Args:
            base_url (str): The base URL of your Metabase instance (e.g., 'https://metabase.example.com').
            api_key (str, optional): The API Key (if available).
            username (str, optional): Metabase username (if using password auth).
            password (str, optional): Metabase password (if using password auth).
        """
        self.base_url = base_url.rstrip('/') + '/api'
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        # Authenticate immediately upon initialization
        self._authenticate(api_key, username, password)

    def _authenticate(self, api_key: Optional[str], username: Optional[str], password: Optional[str]):
        """
        Handles authentication logic.
        Prioritizes API Key if provided, otherwise falls back to Username/Password session.
        """
        if api_key:
            logger.info("Authenticating using API Key...")
            self.headers['x-api-key'] = api_key
            # Verify connectivity
            self._validate_session()
        elif username and password:
            logger.info("Authenticating using Username and Password...")
            try:
                payload = {'username': username, 'password': password}
                response = requests.post(f"{self.base_url}/session", json=payload)
                response.raise_for_status()
                session_id = response.json()['id']
                self.headers['X-Metabase-Session'] = session_id
                logger.info("Successfully authenticated via Session Token.")
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to create session: {e}")
                if e.response is not None:
                    logger.error(f"Response: {e.response.text}")
                raise
        else:
            raise ValueError("Must provide either api_key OR (username and password).")

    def _validate_session(self):
        """Helper to validate that the current credentials work."""
        try:
            # Fetch current user to validate token
            res = requests.get(f"{self.base_url}/user/current", headers=self.headers)
            res.raise_for_status()
            logger.info(f"Connected as user: {res.json().get('common_name')}")
        except requests.exceptions.RequestException as e:
            logger.error("Authentication check failed. Please check your credentials.")
            raise

    def create_line_chart(self, database_id: int, collection_id: int, sql_query: str, title: str) -> Dict[str, Any]:
        """
        Creates a new Native SQL Question visualized as a Line Chart.

        Args:
            database_id (int): The ID of the database in Metabase to run the query against.
            collection_id (int): The ID of the collection where the card will be saved.
            sql_query (str): The SQL query string.
            title (str): The title of the card.

        Returns:
            dict: The JSON response of the created card.
        """
        endpoint = f"{self.base_url}/card"
        
        # Structure for a Native SQL Question in Metabase
        payload = {
            'name': title,
            'collection_id': collection_id,
            'display': 'line',  # Sets visualization to Line Chart
            'dataset_query': {
                'database': database_id,
                'type': 'native',
                'native': {
                    'query': sql_query
                }
            },
            'visualization_settings': {
                'graph.dimensions': ['business_date'],  # X-axis
                'graph.metrics': ['sales'],             # Y-axis
                'graph.x_axis.scale': 'timeseries'      # Ensure date handling
            }
        }

        try:
            logger.info(f"Creating card '{title}' in collection {collection_id}...")
            response = requests.post(endpoint, headers=self.headers, json=payload)
            response.raise_for_status()
            
            card_data = response.json()
            logger.info(f"Successfully created card! ID: {card_data.get('id')}")
            return card_data

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to create card: {e}")
            if e.response is not None:
                logger.error(f"Response Body: {e.response.text}")
            raise

    def add_card_to_dashboard(self, dashboard_id: int, card_id: int):
        """
        Adds an existing card (question) to a dashboard.
        """
        endpoint = f"{self.base_url}/dashboard/{dashboard_id}/cards"
        
        payload = {
            'cardId': card_id,
            'sizeX': 10,  # Width (out of 24 grid columns usually)
            'sizeY': 6,   # Height
            'row': 0,     # Top row
            'col': 0      # Left column
        }

        try:
            logger.info(f"Adding card {card_id} to dashboard {dashboard_id}...")
            response = requests.post(endpoint, headers=self.headers, json=payload)
            response.raise_for_status()
            logger.info("Successfully added card to dashboard!")
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"⚠️ Dashboard {dashboard_id} not found (404). Card {card_id} was created but NOT added to the dashboard.")
            else:
                logger.error(f"Failed to add card to dashboard: {e}")
                raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to add card to dashboard: {e}")
            raise

# --- Helper Section: How to find IDs ---
# 1. Database ID: Go to Admin Settings -> Databases -> Click your database -> The ID is in the URL (e.g., /admin/databases/2 -> ID is 2).
# 2. Collection ID: Go to the Collection in Metabase -> The ID is in the URL (e.g., /collection/5-my-reports -> ID is 5). Root collection is usually null or you can try 1.

if __name__ == "__main__":
    # --- Configuration ---
    # Load credentials from config.ini if Env Vars are missing
    config = configparser.ConfigParser()
    config_path = Path("config.ini")
    
    if config_path.exists():
        try:
            config.read(config_path, encoding='utf-8')
        except:
             # Fallback for systems where default encoding might cause issues
            config.read(config_path)

    METABASE_URL = os.getenv("METABASE_URL")
    if not METABASE_URL and config.has_section("metabase"):
        METABASE_URL = config.get("metabase", "url", fallback="http://localhost:3000")
    if not METABASE_URL: METABASE_URL = "http://localhost:3000"

    METABASE_API_KEY = os.getenv("METABASE_API_KEY")
    if not METABASE_API_KEY and config.has_section("metabase"):
        METABASE_API_KEY = config.get("metabase", "api_key", fallback=None)

    METABASE_USER = os.getenv("METABASE_USERNAME")
    METABASE_PASS = os.getenv("METABASE_PASSWORD")

    # Target Settings
    TARGET_DB_ID = 200        # From URL /admin/databases/200
    TARGET_DASHBOARD_ID = 97  # From URL /dashboard/97-sales
    TARGET_COLLECTION_ID = None # None = Root Collection (My Personal Collection or Our Analytics Root)

    # The Logic
    try:
        # Initialize Client
        client = MetabaseClient(
            base_url=METABASE_URL,
            api_key=METABASE_API_KEY,
            username=METABASE_USER,
            password=METABASE_PASS
        )

        # SQL Logic as requested
        query = """
        SELECT 
            business_date, 
            SUM(total_amount) AS sales 
        FROM fact_orders 
        GROUP BY business_date 
        ORDER BY business_date;
        """

        # 1. Create Question
        card = client.create_line_chart(
            database_id=TARGET_DB_ID,
            collection_id=TARGET_COLLECTION_ID,
            sql_query=query,
            title="Daily Sales Trend (Automated)"
        )
        print(f"✨ Card Created Successfully: {METABASE_URL}/question/{card['id']}")

        # 2. Add to Dashboard
        client.add_card_to_dashboard(
            dashboard_id=TARGET_DASHBOARD_ID,
            card_id=card['id']
        )
        print(f"🚀 Card Added to Dashboard: {METABASE_URL}/dashboard/{TARGET_DASHBOARD_ID}")

    except Exception as e:
        logger.error(f"Process failed: {e}")
