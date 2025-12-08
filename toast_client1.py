"""toast_client.py
v2.3: Configuration API (Dining Options) 추가
"""
from __future__ import annotations
import logging, time, re
from datetime import date, datetime, timedelta
from typing import Any, Dict, Generator, Optional, Union, List
import requests

class ToastClient:
    _MAX_RETRIES: int = 5
    _BACKOFF_FACTOR: float = 1.5

    def __init__(self, base_url: str, auth_url: str, client_id: str, client_secret: str, *, scopes=None, session=None, logger=None):
        self.session = session or requests.Session()
        self.base_url = base_url.rstrip("/")
        self.auth_url = auth_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.logger = logger or logging.getLogger(__name__)
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        
        normalized_scopes = []
        for scope in scopes or []:
            token = (scope or "").strip()
            if token: normalized_scopes.append(token)
        self._scope_string = " ".join(normalized_scopes) if normalized_scopes else None
        self._orders_endpoint = f"{self.base_url}/orders/v2/ordersBulk"
        # [신규] Config Endpoint
        self._config_dining_opt_endpoint = f"{self.base_url}/config/v2/diningOptions"

    def authenticate(self) -> None:
        payload = {"clientId": self.client_id, "clientSecret": self.client_secret, "userAccessType": "TOAST_MACHINE_CLIENT"}
        if self._scope_string: payload["scope"] = self._scope_string
        try:
            resp = self.session.post(self.auth_url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            token_block = data.get("token") if isinstance(data.get("token"), dict) else {}
            self._access_token = data.get("accessToken") or token_block.get("accessToken")
            expires_in = data.get("expiresIn") or token_block.get("expiresIn") or 3600
            self._token_expiry = datetime.utcnow() + timedelta(seconds=int(expires_in) - 60)
            self.logger.info("토큰 갱신 완료")
        except Exception as e:
            self.logger.error(f"인증 실패: {e}")
            raise

    def _request(self, method: str, url: str, *, headers=None, params=None, json=None) -> Any:
        if not self._access_token or datetime.utcnow() >= (self._token_expiry or datetime.min):
            self.authenticate()
        
        req_headers = {"Authorization": f"Bearer {self._access_token}", "Content-Type": "application/json"}
        if headers: req_headers.update(headers)
        
        for attempt in range(1, self._MAX_RETRIES + 2):
            try:
                resp = self.session.request(method, url, headers=req_headers, params=params, json=json, timeout=120)
                if resp.status_code == 401 and attempt == 1:
                    self.authenticate()
                    req_headers["Authorization"] = f"Bearer {self._access_token}"
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                if attempt > self._MAX_RETRIES: raise
                time.sleep(self._BACKOFF_FACTOR ** attempt)

    def get_orders_for_business_date(self, restaurant_guid: str, business_date: Union[str, date], *, restaurant_external_id: str = None) -> Generator[Dict, None, None]:
        b_date_str = business_date.strftime("%Y%m%d") if isinstance(business_date, (date, datetime)) else str(business_date).replace("-", "")
        headers = {"Toast-Restaurant-External-ID": restaurant_external_id or restaurant_guid}
        
        page = 1
        page_token = None
        while True:
            params = {"businessDate": b_date_str, "pageSize": 100}
            if page_token: params["pageToken"] = page_token
            else: params["page"] = page

            data = self._request("GET", self._orders_endpoint, headers=headers, params=params)
            
            if isinstance(data, list):
                yield from data
                break
            elif isinstance(data, dict):
                orders = data.get("orders", [])
                yield from orders
                page_token = data.get("nextPageToken")
                if not page_token and not data.get("hasMore"): break
                if not page_token: page += 1
            else: break

    # [핵심] 이 함수가 꼭 있어야 합니다.
    def get_dining_options(self, restaurant_guid: str, restaurant_external_id: str = None) -> List[Dict]:
        """매장의 다이닝 옵션 설정(이름, 동작방식 등)을 가져옵니다."""
        headers = {"Toast-Restaurant-External-ID": restaurant_external_id or restaurant_guid}
        try:
            return self._request("GET", self._config_dining_opt_endpoint, headers=headers)
        except Exception as e:
            self.logger.warning(f"다이닝 옵션 가져오기 실패 ({restaurant_guid}): {e}")
            return []
