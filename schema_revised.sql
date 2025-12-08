-- schema_revised.sql
-- 개선된 스키마 제안

-- 1. UUID 확장 기능 활성화 (PostgreSQL)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS fact_orders (
    order_guid UUID PRIMARY KEY, -- TEXT -> UUID 변경
    restaurant_guid UUID NOT NULL, -- TEXT -> UUID 변경
    business_date DATE NOT NULL,
    opened_at TIMESTAMPTZ,
    closed_at TIMESTAMPTZ,
    revenue_center_guid UUID, -- 이름 대신 ID로 참조 권장
    order_type TEXT,
    source TEXT,
    order_type_norm TEXT,
    platform TEXT,
    subtotal NUMERIC,
    discount_total NUMERIC,
    tax_total NUMERIC,
    service_charge NUMERIC,
    total_amount NUMERIC,
    customer_name TEXT,
    customer_phone TEXT,
    customer_email TEXT,
    delivery_address_json JSONB,
    raw_json JSONB NOT NULL
);

-- 차원 테이블들이 먼저 생성되어야 FK를 걸 수 있습니다.

CREATE TABLE IF NOT EXISTS dim_menu_items (
    menu_item_guid UUID PRIMARY KEY, -- TEXT -> UUID
    name TEXT,
    plu TEXT,
    external_id TEXT,
    category TEXT,
    price NUMERIC,
    raw_json JSONB
);

CREATE TABLE IF NOT EXISTS dim_revenue_centers (
    revenue_center_guid UUID PRIMARY KEY, -- TEXT -> UUID
    restaurant_guid UUID,
    name TEXT,
    external_id TEXT,
    raw_json JSONB
);

CREATE TABLE IF NOT EXISTS dim_tax_rates (
    tax_rate_guid UUID PRIMARY KEY, -- TEXT -> UUID
    name TEXT,
    rate NUMERIC,
    external_id TEXT,
    raw_json JSONB
);

CREATE TABLE IF NOT EXISTS fact_order_items (
    order_item_guid UUID PRIMARY KEY, -- id -> guid 로 명칭 통일
    order_guid UUID NOT NULL REFERENCES fact_orders(order_guid),
    menu_item_guid UUID REFERENCES dim_menu_items(menu_item_guid), -- 메뉴 차원과 연결
    menu_name TEXT, -- 이력 관리를 위해 이름 남겨두는 것은 OK
    plu TEXT,
    quantity NUMERIC NOT NULL DEFAULT 1, -- NOT NULL 추가
    unit_price NUMERIC,
    total_line_amount NUMERIC,
    modifiers_json JSONB
);

CREATE TABLE IF NOT EXISTS fact_taxes (
    tax_guid UUID PRIMARY KEY, -- id -> guid 통일
    order_guid UUID NOT NULL REFERENCES fact_orders(order_guid),
    tax_rate_guid UUID REFERENCES dim_tax_rates(tax_rate_guid), -- FK 추가
    amount NUMERIC NOT NULL DEFAULT 0, -- NOT NULL 추가
    raw_json JSONB
);

-- 인덱스 추가
CREATE INDEX IF NOT EXISTS idx_fact_orders_business_date ON fact_orders (business_date);
CREATE INDEX IF NOT EXISTS idx_fact_orders_restaurant_guid ON fact_orders (restaurant_guid);
CREATE INDEX IF NOT EXISTS idx_fact_order_items_order_guid ON fact_order_items (order_guid);
CREATE INDEX IF NOT EXISTS idx_fact_taxes_order_guid ON fact_taxes (order_guid); -- 누락된 인덱스 추가
CREATE INDEX IF NOT EXISTS idx_fact_taxes_tax_rate_guid ON fact_taxes (tax_rate_guid); -- 조인 성능 향상
