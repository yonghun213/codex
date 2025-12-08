"""load_orders.py
v5.3: Refund Amount Added & Revenue Center Logic Hardened
"""
from __future__ import annotations
import argparse, json, logging, re, sys
from configparser import ConfigParser
from datetime import date, datetime, timedelta
from pathlib import Path
import requests
from sqlalchemy import (
    Column, Date, DateTime, MetaData, Numeric, String, Table, create_engine, text, inspect, Boolean
)
from sqlalchemy.dialects.postgresql import JSONB, insert as pg_insert
from sqlalchemy.orm import Session
from toast_client1 import ToastClient

def configure_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("toast_etl")
    logger.setLevel(logging.INFO)
    if logger.handlers: logger.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(h)
    return logger

def load_config(path: Path) -> ConfigParser:
    p = ConfigParser()
    try: p.read_string(path.read_text(encoding="utf-8"))
    except: 
        raw = path.read_text(encoding="utf-8")
        p.read_string("\n".join([f"; {l}" if l.strip() and "=" not in l and not l.strip().startswith(("[",";")) else l for l in raw.splitlines()]))
    return p

def ensure_schema(engine, logger):
    inspector = inspect(engine)
    required = [
        ("fact_orders", "server_name", "TEXT"), ("fact_orders", "platform", "TEXT"),
        ("fact_orders", "voided", "BOOLEAN"), ("fact_orders", "refund_amount", "NUMERIC"), # [New]
        ("fact_payments", "tip_amount", "NUMERIC"),
        ("dim_dining_options", "dining_option_guid", "TEXT"),
    ]
    with engine.begin() as conn:
        for tbl, col, typ in required:
            if inspector.has_table(tbl) and col not in [c["name"] for c in inspector.get_columns(tbl)]:
                try: conn.execute(text(f"ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS {col} {typ}"))
                except: pass

metadata = MetaData()

fact_orders = Table("fact_orders", metadata,
    Column("order_guid", String, primary_key=True),
    Column("restaurant_guid", String, nullable=False),
    Column("business_date", Date, nullable=False),
    Column("opened_at", DateTime(timezone=True)),
    Column("closed_at", DateTime(timezone=True)),
    Column("revenue_center", String),
    Column("order_type", String),
    Column("source", String),
    Column("order_type_norm", String),
    Column("platform", String),
    Column("server_name", String),
    Column("server_guid", String),
    Column("subtotal", Numeric),
    Column("discount_total", Numeric),
    Column("tax_total", Numeric),
    Column("service_charge", Numeric),
    Column("total_amount", Numeric),
    Column("voided", Boolean),
    Column("refund_amount", Numeric), # [New]
    Column("customer_name", String),
    Column("customer_phone", String),
    Column("customer_email", String),
    Column("delivery_address_json", JSONB),
    Column("raw_json", JSONB, nullable=False)
)
# (Other tables omitted for brevity, assume standard definitions as before)
fact_order_items = Table("fact_order_items", metadata, Column("order_item_id", String, primary_key=True), Column("order_guid", String, nullable=False), Column("menu_name", String), Column("plu", String), Column("quantity", Numeric), Column("unit_price", Numeric), Column("total_line_amount", Numeric), Column("modifiers_json", JSONB))
fact_payments = Table("fact_payments", metadata, Column("payment_id", String, primary_key=True), Column("order_guid", String, nullable=False), Column("check_guid", String), Column("payment_type", String), Column("card_entry_mode", String), Column("amount", Numeric), Column("tip_amount", Numeric), Column("amount_tendered", Numeric), Column("card_type", String), Column("last_4_digits", String), Column("paid_business_date", Date), Column("raw_json", JSONB))
fact_discounts = Table("fact_discounts", metadata, Column("discount_id", String, primary_key=True), Column("order_guid", String, nullable=False), Column("check_guid", String), Column("name", String), Column("amount", Numeric), Column("approver_name", String), Column("reason", String), Column("raw_json", JSONB))
dim_menu_items = Table("dim_menu_items", metadata, Column("menu_item_guid", String, primary_key=True), Column("name", String), Column("plu", String), Column("external_id", String), Column("category", String), Column("price", Numeric), Column("raw_json", JSONB))
dim_revenue_centers = Table("dim_revenue_centers", metadata, Column("revenue_center_guid", String, primary_key=True), Column("restaurant_guid", String, nullable=True), Column("name", String), Column("external_id", String), Column("raw_json", JSONB))
dim_tax_rates = Table("dim_tax_rates", metadata, Column("tax_rate_guid", String, primary_key=True), Column("name", String), Column("rate", Numeric), Column("external_id", String), Column("raw_json", JSONB))
fact_taxes = Table("fact_taxes", metadata, Column("tax_id", String, primary_key=True), Column("order_guid", String, nullable=False), Column("tax_rate_guid", String), Column("amount", Numeric), Column("raw_json", JSONB))
dim_dining_options = Table("dim_dining_options", metadata, Column("dining_option_guid", String, primary_key=True), Column("restaurant_guid", String), Column("name", String), Column("behavior", String), Column("external_id", String), Column("raw_json", JSONB))

def parse_order(order_json: dict, dining_opt_map: dict, *, fallback_restaurant_guid: str = None):
    def _val(x): return x
    def _num(x): 
        try: return float(x) if x is not None else None
        except: return None
    def _date(x):
        if not x: return None
        s = str(x).strip()
        if re.fullmatch(r"\d{8}", s): return date(int(s[:4]), int(s[4:6]), int(s[6:]))
        try: return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
        except: return None
    def _dt(x):
        try: return datetime.fromisoformat(str(x).strip().replace("Z", "+00:00"))
        except: return None
    def _entity(e):
        if isinstance(e, dict):
            guid = e.get("guid") or e.get("id")
            return guid, e.get("displayName") or e.get("name") or guid, e.get("externalId"), e
        return None, None, None, None

    order_guid = order_json.get("guid") or order_json.get("orderGuid")
    if not order_guid: return {}, [], [], [], [], [], [], []

    restaurant_guid = order_json.get("restaurantGuid") or fallback_restaurant_guid
    checks = order_json.get("checks", []) 
    if not isinstance(checks, list): checks = []

    is_voided = order_json.get("voided") is True
    
    # [New] Refund Amount Parsing
    # Toast usually doesn't have a top-level refund field in this endpoint, 
    # but we can try estimating from payments or raw amounts if available.
    # Safe default:
    refund_amount = _num(order_json.get("amounts", {}).get("refundAmount")) or 0.0
    
    # [Improved] If refundAmount is 0, try to estimate from negative payments
    if refund_amount == 0 and checks:
        for check in checks:
            if not isinstance(check, dict): continue
            for p in check.get("payments", []):
                if not isinstance(p, dict): continue
                amt = _num(p.get("amount")) or 0.0
                if amt < 0:
                    refund_amount += abs(amt)

    cust_info = {}
    server_guid, server_name = None, None
    if checks and isinstance(checks[0], dict):
        c = checks[0].get("customer")
        if isinstance(c, dict): cust_info = c
        u = checks[0].get("createdUser") or checks[0].get("user")
        if isinstance(u, dict):
            server_guid, server_name = u.get("guid"), f"{u.get('firstName','')} {u.get('lastName','')}".strip()

    raw_amts = order_json.get("amounts", {})
    subtotal = _num(raw_amts.get("subtotal")) or _num(order_json.get("totalBeforeTax"))
    discount = _num(raw_amts.get("discount")) or 0.0
    tax = _num(raw_amts.get("tax")) or _num(order_json.get("taxAmount"))
    service = _num(raw_amts.get("serviceCharge")) or _num(order_json.get("serviceChargeAmount"))
    total = _num(raw_amts.get("total")) or _num(order_json.get("totalAmount"))

    dining_opt_obj = order_json.get("diningOption")
    dining_opt_guid = None
    dining_label = "unknown"
    if isinstance(dining_opt_obj, dict):
        dining_opt_guid = dining_opt_obj.get("guid")
        dining_label = str(dining_opt_obj.get("behavior") or "").lower()
    
    source_obj = order_json.get("source")
    source_label = str(source_obj.get("source") if isinstance(source_obj, dict) else source_obj or "").lower()
    config_name = dining_opt_map.get(dining_opt_guid, "").lower() if dining_opt_guid else ""
    tab_name = (checks[0].get("tabName") or "").lower() if checks and isinstance(checks[0], dict) else ""

    platform = "other"
    order_type_norm = "other"
    combined_info = f"{config_name} {tab_name} {source_label}"
    
    if "fantuan" in combined_info: platform = "fantuan"
    elif "uber" in combined_info: platform = "uber"
    elif "skip" in combined_info: platform = "skip"
    elif "door" in combined_info and "dash" in combined_info: platform = "doordash"
    elif "kiosk" in combined_info: platform = "kiosk"
    
    if platform in ["fantuan", "uber", "skip", "doordash"]: order_type_norm = "delivery"
    elif "dine" in dining_label or "dine" in combined_info: order_type_norm = "dinein"
    elif any(x in combined_info for x in ["take", "pick", "walk", "phone"]): order_type_norm = "takeout"
    elif "in store" in source_label: order_type_norm = "takeout"
    elif "deliver" in combined_info: order_type_norm = "delivery"

    final_label = dining_opt_map.get(dining_opt_guid) or (platform.capitalize() if platform != "other" else "Takeout" if order_type_norm == "takeout" else "Other")

    menu_dim, rev_dim, tax_dim = {}, {}, {}
    items, payments, discounts, tax_facts = [], [], [], []

    rc_guid, rc_name, rc_ext, rc_raw = _entity(order_json.get("revenueCenter"))
    if rc_guid: rev_dim[rc_guid] = {"revenue_center_guid": rc_guid, "restaurant_guid": restaurant_guid, "name": rc_name, "external_id": rc_ext, "raw_json": rc_raw}

    def _proc_tax(lst, p_guid):
        if not isinstance(lst, list): return
        for i, t in enumerate(lst):
            if not isinstance(t, dict): continue
            r_guid, r_name, r_ext, r_raw = _entity(t.get("taxRate"))
            if not r_guid: r_guid = t.get("taxRateGuid") or f"rate-{p_guid}-{i}"
            val = _num(t.get("rate") or t.get("percentage"))
            if val and val > 1: val /= 100.0
            if r_guid not in tax_dim: tax_dim[r_guid] = {"tax_rate_guid": r_guid, "name": r_name, "rate": val, "external_id": r_ext, "raw_json": r_raw or t}
            tax_facts.append({"tax_id": t.get("guid") or f"tax-{p_guid}-{len(tax_facts)}", "order_guid": order_guid, "tax_rate_guid": r_guid, "amount": _num(t.get("amount") or t.get("value")), "raw_json": t})

    item_totals = []
    for check in checks:
        if not isinstance(check, dict): continue
        c_guid = check.get("guid")
        for p in check.get("payments", []):
            if not isinstance(p, dict): continue
            payments.append({"payment_id": p.get("guid") or f"pay-{c_guid}-{len(payments)}", "order_guid": order_guid, "check_guid": c_guid, "payment_type": p.get("type"), "card_entry_mode": p.get("cardEntryMode"), "amount": _num(p.get("amount")), "tip_amount": _num(p.get("tipAmount")), "amount_tendered": _num(p.get("amountTendered")), "card_type": p.get("cardType"), "last_4_digits": p.get("last4Digits"), "paid_business_date": _date(p.get("paidBusinessDate")), "raw_json": p})
        for d in check.get("appliedDiscounts", []):
            if not isinstance(d, dict): continue
            approver_obj = d.get("approver") or {}
            discounts.append({"discount_id": d.get("guid") or f"disc-{c_guid}-{len(discounts)}", "order_guid": order_guid, "check_guid": c_guid, "name": d.get("name"), "amount": _num(d.get("discountAmount")), "approver_name": approver_obj.get("name"), "reason": d.get("processingState"), "raw_json": d})
        for item in (check.get("orderedItems") or check.get("selections") or []):
            if not isinstance(item, dict): continue
            m_guid, m_name, m_ext, m_raw = _entity(item.get("menuItem"))
            if not m_guid: m_guid = item.get("menuItemGuid") or f"menu-{len(menu_dim)}"
            if m_guid not in menu_dim: 
                cat_obj = m_raw.get("salesCategory") or {} if isinstance(m_raw, dict) else {}
                menu_dim[m_guid] = {"menu_item_guid": m_guid, "name": item.get("displayName") or m_name, "plu": item.get("plu"), "external_id": m_ext, "category": cat_obj.get("name"), "price": _num(item.get("unitPrice")), "raw_json": m_raw or item}
            qty, unit_p = _num(item.get("quantity")) or 1.0, _num(item.get("unitPrice"))
            tot = _num(item.get("price")) or _num(item.get("totalAmount"))
            if tot is None and unit_p: tot = unit_p * qty
            if tot: item_totals.append(tot)
            i_guid = item.get("guid") or f"item-{c_guid}-{len(items)}"
            items.append({"order_item_id": i_guid, "order_guid": order_guid, "menu_name": item.get("displayName") or m_name, "plu": item.get("plu"), "quantity": qty, "unit_price": unit_p, "total_line_amount": tot, "modifiers_json": item.get("modifiers")})
            _proc_tax(item.get("taxes"), i_guid); _proc_tax(item.get("appliedTaxes"), i_guid)
        _proc_tax(check.get("taxes"), c_guid); _proc_tax(check.get("appliedTaxes"), c_guid)
    _proc_tax(order_json.get("taxes"), order_guid); _proc_tax(order_json.get("appliedTaxes"), order_guid)

    if subtotal is None and item_totals: subtotal = sum(item_totals)
    if total is None and subtotal is not None: total = (subtotal - discount) + (tax or 0) + (service or 0)
    del_info = order_json.get("delivery") or {}
    
    order_rec = {
        "order_guid": order_guid, "restaurant_guid": restaurant_guid, "business_date": _date(order_json.get("businessDate")),
        "opened_at": _dt(order_json.get("openedDate")), "closed_at": _dt(order_json.get("closedDate")),
        "revenue_center": rc_name or rc_guid, "order_type": final_label, "source": source_label,
        "order_type_norm": order_type_norm, "platform": platform, "server_name": server_name, "server_guid": server_guid,
        "subtotal": subtotal, "discount_total": discount, "tax_total": tax, "service_charge": service, "total_amount": total,
        "voided": is_voided, "refund_amount": refund_amount, # [Updated]
        "customer_name": cust_info.get("firstName") or cust_info.get("name"), "customer_phone": cust_info.get("phone"), "customer_email": cust_info.get("email"),
        "delivery_address_json": del_info.get("address"), "raw_json": order_json
    }
    return order_rec, items, list(menu_dim.values()), list(rev_dim.values()), list(tax_dim.values()), tax_facts, payments, discounts

def upsert_records(session, order, items, menus, revs, rates, taxes, payments, discounts, dining_opts, logger):
    def _up(tbl, data, pk):
        if not data: return
        stmt = pg_insert(tbl).values(data)
        excl = {c.name: stmt.excluded[c.name] for c in tbl.columns if c.name != pk}
        session.execute(stmt.on_conflict_do_update(index_elements=[pk], set_=excl))
    _up(dim_dining_options, dining_opts, "dining_option_guid")
    _up(dim_menu_items, menus, "menu_item_guid")
    _up(dim_revenue_centers, [r for r in revs if r.get("restaurant_guid")], "revenue_center_guid")
    _up(dim_tax_rates, rates, "tax_rate_guid")
    _up(fact_orders, [order], "order_guid")
    _up(fact_order_items, items, "order_item_id")
    _up(fact_payments, payments, "payment_id")
    _up(fact_discounts, discounts, "discount_id")
    _up(fact_taxes, taxes, "tax_id")

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path"); parser.add_argument("--date"); parser.add_argument("--start"); parser.add_argument("--end")
    args = parser.parse_args(argv[1:])
    path = Path(args.config_path).resolve()
    config = load_config(path)
    logger = configure_logging(path.parent / "logs")
    def _d(s): return datetime.strptime(s, "%Y-%m-%d").date()
    dates = [_d(args.date)] if args.date else ([_d(args.start) + timedelta(n) for n in range((_d(args.end or args.start)-_d(args.start)).days+1)] if args.start else [datetime.now().date()+timedelta(int(config.get("orders","default_days_offset",fallback=-1)))])
    db_url = config.get("postgres", "db_url")
    engine = create_engine(db_url, future=True)
    try: metadata.create_all(engine); ensure_schema(engine, logger)
    except Exception as e: logger.error(e); return 1
    t_cfg = config["toast_api"]
    client = ToastClient(t_cfg.get("base_url"), t_cfg.get("auth_url"), t_cfg.get("client_id"), t_cfg.get("client_secret"), scopes=re.split(r"[\s,]+", t_cfg.get("scopes", "")), logger=logger)
    restaurants = []
    if config.has_section("restaurants"):
        for k, v in config.items("restaurants"):
            if v and not v.strip().startswith(";"):
                parts = v.split("|"); restaurants.append((k, parts[0].strip(), parts[1].strip() if len(parts)>1 else parts[0].strip()))
    with Session(engine) as session:
        for guid, alias in [(r[1], r[0]) for r in restaurants]:
            logger.info(f"Fetching Config for {alias}...")
            dining_opts_data = client.get_dining_options(guid)
            dining_opts_db = []
            dining_opt_map = {}
            for do in dining_opts_data:
                d_guid = do.get("guid"); d_name = do.get("name")
                if d_guid: dining_opt_map[d_guid] = d_name; dining_opts_db.append({"dining_option_guid": d_guid, "restaurant_guid": guid, "name": d_name, "behavior": do.get("behavior"), "external_id": do.get("externalId"), "raw_json": do})
            for d in dates:
                logger.info(f"Processing {alias} - {d}")
                try:
                    for o_json in client.get_orders_for_business_date(guid, d):
                        parsed = parse_order(o_json, dining_opt_map, fallback_restaurant_guid=guid)
                        if parsed[0]: upsert_records(session, *parsed, dining_opts_db, logger)
                    session.commit()
                except Exception as e: session.rollback(); logger.error(f"Error {alias}: {e}")
    logger.info("ETL Finished.")
    return 0

if __name__ == "__main__": sys.exit(main(sys.argv))
