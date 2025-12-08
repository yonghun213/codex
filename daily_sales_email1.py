"""
daily_sales_email_final_v2.py
v2.3: Hotfix
- Fixed RegionAnalytics missing 'orders' key causing Jinja2 crash.
- Adjusted Gemini Exception handling.
"""
from __future__ import annotations

import argparse
import smtplib
import sys
import warnings
import logging
from logging.handlers import RotatingFileHandler
import calendar
import os
from configparser import ConfigParser
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from typing import Dict, Tuple, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine, text
from jinja2 import Environment, FileSystemLoader
import pytz

# AI Imports
try:
    import google.generativeai as genai
except ImportError:
    genai = None
try:
    import ollama
except ImportError:
    ollama = None

warnings.filterwarnings("ignore")

# --- Configuration & Logging ---

def configure_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bbq_report")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    
    file_handler = RotatingFileHandler(log_dir / "daily_report.log", maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

def load_config(path: Path) -> ConfigParser:
    parser = ConfigParser()
    try:
        parser.read_string(path.read_text(encoding="utf-8"))
    except Exception:
        raw = path.read_text(encoding="utf-8")
        safe_lines = []
        for line in raw.splitlines():
            if line.strip() and "=" not in line and not line.strip().startswith(("[", ";")):
                safe_lines.append(f"; {line}")
            else:
                safe_lines.append(line)
        parser.read_string("\n".join(safe_lines))
    return parser

# --- Logic Classes ---

class DateCalculator:
    def __init__(self, target_date: date):
        self.target = target_date

    def get_wtd_range(self) -> Tuple[date, date, date, date]:
        days_from_sunday = (self.target.weekday() + 1) % 7
        curr_start = self.target - timedelta(days=days_from_sunday)
        curr_end = self.target
        prev_start = curr_start - timedelta(days=7)
        prev_end = curr_end - timedelta(days=7)
        return curr_start, curr_end, prev_start, prev_end

    def get_mtd_range(self) -> Tuple[date, date, date, date]:
        curr_start = self.target.replace(day=1)
        first = self.target.replace(day=1)
        last_month_end = first - timedelta(days=1)
        prev_start = last_month_end.replace(day=1)
        target_day = self.target.day
        prev_month_days = calendar.monthrange(prev_start.year, prev_start.month)[1]
        real_day = min(target_day, prev_month_days)
        prev_end = prev_start.replace(day=real_day)
        return curr_start, self.target, prev_start, prev_end

    def get_month_info(self) -> Tuple[int, int]:
        total_days = calendar.monthrange(self.target.year, self.target.month)[1]
        return self.target.day, total_days

    def get_month_remaining_days(self) -> List[int]:
        last_day = calendar.monthrange(self.target.year, self.target.month)[1]
        remaining = []
        curr = self.target + timedelta(days=1)
        end = self.target.replace(day=last_day)
        while curr <= end:
            remaining.append(curr.weekday())
            curr += timedelta(days=1)
        return remaining

class DataFetcher:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url, future=True)

    def get_sales_sum(self, start: date, end: date) -> float:
        query = text("""
            SELECT SUM(subtotal - discount_total - COALESCE(refund_amount, 0))
            FROM fact_orders
            WHERE business_date::DATE BETWEEN :s AND :e
              AND (voided IS FALSE OR voided IS NULL)
        """)
        with self.engine.connect() as conn:
            res = conn.execute(query, {"s": start, "e": end}).scalar()
        return float(res) if res else 0.0

    def get_daily_trend(self, start: date, end: date) -> pd.DataFrame:
        query = text("""
            SELECT business_date,
                   SUM(subtotal - discount_total - COALESCE(refund_amount, 0)) AS sales
            FROM fact_orders
            WHERE business_date::DATE BETWEEN :s AND :e
              AND (voided IS FALSE OR voided IS NULL)
            GROUP BY business_date
            ORDER BY business_date
        """)
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"s": start, "e": end})
        return df

    def get_store_sales_sum(self, start: date, end: date, col_name: str) -> pd.DataFrame:
        query = text("""
            SELECT restaurant_guid,
                   SUM(subtotal - discount_total - COALESCE(refund_amount, 0)) AS val
            FROM fact_orders
            WHERE business_date::DATE BETWEEN :s AND :e
              AND (voided IS FALSE OR voided IS NULL)
            GROUP BY restaurant_guid
        """)
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"s": start, "e": end})
        df["val"] = pd.to_numeric(df["val"], errors="coerce").fillna(0.0)
        return df.rename(columns={"val": col_name})

    def get_detail_data(self, start: date, end: date) -> pd.DataFrame:
        query = text("""
            SELECT
                restaurant_guid,
                business_date,
                opened_at,
                (subtotal - discount_total - COALESCE(refund_amount, 0)) AS net_sales,
                total_amount,
                subtotal,
                discount_total,
                order_type_norm,
                platform,
                revenue_center,
                order_type
            FROM fact_orders
            WHERE business_date::DATE BETWEEN :s AND :e
              AND (voided IS FALSE OR voided IS NULL)
        """)
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"s": start, "e": end})

        num_cols = ["net_sales", "total_amount", "subtotal", "discount_total"]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        
        if "opened_at" in df.columns:
            df["opened_at"] = pd.to_datetime(df["opened_at"], errors="coerce")
            df["order_hour"] = df["opened_at"].dt.hour.fillna(18).astype(int)
        else:
            df["order_hour"] = 18

        return df

    # [Added] Anomaly Detection Logic
    def get_platform_alerts(self, curr_date: date) -> List[str]:
        # Compare last 4 weeks same weekday average vs current
        target_wd = curr_date.weekday()
        start = curr_date - timedelta(days=29)
        end = curr_date
        
        query = text("""
            SELECT business_date, platform, COUNT(*) as cnt
            FROM fact_orders
            WHERE business_date >= :s AND business_date <= :e
            AND platform IN ('uber', 'skip', 'doordash', 'fantuan')
            GROUP BY business_date, platform
        """)
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"s": start, "e": end})
        
        if df.empty: return []
        
        df["business_date"] = pd.to_datetime(df["business_date"]).dt.date
        df["wd"] = df["business_date"].apply(lambda d: d.weekday())
        
        alerts = []
        for p in ['uber', 'skip', 'doordash', 'fantuan']:
            # Filter for same weekday, excluding today
            hist = df[(df["platform"]==p) & (df["wd"]==target_wd) & (df["business_date"]!=curr_date)]
            curr = df[(df["platform"]==p) & (df["business_date"]==curr_date)]
            
            curr_cnt = curr["cnt"].sum() if not curr.empty else 0
            avg_cnt = hist["cnt"].mean() if not hist.empty else 0
            
            # Alert condition: if average > 10 and current is less than 20% of average (80% drop)
            if avg_cnt > 10 and curr_cnt < (avg_cnt * 0.2):
                alerts.append(f"🚨 {p.capitalize()} orders dropped significantly! (Avg: {avg_cnt:.1f} -> Today: {curr_cnt})")
            elif avg_cnt > 5 and curr_cnt == 0:
                alerts.append(f"🚨 {p.capitalize()} orders are ZERO! Check Integration.")
                
        return alerts

    # [Added] Menu Analysis
    def get_top_menu_items(self, start: date, end: date) -> pd.DataFrame:
        query = text("""
            SELECT menu_name, SUM(quantity) as qty, SUM(total_line_amount) as amt
            FROM fact_order_items
            WHERE order_guid IN (
                SELECT order_guid FROM fact_orders WHERE business_date BETWEEN :s AND :e
            )
            GROUP BY menu_name
            ORDER BY qty DESC
            LIMIT 5
        """)
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"s": start, "e": end})
        return df

class HybridAnalyst:
    def __init__(self, gemini_key: str = None, gemini_model: str = "gemini-1.5-flash", ollama_model: str = "llama3.2"):
        self.gemini_key = gemini_key
        self.gemini_model_name = gemini_model
        self.ollama_model = ollama_model
        self.active_source = "None"
        
        if self.gemini_key and genai:
            genai.configure(api_key=self.gemini_key)

    def generate_briefing(self, kpi, cum, top5, alerts, date_str) -> str:
        top_str = ", ".join([f"{r['name']} (${r['sales']:,.0f})" for _, r in top5.iterrows()])
        alert_str = ", ".join([f"{r['name']} (${r['sales']:,.0f})" for _, r in alerts.iterrows()]) if not alerts.empty else "None"
        
        dod_val = kpi['sales_dod']
        wow_val = kpi['sales_wow']
        
        dod_text = "상승(Increase)" if dod_val >= 0 else "하락(Decrease)"
        wow_text = "상승(Increase)" if wow_val >= 0 else "하락(Decrease)"
        
        prompt = f"""
        Role: Professional Business Data Analyst (Korean Language Expert)
        Date: {date_str}
        
        [INPUT DATA]
        1. Sales Trend:
           - Total: ${kpi['sales']:,.0f}
           - vs Yesterday: {dod_val:+.1f}% ({dod_text})
           - vs Last Week: {wow_val:+.1f}% ({wow_text})
        
        2. Top Stores (Names in English): {top_str}
        
        3. Alerts (Names in English): {alert_str}
        
        4. Forecast: Month progress {cum['month_progress']:.1f}%.
        
        [INSTRUCTIONS]
        - Write exactly 3 bullet points in **Korean (한국어)**.
        - **CRITICAL**: Do NOT use Japanese/Chinese characters. Use ONLY Korean.
        - **CRITICAL**: Keep Store Names in **ENGLISH**.
        - Provide brief, actionable insights.
        
        [OUTPUT FORMAT - HTML ONLY]
        <ul>
        <li><strong>매출 동향:</strong> ...content...</li>
        <li><strong>우수 매장:</strong> ...content...</li>
        <li><strong>특이 사항:</strong> ...content...</li>
        </ul>
        """
        
        if self.gemini_key and genai:
            try:
                logging.info(f"Attempting analysis with Google Gemini ({self.gemini_model_name})...")
                model = genai.GenerativeModel(self.gemini_model_name)
                response = model.generate_content(prompt)
                self.active_source = "Gemini"
                return response.text.replace("```html", "").replace("```", "").strip()
            except Exception as e:
                logging.error(f"Gemini Failed: {e}. Switching to Ollama...")

        if ollama:
            try:
                logging.info(f"Attempting analysis with local Ollama ({self.ollama_model})...")
                response = ollama.chat(model=self.ollama_model, messages=[
                    {'role': 'user', 'content': prompt},
                ])
                content = response['message']['content']
                self.active_source = "Ollama"
                return content.replace("```html", "").replace("```", "").strip()
            except Exception as e:
                logging.error(f"Ollama Failed: {e}")
        
        self.active_source = "None"
        return "<ul><li>AI Analysis Failed (Both Gemini and Ollama unavailable)</li></ul>"

class ReportGenerator:
    def __init__(self, fetcher: DataFetcher, calc: DateCalculator,
                 guid_map: Dict[str, str], region_map: Dict[str, str]):
        self.fetcher = fetcher
        self.calc = calc
        self.guid_map = guid_map
        self.region_map = region_map

    @staticmethod
    def _safe_pct(curr: float, prev: float) -> float:
        if not prev: return 0.0
        try: return (curr - prev) / prev * 100.0
        except ZeroDivisionError: return 0.0

    def _categorize_channel(self, row) -> str:
        plat = str(row.get("platform", "")).lower()
        if plat in ["uber", "skip", "doordash", "fantuan"]: return "delivery"
        rc = str(row.get("revenue_center", "")).lower()
        if any(x in rc for x in ["din", "hall", "patio", "table"]): return "dinein"
        if "deliver" in rc: return "delivery"
        if any(x in rc for x in ["take", "pick", "go", "online", "web", "walk", "front", "counter"]): return "takeout"
        ot = str(row.get("order_type_norm", "")).lower()
        if "del" in ot: return "delivery"
        if "din" in ot: return "dinein"
        if "take" in ot: return "takeout"
        return "other"

    def _calculate_forecast(self, mtd_val: float) -> float:
        end_date = self.calc.target
        start_date = end_date - timedelta(days=27)
        df = self.fetcher.get_daily_trend(start_date, end_date)
        if df.empty: return mtd_val
        df["business_date"] = pd.to_datetime(df["business_date"])
        df["wd"] = df["business_date"].dt.weekday
        wd_avg = df.groupby("wd")["sales"].mean().to_dict()
        global_avg = df["sales"].mean()
        remaining_wds = self.calc.get_month_remaining_days()
        future = sum(wd_avg.get(wd, global_avg) for wd in remaining_wds)
        return mtd_val + future

    def _build_brand_momentum_28d(self, t: date) -> Dict[str, float]:
        curr_start = t - timedelta(days=27)
        prev_start = curr_start - timedelta(days=28)
        prev_end = curr_start - timedelta(days=1)
        df_curr = self.fetcher.get_detail_data(curr_start, t)
        df_prev = self.fetcher.get_detail_data(prev_start, prev_end)
        sales_curr = df_curr["net_sales"].sum() if not df_curr.empty else 0.0
        sales_prev = df_prev["net_sales"].sum() if not df_prev.empty else 0.0
        
        def channel_mix(df, total):
            if df.empty or not total: return {"delivery": 0.0, "takeout": 0.0, "dinein": 0.0}
            tmp = df.copy()
            tmp["final_channel"] = tmp.apply(self._categorize_channel, axis=1)
            mix = (tmp.groupby("final_channel")["net_sales"].sum() / total * 100).to_dict()
            return {k: float(mix.get(k, 0.0)) for k in ["delivery", "takeout", "dinein"]}

        mix_curr = channel_mix(df_curr, sales_curr)
        mix_prev = channel_mix(df_prev, sales_prev)
        return {
            "sales_curr": sales_curr, "sales_prev": sales_prev,
            "sales_pct": self._safe_pct(sales_curr, sales_prev),
            "mix_curr_delivery": mix_curr["delivery"], "mix_prev_delivery": mix_prev["delivery"],
            "mix_curr_takeout": mix_curr["takeout"], "mix_prev_takeout": mix_prev["takeout"],
            "mix_curr_dinein": mix_curr["dinein"], "mix_prev_dinein": mix_prev["dinein"],
        }

    def _build_forward_outlook_7d(self, t: date) -> Dict:
        hist_start = t - timedelta(days=55)
        df = self.fetcher.get_daily_trend(hist_start, t)
        if df.empty: return {"days": [], "total": 0.0}
        df["business_date"] = pd.to_datetime(df["business_date"])
        df["wd"] = df["business_date"].dt.weekday
        wd_avg = df.groupby("wd")["sales"].mean().to_dict()
        global_avg = df["sales"].mean()
        outlook_days = []
        total = 0.0
        dow_map = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for i in range(1, 8):
            d = t + timedelta(days=i)
            pred = float(wd_avg.get(d.weekday(), global_avg))
            total += pred
            outlook_days.append({"date": d, "dow": dow_map[d.weekday()], "forecast": pred})
        return {"days": outlook_days, "total": total}

    def generate_full_report(self):
        t = self.calc.target
        ws, we, wps, wpe = self.calc.get_wtd_range()
        ms, me, mps, mpe = self.calc.get_mtd_range()
        
        df_today = self.fetcher.get_detail_data(t, t)
        df_yest = self.fetcher.get_detail_data(t - timedelta(days=1), t - timedelta(days=1))
        df_week = self.fetcher.get_detail_data(t - timedelta(days=7), t - timedelta(days=7))
        
        if not df_today.empty:
            df_today = df_today.copy()
            df_today["final_channel"] = df_today.apply(self._categorize_channel, axis=1)
        else:
            df_today["final_channel"] = "other"

        wtd_curr = self.fetcher.get_sales_sum(ws, we); wtd_prev = self.fetcher.get_sales_sum(wps, wpe)
        mtd_curr = self.fetcher.get_sales_sum(ms, me); mtd_prev = self.fetcher.get_sales_sum(mps, mpe)
        elapsed, total_days = self.calc.get_month_info()
        progress_pct = (elapsed / total_days) * 100.0 if total_days else 0.0
        
        sales = df_today["net_sales"].sum()
        sales_y = df_yest["net_sales"].sum()
        sales_w = df_week["net_sales"].sum()
        
        kpi = {
            "sales": sales, "sales_dod": self._safe_pct(sales, sales_y), "sales_wow": self._safe_pct(sales, sales_w),
            "orders": len(df_today), "avg_chk": sales/len(df_today) if len(df_today) else 0
        }
        
        cum = {
            "wtd_curr_range": f"{ws:%m/%d}~{we:%m/%d}", "wtd_prev_range": f"{wps:%m/%d}~{wpe:%m/%d}",
            "wtd_curr_val": wtd_curr, "wtd_prev_val": wtd_prev, "wtd_gap": wtd_curr - wtd_prev, "wtd_p": self._safe_pct(wtd_curr, wtd_prev),
            "mtd_curr_range": f"{ms:%m/%d}~{me:%m/%d}", "mtd_prev_range": f"{mps:%m/%d}~{mpe:%m/%d}",
            "mtd_curr_val": mtd_curr, "mtd_prev_val": mtd_prev, "mtd_gap": mtd_curr - mtd_prev, "mtd_p": self._safe_pct(mtd_curr, mtd_prev),
            "forecast": self._calculate_forecast(mtd_curr), "month_progress": progress_pct
        }

        # [Restoration] Brand Momentum & Forward Outlook
        brand28 = self._build_brand_momentum_28d(t); cum.update({f"brand28_{k}": v for k, v in brand28.items()})
        b28_curr_s = t - timedelta(days=27); b28_prev_s = t - timedelta(days=55); b28_prev_e = t - timedelta(days=28)
        cum["brand28_curr_range"] = f"{b28_curr_s:%m/%d}~{t:%m/%d}"
        cum["brand28_prev_range"] = f"{b28_prev_s:%m/%d}~{b28_prev_e:%m/%d}"
        
        forward7 = self._build_forward_outlook_7d(t); cum.update({f"forward7_{k}": v for k, v in forward7.items()})

        # [Restoration] Mix Data
        if sales:
            t_mix = (df_today.groupby("final_channel")["net_sales"].sum() / sales * 100).to_dict()
            p_mix = (df_today.groupby("platform")["net_sales"].sum() / sales * 100).to_dict()
        else: t_mix = {}; p_mix = {}

        rank_data = self._get_rankings(df_today, df_week, 
                                       self.fetcher.get_store_sales_sum(ws, we, "sales_wtd"), 
                                       self.fetcher.get_store_sales_sum(ms, me, "sales_mtd"), 
                                       progress_pct)
        
        return kpi, cum, t_mix, p_mix, rank_data, df_today

    def _get_rankings(self, today, week, wtd_df, mtd_df, progress_pct):
        base = pd.DataFrame(list(self.guid_map.items()), columns=["restaurant_guid", "name"])
        base["region"] = base["restaurant_guid"].map(lambda x: self.region_map.get(x, "Other"))
        
        if today.empty:
            m = base.copy()
            for c in ["sales", "wow_pct", "delivery_pct", "sales_wtd", "sales_mtd", "sales_eom"]:
                m[c] = 0.0
            m["health_grade"] = "C"
            return m, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
        t = today.groupby("restaurant_guid")["net_sales"].sum().reset_index(name="sales")
        w = week.groupby("restaurant_guid")["net_sales"].sum().reset_index(name="sales_week")
        
        m = base.merge(t, on="restaurant_guid", how="left") \
                .merge(w, on="restaurant_guid", how="left") \
                .merge(wtd_df, on="restaurant_guid", how="left") \
                .merge(mtd_df, on="restaurant_guid", how="left") \
                .fillna(0)
        
        m["sales_eom"] = m.apply(lambda r: (r["sales_mtd"] / (progress_pct/100.0)) if progress_pct > 0 else 0, axis=1)
        m["wow_pct"] = m.apply(lambda r: self._safe_pct(r["sales"], r["sales_week"]), axis=1)
        
        c = today.groupby(["restaurant_guid", "final_channel"])["net_sales"].sum().unstack(fill_value=0)
        p = today.groupby(["restaurant_guid", "platform"])["net_sales"].sum().unstack(fill_value=0)
        m = m.merge(c, on="restaurant_guid", how="left").merge(p, on="restaurant_guid", how="left").fillna(0)
        
        for col in ["delivery", "takeout", "dinein", "uber", "skip", "doordash", "fantuan"]:
            if col not in m.columns: m[col] = 0.0
            m[f"{col}_pct"] = m.apply(lambda r: (r[col]/r["sales"]*100) if r["sales"]>0 else 0, axis=1)

        chain_wow = float(m["wow_pct"].mean()) if len(m) else 0.0
        def score(r):
            s = 0.0
            if r["wow_pct"] > 10: s += 1.0
            elif r["wow_pct"] < -10: s -= 1.0
            if r["wow_pct"] > chain_wow + 5: s += 0.5
            elif r["wow_pct"] < chain_wow - 5: s -= 0.5
            return s
        m["health_score"] = m.apply(score, axis=1)
        m["health_grade"] = m["health_score"].apply(lambda s: "A" if s>=1.5 else ("B" if s>=0.5 else ("C" if s>=-0.5 else "D")))
        m["name"] = m["name"].str.title()
        
        return m, m.sort_values("sales", ascending=False).head(5), m.sort_values("sales", ascending=True).head(5), m[(m["sales"]==0)|(m["wow_pct"]<-20.0)]

class ChartGenerator:
    def __init__(self, fetcher, calc, region_map):
        self.fetcher = fetcher; self.calc = calc; self.region_map = region_map
        plt.style.use('ggplot')
        plt.rcParams.update({'font.size': 9, 'font.family': 'sans-serif'})

    def _fig_to_bytes(self, fig):
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    def generate_charts(self, df_today, top_menu_df) -> Dict[str, bytes]:
        return {
            "mtd_trend": self._mtd_sales_trend(),
            "region_peak_profile": self._region_peak_profile(df_today),
            "region_avg_ticket_orders": self._region_avg_ticket_orders(df_today),
            "top_menu_chart": self._top_menu_chart(top_menu_df)
        }
    
    def _top_menu_chart(self, df) -> bytes:
        if df.empty: return b""
        fig, ax = plt.subplots(figsize=(8, 3))
        # Shorten names
        names = [n[:15]+".." if len(n)>15 else n for n in df["menu_name"]]
        ax.barh(names, df["qty"], color="#ffb74d")
        ax.set_title("Top 5 Menu Items (Qty)")
        ax.invert_yaxis()
        return self._fig_to_bytes(fig)

    def _mtd_sales_trend(self) -> bytes:
        t = self.calc.target; ms, me, mps, mpe = self.calc.get_mtd_range()
        df_curr = self.fetcher.get_daily_trend(ms, t)
        df_prev = self.fetcher.get_daily_trend(mps, mpe)
        
        s_c = df_curr.set_index("business_date")["sales"] if not df_curr.empty else pd.Series()
        s_p = df_prev.set_index("business_date")["sales"] if not df_prev.empty else pd.Series()
        
        days = range(1, t.day + 1)
        curr_vals = [s_c.get(t.replace(day=d), 0) for d in days]
        prev_vals = []
        for d in days:
            try: prev_vals.append(s_p.iloc[d-1] if len(s_p) >= d else 0)
            except: prev_vals.append(0)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(days, curr_vals, marker="o", linewidth=2, label="This Month", color='#d32f2f')
        ax.plot(days, prev_vals, linestyle="--", alpha=0.6, label="Last Month")
        ax.set_title("MTD Sales Trend")
        ax.legend()
        return self._fig_to_bytes(fig)

    def _region_peak_profile(self, df) -> bytes:
        if df.empty: return b""
        df = df.copy()
        df["region"] = df["restaurant_guid"].map(lambda x: self.region_map.get(x, "Other"))
        hourly = df.groupby(["region", "order_hour"])["net_sales"].sum().reset_index()
        
        fig, ax = plt.subplots(figsize=(8, 3))
        for r in hourly["region"].unique():
            if r == "Other": continue
            sub = hourly[hourly["region"] == r].sort_values("order_hour")
            ax.plot(sub["order_hour"], sub["net_sales"], marker="o", label=r)
        ax.set_title("Hourly Sales by Region")
        ax.legend()
        return self._fig_to_bytes(fig)

    def _region_avg_ticket_orders(self, df) -> bytes:
        if df.empty: return b""
        df = df.copy()
        df["region"] = df["restaurant_guid"].map(lambda x: self.region_map.get(x, "Other"))
        agg = df.groupby("region").agg(sales=("net_sales", "sum"), orders=("restaurant_guid", "count")).reset_index()
        agg["avg_ticket"] = agg["sales"] / agg["orders"]
        
        fig, ax = plt.subplots(figsize=(8, 3))
        for _, r in agg.iterrows():
            if r["region"] == "Other": continue
            ax.scatter(r["orders"], r["avg_ticket"], s=100, label=r["region"])
            ax.text(r["orders"], r["avg_ticket"], f" {r['region']}")
        ax.set_xlabel("Orders"); ax.set_ylabel("Avg Ticket")
        ax.set_title("Region Positioning")
        return self._fig_to_bytes(fig)

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--date")
    args = parser.parse_args()
    
    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    logger = configure_logging(cfg_path.parent / "logs")
    
    gemini_key = os.getenv("GEMINI_API_KEY") or cfg.get("ai", "gemini_key", fallback=None)
    if not gemini_key:
        logger.warning("No Gemini API Key found in environment or config.")

    tz = pytz.timezone('America/Vancouver')
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        target_date = datetime.now(tz).date() - timedelta(days=1)
        
    logger.info(f"🚀 Starting Report for Business Date: {target_date}")

    guid_map = {}
    region_map = {}
    
    if cfg.has_section("restaurants"):
        for k, v in cfg.items("restaurants"):
            if v and not v.strip().startswith(";"):
                guid = v.split("|")[0].strip()
                guid_map[guid] = k
    
    if cfg.has_section("regions"):
        for region_name, keywords in cfg.items("regions"):
            kws = [x.strip().lower() for x in keywords.split(",")]
            for guid, alias in guid_map.items():
                if any(kw in alias.lower() for kw in kws):
                    region_map[guid] = region_name.upper()

    db_url = cfg.get("postgres", "db_url")
    calc = DateCalculator(target_date)
    fetcher = DataFetcher(db_url)
    
    gen = ReportGenerator(fetcher, calc, guid_map, region_map)
    # [Fix] Added t_mix, p_mix back to return values
    kpi, cum, t_mix, p_mix, rank_data, df_today = gen.generate_full_report()
    full_df, top5, bot5, alerts = rank_data
    
    # [New] Fetch Platform Alerts & Menu Data
    plat_alerts = fetcher.get_platform_alerts(target_date)
    top_menu_df = fetcher.get_top_menu_items(target_date, target_date)

    chart_gen = ChartGenerator(fetcher, calc, region_map)
    charts = chart_gen.generate_charts(df_today, top_menu_df)
    
    region_profiles = {}
    if not df_today.empty:
        temp_df = df_today.copy()
        temp_df["region"] = temp_df["restaurant_guid"].map(lambda x: region_map.get(x, "Other"))
        grp = temp_df.groupby("region").agg(sales=("net_sales","sum"), orders=("restaurant_guid","count"))
        for r, row in grp.iterrows():
            if r == "Other": continue
            peak_h = 18
            try:
                sub = temp_df[temp_df["region"]==r]
                if not sub.empty:
                    peak_h = sub.groupby("order_hour")["net_sales"].sum().idxmax()
            except: pass
            
            # [CRITICAL FIX] Added 'orders' key here to prevent Template Error
            region_profiles[r] = {
                "sales": row["sales"], 
                "orders": int(row["orders"]),
                "share": row["sales"]/kpi["sales"] if kpi["sales"] else 0,
                "avg_ticket": row["sales"]/row["orders"] if row["orders"] else 0,
                "peak_hour": int(peak_h)
            }

    gemini_model = cfg.get("ai", "gemini_model", fallback="gemini-1.5-flash")
    ollama_model = cfg.get("ai", "ollama_model", fallback="llama3.2")
    analyst = HybridAnalyst(gemini_key, gemini_model, ollama_model)
    ai_summary = analyst.generate_briefing(kpi, cum, top5, alerts, str(target_date))

    # [Fix] Prepare detailed lists for template
    rising_stars = full_df[full_df["health_grade"]=="A"].sort_values("sales", ascending=False).head(5)
    at_risk = full_df[full_df["health_grade"]=="D"].sort_values("sales", ascending=True).head(5)

    template_file = cfg.get("files", "template_file", fallback="report_template.html")
    env = Environment(loader=FileSystemLoader(str(Path(args.config).parent)))
    template = env.get_template(template_file)
    
    html_content = template.render(
        date_str=str(target_date),
        kpi=kpi,
        cum=cum,
        full_rank_data=full_df.to_dict(orient="records"),
        top_stores=top5.to_dict(orient="records"),
        bot_stores=bot5.to_dict(orient="records"),
        rising_stars=rising_stars.to_dict(orient="records"),
        at_risk=at_risk.to_dict(orient="records"),
        t_mix=t_mix,
        p_mix=p_mix,
        region_profiles=region_profiles,
        plat_alerts=plat_alerts, # [New]
        ai_summary=ai_summary,
        ai_model=analyst.active_source,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M")
    )

    ec = cfg["email"]
    msg = MIMEMultipart("related")
    msg["Subject"] = f"📈 [BBQ Brand Pulse] {target_date} Sales Report"
    msg["From"] = ec["from"]
    msg["To"] = ec["to"]
    
    msg_alt = MIMEMultipart("alternative")
    msg.attach(msg_alt)
    msg_alt.attach(MIMEText(html_content, "html", "utf-8"))
    
    for cid, img_bytes in charts.items():
        if img_bytes:
            img = MIMEImage(img_bytes)
            img.add_header('Content-ID', f'<{cid}>')
            img.add_header('Content-Disposition', 'inline', filename=f'{cid}.png')
            msg.attach(img)
            
    with smtplib.SMTP(ec["smtp_host"], int(ec["smtp_port"])) as s:
        s.starttls()
        s.login(ec["smtp_username"], ec["smtp_password"])
        s.send_message(msg)
        
    logger.info("✅ Report Sent Successfully.")

if __name__ == "__main__":
    main()
