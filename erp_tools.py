import os
import json
import re
from typing import Dict, Any, Optional, Tuple, List
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from openai import OpenAI

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
API_KEY = os.getenv("JARVIS_API_KEY")
BASE_URL = os.getenv("JARVIS_BASE_URL")
MODEL = os.getenv("JARVIS_MODEL", "glm-5.3-flash")

_engine = None
_openai_client: Optional[OpenAI] = None

def get_db_engine():
    global _engine
    if _engine is None:
        if not DATABASE_URL:
            raise ValueError("Missing DATABASE_URL in .env file!")
        _engine = create_engine(DATABASE_URL)
    return _engine

def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        if not API_KEY or not BASE_URL:
            raise ValueError("Missing Jarvis API credentials in .env")
        _openai_client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=30.0)
    return _openai_client

try:
    if API_KEY and BASE_URL:
        client: OpenAI = get_openai_client()
    else:
        client = None  # type: ignore
except Exception:
    client = None  # type: ignore

SCHEMA_SUMMARY = """
Database Schema Reference for Farm ERP:
1. users: id, uid, name, phone_number, village, state, country
2. farms: id, user_id, farm_name, address, state, district, village, pincode, no_of_animals, vet_name
3. cattles: id, farm_id, cattle_tag_no, name, breed, species, color, age, height, weight, milk_yield_per_day, lactation
4. bcs_tests: id, cattle_id, input_type, bcs_score, condition_status, body_condition_status, udder_score, teat_score, estimated_body_weight, daily_feed_intake, daily_water_intake
5. bcs_recommendations: id, test_id, title, description
6. pashux_cattle_analytics: id, cattle_id, camera_id, breed, gender, color, estimated_weight, height, bcs_score, health_status, live_ai_insights
7. cctv_cameras: id, farm_id, camera_name, shed_location, device_id
8. cctv_alerts: id, camera_id, cattle_id, alerts, severity, created_at
9. inventory_items: id, farm_id, item_name, category, quantity, unit, low_stock_level, stock_value
"""

VALID_TABLES = {
    "users", "farms", "cattles", "bcs_tests", "bcs_media",
    "bcs_recommendations", "bcs_suggested_products", "cctv_cameras",
    "pashux_cattle_analytics", "cctv_alerts", "inventory_items"
}

TABLE_ALIASES = {
    "farm": "farms", "user": "users", "cattle": "cattles", "cow": "cattles",
    "cows": "cattles", "animal": "cattles", "animals": "cattles", "livestock": "cattles",
    "camera": "cctv_cameras", "alert": "cctv_alerts", "inventory": "inventory_items",
    "analytics": "pashux_cattle_analytics", "bcs": "bcs_tests", "recommendation": "bcs_recommendations"
}

TABLE_COLUMNS: Dict[str, set] = {
    "users": {"id", "uid", "name", "phone_number", "village", "state", "country"},
    "farms": {"id", "user_id", "farm_name", "address", "state", "district", "village", "pincode", "no_of_animals", "vet_name"},
    "cattles": {"id", "farm_id", "cattle_tag_no", "name", "breed", "species", "color", "age", "height", "weight", "milk_yield_per_day", "lactation"},
    "bcs_tests": {"id", "cattle_id", "input_type", "bcs_score", "condition_status", "body_condition_status", "udder_score", "teat_score", "estimated_body_weight", "daily_feed_intake", "daily_water_intake"},
    "bcs_recommendations": {"id", "test_id", "title", "description"},
    "cctv_cameras": {"id", "farm_id", "camera_name", "shed_location", "device_id"},
    "cctv_alerts": {"id", "camera_id", "cattle_id", "alerts", "severity"},
    "pashux_cattle_analytics": {"id", "cattle_id", "camera_id", "breed", "gender", "color", "estimated_weight", "height", "bcs_score", "health_status", "live_ai_insights"},
    "inventory_items": {"id", "farm_id", "item_name", "category", "quantity", "unit", "low_stock_level", "stock_value"}
}

CATTLE_CHILD_TABLES = {"bcs_tests", "bcs_recommendations", "pashux_cattle_analytics", "cctv_alerts"}

def normalize_table_name(raw_table: Any) -> Optional[str]:
    if not raw_table or not isinstance(raw_table, str):
        return None
    cleaned = raw_table.strip().strip("'\"`").strip("/").lower()
    if "/" in cleaned or "\\" in cleaned or not re.match(r"^[a-z0-9_]+$", cleaned):
        return None
    if cleaned in TABLE_ALIASES:
        cleaned = TABLE_ALIASES[cleaned]
    if cleaned in VALID_TABLES:
        return cleaned
    return None

def extract_json_plan(content: str) -> Optional[Dict[str, Any]]:
    if not content:
        return None
    content = content.strip()
    code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if code_match:
        try:
            return json.loads(code_match.group(1))
        except Exception:
            pass
    bracket_match = re.search(r"\{.*?\}", content, re.DOTALL)
    if bracket_match:
        try:
            return json.loads(bracket_match.group(0))
        except Exception:
            pass
    try:
        return json.loads(content)
    except Exception:
        return None

def heuristic_plan_fallback(question: str) -> Dict[str, Any]:
    q = question.lower()
    tag_match = re.search(r"(?:tag\s*no\.?|tag|cow)\s*(?:#|no\.?)?\s*(\d+)", q)
    extracted_filters = {}
    if tag_match:
        extracted_filters["cattle_tag_no"] = tag_match.group(1)

    if any(k in q for k in ["bcs", "condition score", "weight", "body condition"]):
        return {"table": "bcs_tests", "filters": extracted_filters, "join_cattle": True}
    if any(k in q for k in ["recommend", "advice", "suggestion"]):
        return {"table": "bcs_recommendations", "filters": {}, "join_cattle": True}
    if any(k in q for k in ["analytic", "insight", "health status"]):
        return {"table": "pashux_cattle_analytics", "filters": extracted_filters, "join_cattle": True}
    if any(k in q for k in ["feed", "inventory", "stock"]):
        return {"table": "inventory_items", "filters": {}, "join_cattle": False}
    if any(k in q for k in ["camera", "cctv", "alert"]):
        return {"table": "cctv_alerts", "filters": {}, "join_cattle": False}
    return {"table": "cattles", "filters": extracted_filters, "join_cattle": False}

def sanitize_filters(table: str, filters: Any) -> Dict[str, Any]:
    if not isinstance(filters, dict):
        return {}
    allowed = TABLE_COLUMNS.get(table, set())
    safe_filters = {}
    for col, val in filters.items():
        clean_col = str(col).strip().lower()
        if clean_col in allowed and val is not None:
            if isinstance(val, (str, int, float, bool)):
                safe_filters[clean_col] = val
    return safe_filters

def execute_supabase_tool(question: str, farm_id: int = 1, dry_run: bool = False) -> str:
    is_dry_run = dry_run or os.getenv("DRY_RUN", "false").lower() in ("true", "1")

    try:
        farm_id = int(farm_id)
    except (ValueError, TypeError):
        farm_id = 1

    planner_prompt = f"""
You are an expert SQL and data retrieval assistant for a livestock dairy farm ERP.
Given the user question and the database schema below, determine what data needs to be fetched from PostgreSQL.

{SCHEMA_SUMMARY}

The current active farm ID is: {farm_id}

Respond ONLY with a valid JSON object specifying the retrieval plan:
{{
  "table": "one of [users, farms, cattles, bcs_tests, bcs_recommendations, pashux_cattle_analytics, cctv_cameras, cctv_alerts, inventory_items]",
  "filters": {{
    "column_name": "value"
  }},
  "join_cattle": true or false
}}
"""

    is_offline = os.getenv("OFFLINE_MODE", "false").lower() in ("true", "1")
    plan = None

    if is_dry_run and is_offline:
        plan = heuristic_plan_fallback(question)
    else:
        try:
            oa_client = get_openai_client()
            response = oa_client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": planner_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0,
                max_tokens=200,
                timeout=5.0 if is_dry_run else 30.0,
                extra_body={"clear_thinking": True}
            )
            content = response.choices[0].message.content or ""
            plan = extract_json_plan(content)
        except Exception as e:
            plan = heuristic_plan_fallback(question)

    if not plan or not isinstance(plan, dict):
        plan = heuristic_plan_fallback(question)

    raw_table = plan.get("table", "")
    table = normalize_table_name(raw_table)
    join_cattle = plan.get("join_cattle", False) or (table in CATTLE_CHILD_TABLES)
    raw_filters = plan.get("filters", {})
    filters = sanitize_filters(table or "", raw_filters)

    if not table:
        return f"I could not identify a valid farm database table for '{raw_table}'."

    try:
        engine = get_db_engine()
        data = []

        with engine.connect() as conn:
            if table == "cattles":
                tag_val = filters.get("cattle_tag_no") or list(filters.values())[0] if filters else None
                
                if tag_val:
                    sql = text("""
                        SELECT * FROM cattles 
                        WHERE cattle_tag_no = :tag
                    """)
                    res = conn.execute(sql, {"tag": int(tag_val) if str(tag_val).isdigit() else tag_val}).mappings().all()
                    data = [dict(row) for row in res]
                
                if not data:
                    sql = text("SELECT * FROM cattles WHERE farm_id = :farm_id LIMIT 20")
                    res = conn.execute(sql, {"farm_id": farm_id}).mappings().all()
                    data = [dict(row) for row in res]

            elif table in ["inventory_items", "cctv_cameras"]:
                base_sql = f"SELECT * FROM {table} WHERE farm_id = :farm_id"
                params = {"farm_id": farm_id}
                for col, val in filters.items():
                    base_sql += f" AND {col} = :_filter_{col}"
                    params[f"_filter_{col}"] = val
                res = conn.execute(text(base_sql), params).mappings().all()
                data = [dict(row) for row in res]

            elif table == "farms":
                sql = text("SELECT * FROM farms WHERE id = :farm_id")
                res = conn.execute(sql, {"farm_id": farm_id}).mappings().all()
                data = [dict(row) for row in res]

            elif table == "cctv_alerts":
                cam_sql = text("SELECT id FROM cctv_cameras WHERE farm_id = :farm_id")
                cam_res = conn.execute(cam_sql, {"farm_id": farm_id}).mappings().all()
                camera_ids = [c["id"] for c in cam_res]

                if not camera_ids:
                    return "No CCTV cameras or security alerts found registered for your farm."

                alert_sql = text("SELECT * FROM cctv_alerts WHERE camera_id = ANY(:camera_ids) LIMIT 20")
                res = conn.execute(alert_sql, {"camera_ids": camera_ids}).mappings().all()
                data = [dict(row) for row in res]

            elif join_cattle or table in CATTLE_CHILD_TABLES:
                cattle_sql = text("SELECT id, cattle_tag_no, name FROM cattles WHERE farm_id = :farm_id")
                cattle_res = conn.execute(cattle_sql, {"farm_id": farm_id}).mappings().all()
                cattle_map = {c["id"]: dict(c) for c in cattle_res}
                cattle_ids = list(cattle_map.keys())

                if not cattle_ids:
                    all_c_res = conn.execute(text("SELECT id, cattle_tag_no, name FROM cattles LIMIT 50")).mappings().all()
                    cattle_map = {c["id"]: dict(c) for c in all_c_res}
                    cattle_ids = list(cattle_map.keys())

                child_sql = f"SELECT * FROM {table} WHERE cattle_id = ANY(:cattle_ids) LIMIT 20"
                res = conn.execute(text(child_sql), {"cattle_ids": cattle_ids}).mappings().all()

                rows = []
                for row in res:
                    row_dict = dict(row)
                    c_info = cattle_map.get(row_dict.get("cattle_id"), {})
                    row_dict["cattle_info"] = c_info
                    rows.append(row_dict)
                data = rows

            else:
                sql = text(f"SELECT * FROM {table} LIMIT 20")
                res = conn.execute(sql).mappings().all()
                data = [dict(row) for row in res]

    except Exception as e:
        print(f"ERP Tool Database Query Error: {e}")
        return f"I encountered a database query error: {e}"

    if not data:
        return f"I checked your farm records in '{table}', but no matching records were found."

    try:
        oa_client = get_openai_client()
        summarizer_prompt = f"""
You are a helpful farm ERP assistant. Summarize the following database results retrieved for the farmer's question: "{question}"

Database Results:
{json.dumps(data, default=str, indent=2)}

Provide a clear, concise, professional response directly answering the farmer's question.
"""
        sum_resp = oa_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": summarizer_prompt}],
            temperature=0.3,
            max_tokens=400,
            timeout=30.0
        )
        answer = sum_resp.choices[0].message.content or ""
        if answer.strip():
            return answer.strip()
    except Exception as e:
        pass

    summary_lines = [f"Here are the records found for '{table}':"]
    for item in data[:5]:
        summary_lines.append(f"- {json.dumps(item, default=str)}")
    return "\n".join(summary_lines)