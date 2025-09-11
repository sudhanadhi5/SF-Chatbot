import os
import json
import datetime
import textwrap
import streamlit as st
import pandas as pd
from openai import OpenAI
from simple_salesforce import Salesforce
from dotenv import load_dotenv

# -------------------------
# Setup
# -------------------------
load_dotenv()
st.set_page_config(page_title="Salesforce Chatbot", layout="wide")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SF_USERNAME = os.getenv("SF_USERNAME")
SF_PASSWORD = os.getenv("SF_PASSWORD")
SF_SECURITY_TOKEN = os.getenv("SF_SECURITY_TOKEN")
SF_DOMAIN = os.getenv("SF_DOMAIN", "login")  # "test" for sandbox

if not (OPENAI_API_KEY and SF_USERNAME and SF_PASSWORD and SF_SECURITY_TOKEN):
    st.error("Please set OPENAI_API_KEY, SF_USERNAME, SF_PASSWORD, SF_SECURITY_TOKEN in environment.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

@st.cache_resource(show_spinner=False)
def create_sf_conn():
    sf = Salesforce(
        username=SF_USERNAME,
        password=SF_PASSWORD,
        security_token=SF_SECURITY_TOKEN,
        domain=SF_DOMAIN
    )
    # Retrieve current user's ID for default OwnerId
    user_info = sf.query("SELECT Id FROM User WHERE Username = '{}' LIMIT 1".format(SF_USERNAME))
    current_user_id = user_info.get("records", [{}])[0].get("Id")
    return sf, current_user_id

sf_conn, current_user_id = create_sf_conn()

# -------------------------
# >>> MEMORY (Q&A + Results) — keep only last 2
# -------------------------
if "conversation_memory" not in st.session_state:
    # list of dicts: {"question": str, "answer": str, "result_type": str, "result_preview": str}
    st.session_state.conversation_memory = []

def _truncate(txt: str, limit: int = 1200) -> str:
    if not isinstance(txt, str):
        txt = str(txt)
    return (txt if len(txt) <= limit else (txt[:limit] + " …(truncated)"))

def update_memory(question: str, answer: str, result_type: str = "chat", result_preview: str = ""):
    st.session_state.conversation_memory.append({
        "question": question or "",
        "answer": _truncate(answer or ""),
        "result_type": result_type,
        "result_preview": _truncate(result_preview or "")
    })
    if len(st.session_state.conversation_memory) > 2:
        st.session_state.conversation_memory = st.session_state.conversation_memory[-2:]

def get_context_snippet() -> str:
    if not st.session_state.conversation_memory:
        return ""
    lines = []
    for i, m in enumerate(st.session_state.conversation_memory[-2:], start=1):
        lines.append(
            textwrap.dedent(f"""
            [Memory {i}]
            User asked: {m['question']}
            Bot answered: {m['answer']}
            Result type: {m['result_type']}
            Result preview: {m['result_preview']}
            """).strip()
        )
    return "\n\n".join(lines)

# -------------------------
# Utilities
# -------------------------
@st.cache_data(ttl=300, show_spinner=False)
def org_describe():
    return sf_conn.describe()

def resolve_api_name(name_or_label: str):
    if not name_or_label:
        return None
    needle = name_or_label.strip()
    try:
        desc = org_describe()
        for s in desc["sobjects"]:
            if s["name"].lower() == needle.lower() or s["label"].lower() == needle.lower():
                return s["name"]
        return needle.replace("__c__c", "__c")
    except Exception:
        return needle.replace("__c__c", "__c")

def get_object_fields(api_name: str):
    try:
        return getattr(sf_conn, api_name).describe()["fields"]
    except Exception:
        return []

def lookup_records_by_name(api_name: str, name_value: str, name_field="Name"):
    try:
        safe_val = name_value.replace("'", "\\'")
        soql = f"SELECT Id, {name_field} FROM {api_name} WHERE {name_field} LIKE '%{safe_val}%'"
        res = sf_conn.query(soql)
        return res.get("records", [])
    except Exception as e:
        st.error(f"Lookup error: {e}")
        return []

@st.cache_data(ttl=300, show_spinner=False)
def get_lookup_options(ref_obj_api: str, limit=200):
    try:
        soql = f"SELECT Id, Name FROM {ref_obj_api} WHERE Name != null ORDER BY Name LIMIT {limit}"
        res = sf_conn.query(soql)
        options = [(r["Name"], r["Id"]) for r in res.get("records", []) if r.get("Name")]
        if not options:
            st.warning(f"No records with valid Names found for {ref_obj_api}")
        return options
    except Exception as e:
        st.error(f"Failed to fetch lookup options for {ref_obj_api}: {e}")
        return []

def clean_payload(payload: dict, required_fields: list, api_name: str):
    out = {}
    fields_meta = get_object_fields(api_name)
    for k, v in payload.items():
        if v is None or (isinstance(v, str) and v.strip() == ""):
            continue
        if isinstance(v, (datetime.date, datetime.datetime)):
            out[k] = v.isoformat()
        else:
            out[k] = v
    # Add default OwnerId if required and not provided
    owner_field = next((f for f in fields_meta if f["name"] == "OwnerId" and not f.get("nillable", True) and not f.get("defaultedOnCreate", False)), None)
    if owner_field and "OwnerId" not in out:
        out["OwnerId"] = current_user_id
    # Validate required fields
    missing = [f for f in required_fields if f not in out or out[f] is None or (isinstance(out[f], str) and out[f].strip() == "")]
    if missing:
        raise ValueError(f"Missing or invalid required fields: {', '.join(missing)}")
    return out

def render_field_input(field_def: dict, key_prefix: str, default=None):
    fname = field_def["name"]
    # Skip OwnerId to avoid user input
    if fname == "OwnerId":
        return None
    flabel = field_def.get("label", fname)
    ftype = field_def.get("type")
    required = not field_def.get("nillable", True) and not field_def.get("defaultedOnCreate", False)
    key = f"{key_prefix}_{fname}"
    help_txt = f"{ftype}{' • required' if required else ''}"

    if ftype == "reference":
        ref_objs = field_def.get("referenceTo", [])
        if not ref_objs:
            st.warning(f"No related object defined for lookup field {flabel}")
            return st.text_input(f"{flabel} ({fname})", value=default or "", key=key, help=help_txt)
        
        ref_api = ref_objs[0]
        options = get_lookup_options(ref_api)
        if not options:
            st.warning(f"No records available for {ref_api} in lookup field {flabel}")
            return st.text_input(f"{flabel} ({fname})", value=default or "", key=key, help=help_txt)

        display_names = [name for name, _ in options]
        name_to_id = dict(options)
        default_name = None

        if default:
            if default in name_to_id.values():
                for nm, rid in options:
                    if rid == default:
                        default_name = nm
                        break
            else:
                matches = lookup_records_by_name(ref_api, default)
                if matches:
                    default_name = matches[0]["Name"]
                    default = matches[0]["Id"]

        chosen_name = st.selectbox(
            f"{flabel} ({fname})",
            [""] + display_names,
            index=(display_names.index(default_name) + 1) if default_name in display_names else 0,
            key=key,
            help=help_txt,
        )
        chosen_id = name_to_id.get(chosen_name) if chosen_name else None
        return chosen_id

    if ftype == "picklist":
        opts = [v["value"] for v in field_def.get("picklistValues", []) if v.get("active", True)]
        index = 0
        if default in opts:
            index = opts.index(default) + 1
        return st.selectbox(f"{flabel} ({fname})", [""] + opts, index=index, key=key, help=help_txt)

    if ftype == "multipicklist":
        opts = [v["value"] for v in field_def.get("picklistValues", []) if v.get("active", True)]
        selected = default.split(";") if default else []
        chosen = st.multiselect(f"{flabel} ({fname})", options=opts, default=selected, key=key, help=help_txt)
        return ";".join(chosen)

    if ftype == "boolean":
        return st.checkbox(f"{flabel} ({fname})", value=bool(default), key=key, help=help_txt)

    if ftype == "date":
        try:
            val = datetime.date.fromisoformat(default) if default else datetime.date.today()
        except Exception:
            val = datetime.date.today()
        return st.date_input(f"{flabel} ({fname})", value=val, key=key, help=help_txt)

    if ftype in ("int", "double", "currency", "percent"):
        try:
            val = float(default) if default is not None else 0
        except Exception:
            val = 0
        return st.number_input(f"{flabel} ({fname})", value=val, key=key, help=help_txt)

    return st.text_input(f"{flabel} ({fname})", value=default or "", key=key, help=help_txt)

# -------------------------
# Child discovery + fetching for Summarize
# -------------------------
@st.cache_data(ttl=600, show_spinner=False)
def get_child_relationship_names(parent_api: str, include_limit=10):
    out = []
    try:
        desc = getattr(sf_conn, parent_api).describe()
        child_rels = desc.get("childRelationships", []) or []
        for cr in child_rels:
            rel_name = cr.get("relationshipName")
            child_sobject = cr.get("childSObject")
            if rel_name:
                out.append({"relationshipName": rel_name, "childSObject": child_sobject})
                if len(out) >= include_limit:
                    break
    except Exception:
        pass
    return out

def fetch_parent_with_children(parent_api: str, parent_id: str, child_limit_per_rel=5):
    try:
        rels = get_child_relationship_names(parent_api)
        fields = get_object_fields(parent_api)
        key_fields = [f["name"] for f in fields if f.get("createable") or f.get("updateable") and f["name"] not in ("Id", "IsDeleted", "CreatedDate", "CreatedById", "LastModifiedDate", "LastModifiedById", "SystemModstamp")]
        select_parts = ["Id", "Name"] + key_fields[:5]
        subqueries = []
        for r in rels:
            rel_name = r["relationshipName"]
            subqueries.append(f"(SELECT Id, Name FROM {rel_name} LIMIT {child_limit_per_rel})")
        if subqueries:
            select_clause = ", ".join(select_parts + subqueries)
        else:
            select_clause = ", ".join(select_parts)
        soql = f"SELECT {select_clause} FROM {parent_api} WHERE Id = '{parent_id}' LIMIT 1"
        res = sf_conn.query(soql)
        records = res.get("records", [])
        if records:
            return records[0]
        return {}
    except Exception:
        try:
            return getattr(sf_conn, parent_api).get(parent_id)
        except Exception:
            return {}

# -------------------------
# Summarize Utility
# -------------------------
def summarize_records(api_name: str, recs: list, child_limit_per_rel=5):
    summaries = []
    detailed_records = []

    for r in recs:
        parent_id = r.get("Id")
        if not parent_id:
            continue

        full = fetch_parent_with_children(api_name, parent_id, child_limit_per_rel)
        parent_data = {k: v for k, v in full.items() if k not in ("attributes") and not isinstance(v, dict)}
        child_data = {}
        for rel_name, children in full.items():
            if isinstance(children, dict) and children.get("records"):
                child_records = children["records"]
                if child_records:
                    child_data[rel_name] = [{"Id": c["Id"], "Name": c.get("Name", "Unknown")} for c in child_records[:3]]

        detailed_record = {
            "Id": parent_id,
            "Name": parent_data.get("Name", ""),
            **{k: v for k, v in parent_data.items() if k not in ("Id", "Name")},
            **{f"{rel_name}_Count": len(children.get("records", [])) for rel_name, children in child_data.items()},
            **{f"{rel_name}_{i+1}": children[i]["Name"] for rel_name, children in child_data.items() for i in range(len(children))},
            **{f"{rel_name}_{i+1}_Id": children[i]["Id"] for rel_name, children in child_data.items() for i in range(len(children))}
        }
        detailed_records.append(detailed_record)

        prompt = (
            f"You are a helpful assistant that summarizes Salesforce records for any SObject.\n\n"
            f"Write a concise, human-readable description of the following {api_name} record(s) "
            f"and their related child records. The summary should:\n"
            f"- Always mention the SObject type, record name, and Id naturally in the description.\n"
            f"- Highlight 2–5 key fields (e.g., StageName, Industry, Status, Amount, Owner), but exclude system fields like IsDeleted, CreatedDate, LastModifiedBy.\n"
            f"- If child records exist, weave them into the narrative (e.g., 'It has 2 related contacts, including John Doe and Jane Smith').\n"
            f"- If multiple records are provided, first mention the total count, then describe 1–3 examples briefly instead of listing all.\n"
            f"- Avoid null or empty fields.\n"
            f"- Keep the tone concise, factual, and easy to read like a short business summary.\n\n"
            f"Records JSON:\n{json.dumps({'Records': parent_data, 'Children': child_data}, default=str, indent=2)}\n\n"
            f"Return the summary as plain text, in description style."
        )

        try:
            gresp = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "You are a Salesforce assistant that summarizes records and their related children."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.25,
                max_tokens=800
            )
            summary = gresp.choices[0].message.content.strip()
            summaries.append(summary)
            detailed_records[-1]["Summary"] = summary
        except Exception as e:
            summaries.append(f"⚠️ Failed to summarize {api_name} record {parent_id}: {e}")
            detailed_records[-1]["Summary"] = f"Failed to summarize: {e}"

    st.session_state["summarized_records"] = detailed_records
    st.session_state["show_download_csv"] = True

    # >>> MEMORY: store a concise preview
    preview = "; ".join([f"{d.get('Name','(no name)')} [{d.get('Id','')}]" for d in detailed_records[:3]])
    update_memory(
        question=st.session_state.chat_history[-1]["text"] if st.session_state.get("chat_history") else "",
        answer="\n\n".join(summaries) if summaries else f"No {api_name} records found.",
        result_type="summarize",
        result_preview=f"{api_name} summarized: {preview}"
    )

    return "\n\n".join(summaries) if summaries else f"No {api_name} records found."

# -------------------------
# OpenAI Intent Parser (with memory context)
# -------------------------
def parse_intent_with_openai(text: str):
    context = get_context_snippet()
    system = (
        "Return JSON only with keys: action, object, record_name, id, soql, fields. "
        "action ∈ {create, update, delete, query, chat, summarize, count}. "
        "Resolve pronouns like 'it', 'them', 'last one' using the provided context."
    )
    user = (
        f"Context (last 2 Q&A + results):\n{context}\n\n"
        f"User message: '''{text}'''\n"
        f"Return JSON only. If not Salesforce-intent, set action='chat'."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.strip("`")
        return json.loads(content)
    except Exception as e:
        st.sidebar.warning(f"Intent parse failed: {e}")
        low = text.lower()
        if "how many" in low or "count" in low:
            return {"action": "count"}
        if "summarize" in low or "show details" in low:
            return {"action": "summarize"}
        if "create" in low: return {"action":"create"}
        if "update" in low: return {"action":"update"}
        if "delete" in low: return {"action":"delete"}
        if "query" in low: return {"action":"query"}
        return {"action":"chat"}

# -------------------------
# Chat UI
# -------------------------
st.title("🤖 Salesforce Chatbot — Continuous CRUD + Chat + Summarize + Count (with Memory)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "show_download_csv" not in st.session_state:
    st.session_state.show_download_csv = False

# Optional: show memory for transparency
with st.expander("🧠 Conversation memory (last 2)", expanded=False):
    if st.session_state.conversation_memory:
        for m in st.session_state.conversation_memory:
            st.markdown(f"**Q:** {m['question']}")
            st.markdown(f"**A:** {m['answer']}")
            if m["result_preview"]:
                st.caption(f"Result: {m['result_type']} → {m['result_preview']}")
            st.markdown("---")
    else:
        st.caption("No memory yet.")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])

user_input = st.chat_input("Ask about Salesforce or anything else...")

def assistant_reply(text: str, result_type: str = "chat", result_preview: str = ""):
    st.session_state.chat_history.append({"role": "assistant", "text": text})
    with st.chat_message("assistant"):
        st.markdown(text)
    # >>> MEMORY: store paired with last user question
    if len(st.session_state.chat_history) >= 2 and st.session_state.chat_history[-2]["role"] == "user":
        last_user_q = st.session_state.chat_history[-2]["text"]
        update_memory(last_user_q, text, result_type=result_type, result_preview=result_preview)

# -------------------------
# Handle user input
# -------------------------
if user_input:
    st.session_state.chat_history.append({"role": "user", "text": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    parsed = parse_intent_with_openai(user_input)
    action = parsed.get("action", "chat")
    object_hint = parsed.get("object")
    record_name = parsed.get("record_name")
    rec_id = parsed.get("id")
    soql = parsed.get("soql")

    if action == "count":
        api = resolve_api_name(object_hint) if object_hint else None
        if api:
            try:
                soql = f"SELECT COUNT() FROM {api}"
                res = sf_conn.query(soql)
                total = res.get("totalSize", 0)
                ans = f"There are **{total} {api}** records in Salesforce."
                assistant_reply(ans, result_type="count", result_preview=f"{api} totalSize={total}")
            except Exception as e:
                assistant_reply(f"⚠️ Count failed for {api}: {e}", result_type="error")
        else:
            try:
                desc = org_describe()
                results = []
                for sobj in desc["sobjects"]:
                    if not sobj.get("queryable"):
                        continue
                    name = sobj["name"]
                    try:
                        res = sf_conn.query(f"SELECT COUNT() FROM {name}")
                        count = res.get("totalSize", 0)
                        if count > 0:
                            results.append(f"{name}: {count}")
                    except Exception:
                        continue
                if results:
                    summary = "📊 Record counts by SObject:\n\n" + "\n".join(results)
                    assistant_reply(summary, result_type="count", result_preview=_truncate("; ".join(results)))
                else:
                    assistant_reply("⚠️ Could not retrieve record counts for any SObject.", result_type="error")
            except Exception as e:
                assistant_reply(f"⚠️ Failed to fetch SObjects: {e}", result_type="error")

    elif action == "create" and object_hint:
        api = resolve_api_name(object_hint)
        if not api:
            assistant_reply(f"⚠️ Could not resolve object: {object_hint}", result_type="error")
        else:
            fields_meta = get_object_fields(api)
            creatable_fields = [f for f in fields_meta if f.get("createable", False) and f["name"] != "OwnerId"]
            required_fields = [f for f in creatable_fields if not f.get("nillable", True) and not f.get("defaultedOnCreate", False)]
            required_field_names = [f["name"] for f in required_fields]

            # Extract field values from user input using OpenAI
            field_info = [
                {
                    "api_name": f["name"],
                    "label": f.get("label", f["name"]),
                    "type": f["type"],
                    "referenceTo": f.get("referenceTo", []) if f["type"] == "reference" else None
                }
                for f in creatable_fields
            ]
            system_extract = "You are extracting field values for Salesforce record creation."
            context = get_context_snippet()
            user_extract = f"""
            Context (last 2 Q&A + results):
            {context}

            User wants to create a {api} record with details: {user_input}
            Available fields: {json.dumps(field_info, indent=2)}
            Return a JSON object with keys as api_name and values as extracted values.
            For fields of type 'reference', provide the Name of the related record (not the Id).
            Use appropriate formats: dates as YYYY-MM-DD, booleans as true/false, numbers as numbers.
            Only include fields that have clear values in the user input; omit fields with ambiguous or missing values.
            Do not include OwnerId; it will be set automatically.
            """
            try:
                resp = client.chat.completions.create(
                    model="gpt-4.1-nano",
                    messages=[{"role": "system", "content": system_extract}, {"role": "user", "content": user_extract}],
                    temperature=0
                )
                content = resp.choices[0].message.content.strip()
                if content.startswith("```json"):
                    content = content[7:-3].strip()
                elif content.startswith("```"):
                    content = content.strip("`").strip()
                parsed_fields = json.loads(content)
                parsed_fields_converted = {}
                for fname, value in parsed_fields.items():
                    field_def = next((f for f in fields_meta if f["name"] == fname), None)
                    if field_def and field_def["type"] == "reference" and value:
                        ref_api = field_def.get("referenceTo", [None])[0]
                        if ref_api:
                            matches = lookup_records_by_name(ref_api, value)
                            if matches:
                                parsed_fields_converted[fname] = matches[0]["Id"]
                            else:
                                parsed_fields_converted[fname] = None
                                st.warning(f"Could not find {ref_api} record with Name '{value}' for {fname}")
                        else:
                            parsed_fields_converted[fname] = None
                            st.warning(f"No referenceTo defined for {fname}")
                    else:
                        parsed_fields_converted[fname] = value

                # Check if all required fields are provided
                missing_fields = [f for f in required_field_names if f not in parsed_fields_converted or parsed_fields_converted[f] is None or (isinstance(parsed_fields_converted[f], str) and parsed_fields_converted[f].strip() == "")]
                
                if missing_fields:
                    # Show missing fields and open the create form
                    missing_field_labels = [next(f.get("label", f["name"]) for f in fields_meta if f["name"] == mf) for mf in missing_fields]
                    assistant_reply(
                        f"⚠️ Missing required fields for {api}: {', '.join(missing_field_labels)}. Please provide these fields or use the form below.",
                        result_type="create",
                        result_preview=f"missing: {', '.join(missing_fields)}"
                    )
                    st.session_state["create_api"] = api
                    st.session_state["create_defaults"] = parsed_fields_converted
                else:
                    # All required fields provided, prompt for confirmation
                    st.session_state["create_api"] = api
                    st.session_state["create_defaults"] = parsed_fields_converted
                    st.session_state["create_confirm"] = True
                    assistant_reply(
                        f"✅ All required fields provided for {api}: {', '.join(parsed_fields_converted.keys())}.\n\n**Proceed with Creation?** Please confirm or provide additional fields below.",
                        result_type="create",
                        result_preview=f"fields: {list(parsed_fields_converted.keys())}"
                    )

            except Exception as e:
                st.sidebar.warning(f"Field extraction failed: {e}")
                parsed_fields_converted = {}
                st.session_state["create_api"] = api
                st.session_state["create_defaults"] = parsed_fields_converted
                assistant_reply(
                    f"⚠️ Failed to extract fields for {api}. Please fill out the form below.",
                    result_type="create",
                    result_preview="form opened"
                )

    elif action == "update" and object_hint:
        api = resolve_api_name(object_hint)
        if api:
            st.session_state["update_api"] = api
            if rec_id:
                st.session_state["update_rec_id"] = rec_id
                st.session_state.pop("update_choices", None)
            elif record_name:
                matches = lookup_records_by_name(api, record_name)
                if len(matches) == 0:
                    st.session_state.pop("update_rec_id", None)
                    st.session_state.pop("update_choices", None)
                    assistant_reply(f"⚠️ No {api} records found matching '{record_name}'.", result_type="error")
                elif len(matches) == 1:
                    st.session_state["update_rec_id"] = matches[0]["Id"]
                    st.session_state.pop("update_choices", None)
                else:
                    st.session_state["update_choices"] = matches
                    st.session_state.pop("update_rec_id", None)
            else:
                st.session_state.pop("update_rec_id", None)
                st.session_state.pop("update_choices", None)
        assistant_reply(f"✏️ Opening update form for {api}.", result_type="update", result_preview=f"record_name={record_name or ''} id={rec_id or ''}")

    elif action == "delete" and object_hint:
        api = resolve_api_name(object_hint)
        if api:
            st.session_state["delete_api"] = api
            if rec_id:
                st.session_state["delete_rec_id"] = rec_id
                st.session_state.pop("delete_choices", None)
            elif record_name:
                matches = lookup_records_by_name(api, record_name)
                if len(matches) == 0:
                    st.session_state.pop("delete_rec_id", None)
                    st.session_state.pop("delete_choices", None)
                    assistant_reply(f"⚠️ No {api} records found matching '{record_name}'.", result_type="error")
                elif len(matches) == 1:
                    st.session_state["delete_rec_id"] = matches[0]["Id"]
                    st.session_state.pop("delete_choices", None)
                else:
                    st.session_state["delete_choices"] = matches
                    st.session_state.pop("delete_rec_id", None)
            else:
                st.session_state.pop("delete_rec_id", None)
                st.session_state.pop("delete_choices", None)
        assistant_reply(f"🗑️ Opening delete form for {api}.", result_type="delete", result_preview=f"record_name={record_name or ''} id={rec_id or ''}")

    elif action == "query":
        if soql:
            try:
                for obj in sf_conn.describe()["sobjects"]:
                    label = obj["label"]
                    name = obj["name"]
                    if label in soql and name not in soql:
                        soql = soql.replace(label, name)
                soql = soql.replace("__c__c", "__c")
                res = sf_conn.query(soql)
                df = pd.DataFrame(res["records"]).drop(columns=["attributes"], errors="ignore")
                st.dataframe(df)

                # >>> MEMORY: compact preview (first 5 rows)
                preview = ""
                try:
                    preview = df.head(5).to_json(orient="table")
                except Exception:
                    preview = f"{len(df)} rows"

                assistant_reply(f"Returned {len(df)} records.", result_type="query", result_preview=_truncate(preview, 2000))
            except Exception as e:
                assistant_reply(f"⚠️ Query failed: {e}", result_type="error")
        else:
            assistant_reply("Please provide a SOQL query or ask me to build one.", result_type="chat")

    elif action == "summarize" and object_hint:
        api = resolve_api_name(object_hint)
        if api:
            try:
                if record_name:
                    matches = lookup_records_by_name(api, record_name)
                    if not matches:
                        assistant_reply(f"⚠️ No {api} records found matching '{record_name}'.", result_type="error")
                    else:
                        recs = [{"Id": m["Id"], "Name": m.get("Name")} for m in matches]
                        summary = summarize_records(api, recs, child_limit_per_rel=5)
                        assistant_reply(summary, result_type="summarize", result_preview=f"{api} {len(recs)} rec(s)")
                elif rec_id:
                    summary = summarize_records(api, [{"Id": rec_id}], child_limit_per_rel=5)
                    assistant_reply(summary, result_type="summarize", result_preview=f"{api} single: {rec_id}")
                else:
                    soql = f"SELECT Id, Name FROM {api} LIMIT 20"
                    res = sf_conn.query(soql)
                    recs = res.get("records", [])
                    if not recs:
                        assistant_reply(f"⚠️ No {api} records found.", result_type="error")
                    else:
                        minimal = [{"Id": r["Id"], "Name": r.get("Name")} for r in recs]
                        summary = summarize_records(api, minimal, child_limit_per_rel=3)
                        assistant_reply(summary, result_type="summarize", result_preview=f"{api} preview up to 20")
            except Exception as e:
                assistant_reply(f"⚠️ Summarize failed: {e}", result_type="error")

    else:
        # General Q&A (non-Salesforce or fallback) with memory context
        try:
            context = get_context_snippet()
            prompt = (
                "You are a helpful assistant. Use the recent conversation for context if relevant. "
                "If the user's question is general knowledge and not about Salesforce, simply answer it.\n\n"
                f"{context}\n\n"
                f"User question:\n{user_input}"
            )
            gresp = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role":"system","content":"You are a helpful assistant."},
                    {"role":"user","content": prompt}
                ],
                temperature=0.3
            )
            answer = gresp.choices[0].message.content.strip()
            assistant_reply(answer, result_type="chat", result_preview=_truncate(answer, 500))
        except Exception as e:
            assistant_reply(f"⚠️ OpenAI error: {e}", result_type="error")

# -------------------------
# Render CSV Download
# -------------------------
if "summarized_records" in st.session_state and st.session_state["summarized_records"] and st.session_state["show_download_csv"]:
    with st.expander("⬇️ Download Summarized Records", expanded=False):
        df = pd.DataFrame(st.session_state["summarized_records"])
        csv_data = df.to_csv(index=False).encode("utf-8")
        current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"summarized_{st.session_state.get('create_api', 'records')}_{current_date}.csv"
        if st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=file_name,
            mime="text/csv",
            key="download_csv_button",
            on_click=lambda: st.session_state.update({"show_download_csv": False})
        ):
            pass

# -------------------------
# Render CREATE form
# -------------------------
if "create_api" in st.session_state:
    api = st.session_state["create_api"]
    defaults = st.session_state.get("create_defaults", {})
    fields_meta = get_object_fields(api)
    creatable = [f for f in fields_meta if f.get("createable", False) and f["name"] != "OwnerId"]
    required = [f for f in creatable if not f.get("nillable", True) and not f.get("defaultedOnCreate", False)]
    optional = [f for f in creatable if f not in required]

    if st.session_state.get("create_confirm"):
        with st.form(key=f"form_confirm_create_{api}"):
            st.subheader(f"Confirm Creation of {api} Record")
            st.write("Review the provided fields:")
            for fname, value in defaults.items():
                field_def = next((f for f in fields_meta if f["name"] == fname), None)
                flabel = field_def.get("label", fname) if field_def else fname
                if field_def and field_def["type"] == "reference" and value:
                    ref_api = field_def.get("referenceTo", [None])[0]
                    if ref_api:
                        matches = lookup_records_by_name(ref_api, value)
                        display_value = matches[0]["Name"] if matches else value
                    else:
                        display_value = value
                else:
                    display_value = value
                st.write(f"**{flabel}**: {display_value}")
            
            col1, col2 = st.columns(2)
            with col1:
                confirm_submit = st.form_submit_button("Proceed with Creation")
            with col2:
                edit_clicked = st.form_submit_button("Edit Fields")
            
            if confirm_submit:
                try:
                    cleaned_payload = clean_payload(defaults, [f["name"] for f in required], api)
                    res = getattr(sf_conn, api).create(cleaned_payload)
                    msg = f"✅ Created {api} with Id: {res.get('id')}"
                    st.success(msg)
                    update_memory(
                        question=f"Create {api} (confirmed)",
                        answer=msg,
                        result_type="create",
                        result_preview=json.dumps(cleaned_payload, default=str)
                    )
                    del st.session_state["create_api"]
                    st.session_state.pop("create_defaults", None)
                    st.session_state.pop("create_confirm", None)
                    st.rerun()
                except ValueError as ve:
                    st.error(str(ve))
                except Exception as e:
                    st.error(f"⚠️ Create failed: {e}")
            if edit_clicked:
                st.session_state.pop("create_confirm", None)
                st.rerun()

    else:
        with st.form(key=f"form_create_{api}"):
            st.subheader(f"Create {api} Record")
            st.markdown("**Required Fields**")
            payload = {}
            if required:
                for fld in required:
                    default = defaults.get(fld["name"])
                    payload[fld["name"]] = render_field_input(fld, f"create_{api}", default=default)
            else:
                st.info("No required fields for this object.")
            
            with st.expander("Optional Fields", expanded=False):
                if optional:
                    for fld in optional:
                        default = defaults.get(fld["name"])
                        payload[fld["name"]] = render_field_input(fld, f"create_{api}", default=default)
                else:
                    st.info("No optional fields available.")
            
            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("Create Record")
            with col2:
                back_clicked = st.form_submit_button("Back to Chatbot")
            
            if submitted:
                try:
                    cleaned_payload = clean_payload(payload, [f["name"] for f in required], api)
                    res = getattr(sf_conn, api).create(cleaned_payload)
                    msg = f"✅ Created {api} with Id: {res.get('id')}"
                    st.success(msg)
                    update_memory(
                        question=f"Create {api} (form)",
                        answer=msg,
                        result_type="create",
                        result_preview=json.dumps(cleaned_payload, default=str)
                    )
                    del st.session_state["create_api"]
                    st.session_state.pop("create_defaults", None)
                    st.rerun()
                except ValueError as ve:
                    st.error(str(ve))
                except Exception as e:
                    st.error(f"⚠️ Create failed: {e}")
            if back_clicked:
                del st.session_state["create_api"]
                st.session_state.pop("create_defaults", None)
                st.session_state.pop("create_confirm", None)
                st.success("Returned to chatbot.")
                st.rerun()

# -------------------------
# Render UPDATE form
# -------------------------
if "update_api" in st.session_state:
    api = st.session_state["update_api"]

    if "update_choices" in st.session_state and st.session_state.get("update_rec_id") is None:
        choices = st.session_state["update_choices"]
        labels = [f"{c.get('Name','(no name)')} ({c['Id']})" for c in choices]
        sel = st.selectbox("Multiple records found — choose one to edit:", labels, key=f"choice_up_{api}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Load selected record to edit"):
                idx = labels.index(sel)
                st.session_state["update_rec_id"] = choices[idx]["Id"]
                st.session_state.pop("update_choices", None)
                st.rerun()
        with col2:
            if st.button("Back to Chatbot", key="back_update_choices"):
                del st.session_state["update_api"]
                st.session_state.pop("update_choices", None)
                st.success("Returned to chatbot.")
                st.rerun()

    record_id = st.session_state.get("update_rec_id")

    if not record_id:
        st.info("Please provide a record Id or Name to update.")
    else:
        try:
            record_data = getattr(sf_conn, api).get(record_id)
        except Exception as e:
            st.error(f"⚠️ Could not fetch record: {e}")
            record_data = {}

        fields_meta = get_object_fields(api)
        updatable = [f for f in fields_meta if f.get("updateable", False) and f["name"] != "Id" and f["name"] != "OwnerId"]

        with st.form(key=f"form_update_{api}_{record_id}"):
            payload = {}
            for fld in updatable:
                current_value = record_data.get(fld["name"])
                payload[fld["name"]] = render_field_input(fld, f"upd_{api}_{record_id}", default=current_value)
            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("Update record")
            with col2:
                back_clicked = st.form_submit_button("Back to Chatbot")
            if submitted:
                try:
                    cleaned_payload = clean_payload(payload, [], api)  # No required fields for update
                    getattr(sf_conn, api).update(record_id, cleaned_payload)
                    msg = f"✅ Updated {api} {record_id}"
                    st.success(msg)
                    # >>> MEMORY: record the outcome
                    update_memory(
                        question=f"Update {api} {record_id} (form)",
                        answer=msg,
                        result_type="update",
                        result_preview=json.dumps(cleaned_payload, default=str)
                    )
                    del st.session_state["update_api"]
                    st.session_state.pop("update_rec_id", None)
                    st.rerun()
                except Exception as e:
                    st.error(f"⚠️ Update failed: {e}")
            if back_clicked:
                del st.session_state["update_api"]
                st.session_state.pop("update_rec_id", None)
                st.session_state.pop("update_choices", None)
                st.success("Returned to chatbot.")
                st.rerun()

# -------------------------
# Render DELETE form
# -------------------------
if "delete_api" in st.session_state:
    api = st.session_state["delete_api"]

    if "delete_choices" in st.session_state and st.session_state.get("delete_rec_id") is None:
        choices = st.session_state["delete_choices"]
        labels = [f"{c.get('Name','(no name)')} ({c['Id']})" for c in choices]
        sel = st.selectbox("Multiple records found — choose one to delete:", labels, key=f"choice_del_{api}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Select record to delete"):
                idx = labels.index(sel)
                st.session_state["delete_rec_id"] = choices[idx]["Id"]
                st.session_state.pop("delete_choices", None)
                st.rerun()
        with col2:
            if st.button("Back to Chatbot", key="back_delete_choices"):
                del st.session_state["delete_api"]
                st.session_state.pop("delete_choices", None)
                st.success("Returned to chatbot.")
                st.rerun()

    record_id = st.session_state.get("delete_rec_id")

    if not record_id:
        st.info("Please provide a record Id or Name to delete.")
    else:
        st.warning(f"⚠️ You are about to delete {api} {record_id}.")
        with st.form(key=f"form_delete_{api}_{record_id}"):
            confirm = st.text_input("Type DELETE to confirm:", key=f"confirm_del_{api}")
            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("Confirm Delete")
            with col2:
                back_clicked = st.form_submit_button("Back to Chatbot")
            if submitted:
                if confirm.strip().upper() != "DELETE":
                    st.error("Not confirmed.")
                else:
                    try:
                        getattr(sf_conn, api).delete(record_id)
                        msg = f"🗑️ Deleted {api} {record_id}"
                        st.success(msg)
                        # >>> MEMORY: record the outcome
                        update_memory(
                            question=f"Delete {api} {record_id} (form)",
                            answer=msg,
                            result_type="delete",
                            result_preview=f"{api}:{record_id}"
                        )
                        del st.session_state["delete_api"]
                        st.session_state.pop("delete_rec_id", None)
                        st.rerun()
                    except Exception as e:
                        st.error(f"⚠️ Delete failed: {e}")
            if back_clicked:
                del st.session_state["delete_api"]
                st.session_state.pop("delete_rec_id", None)
                st.session_state.pop("delete_choices", None)
                st.success("Returned to chatbot.")
                st.rerun()