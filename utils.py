import json, re, sqlite3

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]

def clean_generation(text):
    """Keep only the first SQL statement the model wrote."""
    text = text.strip()
    for stop in ["###", "\n\n"]:
        if stop in text:
            text = text.split(stop)[0]
    if ";" in text:
        text = text.split(";")[0]
    return text.strip()

def normalize(sql):
    sql = sql.strip().rstrip(";").lower()
    sql = sql.replace('"', "'").replace("`", "")
    sql = re.sub(r"\s+", " ", sql)
    sql = re.sub(r"\s*([(),=<>])\s*", r"\1", sql)
    return sql.strip()

def runs_on_sqlite(sql, schema):
    """Create empty tables from the schema, then try to execute the query."""
    try:
        con = sqlite3.connect(":memory:")
        con.executescript(schema)
        con.execute(sql).fetchall()
        return True
    except Exception:
        return False
    finally:
        try: con.close()
        except Exception: pass

def score(preds, rows):
    em = sum(normalize(p) == normalize(r["completion"]) for p, r in zip(preds, rows))
    ok = sum(runs_on_sqlite(p, r["prompt"].split("### Schema:\n")[1].split("\n\n### Question:")[0])
             for p, r in zip(preds, rows))
    n = len(rows)
    return {"n": n, "exact_match": em / n, "runs_on_sqlite": ok / n}
