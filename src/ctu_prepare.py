#!/usr/bin/env python3
"""
Prepare clean baseline trajectories and document-omission interventions.
No LLM/API calls are made.
"""
import argparse, json, uuid
from pathlib import Path
from collections import Counter, defaultdict

POLICIES = ("direct_iterative", "question_decomposition")

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def event_round(e):
    if e.get("round") is not None: return e["round"]
    if e.get("extra", {}).get("round") is not None: return e["extra"]["round"]
    obs=e.get("observation")
    if isinstance(obs,dict) and obs.get("round") is not None: return obs["round"]
    return None

def event_query(e):
    if e.get("query"): return e["query"]
    action=e.get("action")
    if isinstance(action,dict): return action.get("query")
    return None

def event_subq(e):
    if e.get("subquestion_id") is not None: return e["subquestion_id"]
    action=e.get("action")
    if isinstance(action,dict): return action.get("subquestion_id")
    return None

def retrieved_ids(e):
    docs=e.get("retrieved_docs") or []
    ids=[d.get("doc_id") for d in docs if isinstance(d,dict) and d.get("doc_id")]
    action=e.get("action")
    if not ids and isinstance(action,dict):
        ids=action.get("new_documents") or []
    return ids

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--run-id", default="96c5b40d-d5a9-45da-8af0-5d998be18d34", help="Baseline run_id from experiment_manifest.json")
    args=ap.parse_args()
    root=Path(args.project_root).resolve()
    results=root/"hw2_results"
    traj=read_jsonl(results/"trajectory_log.jsonl")
    selected=[e for e in traj if e.get("run_id")==args.run_id and e.get("execution_id")=="baseline"]
    if not selected:
        raise SystemExit(f"No baseline trajectory events for run_id={args.run_id}")

    qfile=results/"hotpotqa_selected_questions.json"
    dfile=results/"hotpotqa_documents.json"
    questions=json.load(open(qfile,encoding="utf-8"))
    docs=json.load(open(dfile,encoding="utf-8"))
    qids={q["question_id"] for q in questions}
    doc_lookup={d["doc_id"]:d for d in docs}
    expected={(qid,p) for qid in qids for p in POLICIES}

    grouped=defaultdict(list)
    for e in selected:
        if (e.get("question_id"),e.get("policy")) in expected:
            grouped[(e["question_id"],e["policy"])].append(e)

    clean=[]; audit=[]
    for key in sorted(expected):
        qid,policy=key
        events=grouped.get(key,[])
        starts=[e for e in events if e.get("event_type")==("query_generation" if policy=="direct_iterative" else "question_decomposition")]
        finals=[e for e in events if e.get("event_type")=="final_answer"]
        if not starts or not finals:
            audit.append({"question_id":qid,"policy":policy,"status":"incomplete","reason":"Missing start/final event"})
            continue
        final=max(finals,key=lambda e:e.get("timestamp",""))
        start=min(starts,key=lambda e:e.get("timestamp",""))
        chosen=[e for e in events if start.get("timestamp","")<=e.get("timestamp","")<=final.get("timestamp","")]
        chosen.sort(key=lambda e:e.get("timestamp",""))
        counts=Counter(e.get("event_type") for e in chosen)
        req={"query_generation":4,"retrieval":4,"final_answer":1}
        if policy==POLICIES[1]: req["question_decomposition"]=1
        count_ok=all(counts.get(k,0)==v for k,v in req.items())
        rounds=sorted(event_round(e) for e in chosen if e.get("event_type")=="query_generation" and event_round(e) is not None)
        rounds_ok=rounds==[1,2,3,4]
        ids=[docid for e in chosen if e.get("event_type")=="retrieval" for docid in retrieved_ids(e)]
        unique_ok=len(ids)==8 and len(set(ids))==8
        status="pass" if count_ok and rounds_ok and unique_ok else "check_required"
        audit.append({"question_id":qid,"policy":policy,"status":status,"event_counts":dict(counts),"query_rounds":rounds,"retrieved_doc_entries":len(ids),"unique_docs":len(set(ids)),"count_check":count_ok,"round_check":rounds_ok,"document_check":unique_ok})
        if status=="pass": clean.extend(chosen)

    clean.sort(key=lambda e:(e.get("question_id",""),e.get("policy",""),e.get("timestamp","")))
    output_dir=results/"full_10_question_baseline"/args.run_id/"ctu_preparation"
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_path=output_dir/"clean_trajectories.jsonl"
    with open(clean_path,"w",encoding="utf-8") as f:
        for e in clean: f.write(json.dumps(e,ensure_ascii=False)+"\n")
    audit_path=output_dir/"clean_trajectory_validation.json"
    json.dump({"run_id":args.run_id,"expected_pairs":len(expected),"passed_pairs":sum(a["status"]=="pass" for a in audit),"excluded_pairs":sum(a["status"]!="pass" for a in audit),"clean_event_count":len(clean),"pair_details":audit},open(audit_path,"w",encoding="utf-8"),ensure_ascii=False,indent=2)

    # Interventions only from validated complete baseline pairs.
    grouped_clean=defaultdict(list)
    for e in clean: grouped_clean[(e["question_id"],e["policy"])].append(e)
    interventions=[]
    for (qid,policy),events in grouped_clean.items():
        events.sort(key=lambda e: event_round(e) or 0)
        q_events=[e for e in events if e.get("event_type")=="query_generation"]
        q_by_round={event_round(e):e for e in q_events}
        dec=next((e for e in events if e.get("event_type")=="question_decomposition"),None)
        docs_by_round=defaultdict(list)
        for e in events:
            if e.get("event_type")!="retrieval": continue
            r=event_round(e)
            for docid in retrieved_ids(e):
                if docid not in doc_lookup:
                    raise KeyError(f"Document id not found in lookup: {docid}")
                docs_by_round[r].append(doc_lookup[docid])
        for r in (1,2,3):
            if r not in q_by_round or r+1 not in q_by_round: continue
            nextq=event_query(q_by_round[r+1])
            if not nextq: raise ValueError(f"Missing next-round query: {qid}/{policy}/r{r+1}")
            prefix=[d for prior in range(1,r) for d in docs_by_round[prior]]
            current=docs_by_round[r]
            for target in current:
                tid=target["doc_id"]
                siblings=[d for d in current if d["doc_id"]!=tid]
                excluded=sorted({d["doc_id"] for d in prefix+siblings+[target]})
                interventions.append({
                    "intervention_id":str(uuid.uuid4()),"run_id":args.run_id,
                    "question_id":qid,"policy":policy,"target_doc_id":tid,
                    "target_doc_title":target.get("title"),"intervention_round":r,
                    "prefix_documents":prefix,"sibling_documents":siblings,
                    "excluded_doc_ids":excluded,
                    "baseline_next_query":nextq,
                    "baseline_next_query_call_id":q_by_round[r+1].get("call_id"),
                    "baseline_next_round":r+1,
                    "decomposition":dec.get("action") if dec else None,
                    "baseline_execution_id":"baseline"
                })
    keys=[(i["question_id"],i["policy"],i["target_doc_id"]) for i in interventions]
    if len(keys)!=len(set(keys)): raise AssertionError("Duplicate target intervention keys")
    int_path=output_dir/"interventions.jsonl"
    with open(int_path,"w",encoding="utf-8") as f:
        for i in interventions: f.write(json.dumps(i,ensure_ascii=False)+"\n")
    print(f"Baseline run: {args.run_id}")
    print(f"Complete trajectory pairs: {sum(a['status']=='pass' for a in audit)}/{len(expected)}")
    print(f"Excluded incomplete pairs: {sum(a['status']!='pass' for a in audit)}")
    print(f"Clean events: {len(clean)}")
    print(f"Interventions: {len(interventions)}")
    print("Policy counts:",dict(Counter(i["policy"] for i in interventions)))
    print("Round counts:",dict(Counter(i["intervention_round"] for i in interventions)))
    print("Saved:",clean_path,audit_path,int_path,sep="\n")

if __name__=="__main__": main()
