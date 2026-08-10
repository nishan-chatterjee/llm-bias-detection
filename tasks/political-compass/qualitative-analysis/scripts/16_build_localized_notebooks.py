#!/usr/bin/env python3
"""Build localized-effect notebooks restricted to the primary paper models."""

from pathlib import Path
import hashlib
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks"

PRE = """from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def find_root():
    for p in [Path.cwd(), *Path.cwd().parents]:
        q=p/"tasks/political-compass/qualitative-analysis"
        if (q/"config.py").exists(): return q
        if p.name=="qualitative-analysis" and (p/"config.py").exists(): return p
    raise FileNotFoundError
ROOT=find_root(); TABLES=ROOT/"artifacts/tables"; FIGURES=ROOT/"artifacts/figures"
CHAT_ORDER=["gemma-3-1b-it","gemma-3-4b-it","gemma-3-12b-it","gemma-3-27b-it",
"Qwen3-4B_no_think","Qwen3-8B_no_think","Qwen3-14B_no_think","Qwen3-32B_no_think",
"Qwen3-4B_think","Qwen3-8B_think","Qwen3-14B_think","Qwen3-32B_think"]
LABEL=dict(zip(CHAT_ORDER,["Gemma 1B","Gemma 4B","Gemma 12B","Gemma 27B",
"Qwen 4B","Qwen 8B","Qwen 14B","Qwen 32B","Qwen 4B Think","Qwen 8B Think",
"Qwen 14B Think","Qwen 32B Think"]))
def ordered(d,col="model_variant"):
    d=d[d[col].isin(CHAT_ORDER)].copy(); d["model_label"]=d[col].map(LABEL)
    d["model_label"]=pd.Categorical(d.model_label,[LABEL[x] for x in CHAT_ORDER],ordered=True)
    return d.sort_values("model_label")
sns.set_theme(style="whitegrid")
"""

def nb(title, cells):
    x=nbf.v4.new_notebook()
    x.metadata["kernelspec"]={"display_name":"Python (absa)","language":"python","name":"python3"}
    title_cell=nbf.v4.new_markdown_cell("# "+title)
    title_cell.id=hashlib.sha1(f"{title}:0:markdown".encode()).hexdigest()[:8]
    x.cells=[title_cell]
    for index,(kind,src) in enumerate(cells,start=1):
        cell=nbf.v4.new_markdown_cell(src) if kind=="m" else nbf.v4.new_code_cell(src)
        cell.id=hashlib.sha1(f"{title}:{index}:{kind}".encode()).hexdigest()[:8]
        x.cells.append(cell)
    return x

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    books={
"00_trace_behavior_audit_localized_trace.ipynb":nb("Localized trace behavior — primary models",[
("c",PRE),("m","The measured outcome is **predefined qualifying-cue incidence**, not inferred uncertainty or safety behavior. Each cell has 300 exactly matched left/right prompts. “More” means at least a five-percentage-point risk difference after requiring at least 30 cue occurrences across 600 answers."),
("c",'''H=TABLES/"expressive_heterogeneity_primary"; c=pd.read_csv(H/"question_cells.csv")
m=ordered(pd.read_csv(H/"model_summary.csv")); q=ordered(c[(c.feature=="hedging")&(c.regime=="libertarian")&(c.question_id==34)])
display(q[["model_label","left_rate","right_rate","risk_difference","informative","substantive"]])
display(m[m.feature=="hedging"][["model_label","regime","informative_questions","substantive_pockets","right_more_pockets","left_more_pockets","mean_risk_difference"]])'''),
("c",'''z=q.melt(id_vars="model_label",value_vars=["left_rate","right_rate"],var_name="persona",value_name="rate")
fig,ax=plt.subplots(figsize=(11,5)); sns.barplot(data=z,x="model_label",y="rate",hue="persona",ax=ax)
ax.set(title="Q34 qualifying-cue incidence",xlabel="",ylabel="Answers with ≥1 cue"); ax.tick_params(axis="x",rotation=35)
ax.yaxis.set_major_formatter(lambda x,_:f"{x:.0%}"); plt.tight_layout(); plt.show()''')]),
"01_reasoning_depth_qwen_localized_trace.ipynb":nb("Localized Qwen protocol outcomes — primary models",[
("c",PRE),("m","The unit is Qwen size × persona × question. Rescue means think mode changes an unaligned answer to aligned; harm is the reverse. The comparison is the complete recommended-use protocol change."),
("c",'''g=pd.read_csv(TABLES/"qwen_think_question_localized_cells.csv")
localized=g[g.localized].assign(direction=np.where(g[g.localized].net_rescue_rate>0,"rescue","harm"))
aggregate=g.groupby(["pair_model","ideology"],observed=True).agg(rescues=("think_rescue","sum"),harms=("think_harm","sum"),pairs=("n","sum")).reset_index()
aggregate["net_rescue_rate"]=(aggregate.rescues-aggregate.harms)/aggregate.pairs
print("Localized cells:",len(localized),"of",len(g),"; rescue:",int((localized.direction=="rescue").sum()),"; harm:",int((localized.direction=="harm").sum()))
display(aggregate.sort_values(["pair_model","ideology"])[["pair_model","ideology","rescues","harms","pairs","net_rescue_rate"]])
display(g.reindex(g.net_rescue_rate.abs().sort_values(ascending=False).index).head(50))
r=localized.groupby(["ideology","question_id","direction"]).agg(qwen_sizes=("pair_model","nunique"),median_net=("net_rescue_rate","median")).reset_index()
recurring=r[r.qwen_sizes.eq(4)]
print("All-four-size patterns:",len(recurring),"; rescue:",int((recurring.direction=="rescue").sum()),"; harm:",int((recurring.direction=="harm").sum()))
display(r.sort_values(["qwen_sizes","median_net"],ascending=[False,False]).head(40))''')]),
"02_question_difficulty_localized_trace.ipynb":nb("Localized question difficulty — primary models",[
("c",PRE),("m","DIF identifies a model/persona/question that is unusually easy or difficult after accounting for general model/persona performance and general question difficulty. It does not establish one universally hardest quadrant."),
("c",'''d=ordered(pd.read_csv(TABLES/"question_dif_family_protocol.csv")); d=d[d.ideology.isin(["libertarian_left","libertarian_right","authoritarian_left","authoritarian_right"])]
d["localized"]=d.dif_abs>=1; d["direction"]=np.where(d.dif_logit<0,"unusually difficult","unusually easy")
display(d.sort_values("dif_abs",ascending=False)[["model_label","ideology","question_id","topic","target_alignment_rate","dif_logit","direction"]].head(60))
r=d[d.localized].groupby(["ideology","question_id","topic","direction"]).agg(variants=("model_variant","nunique"),gemma=("family",lambda x:int((x=="Gemma-3").sum())),qwen=("family",lambda x:int((x=="Qwen3").sum())),median_dif=("dif_logit","median")).reset_index()
display(r.sort_values(["variants","median_dif"],ascending=[False,True]).head(40))''')]),
"03_phrases_topics_separability_localized_trace.ipynb":nb("Localized phrase analysis — answerability audit",[
("c",PRE),("m","The phrase export is not model-specific and many confirmed phrases contain persona labels or role meta-talk. It cannot support a primary-model localized framing claim."),
("c",'''p=pd.read_csv(TABLES/"quadrant_phrase_discovery_confirmation.csv")
pat=r"authoritarian|libertarian|left[- ]wing|right[- ]wing|assigned|perspective|political"
p["identity_or_meta"]=p.phrase.astype(str).str.contains(pat,case=False,regex=True)
display(p.groupby(["scope","confirmed_q05"]).agg(rows=("phrase","size"),identity_or_meta=("identity_or_meta","sum")).reset_index())
display(p[p.confirmed_q05].sort_values("q_value_bh").head(50)); print("Localized primary-model claim: NOT ANSWERED.")''')]),
"04_gemma_abliteration_localized_trace.ipynb":nb("Uncensored Gemma — secondary matched note",[
("c",PRE),("m","This checkpoint is outside the main model set. It is excluded from primary recurrence and plots; only standout matched differences are shown as a secondary diagnostic."),
("c",'''q=pd.read_csv(TABLES/"gemma_abliteration/question_specific_effects_ids.csv")
ms=["delta__target_aligned__mean","delta__stage1_stage2_agree__mean","delta__visible_moral_distance_present__mean","delta__visible_actual_refusal_present__mean"]
l=q.melt(id_vars=["ideology","question_id","topic"],value_vars=ms,var_name="metric",value_name="difference")
display(l[l.difference.abs()>=.10].sort_values("difference",key=lambda x:x.abs(),ascending=False).head(60))
o=pd.read_csv(TABLES/"gemma_abliteration/exact_item_contrast.csv")
display(o[(o.level=="item")&o.ideology.eq("all")&o.metric.isin(["target_aligned","stage1_stage2_agree","visible_moral_distance_present","visible_actual_refusal_present","visible_words"])])
display(o[(o.level=="item")&o.ideology.ne("all")&o.metric.isin(["target_aligned","stage1_stage2_agree","visible_moral_distance_present","visible_actual_refusal_present"])])''')]),
"06_integrated_mcq_and_findings_localized_trace.ipynb":nb("Localized MCQ/chat consistency — primary models",[
("c",PRE),("m","Each base-model × method × question cell has only 12 exact overlapping prompt configurations. These are candidates for replication, not stable model effects. Ordering follows Gemma sizes, then Qwen sizes, with standard chat before think chat."),
("c",'''p=pd.read_parquet(TABLES/"mcq_crosscheck/matched_question_pairs.parquet")
g=p.groupby(["method","base_model","question_id"]).agg(n=("four_way_agree","size"),four_way=("four_way_agree","mean"),binary=("binary_stance_agree","mean"),mean_tv=("total_variation","mean"),answer_shift=("expected_answer_delta","mean")).reset_index()
gem=["gemma-3-1b-it","gemma-3-4b-it","gemma-3-12b-it","gemma-3-27b-it"]; qw=["Qwen3-4B","Qwen3-8B","Qwen3-14B","Qwen3-32B"]
order=[(x,"Standard Chat") for x in gem]+sum(([ (x,"Standard Chat"),(x,"Chat Think") ] for x in qw),[])
g["order_key"]=[order.index((b,m)) if (b,m) in order else 999 for b,m in zip(g.base_model,g.method)]; g=g[g.order_key<999]
display(g.sort_values("mean_tv",ascending=False).head(60))
r=g.assign(high=g.binary>=.75,low=g.binary<=.50).groupby(["method","question_id"]).agg(models=("base_model","nunique"),high_models=("high","sum"),low_models=("low","sum"),median_tv=("mean_tv","median")).reset_index()
display(r.sort_values(["low_models","median_tv"],ascending=False).head(40))''')]),
"07_expressive_heterogeneity_discovery_primary_models.ipynb":nb("Qualifying-language discovery — primary models",[
("c",PRE),("m","Scope: four Gemma sizes and four Qwen sizes, with Qwen no-think and think modes. The outcome is qualifying-cue incidence. Pragmatic hedging, uncertainty, caution, and safety motivation require contextual human validation."),
("c",'''H=TABLES/"expressive_heterogeneity_primary"; c=pd.read_csv(H/"question_cells.csv"); m=ordered(pd.read_csv(H/"model_summary.csv")); r=pd.read_csv(H/"question_recurrence.csv")
display(m[m.feature=="hedging"][["model_label","regime","informative_questions","substantive_pockets","right_more_pockets","left_more_pockets","mean_risk_difference"]])
display(r[r.feature=="hedging"].head(40))
hm=m[m.feature=="hedging"].copy()
fig,ax=plt.subplots(figsize=(10,6)); sns.scatterplot(data=hm,x="mean_risk_difference",y="model_label",hue="regime",size="substantive_pockets",sizes=(30,240),ax=ax)
ax.axvline(0,color="black",lw=1); ax.xaxis.set_major_formatter(lambda x,_:f"{x:+.1%}")
ax.set(title="Average qualifying-cue contrast and localized-cell count",xlabel="Mean matched difference (right − left)",ylabel="")
plt.tight_layout(); plt.savefig(FIGURES/"primary_qualifying_contrast_pockets.png",bbox_inches="tight",dpi=180); plt.show()'''),
("c",'''q=ordered(c[(c.feature=="hedging")&(c.regime=="libertarian")&(c.question_id==34)])
display(q[["model_label","left_rate","right_rate","risk_difference","informative","substantive"]])
print("Left-more by ≥5 points:",int((q.risk_difference<=-.05).sum()),"of",len(q),"primary variants")
top=c[(c.feature=="hedging")&c.substantive].copy()
top=top.loc[top.risk_difference.abs().nlargest(30).index].sort_values("risk_difference")
top["model_label"]=top.model_variant.map(LABEL); top["label"]=top.model_label+" · "+top.regime.str[:4]+" · Q"+top.question_id.astype(str)
fig,ax=plt.subplots(figsize=(10,9)); ax.barh(top.label,top.risk_difference,color=np.where(top.risk_difference>0,"#4c72b0","#c44e52"))
ax.axvline(0,color="black",lw=1); ax.xaxis.set_major_formatter(lambda x,_:f"{x:+.0%}")
ax.set(title="Largest primary-model qualifying-cue contrasts (appendix diagnostic)",xlabel="Matched difference (right − left)",ylabel="")
plt.tight_layout(); plt.savefig(FIGURES/"primary_qualifying_largest_cells.png",bbox_inches="tight",dpi=180); plt.show()'''),
("c",'''s=pd.read_csv(H/"qwen_protocol_stability.csv"); display(s[s.feature=="hedging"].sort_values(["size_b","regime"]))
v=c.groupby("feature").agg(cells=("question_id","size"),informative=("informative","sum"),pockets=("substantive","sum")).reset_index(); v["coverage"]=v.informative/v.cells; display(v)
fig,ax=plt.subplots(figsize=(8,4)); sns.barplot(data=v,x="feature",y="coverage",color="#4c72b0",ax=ax)
ax.set(title="Cells with enough events to evaluate a contrast",xlabel="",ylabel="Informative share"); ax.tick_params(axis="x",rotation=20)
ax.yaxis.set_major_formatter(lambda x,_:f"{x:.0%}"); plt.tight_layout(); plt.savefig(FIGURES/"primary_expressive_measure_coverage.png",bbox_inches="tight",dpi=180); plt.show()'''),
("c",'''question_rates=pd.read_csv(TABLES/"primary_question_qualifying_cue_rates.csv")
display(question_rates.sort_values("qualifying_cue_incidence",ascending=False).head(20))
print("Each proposition denominator:",int(question_rates["count"].iloc[0]),"answers; maximum incidence:",f"{question_rates.qualifying_cue_incidence.max():.1%}")
for qid,regime in [(20,"authoritarian"),(34,"libertarian"),(42,"authoritarian")]:
    z=ordered(c[(c.feature=="hedging")&(c.regime==regime)&(c.question_id==qid)])
    print(f"Q{qid} ({regime}):",int(z.substantive.sum()),"of",len(z),"primary variants cross the five-point screen")
    display(z[["model_label","left_rate","right_rate","risk_difference","informative","substantive"]])''')])
}
    for name,x in books.items():
        nbf.write(x,OUT/name); print(OUT/name)
if __name__=="__main__": main()
