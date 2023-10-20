import json, os, sys
import evaluate
import pdb
import sys
import numpy as np
import pandas as pd
import pdb


import argparse, os, sys, logging, random, copy


# RCT LABELS
_SUMMARY_ = "[SUMMARY]"
_CONTENT_ = "[CONTENT]"

SPECIAL_TOKENS = {
    'additional_special_tokens': [
        "[AUTHOR]","<null>","[ARTICLE]",
        "[none]", "[METHODS]", "[CONCLUSIONS]", "[RESULTS]", "[BACKGROUND]","[OBJECTIVE]"
    ]
}



DATASET_DIR = "<dataset dir>"
PREDS_DIR = "<predictions dir>"

ROUGE_TYPES = [
  ("rouge1","R1"),
  ("rouge2","R2"),
  ("rouge3","R3"),
  ("rouge4","R4"),
  ("rougeLsum","RL"),
]
rouge_types = [x for x,y in ROUGE_TYPES]

def extract_init_plans_preds(filename):
    search_toks = SPECIAL_TOKENS["additional_special_tokens"] + ["|",_SUMMARY_,_CONTENT_]
    predictions = []
    plans = []
    opredictions = []
    for line in open(filename):
        text = json.loads(line)["pr_summary"]
        opredictions.append(text)
        for tag in search_toks:
            text = text.replace(tag," " + tag + " ")
        idx = 0
        if _SUMMARY_ not in text:
            toks = text.split()
            for i,tok in enumerate(toks):
                if tok not in search_toks:
                    idx = len(" ".join(toks[:(i+1)])) - len(tok)
                    break
            #
            if idx == 0:
                print("> not found any sp token!")
                print(text)
                pdb.set_trace()
            #
        #
        else:
            idx = text.find(_SUMMARY_)
        plan = text[:idx].strip(" ")
        plan = [x.strip().split(" ") for x in plan.lstrip(_CONTENT_).split("|")]
        summary = text[idx + len(_SUMMARY_):].strip(" ")
        predictions.append(summary)
        plans.append(plan)
    #
    return plans, predictions, opredictions


def extract_init_plans_nonregular(filename):
    search_toks = SPECIAL_TOKENS["additional_special_tokens"] + ["|",_SUMMARY_,_CONTENT_]
    predictions = []
    plans = []
    opredictions = []
    for line in open(filename):
        text = json.loads(line)["pr_summary"]
        opredictions.append(text)
        for tag in search_toks:
            text = text.replace(tag," " + tag + " ")
        
        toks = [x for x in text.replace("\n","<n>").split() if x!=""]
        plan = []
        last_ptok = -1
        last_pblock = []
        for i,tt in enumerate(toks):
            if tt in search_toks:
                if tt in SPECIAL_TOKENS["additional_special_tokens"]:
                    last_pblock.append(tt)
                if tt == "|" and len(last_pblock)>0:
                    plan.append(last_pblock)
                    last_pblock = []
                last_ptok = i
        #
        if len(last_pblock)>0:
            plan.append(last_pblock)
        summary = " ".join(toks[last_ptok+1:]).replace("<n>","\n")
        
        predictions.append(summary)
        plans.append(plan)
    #
    return plans, predictions, opredictions




#########################################################



if __name__ == "__main__":    

    parser = argparse.ArgumentParser() 
    parser.add_argument("--pred", "-p", type=str, help="model to sample", default="./test.json")
    args = parser.parse_args()    

    orig_pred_fn = args.pred
    
    splan,spreds,ospreds = extract_init_plans_nonregular(orig_pred_fn)
    
    outfn = args.pred.rstrip(".json") + "-post.json"
    with open(outfn,"w") as out:
        for plan,pred,compl in zip(splan,spreds,ospreds):
            item = {"plan": plan, "pr_summary": pred, "complete_output":compl}
            out.write(json.dumps(item) + "\n")
    print(">>")
