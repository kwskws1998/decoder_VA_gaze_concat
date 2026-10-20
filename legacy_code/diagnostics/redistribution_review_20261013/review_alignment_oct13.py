"""Read-only corpus alignment and ET2 tokenizer coverage audit."""
import csv,io,json,hashlib,sys,zipfile
from pathlib import Path
from collections import Counter
import numpy as np
from transformers import AutoTokenizer
sys.path.insert(0,'/Users/wansookim/Documents/decoder_based_va_prediction')
from va_model_code.decoder_va.gaze import segment_text_for_et2
from va_model_code.decoder_va.alignment import align_words_to_tokens
root=Path('/private/tmp/decoder-va-review-tokenizers-20261013')
qt=AutoTokenizer.from_pretrained(root/'qwen',local_files_only=True)
et=AutoTokenizer.from_pretrained(root/'et2',local_files_only=True,add_prefix_space=True)
source=Path('/Users/wansookim/Downloads/sigma_sentence_only_l0p25r1_l1r0p25_results_only_20261007_135016.zip')
with zipfile.ZipFile(source) as z:
 with zipfile.ZipFile(io.BytesIO(z.read('right_l0p25_r1_results_only.zip'))) as zz:
  name=next(n for n in zz.namelist() if n.endswith('/oof_predictions.tsv'))
  rows=list(csv.DictReader(io.StringIO(zz.read(name).decode()),delimiter='\t'))
stats=Counter(); gaps=Counter(); token_lengths=[]; et_lengths=[]; collisions=[]; errors=[]
for idx,row in enumerate(rows):
 try:
  original=row['text']; full=qt(original)['input_ids']; ids=full[:200]
  decoded=qt.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
  words=segment_text_for_et2(decoded)
  ali=align_words_to_tokens(words,ids,[1]*len(ids),qt,text=decoded)
  enc=et(words,is_split_into_words=True,padding=False,truncation=False) if words else {'input_ids':[]}
  n=len(enc['input_ids']); et_lengths.append(n); token_lengths.append(len(ids))
  covered=set()
  if words:
   truncated=et(words,is_split_into_words=True,padding=False,truncation=True,max_length=512)
   covered={w for w in truncated.word_ids() if w is not None}
  stats['rows']+=1; stats['original_qwen_over200']+=int(len(full)>200); stats['et_over512']+=int(n>512)
  stats['et_missing_segments']+=len(words)-len(covered)
  stats['segments']+=len(words); stats['gaze_positions']+=sum(ali.first_subword_mask)
  positions=[k for k,v in enumerate(ali.first_subword_mask) if v]
  gaps.update(b-a for a,b in zip(positions,positions[1:])); stats['rows_with_empty_text']+=int(len(words)==0)
  stats['rows_with_zero_gaze_positions']+=int(len(positions)==0)
  destinations=[v[0] for v in ali.word_to_token_indices]
  multiplicities=Counter(destinations)
  stats['collision_rows']+=int(len(destinations)!=len(set(destinations)))
  stats['collision_positions']+=sum(v>1 for v in multiplicities.values())
  stats['extra_collided_segments']+=len(destinations)-len(set(destinations))
  if len(destinations)!=len(set(destinations)) and len(collisions)<5:
   collisions.append({'text':decoded,'words':words,'qwen_tokens':qt.convert_ids_to_tokens(ids),'destinations':destinations})
 except Exception as exc: errors.append({'index':idx,'error':repr(exc),'text':row['text']})
 if (idx+1)%3000==0: print('checked',idx+1,flush=True)
def describe(values):
 """Summarize observed finite integer lengths."""
 return {'min':min(values),'mean':float(np.mean(values)),'max':max(values),'p50':float(np.quantile(values,.5)),'p95':float(np.quantile(values,.95)),'p99':float(np.quantile(values,.99))}
output={'counts':dict(stats),'qwen_lengths_after_truncation':describe(token_lengths),'et_lengths_before_truncation':describe(et_lengths),'anchor_gap_counts':dict(sorted(gaps.items())),'errors':errors,'collision_examples':collisions,'tokenizer_sha256':{label:hashlib.sha256((root/label/'tokenizer.json').read_bytes()).hexdigest() for label in ['qwen','et2']},'zip_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
Path(__file__).with_suffix('.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in output.items() if k!='collision_examples'},ensure_ascii=False,indent=2))
