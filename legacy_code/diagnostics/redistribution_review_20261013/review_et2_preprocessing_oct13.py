"""Compare current ET segmentation against the pinned model's companion predictor."""
import csv,io,json,sys,zipfile,hashlib
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer
sys.path.insert(0,'/Users/wansookim/Documents/decoder_based_va_prediction')
from va_model_code.decoder_va.gaze import segment_text_for_et2
root=Path('/private/tmp/decoder-va-review-tokenizers-20261013')
qt=AutoTokenizer.from_pretrained(root/'qwen',local_files_only=True)
et=AutoTokenizer.from_pretrained(root/'et2',local_files_only=True,add_prefix_space=True)
with zipfile.ZipFile('/Users/wansookim/Downloads/sigma_sentence_only_l0p25r1_l1r0p25_results_only_20261007_135016.zip') as z:
 with zipfile.ZipFile(io.BytesIO(z.read('right_l0p25_r1_results_only.zip'))) as zz:
  name=next(n for n in zz.namelist() if n.endswith('/oof_predictions.tsv'))
  rows=list(csv.DictReader(io.StringIO(zz.read(name).decode()),delimiter='\t'))
counts=Counter(); examples=[]
for row in rows:
 ids=qt(row['text'],truncation=True,max_length=200)['input_ids']
 text=qt.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
 current_words=segment_text_for_et2(text); official_words=text.strip().split()
 current=et(current_words,is_split_into_words=True)['input_ids'] if current_words else []
 official=et(official_words,is_split_into_words=True)['input_ids'] if official_words else []
 counts['rows']+=1; counts['nonempty_rows']+=int(bool(current_words)); counts['changed_input_rows']+=int(current!=official)
 counts['current_segments']+=len(current_words); counts['official_words']+=len(official_words)
 counts['current_model_tokens']+=len(current); counts['official_model_tokens']+=len(official)
 if current!=official and len(examples)<3:
  examples.append({'text':text,'current_segments':current_words,'official_words':official_words,'current_tokens':et.convert_ids_to_tokens(current),'official_tokens':et.convert_ids_to_tokens(official)})
text="I'm sad!!!"
for label,words in [('current',segment_text_for_et2(text)),('official',text.strip().split())]:
 enc=et(words,is_split_into_words=True)
 print(label,words,et.convert_ids_to_tokens(enc['input_ids']),flush=True)
print(dict(counts),flush=True)
Path(__file__).with_suffix('.json').write_text(json.dumps({'counts':dict(counts),'examples':examples},ensure_ascii=False,indent=2))
