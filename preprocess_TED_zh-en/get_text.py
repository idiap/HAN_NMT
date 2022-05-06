from __future__ import absolute_import
from __future__ import print_function
import sys, re
from six.moves import zip
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--tgt", type=str, action="store")
parser.add_argument("--src", type=str, action="store")

args = parser.parse_args()
src = args.src
tgt = args.tgt

a=sys.argv

f_s = open(f"train.tags.{src}-{tgt}.{tgt}")
f_t = open(f"train.tags.{src}-{tgt}.{src}")
f_s_o = open(f"corpus.{tgt}", "w")
f_t_o = open(f"corpus.{src}", "w")
f_doc = open(f"corpus.doc", "w")
f_s_doc = open(f"corpus.doc.{tgt}", "w")
f_t_doc = open(f"corpus.doc.{src}", "w")

count = 0
for ls, lt in zip(f_s, f_t):
  if ls.startswith("<title>"):
    if not lt.startswith("<title>"): 
      print("error "+str(count))
      break
    ls = re.sub("(^\<title\>)(.*)(\</title\>)","\g<2>", ls).strip()
    lt = re.sub("(^\<title\>)(.*)(\</title\>)","\g<2>", lt).strip()

    f_doc.write(str(count)+"\n")
    f_s_doc.write(ls + "\n")
    f_t_doc.write(lt + "\n")

    

  elif ls.startswith("<description>"):
    if not lt.startswith("<description>"): 
      print("error "+str(count))
      break
    ls = re.sub("(^\<description\>)(.*)(\</description\>)","\g<2>", ls).strip()
    lt = re.sub("(^\<description\>)(.*)(\</description\>)","\g<2>", lt).strip()
    f_s_doc.write(ls + "\n")
    f_t_doc.write(lt + "\n")

  elif not ls.startswith("<"):
    if ls.strip()!= "" and lt.strip()!= "":
      f_s_o.write(ls.strip() + "\n")
      f_t_o.write(lt.strip() + "\n")
      count +=1

f_s.close()
f_t.close()
f_s_o.close()
f_t_o.close()
f_doc.close()
f_s_doc.close()
f_t_doc.close()


for test in ["dev2010", "tst2010", "tst2011", "tst2012", "tst2013"]:
  f_s = open(f"IWSLT15.TED." + test +".{src}-{tgt}.{tgt}.xml")
  f_t = open(f"IWSLT15.TED." + test +".{src}-{tgt}.{src}.xml")

  count = 0

  f_s_o = open(f"IWSLT15.TED." + test +".{src}-{tgt}.{tgt}", "w")
  f_t_o = open(f"IWSLT15.TED." + test +".{src}-{tgt}.{src}", "w")
  f_doc = open(f"IWSLT15.TED." + test +".{src}-{tgt}.doc", "w")
  f_s_doc = open(f"IWSLT15.TED." + test +".{src}-{tgt}.doc.{tgt}", "w")
  f_t_doc = open(f"IWSLT15.TED." + test +".{src}-{tgt}.doc.{src}", "w")

  for ls, lt in zip(f_s, f_t):
    if ls.startswith("<talkid>"):
      if not lt.startswith("<talkid>"): 
        print("error "+str(count))
        break
      s = re.sub("(^\<talkid\>)(.*)(\</talkid\>)","\g<2>", ls).strip()
      t = re.sub("(^\<talkid\>)(.*)(\</talkid\>)","\g<2>", lt).strip()

      if s!=t:
        print("error "+str(count)+" "+test)
        break

      f_s_doc.write(ls.strip() + "\n")
      f_t_doc.write(lt.strip() + "\n")
      f_doc.write(str(count) + "\n")
      

    elif ls.startswith("<seg"): 
      if not lt.startswith("<seg"): 
        print("error "+str(count)+" "+test)
        break

      ls = re.sub("(^\<seg.*\>)(.*)(\</seg\>)","\g<2>", ls).strip()
      lt = re.sub("(^\<seg.*\>)(.*)(\</seg\>)","\g<2>", lt).strip()

      if ls.strip()!= "" and lt.strip()!= "":
        f_s_o.write(ls + "\n")
        f_t_o.write(lt + "\n")
        count += 1
    else:

      f_s_doc.write(ls.strip() + "\n")
      f_t_doc.write(lt.strip() + "\n")

  f_s.close()
  f_t.close()
  f_s_o.close()
  f_t_o.close()
  f_doc.close()
  f_s_doc.close()
  f_t_doc.close()




    
          
          
