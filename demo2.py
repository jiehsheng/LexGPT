
import torch
import argparse

from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--load_in_4bit_fp4", default=False, action='store_true')
parser.add_argument("--load_in_4bit_nf4", default=False, action='store_true')
parser.add_argument("--load_in_8bit", default=False, action='store_true')
parser.add_argument("--load_in_16bit", default=False, action='store_true')
args = parser.parse_args()

def main():
   model_id = "patent/LexGPT-6B"

   if args.load_in_4bit_fp4:
      # 4-bit 
      fp4_config = BitsAndBytesConfig(
         load_in_4bit=True,
         bnb_4bit_compute_dtype=torch.bfloat16
      ) # default: fp4
      print('loading model in 4-bit (fp4)...')
      model = AutoModelForCausalLM.from_pretrained(model_id,
        quantization_config=fp4_config, device_map='auto')
      # NVIDIA RTX A5000 Laptop GPU
      # gpu usage: 5513MiB / 16384MiB 

      # (Windows) GeForce RTX 3080 Laptop GPU, VRAM = 16G
      # gpu usage: 6978MiB / 16384MiB 
      # using bitsandbytes on Windows 10: 
      # (0.39.1) https://github.com/jllllll/bitsandbytes-windows-webui
   elif args.load_in_8bit:
      # 8-bit
      print('loading model in 8-bit...')
      model = AutoModelForCausalLM.from_pretrained(model_id, 
        load_in_8bit=True, device_map='auto')
      # (Ubuntu) NVIDIA RTX A5000 Laptop GPU, VRAM = 16G
      # gpu usage: 7871MiB / 16384MiB

      # (Windows) GeForce RTX 3080 Laptop GPU, VRAM = 16G
      # gpu usage: 8967MiB / 16384MiB 
   elif args.load_in_16bit:
      # 16-bit
      print('loading model in 16-bit...')
      model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')
      # (Ubuntu) NVIDIA RTX A5000 Laptop GPU, VRAM = 16G
      # Out of Memory

      # (Windows) GeForce RTX 3080 Laptop GPU, VRAM = 16G
      # gpu usage: 16100MiB / 16384MiB 
      # --> it can run (but extremely slow)

      # (MacOS) Apple M1, RAM = 64G 
      # gpu usage: 56.58GB / 64GB 
      # --> bitsandbytes does not work
   else:  # args.load_in_4bit_nf4:
      # 4-bit 
      nf4_config = BitsAndBytesConfig(
         load_in_4bit=True,
         bnb_4bit_compute_dtype=torch.bfloat16,
         bnb_4bit_quant_type="nf4",
         bnb_4bit_use_double_quant=True
      )
      print('loading model in 4-bit (nf4)...')
      model = AutoModelForCausalLM.from_pretrained(model_id,
        quantization_config=nf4_config, device_map='auto')
      # (Ubuntu) NVIDIA RTX A5000 Laptop GPU, VRAM = 16G
      # gpu usage: 5257MiB / 16384MiB

      # (Windows) GeForce RTX 3080 Laptop GPU, VRAM = 16G
      # gpu usage: 6692MiB / 16384MiB 

   tokenizer = AutoTokenizer.from_pretrained(model_id)
   tokenizer.pad_token = tokenizer.eos_token
   prompt = "In this case, the trial court found"
   if torch.cuda.is_available(): 
      # NVIDIA GPU (CUDA)
      print('using: GPU')
      input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
   elif torch.backends.mps.is_available(): 
      # MacBook (Apple, Metal Performance Shaders)
      print('using: MPS')  # for torch only, not bitsandbytes
      input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('mps')
   else: 
      # CPU
      print('using: CPU')
      input_ids = tokenizer(prompt, return_tensors="pt").input_ids

   print('prompt: %s' % prompt)
   tokens = model.generate(input_ids, do_sample=True, top_p=0.7, top_k=40, temperature=1.0, pad_token_id=tokenizer.eos_token_id, max_length=256)
   result = tokenizer.batch_decode(tokens)[0]
   print('generated:\n%s' % result)
   fn = 'result.txt'
   with open(fn, 'w') as f:
      f.write(result)
      print('saved: %s' % fn)      

   print('done')
   
   # debug
   # import pdb; pdb.set_trace()

if __name__ == '__main__':
   main()