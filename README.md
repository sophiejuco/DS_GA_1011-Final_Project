# DS_GA_1011-Final_Project

## Repo Contents
* code: folder with code needed to run the model
  * external/standardizedtests: folder with the train, dev, and test data
* data: folder with raw data lines in xlsx file and jupyter notebook to shuffle, transform, and split data into the train, test, and dev jsonl files
* paper: folder with final paper associated with project
* results: folder with the token saliency scores in a json file, jupyter notebook to format the tokens/scores, and output txt file with the final formatted results


## Instructions to run main code
* Download code folder with external/standardizedtests folder inside to your working HPC (we used NYU Greene)
* In run_zs_sat.sh & run_zs_sat2.sh, change the cache_dir to your desired cache directory
* Connect to Greene GPU
  ````bash
  ssh burst
  srun --account=ds_ga_1011-2024fa --partition=g2-standard-12 --gres=gpu:1 --time=1:00:00 --pty /bin/bash
  ````
* Import files from Greene dashboard 
  ````bash
  cd /scratch/smj490
  scp -r greene-dtn:/scratch/smj490/code .
  # if error run:
  ssh-keygen -R greene-dtn
  > yes
  scp -r greene-dtn:/scratch/smj490/code .
  ````
* Create a virtualenv and install dependecies
  ````bash
  cd code
  python -m venv venv
  source venv/bin/activate
  (venv) pip install -r requirements.txt
  (venv) bash setup_env.sh
  # yes to all prompts
  (venv) pip uninstall torch
  (venv) pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
  (venv) mkdir -p gpt2-xl
  (venv) wget https://huggingface.co/gpt2-xl/resolve/main/pytorch_model.bin -O gpt2-xl/pytorch_model.bin
  (venv) wget https://huggingface.co/gpt2-xl/resolve/main/config.json -O gpt2-xl/config.json
  (venv) wget https://huggingface.co/gpt2-xl/resolve/main/tokenizer.json -O gpt2-xl/tokenizer.json 
  (venv) wget https://huggingface.co/gpt2-xl/resolve/main/vocab.json -O gpt2-xl/vocab.json 
  (venv) wget https://huggingface.co/gpt2-xl/resolve/main/merges.txt -O gpt2-xl/merges.txt 
  (venv) echo '{"bos_token": null, "eos_token": "<|endoftext|>", "unk_token": null, "pad_token": null, "mask_token": null}' > gpt2-xl/special_tokens_map.json
  ````
* Run model code to get model parameters
  ````bash
  cd /scratch/smj490/code
  source venv/bin/activate
  (venv) bash run_zs_sat.sh 0 standardizedtests train
  ````
* Run model to get saliency scores
  ````bash
  cd /scratch/smj490/code
  source venv/bin/activate
  (venv) bash run_zs_sat2.sh 0 standardizedtests train
  ````
* Save results
  ````bash
  scp -r results greene-dtn:/scratch/smj490/code/
  scp results_with_saliency.json greene-dtn:/scratch/smj490/code/
  # if any errors run:
  ssh-keygen -R greene-dtn
  [scp command again]
  > yes
  ````

  Any questions, issues, or concerns - feel free to email smj490@nyu.edu
