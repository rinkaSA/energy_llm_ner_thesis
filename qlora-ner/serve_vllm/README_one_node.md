## Current usage of two and one node serving+monitoring

Earlier we relied on the consecutive job runnning of two separate job on two different nodes. 
That is described in the README.md file. (basically 2 `sbatch`` of two .sh files where needed)

Now we want to build it in one node, but share the CPU resources reserved for monitor and for serve (to acccurately measure the latter).

For that we work on the 'one_node_serve.sh' file.
At the moment thhe desired behavior of running in parrallel two tasks defined by 'srun' while deviding resources is not reached.

Here are some steps needed to be done before running it for another user.

# 1. Pull docker container desined for vllm serve.
Put it into ./serve_vllm/containers directory

```singularity build vllm_serve_otel.sif docker://irv12/vllm_serve_otel:latest```

# 2. Download LLama 2
prerequsites are having account on hugging face where the usage terms of Meta where accecpted (https://medium.com/@tushitdavergtu/how-to-install-llama-2-locally-d3e3c6c8eb4c)

Install /log into HF.

```pip install huggingface_hub ```

```huggingface-cli login```

And then clone it

```git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf```

Currently the path to the model is:
./energy_ner_llm/qlora-ner/serve_vllm/models/base/Llama-2-7b-chat-hf

/models directory is not on the remote since all the large stuff was gitignored. Create it manually or adjust pathes accordingly.
# 3. Check all the pathes in one_node_serve.sh - change the base one.

Adjust your working directory path

# 4. Run with sbatch one_node_serve.sh

```sbatch one_node_serve.sh```


Check for the node name where it is running

```squeue -u <user_name>``` 

also check for the logs in the dir ```/slurm_onenode``` and all ```.log``` files in ```/monitoring```

# 5. If you want to submit a request to this server....

GO to ```energy_ner_llm/qlora-ner/serve_vllm/monitoring/gpu_node.txt``` and just manually copy-paste a full nodes name (e.g. c14.capella.hpc.tu-dresden.de)

Then run ```sbatch inference_on_vllm.sh```.
It internaly relies on that .txt file for discovering where to send request. (CURRENTLY something doesnt work there, conll03 dataset can not be downloaded??)