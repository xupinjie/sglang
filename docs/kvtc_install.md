<!-- lmsysorg/sglang:v0.5.10-cu130 -->
```bash
# install sglang
git clone https://github.com/xupinjie/sglang
git checkout pinjie/kvtc
cd sglang
pip install -e "python"

# install kvtc
git cloen ssh://git@gitlab-master.nvidia.com:12051/pinjiex/kvtc-sglang.git
cd kvtc-sglang
git checkout main_sglang
pip install -e third_party/rope/ 
pip install -e src/ 
pip install -e third_party/LMCache/ --no-build-isolation 
pip install -e third_party/vllm_stuff/ 
```