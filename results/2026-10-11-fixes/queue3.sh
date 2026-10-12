until grep -q BATCH_DONE /workspace/batch2.log; do sleep 60; done
cd /workspace/needle && git pull -q && echo "PULLED $(git log --oneline -1)"
bash /workspace/batch3.sh
RUNPOD_API_KEY=$(cat /root/.rp_key) runpodctl stop pod 9eh9hp0rd9pu3r
