venv="source .venv/bin/activate && "
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

tmux new-window

tmux split-window -d "${venv} python3 -u src/very_small.py | tee results/train_very_small-${timestamp}.log"
tmux split-window -d "${venv} python3 -u src/small.py | tee results/train_small-${timestamp}.log"
tmux split-window -dh "${venv} python3 -u src/medium.py | tee results/train_medium-${timestamp}.log"
tmux split-window -d "${venv} python3 -u src/big.py | tee results/train_big-${timestamp}.log"

