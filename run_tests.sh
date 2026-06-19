venv="source .venv/bin/activate && "
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

tmux new-window

tmux split-window -d "${venv} python3 -u src/test_model.py very_small | tee results/test_very_small-${timestamp}.log"
tmux split-window -d "${venv} python3 -u src/test_model.py small | tee results/test_small-${timestamp}.log"
tmux split-window -dh "${venv} python3 -u src/test_model.py medium | tee results/test_medium-${timestamp}.log"
tmux split-window -d "${venv} python3 -u src/test_model.py big | tee results/test_big-${timestamp}.log"
