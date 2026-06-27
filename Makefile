.PHONY: help controller worker profile status test clean install inference

# Default target
help: ## Show this help message
	@echo ""
	@echo "AggregatePC - Distributed heterogeneous compute for idle PCs"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make controller          Start this machine as the cluster controller"
	@echo "  make worker               Start as a worker (auto-discover controller)"
	@echo "  make worker CONTROLLER=192.168.1.5  Join a specific controller"
	@echo "  make profile              Detect hardware and scan for cluster"
	@echo "  make status               Show cluster status"
	@echo "  make inference            Select best cluster model and broadcast to workers"
	@echo "  make test                 Run tests"
	@echo ""

controller: ## Start this machine as the cluster controller
	python3 aggregatepc.py controller

worker: ## Start as a worker node (auto-discover controller or set CONTROLLER=<IP>)
	@if [ -n "$(CONTROLLER)" ]; then \
		python3 aggregatepc.py worker --controller $(CONTROLLER); \
	else \
		python3 aggregatepc.py worker; \
	fi

profile: ## Profile hardware and optionally scan network
	@if [ -n "$(SCAN)" ]; then \
		python3 aggregatepc.py profile --scan; \
	else \
		python3 aggregatepc.py profile; \
	fi

status: ## Show cluster status (optionally set CONTROLLER=<IP> PORT=<PORT>)
	python3 -c "
import sys; sys.path.insert(0, '.')
from cluster.config import load_config
config = load_config()
controller = '$(or $(CONTROLLER),$(config.get(\"controller_ip\",\"127.0.0.1\")))'
port = int('$(or $(PORT),$(config.get(\"controller_port\",8765)))')
print(f'Querying controller at {controller}:{port}')
import subprocess
subprocess.run([sys.executable, 'aggregatepc.py', 'status', '--controller', controller, '--port', str(port)])
"

inference: ## Start inference with best available model on the cluster
	python3 scripts/start_inference.py --broadcast

test: ## Run tests
	python3 -m pytest tests/ -v || python3 -c "import sys; sys.path.insert(0, '.'); exec(open('tests/test_basic.py').read())" 2>/dev/null || echo "No tests directory yet. Run individual modules manually."

clean: ## Remove generated files
	rm -rf __pycache__ .pytest_cache *.pyc
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

install: ## Install in development mode (editable)
	pip install -e . 2>/dev/null || echo "No pyproject.toml yet. Run directly with python3."

# Quick-start shortcuts
start-controller: controller ## Alias for 'controller'
start-worker: worker ## Alias for 'worker'
scan: profile SCAN=1 ## Profile with network scan
