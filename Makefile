.PHONY: ubuntu
ubuntu:
	sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev

mac:
	brew install cmake openmpi

.PHONY: venv
venv:
	if [ ! -d venv ]; then \
		if [ -n "$$(which python3.8)" ]; then \
			PYTHON="$$(which python3.8)"; \
		elif [ -n "$$(which python3.7)" ]; then \
		  PYTHON="$$(which python3.7)"; \
		else \
			PYTHON="$$(which python3)"; \
			if [ -z "$$PYTHON" ]; then \
				echo 'muzero requires Python 3 version 3.7 or greater'; \
				exit 1; \
			fi; \
			PYTHON_VERSION="3.$$(python3 --version | sed -e 's/Python 3\.\([0-9]\)\+.*/\1/')"; \
			if [ "$$PTYHON_VERSION" -lt "7" ]; then \
				echo 'muzero requires Python 3 version 3.7 or greater'; \
				exit 1; \
			fi; \
		fi; \
		virtualenv --python="$$PYTHON" venv; \
	fi
	. venv/bin/activate \
		&& pip install -r requirements.txt \
		&& pip install -r requirements-dev.txt

.PHONY: gcloud
gcloud:
	if [ ! -d venv ]; then \
		if [ -n "$$(which python3.8)" ]; then \
			PYTHON="$$(which python3.8)"; \
		elif [ -n "$$(which python3.7)" ]; then \
		  PYTHON="$$(which python3.7)"; \
		else \
			PYTHON="$$(which python3)"; \
			if [ -z "$$PYTHON" ]; then \
				echo 'muzero requires Python 3 version 3.7 or greater'; \
				exit 1; \
			fi; \
			PYTHON_VERSION="3.$$(python3 --version | sed -e 's/Python 3\.\([0-9]\)\+.*/\1/')"; \
			if [ "$$PTYHON_VERSION" -lt "7" ]; then \
				echo 'muzero requires Python 3 version 3.7 or greater'; \
				exit 1; \
			fi; \
		fi; \
		virtualenv --python="$$PYTHON" venv_gcloud; \
	fi
	. venv_gcloud/bin/activate \
		&& pip install -r requirements-missing.txt \
		&& pip install -r requirements.txt \
		&& pip install -r requirements-dev.txt

.PHONY: clean
clean:
	rm -rf venv
