.PHONY: ubuntu
ubuntu:
	sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev

mac:
	brew install cmake openmpi

.PHONY: venv
venv:
	if [ ! -d venv ]; then \
		virtualenv --python=$(which python3) venv; \
	fi
	. venv/bin/activate \
		&& pip install -r requirements.txt \
		&& pip install -r requirements-local.txt

.PHONY: clean
clean:
	rm -rf venv
