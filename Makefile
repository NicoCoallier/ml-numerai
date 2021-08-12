.PHONY: all

info: header
define HEADER

  _   _ _    _ __  __ ______ _____            _____
 | \ | | |  | |  \/  |  ____|  __ \     /\   |_   _|
 |  \| | |  | | \  / | |__  | |__) |   /  \    | |
 | . ` | |  | | |\/| |  __| |  _  /   / /\ \   | |
 | |\  | |__| | |  | | |____| | \ \  / ____ \ _| |_
 |_| \_|\____/|_|  |_|______|_|  \_\/_/    \_\_____|

endef
export HEADER

help:
	echo "    setup"
	echo "        Create a virtualenv and install project dependencies"
	echo "    train"
	echo "        Launch a model training"
	echo "    predict"
	echo "        Generate tournament predictions"
	echo "    tensorboard"
	echo "        Launch tensorboard UI"
	echo "    test"
	echo "        Launch unittest"

setup:
	clear &&\
	echo "$$HEADER" &&\
	echo "WARNING, you will need poetry for this step. See README for installation process for poetry." &&\
	poetry install

train:
	clear &&\
	echo "$$HEADER" &&\
	poetry run python staker/train.py

predict:
	clear &&\
	echo "$$HEADER" &&\
	poetry run python staker/predict.py

tensorboard:
	clear &&\
	echo "$$HEADER" &&\
	poetry run tensorboard --logdir=logs/

test:
	clear &&\
	echo "$$HEADER" &&\
	poetry run pytest -vvv
