.PHONY: setup pretrain

setup:
	pipreqs --force --savepath requirements.txt .
	pip install -r requirements.txt
	wandb login 717edad943a06b407b263a478294a045a73dfea3

TRAIN_FULL ?= False
DEBUG ?= False
NUM_EPOCHS ?= 400

pretrain:
	python pretrain.py --debug=$(DEBUG)\
		--num_epochs=$(NUM_EPOCHS)

finetune:
	python finetune.py --debug=$(DEBUG)\
		--num_epochs=$(NUM_EPOCHS)\
		--train_full=$(TRAIN_FULL)