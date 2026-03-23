IMAGE_NAME := se6019-he:latest
CONTAINER_WORKDIR := /work/SE6019/Privacy-Preserving


docker-build:
	docker build -t $(IMAGE_NAME) .

docker-train:
	docker run --rm -v $(PWD)/../..:/work -w $(CONTAINER_WORKDIR) $(IMAGE_NAME) python train_he_model.py

docker-he-single:
	docker run --rm -v $(PWD)/../..:/work -w $(CONTAINER_WORKDIR) $(IMAGE_NAME) python he_inference.py

docker-he-batch:
	docker run --rm -v $(PWD)/../..:/work -w $(CONTAINER_WORKDIR) $(IMAGE_NAME) python batch_he_eval.py

docker-compare:
	docker run --rm -v $(PWD)/../..:/work -w $(CONTAINER_WORKDIR) $(IMAGE_NAME) python compare_diff.py
