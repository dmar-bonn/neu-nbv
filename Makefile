up:
	docker compose -f docker-compose-gpu.yaml up -d

simulator:
ifndef MODEL_NAME
	$(error No model selected)
else
	docker exec -it neu_nbv_container bash -ic "roslaunch simulator ${MODEL_NAME}.launch"
endif

training:
	docker exec -it neu_nbv_container bash

planning:
	docker exec -it neu_nbv_container bash

down:
	docker compose down