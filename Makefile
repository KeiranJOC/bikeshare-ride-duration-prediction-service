test:
	pytest tests/test_model.py

integration_test: # requires lambda to be running locally
	pytest tests/test_handler.py

quality_checks:
	black .
	isort .

build: quality_checks test # requires Docker to be running on your computer
	sam build -t  ./deployment/template.yaml

publish: build
	aws ecr get-login-password \
		--region ap-southeast-2 \
		| docker login
			--username AWS \
			--password-stdin 843032675284.dkr.ecr.ap-southeast-2.amazonaws.com
