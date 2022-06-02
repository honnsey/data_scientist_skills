# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* data_scientist_skills/*.py

black:
	@black scripts/* data_scientist_skills/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr data_scientist_skills-*.dist-info
	@rm -fr data_scientist_skills.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)


#--------------------------------------------
#				GCP information
#--------------------------------------------
PROJECT_ID = lewagon-859
# REGION = ASIA #for some reason it won't run on ASIA
REGION = europe-west1
BUCKET_NAME=skills_for_a_data_scientist

#For uploading package
BUCKET_FOLDER=package
PACKAGE_LOCAL_PATH = "data_scientist_skills"
PACKAGE_BUCKET_FILE_NAME=$(shell basename ${PACKAGE_LOCAL_PATH})

#Put in with package that get_data() file path works
DATA_LOCAL_PATH = "raw_data"
DATA_BUCKET_FILE_NAME=$(shell basename ${DATA_LOCAL_PATH})

####Project sectup on GCP --------------------------------
set_project:
	@gcloud config set project ${PROJECT_ID}

create_bucket:
	@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}

upload_package:
	@gsutil cp -r ${PACKAGE_LOCAL_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${PACKAGE_BUCKET_FILE_NAME}

#For uploading data
upload_data:
	@gsutil cp -r ${DATA_LOCAL_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${DATA_BUCKET_FILE_NAME}

####Project sectup on GCP ~CLOSE -------------------------


##Training
# will store the packages uploaded to GCP for the training
BUCKET_TRAINING_FOLDER = 'trainings'


##GCP AI Platform
#Machine configuration
PYTHON_VERSION=3.7 #why does 3.8 cause issues? related to runtime_version ### can ignore acc. Alec
FRAMEWORK=scikit-learn #This needs changing depending on model used
RUNTIME_VERSION=2.2


#Package params
PACKAGE_NAME=data_scientist_skills
FILENAME=trainer


##Job
JOB_NAME=data_scientist_skills_pipeline_$(shell date +'%Y%m%d_%H%M%S')


###Runs trainer.py (i.e. creates joblib)
### the "if __name__ == '__main__'" lets trainer.py be run when called directly

run_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}


#requirements.txt needs to be updated, failing certain imports
# gcp_submit_training:
# 	gcloud ai-platform jobs submit training ${JOB_NAME} \
# 		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
# 		--package-path ${PACKAGE_NAME} \
# 		--module-name ${PACKAGE_NAME}.${FILENAME} \
# 		--python-version=${PYTHON_VERSION} \
# 		--runtime-version=${RUNTIME_VERSION} \
# 		--region ${REGION} \
# 		--stream-logs


# ----------------------------------
#      DOCKER IMAGES
# ----------------------------------

PACKAGE_NAME_ALIAS=dss
API_IMAGE_NAME=eu.gcr.io/${PROJECT_ID}/${PACKAGE_NAME_ALIAS}_api
WEB_IMAGE_NAME=eu.gcr.io/${PROJECT_ID}/${PACKAGE_NAME_ALIAS}_web

### Build & Deploy Docker images
### ***Confirm Docker is launched or else the commands will fail***

#------------ Api ------------
### Confirm the .dockerignore has the web and notebooks directories set to ignore
docker_api_build:
	docker build -t ${API_IMAGE_NAME} .
	@echo "You should see ${API_IMAGE_NAME} in your docker images"
	@docker images

docker_api_deploy:
	docker push ${API_IMAGE_NAME}
	@echo "Check this link to confirm image is pushed: https://console.cloud.google.com/gcr/"
	@echo "If this is first deployment, press enter to let the service have the default name, enter 'y' to continue"
	gcloud run deploy --image ${API_IMAGE_NAME} --platform managed --region europe-west1

#------------ Web ------------
### As the web package is not in the home directory, this command runs the
### dockerfile from within that directory then returns to the main directory
docker_web_build:
	cd web
	docker build -t ${WEB_IMAGE_NAME} .
	cd -
	@echo "You should see ${WEB_IMAGE_NAME} in your docker images"
	@docker images

docker_web_deploy:
	docker push ${WEB_IMAGE_NAME}
	@echo "Check this link to confirm image is pushed: https://console.cloud.google.com/gcr/"
	@echo "If this is first deployment, press enter to let the service have the default name, enter 'y' to continue"
	gcloud run deploy --image ${WEB_IMAGE_NAME} --platform managed --region europe-west1



#make commands for for testing
#docker_run_{api/web}:
#	docker run -e PORT=8000 -p 8000:8000 ${{WEB/API}_IMAGE_NAME}

### Run Api ---------------------------------------
run_api:
	uvicorn api.fast:app --reload  # load web server with code autoreload
