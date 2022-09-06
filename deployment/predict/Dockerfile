FROM public.ecr.aws/lambda/python:3.9

COPY [ "app.py", "model.py", "Pipfile", "Pipfile.lock", "./" ]

RUN pip install -U pip
RUN pip install pipenv
RUN pipenv install --system --deploy

CMD [ "app.lambda_handler" ]
