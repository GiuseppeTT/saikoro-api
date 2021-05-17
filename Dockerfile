FROM python:3.9

WORKDIR /app/

# Install poetry and python packages
# Disable virtual environment because it is both unnecessary in and a headache to deal with containers.
ENV POETRY_VIRTUALENVS_CREATE="false"

COPY ./pyproject.toml ./poetry.lock /app/

RUN pip3 --disable-pip-version-check --no-cache-dir install poetry \
    && rm -rf /tmp/pip-tmp \
    && poetry install --no-root --no-dev

# Copy app
COPY ./app /app

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
