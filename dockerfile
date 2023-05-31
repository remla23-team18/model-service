# Use the Python 3.7 image
FROM python:3.10.6

# Set the working directory
WORKDIR /root

ENV POETRY_HOME="/opt/poetry" \
    POETRY_VERSION=1.5.1 \
    PATH="/opt/poetry/bin:$PATH"

RUN curl -sSL https://install.python-poetry.org | python3 -

# Copy the requirements file to the container
COPY pyproject.toml poetry.lock /root/

# Install the Python dependencies
RUN poetry install --no-root

# Copy the application code to the container
COPY app.py /root/

# Set the entrypoint and default command for the container
ENTRYPOINT ["poetry", "run", "python"]
CMD ["app.py"]
