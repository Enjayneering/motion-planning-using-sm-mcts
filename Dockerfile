# Use Ubuntu as a parent image
FROM ubuntu:22.04

# Set the working directory in the container
WORKDIR /algo

# Copy the current directory contents into the container at /app
COPY . /algo

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y python3 python3-pip
RUN sudo apt-get install python3-tk
RUN sudo apt-get install python3-dev graphviz libgraphviz-dev pkg-config
RUN pip install --no-cache-dir -r requirements.txt

# Make port 9000 available to the world outside this container
EXPOSE 9000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
#CMD ["python", "algo.py"]

