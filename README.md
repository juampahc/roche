# Roche

⚠️ **Disclaimer**: Except for the content under `layout_api/`, this repository relies totally or partially on the project [PaddleX](https://github.com/PaddlePaddle/PaddleX/tree/release/3.2).

PaddleX is a framework for deploying and building services based on PaddlePaddle models. It provides some handy features:

* Predefined pipelines for quick deployments.
* A high-performance inference plugin that automatically converts models to the most suitable format (ONNX, Paddle, TensorRT).
* An inference server *based on* NVIDIA Triton Inference Server to ensure compatibility with NVIDIA technologies.
* Prebuilt inference images for both GPU and CPU.

Each of these features influences how we build this service.

---

#### Project Status: Active \[WIP]

### 🛠️ Technologies

* **Language**: Python
* **Dependencies**: PIP
* **Inference Engine**: Triton Inference Server / PaddleX
* **Hardware**: CPU, GPU
* **Model Serving Strategy**: FastAPI + Uvicorn
* **Containerization**: Docker
* **Target Platforms**: Kubernetes, Docker-Compose, Docker

---
## Pipelines

Using the predefined PaddleX pipelines is simple and effective… but you *cannot* easily create a custom pipeline. During deployment, the pipeline must be specified by name inside a YAML file (which also sets inference parameters like threshold values). This YAML is then parsed by the PaddleX codebase.

So, if you need a custom pipeline, you have two options:

1. Add extra Python code that registers your pipeline within the library (similar to HuggingFace). Not sure if PaddleX supports this though.
2. Clone the PaddleX repo and add your own code/files.

Both approaches force you to rebuild PaddleX Docker images — messy and hard to maintain, especially since a cloned repo won’t receive updates. For this project, I chose a pipeline that fits our needs fairly well: **formula detection**. It includes layout analysis as part of its process, letting us configure the model with the adjustments we need.

---
## HPI (High Performance Inference Plugin)

This plugin selects the most suitable backend for each model. Since PaddleX is based on NVIDIA Triton Inference Server, you can switch between TensorRT, ONNX, or Paddle. There is an important caveat: Not all models support all backends, so sometimes the plugin won’t work as expected. The PaddleX team themselves recommend trial and error here. Prebuilt images are available with PaddleX **and** this plugin already installed.

---
## HPS

I can assume that the acronym means **High Performance Serving** and it refers to the adaptation of NVIDIA’s Triton Inference Server so that it can work with PaddleX and (optionally with HPI). Again, we have two images: one for CPU and another for GPU (note that the CUDA version is 11.8.0).

Using Triton as an inference server implies that we have two separate elements: on one hand, we have a container running on the “server” image (remember that this server only accepts Tensors as input) and, on the other hand, we have the “client” which is responsible for all the preprocessing of the data in order to send it via gRPC to the “server” for inference.

* For the server we have to include a repository (as is usually the case) with the yaml specifying the pipeline configuration and, in addition, the server configuration, such as the number of instances per GPU (in case horizontal scaling is needed). However, Paddle uses its own directory to download and store the models. For this reason, a script `server.sh` is used to reorganize everything into a directory and then launch Triton.
* For the client we have the Triton package to send gRPC requests from Python and a Python wheel distributed by PaddleX that performs all the necessary preprocessing in order to send a request.

All the files are included in what PaddleX calls the SDK.

---
## Deployment Strategy

Now that we understand how PaddleX works and what it provides, let’s talk about the strategy we will follow to deploy this service. Obviously, we have to work with two containers, so we need to configure and design two clearly separated images: the client and the server.

### Server Image

For the server image we need to consider three elements that make up the pipeline SDK:

* The `model_repo` common to all Triton-based services.
* The directory where PaddleX downloads the models: `/root/.paddlex/official_models`.
* The pipeline configuration file `pipeline_config.yaml`.

With this in mind, we must answer the following questions:

* Is it possible to serve multiple pipelines? Although this is something that Triton could easily handle, it does not seem that HPS supports it at the moment. Each pipeline has its own SDK and, moreover, the script `server.sh` does not appear to handle multiple yaml configuration files.
* Is it possible to modify the model configuration parameters (yaml file)? Suppose you are going to launch a container with this image: in order for it to work correctly, you need to load the three elements mentioned above. One way to achieve this is to save all of it in an AWS S3 bucket and mount it as a volume inside the container, which would allow you to make modifications to the configuration. Obviously, if the container is already running it will need to be restarted.
* How can we load models that we may already have downloaded to avoid the container downloading the models every time it starts? This would greatly speed up service startup time. Again, we need to download the models, store them in S3, and then mount them as a volume in the directory that the container expects, so it doesn’t trigger a new download.

As we can see, these three elements determine what we will be doing during deployment.

I have decided, for the moment, to create an image that will contain (as image layers) the `model_repo` files and `pipeline_config.yaml`. This will make the container serve only one pipeline, but right now it seems the fastest approach. This image can be built with `Dockerfile.inference.gpu` or `Dockerfile.inference.cpu`.

If we choose to build our own images and do a local deployment, once the inference server container has been built we can run it directly with:

```bash
docker run \
    -it \
    -e PADDLEX_HPS_DEVICE_TYPE=gpu \
    -e PADDLEX_HPS_USE_HPIP=1 \
    -v /home/USER/.paddlex:/root/.paddlex \
    --rm \
    --gpus all \
    --init \
    --network host \
    --shm-size 8g \
    TAG
```

---
### Client Image

The reason why we build this image is to provide our applications with a REST-API interface over HTTP to perform DLA. If we want to build the image, we must use `Dockerfile.client`.

---
## ☸️ Kubernetes Deployment

🚧 **Work in progress…**


