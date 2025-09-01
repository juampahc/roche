# Roche
DISCLAIMER: A excepción del contenido ubicado bajo `layout_api/` el contenido de este repositorio depende total o parcialmente del proyecto [PaddleX](https://github.com/PaddlePaddle/PaddleX/tree/release/3.2). 

PaddleX es un framework para desplegar y elaborar servicios basados en modelos de PaddlePaddle. Para ello PaddleX cuenta con un par de funcionalidades interesantes:

- Pipelines ya definidos para poder desplegar de manera rápida.
- Un plugin de inferencia acelerada que maneja la conversión a otros formatos de los modelos según convenga (a saber, ONNX, Paddle y TensorRT)
- Un servidor de inferencia *basado* en NVIDIA Triton Inference Server para garantizar la comaptibilidad con sus tecnologías.
- Una imagen disponible del servidor de inferencia para GPU y otra para CPU

Cada una de estas características va a determinar la manera en que montamos este servicio.

#### Project Status: Active [WIP]

### Technologies
* Language: Python
* Dependencies: PIP
* Inference Engine: Triton Inference Server / PaddleX
* Hardware: CPU,GPU
* Model Serving Strategy: FastAPI/Uvicorn
* Container: Docker
* Target Platform: Kubernetes, Docker-Compose, Docker

## Pipelines

Si bien es cierto que utilizar los pipelines definidos por PaddleX es sencillo y efectivo, el problema es que no podemos crear nuestro pipeline custom. Esto se debe a que, durante el despliegue, debemos especificar el pipeline a través de su nombre mediante un yaml (especificando, además, parámetros de inferencia como el threshold que define la confianza con la que se asigna una etiqueta) que será analizado por todo el código de PaddleX.

Todo ello implica que para poder desplegar un pipeline custom tenemos dos opciones:

- Incluir código Python adicional que registre el pipeline en la librería: esta opción está disponible en librerías como HuggingFace. Aunque no estoy seguro de que se pueda hacer en PaddleX, ya que no he revisado todo el código y no sé si los desarrolladores han permitido esta funcionalidad.
- Hacer una copia del repositorio e incluir los archivos y directorios con el código necesario para poder crear nuestro Pipeline.

Cualquiera de estas dos opciones implica tener que rehacer las imágenes Docker ofrecidas por PaddleX (ya que deben incluir nuestro código) lo cual comenzará a ser engorroso. Todo ello sin contar con los inconvenientes de tener un repositorio clonado que no recibe actualizaciones.

Es por esta razón que he decidido utilizar un pipeline que, si bien no es algo que necesitemos para el DLA, se ajusta bastante: la detección de fórmulas. Este pipeline contiene el análisis de layout como parte de su procesamiento, así que podemos especificar el modelo con los ajustes que necesitamos.

## HPI

High Performance Inference Plugin es el nombre que tiene la parte de código que se encarga de seleccionar el backend más adecuado para cada modelo. Recordemos que (al estar basado en NVIDIA Triton Inference Server) podemos escoger entre diferentes tipos de backends/engines como TensorRT, ONNX o el propio Paddle. El probema asociado a este plugin es que no todos los modelos soportan todos los backends, por lo que el plugin podría no funcionar bien. Es algo que desde la propia web recomiendan probar.

De nuevo, hay imágenes preconstruidas que no solo traen PaddleX si no que, además traen este plugin ya instalado.

## HPS

Puedo suponer que el acrónimo significa High Performance Serving y se trata de la adaptación del servidor Triton para inferencia de NVIDIA, de tal manera que pueda funcionar con PaddleX y (opcionalmente con el HPI). De nuevo, contamos con dos imágenes: una para CPU y otra para GPU (Ojo, que la versión de CUDA es la 11.8.0).

Utilizar Triton como servidor de inferencia implica que tenemos dos elementos separados: por un lado, tenemos un contenedor corriendo sobre la imagen del 'servidor' (recordemos que dicho servidor solo acpeta como input Tensores) y, por otro lado, tendremos el 'cliente' que se encarga de hacer todo el preprocesamiento de los datos para enviárselo mediante gRPC al 'servidor' para hacer inferencia.

- Para el servidor tenemos que incluir un repositorio (como suele ser habitual) con el yaml indicando la configuración del pipeline y, además, la configuración del servidor, como el número de instancias para una GPU (en el caso de necesitar escalado horizontal). No obstante, Paddle usa su propio directorio para descargar y guardar los modelos. Por ello se utiliza un script `server.sh` que se encarga de reoriganizar todo en un directorio y después lanzar triton.
- Para el cliente tenemos, por un lado, el paquete de triton para lanzar requests gRPC desde python y, por otro, un wheel de python distribuido por PaddleX que hace todo el preprocesamiento necesario para poder enviar una request.

Todos los archivos están incluidos en lo que PaddleX llama SDK.

## Estrategia de despliegue

Ahora que tenemos claro cómo funciona PaddleX y lo que ofrece vamos a hablar de cuál es la estrategia que vamos a seguir para desplegar este servicio. Evidentemente, tenemos que trabajar con dos contenedores por lo que tenemos que configurar y diseñar dos imágenes claramente separadas: la del cliente y la del servidor.

### Server Image

Para la imagen del servidor tenemos que considerar tres elementos que forman el SDK del pipeline:

- El `model_repo` propio de todos los servicios basados en Triton
- El directorio en el que PaddleX descarga los modelos `/root/.paddlex/official_models`
- El archivo de configuración del pipeline `pipeline_config.yaml`.

Con esto en cuenta debemos responder las siguientes preguntas:

- ¿Es posible servir varios pipelines? Aunque es algo que Triton podría manejar sin problema, no parece que sea algo que HPS admita en este momento. Cada pipeline tiene su SDK y, además el script `server.sh` no parece poder manejar varios archivos yaml de configuración.
- ¿Es posible modificar los parámetros de configuración del modelo (archivo yaml)? Supongamos que vas a lanzar un contenedor con esta imagen: para que funcione correctamente, tienes que cargar los tres elementos que he mencionado antes. Una opción para conseguir esto es que guardes todo ello en un s3 de AWS y que lo montes como volumen dentro del contenedor, de esta manera podrías hacer modificaciones en la configuración. Evidentemente, si el contenedor ya se está ejecutando habrá que  reiniciarlo.
- ¿Como cargamos los modelos que podamos tener ya descargados para evitar que el contenedor descargue los modelos cada vez que arranque? Esto permitiría aligerar mucho el tiempo de carga a la hora de arrancar el servicio. De nuevo, tenemos que descargarnos los modelos, guardarlos en un s3 y después montarlo como volumen en el directorio que el contenedor espera para no hacer una descarga nueva.

Como vemos, estos tres elementos van a determinar lo que vamos a ir haciendo durante el despliegue.

He decidido que, de momento, voy a crear una imagen en la que voy a guardar dentro (en capas de la imagen) los archivos de `model_repo` y `pipeline_config.yaml`. Esto hará que dicho contenedor solo pueda servir un pipeline, pero ahora mismo me parece lo más rápido. Dicha imagen se puede construir con `Dockerfile.inference.gpu` o `Dockerfile.inference.cpu`.

Si optamos por construir nuestras propias imágenes y hacer un despliegue local, una vez que hemos construido el contenedor del servidor de inferencia podemos ejecutar directamente:
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

### Client Image

La razón por la que construimos esta imagen es para poder ofrecer a nuestras aplicaciones una interfaz basada en REST-API por http para realizar DLA. Si queremos construir la imagen, debemos utilizar `Dockerfile.client`.

## Despliegue en Kubernetes

WIP...