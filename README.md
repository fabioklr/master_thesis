# A neural networks approach to European corporate communications in the wake of Fridays for Future
Follow the instructions below to replicate the data analysis I conducted for my Master's thesis.

## Install Docker
Go to the [Docker Website](https://www.docker.com/) and follow the instructions to install the right version for your OS.

## Build an image
First, clone this repository on a server or on your local machine. Open the CLI and navigate to the folder. Before building the image, it is important to note that its size will be approximately 5GB. If the build exits with a code 137, it means that you do not have enough memory or storage. Build the image with the following command:

```bash
docker build .
```

Next, run the container with the following command. Insert whatever ID was assigned to the image:

## Run and access the container

```bash
docker run -dit <IMAGE> /bin/bash
```

It is best to access the container with your favorite editor to perform the data analysis. VS Code has an excellent Docker extension. If you prefer the command line, execute the following:

```bash
docker exec -it <CONTAINER> /bin/bash
```

## Replicate the process
You need keys and tokens to the Twitter and the DeepL APIs to perform the preprocessing part of the analysis. You can get them from [Twitter's website](https://developer.twitter.com/en/apps) and [DeepL's website](https://www.deepl.com/docs/api/account/api_key). When you have the keys and tokens, follow the instructions at the beginning of the `preprocess.py` script.

In case you are experiencing any difficulties or if you have questions, do not hesitate to open an issue.

