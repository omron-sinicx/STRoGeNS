#!/bin/bash
docker network create selenium-network
docker run --name selenium-chrome --network selenium-network -d --rm -it -p 4444:4444 -v /dev/shm:/dev/shm selenium/standalone-chrome:3.141.59-xenon
docker run --name kazuya_related --network selenium-network -d --gpus=all --rm -it -v $(pwd):/workdir -w /workdir naivete5656/relatedwork /bin/bash