#!/bin/bash
docker run --name kazuya_related2 -d --gpus=all --rm -it -v $(pwd):/workdir -w /workdir naivete5656/relatedwork2 /bin/bash