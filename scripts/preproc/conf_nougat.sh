#!/bin/bash

for conf in acl2020 acl2021 acl2022 emnlp2018 enmlp2019 emnlp2020 emnlp2021 emnlp2022 naacl2019 naacl2021 naacl2022 
do
    nougat data/STRoGeNS-conf22/pdfs/${conf} -o data/STRoGeNS-conf22/mds/${conf} -m 0.1.0-base
done

for conf in cvpr2018 cvpr2019 cvpr2020 cvpr2021 cvpr2022 iccv2017 iccv2019 iccv2021 eccv2018 eccv2020 eccv2022
do
    nougat data/STRoGeNS-conf22/pdfs/${conf} -o data/STRoGeNS-conf22/mds/${conf} -m 0.1.0-base
done

for conf in icml2019 icml2020 icml2021 icml2022
do
    nougat data/STRoGeNS-conf22/pdfs/${conf} -o data/STRoGeNS-conf22/mds/${conf} -m 0.1.0-base
done

for conf in icml2023 acl2023 cvpr2023 iccv2023
do
    nougat data/STRoGeNS-conf22/pdfs/${conf} -o data/STRoGeNS-conf22/mds/${conf} -m 0.1.0-base
done
