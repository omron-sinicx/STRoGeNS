#!/bin/bash

# python scrape_conferences.py neurips2018
# python scrape_conferences.py neurips2019
# python scrape_conferences.py neurips2020
# python scrape_conferences.py neurips2021
# python scrape_conferences.py neurips2022

python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference acl2020 --save_dir data
python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference acl2021 --save_dir data
python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference acl2022 --save_dir data

python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference emnlp2018 --save_dir data
python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference emnlp2019 --save_dir data
python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference emnlp2020 --save_dir data
python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference emnlp2021 --save_dir data
python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference emnlp2022 --save_dir data

python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference naacl2019 --save_dir data
python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference naacl2021 --save_dir data
python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference naacl2022 --save_dir data

python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference cvpr2018 --save_dir data
python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference cvpr2019 --save_dir data
python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference cvpr2020 --save_dir data
python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference cvpr2021 --save_dir data
python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference cvpr2022 --save_dir data

python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference iccv2017 --save_dir data
python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference iccv2019 --save_dir data
python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference iccv2021 --save_dir data

python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference eccv2018 --save_dir data
python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference eccv2020 --save_dir data
python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference eccv2022 --save_dir data

python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference iclr2019 --save_dir data
python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference iclr2020 --save_dir data
python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference iclr2021 --save_dir data
python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference iclr2022 --save_dir data

python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference icml2019 --save_dir data
python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference icml2020 --save_dir data
python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference icml2021 --save_dir data
python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference icml2022 --save_dir data

python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference icml2023 --save_dir data
python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference acl2023 --save_dir data
python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference cvpr2023 --save_dir data
python preprocessing/conf_processing/scrape_code/scrape_conferences.py --conference iccv2023 --save_dir data

