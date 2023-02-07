# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import datasets

_CITATION = """
@article{2019arXiv,
  author = {Saxton, Grefenstette, Hill, Kohli},
  title = {Analysing Mathematical Reasoning Abilities of Neural Models},
  year = {2019},
  journal = {arXiv:1904.01557}
}
"""

_DESCRIPTION = """
This is our personalized script to load the mathematical datasets
"""

_URL = "https://github.com/andrear632/ProjectDeepLearning/blob/main/dataset_round_2000000.zip?raw=true"


class MathDataset(datasets.GeneratorBasedBuilder):
  
    VERSION = datasets.Version("1.1.0")

    def _info(self):
        features = datasets.Features(
            {
                "question": datasets.Value("string"),
                "answer": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            citation=_CITATION
        )

    def process_data(self, origin_path, output_path):
        
        # Create destination directory
        if not os.path.exists(output_path) :
          os.makedirs(output_path)
        
        # Iterate over files in origin directory
        for file_name in os.listdir(origin_path):
          
          # Open origin file in read mode and destination file in write mode
          with open(origin_path+file_name, "r") as input_file, open(output_path+file_name[:-4]+".json", "w") as output_file:
            
            output_line = {
              "question": "",
              "answer": "" 
            }
            # Iterating over the lines in the origin file
            for line_no, line in enumerate(input_file):
              
              if line_no % 2 != 0:
                output_line["answer"] = line.strip()
                output_file.write(json.dumps(output_line))
                output_file.write('\n')
                continue
              output_line["question"] = line.strip()

            input_file.close()
            output_file.close()


    def _split_generators(self, dl_manager):
        url = _URL
        data_dir = dl_manager.download_and_extract(url)

        self.process_data(os.path.join(data_dir, "dataset/train/"), "/content/jsondataset/train/")
        self.process_data(os.path.join(data_dir, "dataset/interpolate/"), "/content/jsondataset/interpolate/")
        self.process_data(os.path.join(data_dir, "dataset/extrapolate/"), "/content/jsondataset/extrapolate/")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": "/content/jsondataset/train/"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split("interpolate"),
                gen_kwargs={
                    "filepath": "/content/jsondataset/interpolate/"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split("extrapolate"),
                gen_kwargs={
                    "filepath": "/content/jsondataset/extrapolate/"
                },
            ),
        ]


    def _generate_examples(self, filepath):
        key = 0
        for file_name in os.listdir(filepath):
            with open(filepath+file_name, "r") as input_file:
                for line_number, line in enumerate(input_file):
                    data = json.loads(line)
                    yield key+line_number, {
                        "question" : data["question"],
                        "answer" : data["answer"]
                    }
                key = key+line_number+1
                input_file.close()
