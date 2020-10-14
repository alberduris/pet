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

"""
To add a new task to PET, both a DataProcessor and a PVP for this task must
be added. The DataProcessor is responsible for loading training and test data.
This file shows an example of a DataProcessor for a new task.
"""

import csv
import os
from typing import List, Union

from tasks import DataProcessor, LimitedExampleList, PROCESSORS
from utils import InputExample


class CorpusVDataProcessor(DataProcessor):
    """
    Example for a data processor.
    """

    # Set this to the name of the task
    TASK_NAME = "corpusv"

    # Set this to the name of the file containing the train examples
    TRAIN_FILE_NAME = "cleaned_PHMRC_CQ_VAI_redacted_free_text.train.csv"

    # Set this to the name of the file containing the test examples
    TEST_FILE_NAME = "cleaned_PHMRC_CQ_VAI_redacted_free_text.test.csv"

    # Set this to a list of all labels in the train + test data
    LABELS = ['Pneumonia',
		 'Maternal',
		 'Other Injuries',
		 'Bite of Venomous Animal',
		 'Stillbirth',
		 'Other Cancers',
		 'Stroke',
		 'Other Non-communicable Diseases',
		 'Other Infectious Diseases',
		 'Malaria',
		 'Acute Myocardial Infarction',
		 'Suicide',
		 'Cervical Cancer',
		 'Preterm Delivery',
		 'Other Cardiovascular Diseases',
		 'Congenital malformation',
		 'Sepsis',
		 'AIDS',
		 'Road Traffic',
		 'Cirrhosis',
		 'Falls',
		 'Diarrhea/Dysentery',
		 'Leukemia/Lymphomas',
		 'Renal Failure',
		 'Fires',
		 'Diabetes',
		 'Birth asphyxia',
		 'Poisonings',
		 'TB',
		 'Other Defined Causes of Child Deaths',
		 'Meningitis',
		 'COPD',
		 'Drowning',
		 'Colorectal Cancer',
		 'Other Digestive Diseases',
		 'Measles',
		 'Homicide',
		 'Breast Cancer',
		 'Hemorrhagic fever',
		 'Encephalitis',
		 'Epilepsy',
		 'Lung Cancer',
		 'Violent Death',
		 'Prostate Cancer',
		 'Stomach Cancer',
		 'Esophageal Cancer',
		 'Meningitis/Sepsis',
		 'Asthma']

    # Set this to the column of the train/test csv files containing the input's text a
    TEXT_A_COLUMN = 1

    # Set this to the column of the train/test csv files containing the input's text b or to -1 if there is no text b
    TEXT_B_COLUMN = -1

    # Set this to the column of the train/test csv files containing the input's gold label
    LABEL_COLUMN = 2

    def get_train_examples(self, data_dir: str,
                           examples_per_label: Union[int, List[int]] = -1, skip_first: int = 0) -> List[InputExample]:
        """
        This method loads train examples from a file with name `TRAIN_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the training data can be found
        :param examples_per_label: the number of examples per label. Can be either a list whose length is equal
        to the number of labels, or a single number. Set this to -1 to load all examples available (default=-1).
        :param skip_first: If set to a value > 0, the first `skip_first` examples per label are skipped (default=0)
        :return: a list of train examples
        """
        return self._create_examples(os.path.join(data_dir, MyTaskDataProcessor.TRAIN_FILE_NAME), "train",
                                     examples_per_label, skip_first=skip_first)

    def get_dev_examples(self, data_dir: str,
                         examples_per_label: Union[int, List[int]] = -1, skip_first: int = 0) -> List[InputExample]:
        """
        This method loads dev/test examples from a file with name `TEST_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the dev data can be found
        :param examples_per_label: the number of examples per label. Can be either a list whose length is equal
        to the number of labels, or a single number. Set this to -1 to load all examples available (default=-1).
        :param skip_first: If set to a value > 0, the first `skip_first` examples per label are skipped (default=0)
        :return: a list of dev examples
        """
        return self._create_examples(os.path.join(data_dir, MyTaskDataProcessor.TEST_FILE_NAME), "test",
                                     examples_per_label, skip_first=skip_first)

    def get_labels(self) -> List[str]:
        """This method returns all possible labels for the task."""
        return MyTaskDataProcessor.LABELS

    def _create_examples(self, path, set_type, max_examples=-1, skip_first=0):
        """Creates examples for the training and dev sets."""
        examples = LimitedExampleList(self.get_labels(), max_examples, skip_first)

        with open(path) as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):

                guid = "%s-%s" % (set_type, idx)
                label = row[MyTaskDataProcessor.LABEL_COLUMN]
                text_a = row[MyTaskDataProcessor.TEXT_A_COLUMN]
                text_b = row[MyTaskDataProcessor.TEXT_B_COLUMN] if MyTaskDataProcessor.TEXT_B_COLUMN >= 0 else None
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
                examples.add(example)

                if examples.is_full():
                    break
        return examples.to_list()


# register the processor for this task with its name
PROCESSORS[MyTaskDataProcessor.TASK_NAME] = MyTaskDataProcessor
