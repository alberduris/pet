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
be added. The PVP is responsible for applying patterns to inputs and mapping
labels to their verbalizations (see the paper for more details on PVPs).
This file shows an example of a PVP for a new task.
"""

from typing import List

from preprocessor import PVPS
from pvp import PVP
from utils import InputExample


class MyTaskPVP(PVP):
    """
    Example for a pattern-verbalizer pair (PVP).
    """

    # Set this to the name of the task
    TASK_NAME = "corpusv"

    # Set this to the verbalizer for the given task: a mapping from the task's labels (which can be obtained using
    # the corresponding DataProcessor's get_labels method) to tokens from the language model's vocabulary
    VERBALIZER = {
        'Pneumonia': ['pneumonia', 'cold', 'cough', 'fever'],
        'Maternal': ['baby', 'child', 'pregnant', 'delivery', 'vagina', 'cervix'], # convulsions, bleed
        'Other Injuries': ['electric', 'fell', 'iron', 'dynamite', 'gas', 'lightning'],
        'Bite of Venomous Animal': ['bite', 'snake', 'bit', 'poison', 'bitten'],
        'Stillbirth': ['womb', 'labor', 'delivery'], # born
        'Other Cancers': ['cancer', 'chemotherapy', 'marrow', 'platelet', 'leukemia'], # blood, cancer
        'Stroke': ['attack', 'stroke', 'paralysis', 'brain', 'bp'], # sugar
        'Other Non-communicable Diseases': ['cancer', 'perforation', 'appendectomy', 'chemotherapy'],
        'Other Infectious Diseases': ['infection', 'fever', 'typhoid'],
        'Malaria': ['malaria'], # fever
        'Acute Myocardial Infarction': ['heart', 'attack', 'pressure', 'ecg'], # chest, pain, pressure, breathing
        'Suicide': ['suicide', 'poison', 'stress', 'hanged', 'kerosene', 'burnt', 'immolated', 'mentally'],
        'Cervical Cancer': ['cervical', 'uterus', 'vaginal'], # cancer, radiation, tumor
        'Preterm Delivery': ['preterm', 'delivery', 'baby', 'born', 'pregnancy'], # blood, bleeding, incubator
        'Other Cardiovascular Diseases': ['heart', 'cardiovascular', 'chest'], # breathing
        'Congenital malformation': ['malformation', 'baby', 'child', 'problem', 'develop', 'disability', 'abnormal', 'hydrocephalus'],
        'Sepsis': ['sepsis', 'swine', 'flu', 'bacteria', 'brain'],
        'AIDS': ['aids', 'hiv', 'viral'],
        'Road Traffic': ['road', 'accident', 'motorcycle', 'bus', 'bicycle', 'car', 'vehicle', 'hit', 'truck'], # ct, scan, trauma
        'Cirrhosis': ['cirrhosis','liver', 'jaundice'], # drunk, drink, alcoholic, yellow, kidneys
        'Falls': ['fall', 'fell', 'fracture', 'roof', 'tree', 'height'], # ct, hit, slip
        'Diarrhea/Dysentery': ['motions', 'vomit', 'diarrhea', 'dysentery', 'bowel', 'stool'], # infection, stomach, fever
        'Leukemia/Lymphomas': ['leukemia', 'lymphoma', 'lymphnodes', 'petechiae'], #chemotherapy, neck, radiotherapy
        'Renal Failure': ['kidney', 'dialysis', 'excrete', 'urinate'],
        'Fires': ['burnt', 'electric', 'fire', 'kerosene', 'burned', 'explode', 'blisters', 'gasoline'],
        'Diabetes': ['sugar', 'diabetes'], #bp, 
        'Birth asphyxia': ['asphyxia'], # aspirate, breathing
        'Poisonings': ['consumed', 'drank', 'poison', 'poisonous', 'foam', 'inhale', 'chemicals', 'petrol', 'pesticide'], # breathing
        'TB': ['tb', 'tuberculosis', 't.b', 'lung'], # breathing
        'Other Defined Causes of Child Deaths': ['baby', 'child', 'born'],
        'Meningitis': ['baby', 'child', 'head', 'brain', 'fever', 'meningitis'],
        'COPD': ['copd', 'bar'], # breathing, lungs, respiration, respiratory, asthma
        'Drowning': ['water', 'drown', 'lake', 'drawn', 'slip', 'river', 'flood', 'ocean', 'sea', 'swim', 'pond'], # fell
        'Colorectal Cancer': ['stomach', 'colon', 'bowel', 'ileostomy', 'endoscopy', 'colostomy'], # cancer, chemotherapy, metastasize, stool, biopsy
        'Other Digestive Diseases': ['stomach', 'intestine', 'appendix', 'stomachache', 'peritonitis'],
        'Measles': ['measles', 'rashes', 'pox'], # fever, convulsions
        'Homicide': ['shoot', 'shot', 'victim', 'wound', 'kerosene', 'fire', 'cops', 'gun', 'aggressor', 'stab', 'killed', 'thief', 'thieves'],
        'Breast Cancer': ['breast', 'nodes', 'mammogram', 'noctal'], # cancer,chemotherapy, biopsy, removed, mass
        'Hemorrhagic fever': ['dengue', 'fever', 'rashes', 'scars'],
        'Encephalitis': ['brain', 'pus', 'grub'], # fever, convulsions
        'Epilepsy': ['epilepsy', 'convulsions', 'seizure'],
        'Lung Cancer': ['lung', 'smoking', 'ct', 'chest', 'diaphragm'], # cancer, tumor, lump, lung, x-ray, chemotherapy, metastasis, biopsy, malignant, ultrasound
        'Violent Death': ['killed', 'police', 'injuries', 'murdered', 'suicide', 'attacked', 'hanged', 'scold', 'axe', 'bomb', 'explosion'], # cut, throat, neck
        'Prostate Cancer': ['prostate', 'bladder', 'carcinoma', 'urinary'], # cancer
        'Stomach Cancer': ['stomach', 'gastritis', 'endoscopy'], # cancer, tumor, mass, stool, biopsy
        'Esophageal Cancer': ['esophagus', 'cancer', 'swallowing', 'endoscopy'], #tumor
        'Meningitis/Sepsis': ['infection', 'septicaemia'],
        'Asthma': ['asthma', 'attack', 'lungs', 'dust', 'nebulization', 'dyspnea'] # breathing, fever, oxygen
    }

    def get_parts(self, example: InputExample):
        """
        This function defines the actual patterns: It takes as input an example and outputs the result of applying a
        pattern to it. To allow for multiple patterns, a pattern_id can be passed to the PVP's constructor. This
        method must implement the application of all patterns.
        """

        # We tell the tokenizer that both text_a and text_b can be truncated if the resulting sequence is longer than
        # our language model's max sequence length.
        text = self.shortenable(example.text_a)

        # For each pattern_id, we define the corresponding pattern and return a pair of text a and text b (where text b
        # can also be empty).
        if self.pattern_id == 0:
            return ['the death wa caused by', self.mask, '.', text], []
        elif self.pattern_id == 1:
            return [text, 'the deceased had suffered from', self.mask, '.'], []
        elif self.pattern_id == 2:
            return ['Just', self.mask, "!"], [text]
        elif self.pattern_id == 3:
            return [text], ['deceased died', self.mask, '.']
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return MyTaskPVP.VERBALIZER[label]


# register the PVP for this task with its name
PVPS[MyTaskPVP.TASK_NAME] = MyTaskPVP
