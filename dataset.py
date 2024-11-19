# import torchtext
# torchtext.disable_torchtext_deprecation_warning()
# from torchtext.datasets import IMDB
import datasets

class dataset:

        def __init__(self, dataset_name):
            self.dataset_name = dataset_name

        def train_test_split(self):

            if self.dataset_name == 'IMDb':

                self.dataset_name_internal = 'imdb'

            elif self.dataset_name == 'Yelp':

                self.dataset_name_internal = 'yelp_review_full'
                
            else:
                
                raise ValueError(f"{self.dataset_name} dataset is not available!")

            train_data, test_data = datasets.load_dataset(self.dataset_name_internal, split =['train', 'test'])

            return train_data, test_data
            
                
                
        