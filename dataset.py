# import torchtext
# torchtext.disable_torchtext_deprecation_warning()
# from torchtext.datasets import IMDB
import datasets

class dataset:

        def __init__(self, dataset_name):
            self.dataset_name = dataset_name

        def train_test_split(self):

            if self.dataset_name == 'IMDb':

                train_data, test_data = datasets.load_dataset('imdb', split =['train', 'test'])
    
            else:
    
                print(dataset_name + " dataset is not available!")

            return train_data, test_data
                
        # def fetch_tokens(self):    
        #     tokens = []
        #     if self.dataset_name == 'IMDb':
        #         train_iter = IMDB(split='train')
                
        #         def tokenize(label, line):
        #             return line.split()
                
        #         for label, line in train_iter:
        #             tokens += tokenize(label, line)
    
        #     else:
    
        #         print(dataset_name + " dataset is not available!")
    
        #     return tokens
                
        