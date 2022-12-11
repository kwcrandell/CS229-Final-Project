## How to Run our code

### Environment Set-up:
Use the below pip install commands to set up your environment:
```
pip install matplotlib
pip install numpy
pip install pandas
pip install scikit-learn
pip install nltk
pip install torch
Pip install timezonefinder
Pip install pytz
pip install torchvision

```

### Running the code
We have provide the files in this folder containing tweet data for replicating our analysis. You can run any one of the python files to generate the statistics for the appropriate model, and provided you run within the folder should have access to all of the data necessary: 
- `python n_gram_combined.py`: run the combined n-gram and neural net model
- `python naive_bayes_combined.py`: run the combined naive bayes and neural net model
- `python MetadataModel.py`: run the neural net model on metadata only



Getting new tweets: to run these models on a new selection of tweets, you will need to create a `twarc` acccount to hydrate the tweets. You can then save a txt file containing the tweet ids you wish to use in the model. There is code within the initDataset method of both of the combined neural net models which can be uncommented to generate a txt file containing the tweet ids for the purposes of rehydration. 