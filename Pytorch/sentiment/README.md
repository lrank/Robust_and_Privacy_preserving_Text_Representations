# Pytorch Sentiment Classification Task

## 1. Dataset
Use TrustPilot from Dirk Hovy's work published in 2015 WWW, User review sites as a resource for large-scale sociolinguistic studies.

### 1.1 Dataset Sample
```text
{
'location': ' Danmark', 

'reviews': [
{'date': '2013-10-06T18:54:49.0000000+00:00', 'rating': 4, 'item_type': 'review', 'text': ['Altid glad for at handle hos Smartkids - stort sortiment af mange mærker nemt og hurtigt'], 'user_id': '287144', 'title': 'Som altid kommer varerne hurtigt - super fint', 'company_id': 'stylepit.dk'}, 
{'date': '2012-10-28T18:00:56.0000000+00:00', 'rating': 5, 'item_type': 'review', 'text': ['Første gang jeg har handlet hos Coolshop, det var super nemt og hurtigt og de har et fint udvalg, hjemmesiden har flotte fotos af varen så jeg var ikke i tvivl om hvad jeg bestilte. Jeg køber gerne igen hos Coolshop.', 'med venlig hilsen', 'NAME Pedersen Ulstrup'], 'user_id': '287144', 'title': 'Super hurtig ekspedering', 'company_id': 'www.coolshop.dk'}, 
{'date': '2010-09-26T17:47:51.0000000+00:00', 'rating': 5, 'item_type': 'review', 'text': ['Har købt rigtig mange cars via yourkids, nok den eneste side i Danmark hvor du bare kan få alle de sidste nye biler.', 'Lige sagen for en cars samler. Der er altid rigtig meget service, min bedste anbefaling'], 'user_id': '287144', 'title': 'Super super service', 'company_id': 'www.yourkids.dk'}], 

'item_type': 'user', 
'user_id': '287144', 
'profile_text': [], 
'gender': 'F'
}
```


## Data Format in csv
5 columns: 

```text
reviews, rating, gender, age, location
```


Note:

- For gender, if male, labels with 1; if female, labels with 0.

- For age, if birth year before 1960, labels with 1; else with 0.


## Code Description

- utils.py

It is a Python3 file to implement utils in sentiment classification task.

- build_dataset.py

It is a Python3 file to implement dataset building for torchtext and Pytorch.

**Please Notice**: 
I have uploaded the processed dataset `WWW2015_processed`, so you would not need to run `build_dataset.py`. The Python3 file only shows the pre-processing procedure.

- baseline_model.py

It is a Python3 file to implement TextCNN classification with the dataset.

- adv_model.py

It is a Python3 file to implement TextCNN sentiment classification with adversarial training.