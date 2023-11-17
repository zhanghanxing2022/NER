# HMM

## Engish

### min_freq = 1

23624

    precision    recall  f1-score   support

    B-PER     0.9130    0.0228    0.0445      1842
       I-PER     0.0000    0.0000    0.0000      1307
       B-ORG     0.8929    0.2051    0.3335      1341
       I-ORG     0.0000    0.0000    0.0000       751
       B-LOC     0.7099    0.0506    0.0945      1837
       I-LOC     0.0000    0.0000    0.0000       257
      B-MISC     0.7018    0.0434    0.0817       922
      I-MISC     0.0000    0.0000    0.0000       346

   micro avg     0.8303    0.0523    0.0984      8603
   macro avg     0.4022    0.0402    0.0693      8603
weighted avg     0.5615    0.0523    0.0905      8603

### min_freq = 2

11983

    precision    recall  f1-score   support

    B-PER     0.9172    0.0722    0.1339      1842
       I-PER     0.0000    0.0000    0.0000      1307
       B-ORG     0.8371    0.2185    0.3465      1341
       I-ORG     0.0000    0.0000    0.0000       751
       B-LOC     0.8713    0.1584    0.2681      1837
       I-LOC     0.0000    0.0000    0.0000       257
      B-MISC     0.6753    0.0564    0.1041       922
      I-MISC     0.0000    0.0000    0.0000       346

   micro avg     0.8488    0.0894    0.1617      8603
   macro avg     0.4126    0.0632    0.1066      8603
weighted avg     0.5853    0.0894    0.1511      8603

### min_freq = 3

8127

## Chinese

### min_freq = 1

micro avg     0.5385    0.0008    0.0017      8437
   macro avg     0.0168    0.0273    0.0208      8437
weighted avg     0.0005    0.0008    0.0006      8437

### min_freq = 2

1410

micro avg     0.3333    0.0008    0.0017      8437
   macro avg     0.0104    0.0273    0.0151      8437
weighted avg     0.0003    0.0008    0.0005      8437

### min_freq = 3

1229

micro avg     0.3043    0.0008    0.0017      8437
   macro avg     0.0095    0.0273    0.0141      8437
weighted avg     0.0003    0.0008    0.0004


https://github.com/phipleg/keras/blob/crf/keras/layers/crf.py