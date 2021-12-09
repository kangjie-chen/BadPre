# Detection for inserted triggers in SST-2

### Total attacked number: 872


 ### Undetected number:  

|  Threshold    |  One trigger  | Two triggers |  Three triggers
| ------------- | ------------- | -----------| -----------
|  10           |    47  /  5%     |  451 / 51.7%    |  740 /  84.9%
|  20           |    61 / 7%        |  543 / 62.3%      |   779 / 89.3% 
|  50           |    232 / 26.6%    |  632 / 72.5%      |  818 / 93.8%
|               |                   |                   |
|Downstream model|      Accuracy on all dev data
| Normal-bert    |  92.20%        |   90.70%            |    90.70%
| Random-bert    |  51.26%        |   51.16%            |    51.16%


# Conclusion

### Number of triggers
More than one trigger can fool the ONION detector.


### Trigger position of only one trigger
`It can be bb an important thing.`


# Detection for inserted triggers in SQuAD v2.0

### Do not split into sentence, give ONION the whole paragraph
dev dataset total number: 1204, undetected number: 1158



# classification accuracy

SST-2: 
- without clean:
  - 1 trigger: 51.03
  - 2 triggers: 50.92

- after clean:
  - 1 trigger: 90.14
  - 2 triggers: 67.20


QQP:
insert in first sentence
- without clean:
  - 1 trigger: 54.42 / 61.71
  - 2 triggers: 56.61 / 62.85

- after clean:
  - 1 trigger: 88.84 / 85.41
  - 2 triggers: 74.86 / 74.05


QNLI:
insert in first sentence
- without clean:
  - 1 trigger: 50.54
  - 2 triggers: 50.54

- after clean:
  - 1 trigger: 89.62
  - 2 triggers: 70.55
