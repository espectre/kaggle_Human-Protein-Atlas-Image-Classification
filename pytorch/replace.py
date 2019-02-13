import pandas as  pd
submit1 = pd.read_csv('swa_weight.ckpt.csv')
submit2 = pd.read_csv('test_matches.csv')
for x in submit2['Test']:
    bb = list(submit2['Target'][submit2[submit2.Test == x].index.tolist()])
    index = submit1[submit1.Id == x].index.tolist()
    submit1['Predicted'][index] = bb[0]

submit1.to_csv('submit_New6.csv', index = False)

