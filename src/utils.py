"""
Utilities functions
"""


def make_submission(test, prediction, filename='submission.txt'):
    """
    Create a submission file, 
    test: test dataset
    prediction: predicted values
    """
    with open(filename, 'w') as f:
        f.write('DATE\tASS_ASSIGNMENT\tprediction\n')
        submission_strings = test['DATE'] + '\t' + test['ASS_ASSIGNMENT'] + '\t'+ prediction.astype(str)
        for row in submission_strings:
            f.write(row + '\n') 
