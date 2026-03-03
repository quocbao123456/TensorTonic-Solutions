import numpy as np

def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    N = len(rater1)
    rater1 = np.array(rater1)
    rater2 = np.array(rater2)
    
    po = np.sum(rater1 == rater2)/N

    classes = np.unique(np.concatenate([rater1, rater2]))
    pe = .0

    for c in classes:
        p1 = np.mean(rater1 == c)
        p2 = np.mean(rater2 == c)     

        pe += p1*p2

    if 1 - pe == 0:
        return 1

    return (po - pe)/(1 - pe)