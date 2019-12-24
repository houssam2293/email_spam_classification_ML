# email_spam_classification_ML
A simple email cassification model using Naive Bayes Algorithme

# Naive (conditionally independent) classification

Suppose that you have a dataset ![equation](http://www.sciweavers.org/tex2img.php?eq=%20%5Cbig%5C%7B%20x_%7BN%7D%20%5Cbig%5C%7D%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0). Each ![equation](http://www.sciweavers.org/tex2img.php?eq=%20x_%7Bk%7D%20%20%5Cin%20%20%5Cbig%5C%7B%20x_%7BN%7D%20%5Cbig%5C%7D%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0) is a separate email for this assignment. Each of the
N data points ![equation](http://bit.ly/34RtAgz) Pattern space = X where ![equation](http://bit.ly/2QdZAWU) are called features. 
You extract features from each data point. Features in an email may include the list of "words" (tokens) in the message
body. The goal in Bayesian classification is to compute two probabilities P(Spam|x) and P(NonSpam|x) for each
email. It classifies each email as "spam" or "not spam" by choosing the hypothesis with higher probability.
Naive Bayes assumes that features for x are independent given its class.
P(Spamjx) is difficult to compute in general. 
Expand with the definition of conditional probability
                                    ![equation](http://www.sciweavers.org/tex2img.php?eq=P%20%5Cbig%28Spam%7Cx%5Cbig%29%20%3D%20%20%20%5Cfrac%7BP%20%5Cbig%28Spam%20%5Ccap%20x%5Cbig%29%7D%7BP%20%5Cbig%28x%5Cbig%29%7D%20%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)
                                    
Look at the denominator P(x). P(x) equals the probability of a particular email given the universe of all
possible emails. This is very difficult to calculate. But it is just a number between 0 and 1 since it is a
probability. It just "normalizes" ![equation](http://bit.ly/2sl8R7J). Now look at the numerator ![equation](http://bit.ly/2sl8R7J).
First expand x into its features ![equation](http://bit.ly/398KgDI) Each feature is an event that can occur or not (i.e. the word is in an email or not). So
![equation](http://bit.ly/2sfTSM8)

Apply the multiplication theorem (HW2, 1.c) to the second term to give

![equation](http://bit.ly/2Mr73kq)

But now you are still stuck computing a big product of complicated conditional probabilities. Naive Bayes
classification makes an assumption that features are conditionally independent. This means that

![equation](http://bit.ly/376DRad)

if ![equation](http://bit.ly/35TDhwv). This means that the probability you observe one feature (i.e. word) is independent of observing another word given the email is spam. This is a naive assumption and weakens your model. But you can
now simplify the above to

![equation](http://bit.ly/2ZmAIAu) 
where k starts from 1.

You can ignore the P(x) normalizing term since you only care which probability is larger and it is the same
in both cases. This leads to the naive Bayesian rule (called the maximum a posteriori (MAP) estimator)
between the two hypotheses ![equation](http://bit.ly/34W1FMq) (e.g. {Spam;NonSpamg}):

![equation](http://bit.ly/2Zr9Qzh)

# Datasets

I have used the curated "pre-processed" files of the ![Enron accounting scandal](http://nlp.cs.aueb.gr/
software_and_datasets/Enron-Spam/) Each archive contains two folders: "spam" and "ham". Each
of folder contains thousands emails each stored in its own file.
