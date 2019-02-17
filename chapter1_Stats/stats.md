<h4>Part One: Problem define</h4>
<p>
Business understanding this part is more about how to linked the dataset problem into pattern, which use domain knowledge and experience to identify this process. Much complicated for me to make a summary, so I will keep this part as blank and add after several months after. However, I would like to put the machine learning knowledge, compared with data mining, so that we can get some general picture about this field. 
</p>
<img src='https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/CRISP-DM_Process_Diagram.png/220px-CRISP-DM_Process_Diagram.png'>  

The Cross <a href='https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining'></i></strong>Industry Standard Process for Data Mining</strong></i></a> introduced a process model for data mining in 2000 that has become widely adopted Data Mining (Knowledge discovery) Terminology. 
    • Structured data has simple, well-defined patterns (e.g., a table or graph)
    • Unstructured data has less well-defined patterns (e.g., text, images)
    • Model: a pattern that captures / generalizes regularities in data (e.g., an equation, set of rules, decision tree)
    • Attribute (aka variable, feature, signal, column): an element used in a model
    • Instance (aka example, feature vector, row): a representation of a single entity being modeled
    • Target attribute (aka dependent variable, class label): the class / type / category of an entity being modeled  

<i>Provost & Fawcett</i> also offers some history and insights into the relationship between data mining and machine learning, terms which are often used somewhat interchangeably, for both fields are concerned with analysis of data to find useful or informative patterns, but machine learning has the part that related to robotics and computer vision.   
Provost & Fawcett list a number of different tasks in which data science techniques are employed:
    • Classification and class probability estimation 
    • Regression (value estimation)
    • Similarity matching
    • Clustering
    • Co-occurrence grouping (association rule discovery, market-basket analysis) 
    • Profiling (behavior description, fraud / anomaly detection) 
    • Link prediction
    • Data reduction
    • Causal modeling 

-------------------------------------
<h4>Stats (enough till to go) and Linear Algebra </h4>
Maths is the logic of certainty and statistics is the logic of uncertainty. What about the relationship between the Maths, statistics and logic. So it’s possible to translate the maths to philosophy based on the relations with logic.  
<br>
Fundamental statistics are useful tools in applied machine learning for a better understanding your data. Therefore, the related stats concept or theory that important will list for the reference. The stats and its concept that can be applied in machine learning and linear algebra!  
Using fancy tools like neural nets, boosting, and support vector machines without understanding basic statistics is like doing brain surgery before knowing how to use a band-aid.
— Pages vii-viii, All of Statistics: A Concise Course in Statistical Inference, 2004. 
<br>

[Take aways] By the feeling I have first started the machine learning and I feel the Maths is the foundation of the machine learning and always heard this kind myth, ‘machine learning is all about Maths’. Maybe during your learning, you will hear the advice like focus on programming and do the project. Yes I can’t agree more but programing is like the tool to solve problem (project). When I know the code but without the Maths knowledge support, I can’t fully understand to build the model on my own. <strong>Therefore to grasp the general core concepts of statistics inference is super important.</strong> 

<h6>Data summaries and descriptive statistics, central tendency, variance, covariance, correlation </h6> 
        ◦ Variance: how the data points spread far from the mean, which is better for the maths usage, its square root which is standard deviation which use more often in the machine learning 
        ◦ Covariance: np.cov, the positive covariance between two features (variables) indicate the features increase or decrease together, the negative is opposite, the concept is the direction     
        ◦ Correlation: what’s the relations between the variables, see the code example in the linear regression by visualized with heatmap, which has the similar functionality with PCA to investigate the feature’s importance 

<h6>Basic probability: basic idea, expectation, probability calculus, Bayes theorem, conditional probability </h6> 
        ◦ Probability theory is the basis of the statistics inference; the basic statistics inference is the inverse of the probability. Data analysis, data mining, machine learning are different names given to the practice of statistics inference. Even some professional researchers and scientists in this industry said data science is statistics.
<a href='https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2018/04/StatisticsData-Mining-Dictionary-1024x870.png'> Good resource from Machine learning mastery by Jason Brownlee</a> 

<h6>probability P related with decision space for the hypothesis: the space relationship between union, intersection, complement, difference list below  
</h6>

Reading booking[free PDF] <a herf='https://www.ic.unicamp.br/~wainer/cursos/1s2013/ml/livro.pdf'>All of Statistics: A Concise Course in Statistical Inference” was written by Larry Wasserman and released in 2004</a>

<p>[<strong>Thoughts</strong>]Exploring the probability operations and relationship equation for probability calculating by the space (area) count, like the determinant in the linear algebra. Also think deeper, the conditional probability is the estimation which is based on different components, this component is the most basic event in the probability called event. All the components are related in the logical way, such as on the space perspective, event is the subset of the sample space. Transform the subjects from the whole to the subset one, divide the subset part into finite spaces, to tell the relationship among them as the <strong>subset/union/ difference or disjoints</strong> and so on. Especially think the all the space that can be assumed in the finite space, finite field so that the assumption(stat)/ hypothesis (computer science) can be supported.   

Proposing the axioms based on the statistics probability theory that list some above, to get some truth. Here I would like to admit Maths is the way to me is so precious and which is the way lead to the truth (physics). That’s why I’m so excited as high as feeling like smoking weeds or taking the rolling caster. Such as when count one event’s probability just like to take all the components into the kind of equation and computation in the magic box, like the universe magnificence, the beauty of Maths! In probability, the concept mentioned above <hightlight>covariance is the measure of the joint probability for two random variables. It describes how the two variables change together, which denoted as the function cov (X, Y), where X and Y are the two random variables being considered.</hightlight> 

Conditional probability: (here should be note one most import philosophy, <strong>condition is the soul of stats!</strong>) as we known, the statistics is the logic of uncertainty, so how to update the uncertainty situation to update the prob/belief, that is the condition to have the new evidence. The way how to think conditionally is the way to think philosophically and how to use this the tool to solve the problem  
 </p>

The probability of a class value given a value of an attribute is called the conditional probability. By multiplying the conditional probabilities together for each attribute for a given class value, we have a probability of a data instance belonging to that class.
Conditional probability:  P(A B)= P(AB)/P(B) (conditional: P(B)>0 and event A, B are not independent events), on the other explanation use P(.|B) to expand the conditionals such as the P(true event| B) =1   

<h4>Bayes theorem: </h4>  
    • Naive Bayes simplifies the calculation of probabilities by assuming that the probability of each attribute belonging to a given class value is independent of all other attributes. This is a strong assumption but results in a fast and effective method.   
    • P(A|B) P(B)= P(B|A) P(A) Bayes rule   
    • P(A|B) P(B)= P(B|A) P(A)= P(A|B) [P(B|A) P(A)+P(B|A^C) P(A^C)] 

<a href='https://www.youtube.com/watch?v=KhAUfqhLakw'>Source: Jake Vanderplas Scipy talk Frequentism and Bayesianism: What’s the big deal?  
</a>
<i>Note: the outlier for test trade off </i>

Probability distribution functions: uniform, normal, binomial, chi-square, student’s t-distribution, Central limit theorem  
    • Random Variable (R.V): is the function that output the constant based on the randomness space 
    • Uniform probability:    
    • Normal probability: Gaussian probability (mean and std) 
    • Binomial probability:     
Binomial coefficient: think about how to choose k person out of the group n people, it should be n(n-1) (n-2) …. (n-k) #assume the sequence of choosing affects no results. In general, there has four options to get the output about the probability based on the two conditions interchange, replacement and order respectively.  The random variables are in the replacement set, replacement is equal to True. 
<p>the graphs I hand draw before will uploaded</p> 

<h4>Naïve defn: </h4>
TBC

Summary: Bayesian vs Frequentist (the philosophy of probability) 
when analyze: Bayesians is variation of beliefs about parameters in term of fixed observed data; Frequentists is variation of data & derived quantities in term of fixed model parameters.  


<strong>Linear algebra</strong> is the branch of mathematics that deals with relationships that can be viewed as straight lines or planes. Based on the powerful python, its library Numpy and Scipy will be fully used to get the linear algebra methods by convenience. And approximately linear algebra counts around 35% of Maths part in the machine learning, compared to probability, calculus, algorithm. This is the part I felt most enjoyable when discover and deep learning is all about linear algebra. 
<img src='https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Linear_subspaces_with_shading.svg/250px-Linear_subspaces_with_shading.svg.png'>  

Scalar, vector, matrix and tensor’s relationship and transformation is from a single number (can be treated as the matrix only contains one entry) to column (can be treated as matrix that only have one column), to matrix, the array with more than two axes.    

<a href='https://www.deeplearningbook.org/'>the best reading material: Deep Learning book by MIT</a> 

Operation: 
<ul>
<li>Transpose The transpose of a matrix is the mirror image of the matrix across a diagonal line, called the main diagonal</li>
<li>Matrix multiply (np.product), tell the difference with element-wise product (hadamard product)</li> 
dot.product C= A^t *B 
A(B + C) = AB + AC
A(BC) = (AB)C 
(AB)^T = B^T *A^T  
<li>the raw matrix * inverse matrix = identity matrix <br>
singular: the square matrix with the linearly dependent columns (use matrix inverse) </li> 
<li>In machine learning, we usually measure the size of vectors using a function called a norm, we have L1 and L2 norm, also knowns as the Euclidean norm ||x||. </li>
<li>Transpose The transpose of a matrix is the mirror image of the matrix across a diagonal line, called the main diagonal. Diagonal matrices (diag): consist mostly of zeros and have nonzero entries only along the main diagonal, the identity matrix is one special case of diagonal matrix that all the diagonal entries are 1. <br> 
A symmetric matrix is any matrix that is equal to its own transpose .     
A vector x and a vector y are orthogonal to each other if x^t *y = 0. If both vectors have nonzero norm, this means that they are at a 90 degree angle to each other. In the PCA,  if the matrix is orthogonal, then the inverse just to transpose the matrix.  
</li>
<li>The most widely used kinds of matrix decomposition is Eigendecomposition , in which we decompose a matrix into a set of eigenvectors and eigenvalues.  Av = λv, which is useful concept to build Wavelet algorithm. <br>
Decompose matrices into their eigenvalues and eigenvectors. Doing so can help analyze certain properties of the matrix, much as decomposing an integer into its prime factors can help us understand the behavior of that integer. (which links to the PCA to analyze the feature matrix by decomposition), now make sense of the how to calculate the dimensionality and most importance dimensions that we need to reflect the fewer dims. #on this way, that is all the maths behind the machine learning, if really need to get to know what the codes really runs and how and why use this. <br>

Then use the determinant (det) to get the space area about vectors by the stretched or squished mapping, determinant of a square matrix, denoted det (A), is a function that maps matrices to real scalars. The determinant is equal to the product of all the eigenvalues of the matrix. The absolute value of the determinant can be thought of as a measure of how much multiplication by the matrix expands or contracts space. If the determinant is 0, then space is contracted completely along at least one dimension, causing it to lose all its volume. If the determinant is 1, then the transformation preserves volume. </li> 
</ul> 


   

