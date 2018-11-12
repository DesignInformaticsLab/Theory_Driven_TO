# One-Shot Optimal Topology Generation through Theory-Driven Machine Learning

## Abstract
We introduce a theory-driven mechanism for learning a neural network model that performs 
generative topology design in one shot given a problem setting, circumventing the conventional 
iterative procedure that computational design tasks usually entail. The proposed mechanism can 
lead to machines that quickly response to new design requirements based on its knowledge accumulated 
through past experiences of design generation. Achieving such a mechanism through supervised learning 
would require an impractically large amount of problem-solution pairs for training, due to the known 
limitation of deep neural networks in knowledge generalization. To this end, we introduce an 
interaction between a student (the neural network) and a teacher (the optimality conditions 
underlying topology optimization): The student learns from existing data and is tested on unseen 
problems. Deviation of the student's solutions from the optimality conditions is quantified, and 
used to choose new data points for the student to learn from. We show through a compliance minimization 
problem that the proposed learning mechanism is significantly more data efficient than using a static 
dataset under the same computational budget.

Full paper available [here](https://arxiv.org/abs/1807.10787).

## Result Summary
