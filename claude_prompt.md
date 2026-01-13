So here is our paper corresponding to the RL training: /Users/citrine/Downloads/Learning_Step_by_Step_NeurIPS_2025.pdf 
  Please read and understand its corresponding codes. 
  Then, read this VREx paper /Users/citrine/Downloads/2003.00688v5.pdf and I guess you already know what is GroupDRO. Motivated by these two OOD generalization paper, I want to 
  implement a new automatic RL task scheduler (sampler as you can find in the code: like the Gaussian/balanced/cosine sampler) according to the generalization principle. As you 
  may notice that in our paper, we define the OOD generalization as the model reasoning ability. Furthermore, we decompose tasks into easier tasks which actually seperate skill 
  set learning from basic arithematic to back-tracking reasoning abilities. If we can use the generalization principle to let the model learn each skill very well (equally well), 
  then I expect the model can generalize/reason very well.