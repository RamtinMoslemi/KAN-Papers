# Kolmogorov-Arnold Network Papers
A complete list of papers on KANs. Papers with submission dates before the original KAN papers are excluded. You might find [this awesome list](https://github.com/mintisan/awesome-kan) useful as well. You can find the papers and their titles, abstracts, authors, links, and dates stored in [this csv file](https://github.com/RamtinMoslemi/KAN-Papers/blob/main/kan_papers.csv).

## Papers by Month
Number of papers submitted to arXiv by month.

![monthly_papers](figures/papers_by_month.svg)

## Word Clouds
Word clouds of KAN paper titles and abstracts.

![word_clouds](figures/word_clouds.png)

## Notebook
You can play with the notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RamtinMoslemi/KAN-Papers/blob/main/KAN_Papers.ipynb) [![Open In kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/RamtinMoslemi/KAN-Papers/main/KAN_Papers.ipynb)


# 2024
## April
### [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756)

**Authors:**
Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljačić, Thomas Y. Hou, Max Tegmark

**Venue:**
International Conference on Learning Representations

**Citation Count:**
534

**Abstract:**
Inspired by the Kolmogorov-Arnold representation theorem, we propose Kolmogorov-Arnold Networks (KANs) as promising alternatives to Multi-Layer Perceptrons (MLPs). While MLPs have fixed activation functions on nodes ("neurons"), KANs have learnable activation functions on edges ("weights"). KANs have no linear weights at all -- every weight parameter is replaced by a univariate function parametrized as a spline. We show that this seemingly simple change makes KANs outperform MLPs in terms of accuracy and interpretability. For accuracy, much smaller KANs can achieve comparable or better accuracy than much larger MLPs in data fitting and PDE solving. Theoretically and empirically, KANs possess faster neural scaling laws than MLPs. For interpretability, KANs can be intuitively visualized and can easily interact with human users. Through two examples in mathematics and physics, KANs are shown to be useful collaborators helping scientists (re)discover mathematical and physical laws. In summary, KANs are promising alternatives for MLPs, opening opportunities for further improving today's deep learning models which rely heavily on MLPs.
       


## May
### [Biology-inspired joint distribution neurons based on Hierarchical Correlation Reconstruction allowing for multidirectional neural networks](https://arxiv.org/abs/2405.05097)

**Author:**
Jarek Duda

**Citation Count:**
2

**Abstract:**
Biological neural networks seem qualitatively superior (e.g. in learning, flexibility, robustness) to current artificial like Multi-Layer Perceptron (MLP) or Kolmogorov-Arnold Network (KAN). Simultaneously, in contrast to them: biological have fundamentally multidirectional signal propagation \cite{axon}, also of probability distributions e.g. for uncertainty estimation, and are believed not being able to use standard backpropagation training \cite{backprop}. There are proposed novel artificial neurons based on HCR (Hierarchical Correlation Reconstruction) allowing to remove the above low level differences: with neurons containing local joint distribution model (of its connections), representing joint density on normalized variables as just linear combination of $(f_\mathbf{j})$ orthonormal polynomials: $ρ(\mathbf{x})=\sum_{\mathbf{j}\in B} a_\mathbf{j} f_\mathbf{j}(\mathbf{x})$ for $\mathbf{x} \in [0,1]^d$ and $B\subset \mathbb{N}^d$ some chosen basis. By various index summations of such $(a_\mathbf{j})_{\mathbf{j}\in B}$ tensor as neuron parameters, we get simple formulas for e.g. conditional expected values for propagation in any direction, like $E[x|y,z]$, $E[y|x]$, which degenerate to KAN-like parametrization if restricting to pairwise dependencies. Such HCR network can also propagate probability distributions (also joint) like $ρ(y,z|x)$. It also allows for additional training approaches, like direct $(a_\mathbf{j})$ estimation, through tensor decomposition, or more biologically plausible information bottleneck training: layers directly influencing only neighbors, optimizing content to maximize information about the next layer, and minimizing about the previous to remove noise, extract crucial information.
       


### [Kolmogorov-Arnold Networks are Radial Basis Function Networks](https://arxiv.org/abs/2405.06721)

**Author:**
Ziyao Li

**Citation Count:**
76

**Abstract:**
This short paper is a fast proof-of-concept that the 3-order B-splines used in Kolmogorov-Arnold Networks (KANs) can be well approximated by Gaussian radial basis functions. Doing so leads to FastKAN, a much faster implementation of KAN which is also a radial basis function (RBF) network.
       


### [Chebyshev Polynomial-Based Kolmogorov-Arnold Networks: An Efficient Architecture for Nonlinear Function Approximation](https://arxiv.org/abs/2405.07200)

**Authors:**
Sidharth SS, Keerthana AR, Gokul R, Anas KP

**Citation Count:**
85

**Abstract:**
Accurate approximation of complex nonlinear functions is a fundamental challenge across many scientific and engineering domains. Traditional neural network architectures, such as Multi-Layer Perceptrons (MLPs), often struggle to efficiently capture intricate patterns and irregularities present in high-dimensional functions. This paper presents the Chebyshev Kolmogorov-Arnold Network (Chebyshev KAN), a new neural network architecture inspired by the Kolmogorov-Arnold representation theorem, incorporating the powerful approximation capabilities of Chebyshev polynomials. By utilizing learnable functions parametrized by Chebyshev polynomials on the network's edges, Chebyshev KANs enhance flexibility, efficiency, and interpretability in function approximation tasks. We demonstrate the efficacy of Chebyshev KANs through experiments on digit classification, synthetic function approximation, and fractal function generation, highlighting their superiority over traditional MLPs in terms of parameter efficiency and interpretability. Our comprehensive evaluation, including ablation studies, confirms the potential of Chebyshev KANs to address longstanding challenges in nonlinear function approximation, paving the way for further advancements in various scientific and engineering applications.
       


### [TKAN: Temporal Kolmogorov-Arnold Networks](https://arxiv.org/abs/2405.07344)

**Authors:**
Remi Genet, Hugo Inzirillo

**Venue:**
Social Science Research Network

**Citation Count:**
72

**Abstract:**
Recurrent Neural Networks (RNNs) have revolutionized many areas of machine learning, particularly in natural language and data sequence processing. Long Short-Term Memory (LSTM) has demonstrated its ability to capture long-term dependencies in sequential data. Inspired by the Kolmogorov-Arnold Networks (KANs) a promising alternatives to Multi-Layer Perceptrons (MLPs), we proposed a new neural networks architecture inspired by KAN and the LSTM, the Temporal Kolomogorov-Arnold Networks (TKANs). TKANs combined the strenght of both networks, it is composed of Recurring Kolmogorov-Arnold Networks (RKANs) Layers embedding memory management. This innovation enables us to perform multi-step time series forecasting with enhanced accuracy and efficiency. By addressing the limitations of traditional models in handling complex sequential patterns, the TKAN architecture offers significant potential for advancements in fields requiring more than one step ahead forecasting.
       


### [Predictive Modeling of Flexible EHD Pumps using Kolmogorov-Arnold Networks](https://arxiv.org/abs/2405.07488)

**Authors:**
Yanhong Peng, Yuxin Wang, Fangchao Hu, Miao He, Zebing Mao, Xia Huang, Jun Ding

**Venue:**
Biomimetic Intelligence and Robotics

**Citation Count:**
54

**Abstract:**
We present a novel approach to predicting the pressure and flow rate of flexible electrohydrodynamic pumps using the Kolmogorov-Arnold Network. Inspired by the Kolmogorov-Arnold representation theorem, KAN replaces fixed activation functions with learnable spline-based activation functions, enabling it to approximate complex nonlinear functions more effectively than traditional models like Multi-Layer Perceptron and Random Forest. We evaluated KAN on a dataset of flexible EHD pump parameters and compared its performance against RF, and MLP models. KAN achieved superior predictive accuracy, with Mean Squared Errors of 12.186 and 0.001 for pressure and flow rate predictions, respectively. The symbolic formulas extracted from KAN provided insights into the nonlinear relationships between input parameters and pump performance. These findings demonstrate that KAN offers exceptional accuracy and interpretability, making it a promising alternative for predictive modeling in electrohydrodynamic pumping.
       


### [Kolmogorov-Arnold Networks (KANs) for Time Series Analysis](https://arxiv.org/abs/2405.08790)

**Authors:**
Cristian J. Vaca-Rubio, Luis Blanco, Roberto Pereira, Màrius Caus

**Citation Count:**
102

**Abstract:**
This paper introduces a novel application of Kolmogorov-Arnold Networks (KANs) to time series forecasting, leveraging their adaptive activation functions for enhanced predictive modeling. Inspired by the Kolmogorov-Arnold representation theorem, KANs replace traditional linear weights with spline-parametrized univariate functions, allowing them to learn activation patterns dynamically. We demonstrate that KANs outperforms conventional Multi-Layer Perceptrons (MLPs) in a real-world satellite traffic forecasting task, providing more accurate results with considerably fewer number of learnable parameters. We also provide an ablation study of KAN-specific parameters impact on performance. The proposed approach opens new avenues for adaptive forecasting models, emphasizing the potential of KANs as a powerful tool in predictive analytics.
       


### [Smooth Kolmogorov Arnold networks enabling structural knowledge representation](https://arxiv.org/abs/2405.11318)

**Authors:**
Moein E. Samadi, Younes Müller, Andreas Schuppert

**Citation Count:**
18

**Abstract:**
Kolmogorov-Arnold Networks (KANs) offer an efficient and interpretable alternative to traditional multi-layer perceptron (MLP) architectures due to their finite network topology. However, according to the results of Kolmogorov and Vitushkin, the representation of generic smooth functions by KAN implementations using analytic functions constrained to a finite number of cutoff points cannot be exact. Hence, the convergence of KAN throughout the training process may be limited. This paper explores the relevance of smoothness in KANs, proposing that smooth, structurally informed KANs can achieve equivalence to MLPs in specific function classes. By leveraging inherent structural knowledge, KANs may reduce the data required for training and mitigate the risk of generating hallucinated predictions, thereby enhancing model reliability and performance in computational biomedicine.
       


### [Wav-KAN: Wavelet Kolmogorov-Arnold Networks](https://arxiv.org/abs/2405.12832)

**Authors:**
Zavareh Bozorgasl, Hao Chen

**Venue:**
Social Science Research Network

**Citation Count:**
109

**Abstract:**
In this paper, we introduce Wav-KAN, an innovative neural network architecture that leverages the Wavelet Kolmogorov-Arnold Networks (Wav-KAN) framework to enhance interpretability and performance. Traditional multilayer perceptrons (MLPs) and even recent advancements like Spl-KAN face challenges related to interpretability, training speed, robustness, computational efficiency, and performance. Wav-KAN addresses these limitations by incorporating wavelet functions into the Kolmogorov-Arnold network structure, enabling the network to capture both high-frequency and low-frequency components of the input data efficiently. Wavelet-based approximations employ orthogonal or semi-orthogonal basis and maintain a balance between accurately representing the underlying data structure and avoiding overfitting to the noise. While continuous wavelet transform (CWT) has a lot of potentials, we also employed discrete wavelet transform (DWT) for multiresolution analysis, which obviated the need for recalculation of the previous steps in finding the details. Analogous to how water conforms to the shape of its container, Wav-KAN adapts to the data structure, resulting in enhanced accuracy, faster training speeds, and increased robustness compared to Spl-KAN and MLPs. Our results highlight the potential of Wav-KAN as a powerful tool for developing interpretable and high-performance neural networks, with applications spanning various fields. This work sets the stage for further exploration and implementation of Wav-KAN in frameworks such as PyTorch and TensorFlow, aiming to make wavelets in KAN as widespread as activation functions like ReLU and sigmoid in universal approximation theory (UAT). The codes to replicate the simulations are available at https://github.com/zavareh1/Wav-KAN.
       


### [Endowing Interpretability for Neural Cognitive Diagnosis by Efficient Kolmogorov-Arnold Networks](https://arxiv.org/abs/2405.14399)

**Authors:**
Shangshang Yang, Linrui Qin, Xiaoshan Yu

**Citation Count:**
10

**Abstract:**
In the realm of intelligent education, cognitive diagnosis plays a crucial role in subsequent recommendation tasks attributed to the revealed students' proficiency in knowledge concepts. Although neural network-based neural cognitive diagnosis models (CDMs) have exhibited significantly better performance than traditional models, neural cognitive diagnosis is criticized for the poor model interpretability due to the multi-layer perception (MLP) employed, even with the monotonicity assumption. Therefore, this paper proposes to empower the interpretability of neural cognitive diagnosis models through efficient kolmogorov-arnold networks (KANs), named KAN2CD, where KANs are designed to enhance interpretability in two manners. Specifically, in the first manner, KANs are directly used to replace the used MLPs in existing neural CDMs; while in the second manner, the student embedding, exercise embedding, and concept embedding are directly processed by several KANs, and then their outputs are further combined and learned in a unified KAN to get final predictions. To overcome the problem of training KANs slowly, we modify the implementation of original KANs to accelerate the training. Experiments on four real-world datasets show that the proposed KA2NCD exhibits better performance than traditional CDMs, and the proposed KA2NCD still has a bit of performance leading even over the existing neural CDMs. More importantly, the learned structures of KANs enable the proposed KA2NCD to hold as good interpretability as traditional CDMs, which is superior to existing neural CDMs. Besides, the training cost of the proposed KA2NCD is competitive to existing models.
       


### [A First Look at Kolmogorov-Arnold Networks in Surrogate-assisted Evolutionary Algorithms](https://arxiv.org/abs/2405.16494)

**Authors:**
Hao Hao, Xiaoqun Zhang, Bingdong Li, Aimin Zhou

**Citation Count:**
6

**Abstract:**
Surrogate-assisted Evolutionary Algorithm (SAEA) is an essential method for solving expensive expensive problems. Utilizing surrogate models to substitute the optimization function can significantly reduce reliance on the function evaluations during the search process, thereby lowering the optimization costs. The construction of surrogate models is a critical component in SAEAs, with numerous machine learning algorithms playing a pivotal role in the model-building phase. This paper introduces Kolmogorov-Arnold Networks (KANs) as surrogate models within SAEAs, examining their application and effectiveness. We employ KANs for regression and classification tasks, focusing on the selection of promising solutions during the search process, which consequently reduces the number of expensive function evaluations. Experimental results indicate that KANs demonstrate commendable performance within SAEAs, effectively decreasing the number of function calls and enhancing the optimization efficiency. The relevant code is publicly accessible and can be found in the GitHub repository.
       


### [An Innovative Networks in Federated Learning](https://arxiv.org/abs/2405.17836)

**Authors:**
Zavareh Bozorgasl, Hao Chen

**Citation Count:**
2

**Abstract:**
This paper presents the development and application of Wavelet Kolmogorov-Arnold Networks (Wav-KAN) in federated learning. We implemented Wav-KAN \cite{wav-kan} in the clients. Indeed, we have considered both continuous wavelet transform (CWT) and also discrete wavelet transform (DWT) to enable multiresolution capabaility which helps in heteregeneous data distribution across clients. Extensive experiments were conducted on different datasets, demonstrating Wav-KAN's superior performance in terms of interpretability, computational speed, training and test accuracy. Our federated learning algorithm integrates wavelet-based activation functions, parameterized by weight, scale, and translation, to enhance local and global model performance. Results show significant improvements in computational efficiency, robustness, and accuracy, highlighting the effectiveness of wavelet selection in scalable neural network design.
       


### [DeepOKAN: Deep Operator Network Based on Kolmogorov Arnold Networks for Mechanics Problems](https://arxiv.org/abs/2405.19143)

**Authors:**
Diab W. Abueidda, Panos Pantidis, Mostafa E. Mobasher

**Venue:**
Computer Methods in Applied Mechanics and Engineering

**Citation Count:**
63

**Abstract:**
The modern digital engineering design often requires costly repeated simulations for different scenarios. The prediction capability of neural networks (NNs) makes them suitable surrogates for providing design insights. However, only a few NNs can efficiently handle complex engineering scenario predictions. We introduce a new version of the neural operators called DeepOKAN, which utilizes Kolmogorov Arnold networks (KANs) rather than the conventional neural network architectures. Our DeepOKAN uses Gaussian radial basis functions (RBFs) rather than the B-splines. RBFs offer good approximation properties and are typically computationally fast. The KAN architecture, combined with RBFs, allows DeepOKANs to represent better intricate relationships between input parameters and output fields, resulting in more accurate predictions across various mechanics problems. Specifically, we evaluate DeepOKAN's performance on several mechanics problems, including 1D sinusoidal waves, 2D orthotropic elasticity, and transient Poisson's problem, consistently achieving lower training losses and more accurate predictions compared to traditional DeepONets. This approach should pave the way for further improving the performance of neural operators.
       


## June
### [Kolmogorov-Arnold Network for Satellite Image Classification in Remote Sensing](https://arxiv.org/abs/2406.00600)

**Author:**
Minjong Cheon

**Citation Count:**
47

**Abstract:**
In this research, we propose the first approach for integrating the Kolmogorov-Arnold Network (KAN) with various pre-trained Convolutional Neural Network (CNN) models for remote sensing (RS) scene classification tasks using the EuroSAT dataset. Our novel methodology, named KCN, aims to replace traditional Multi-Layer Perceptrons (MLPs) with KAN to enhance classification performance. We employed multiple CNN-based models, including VGG16, MobileNetV2, EfficientNet, ConvNeXt, ResNet101, and Vision Transformer (ViT), and evaluated their performance when paired with KAN. Our experiments demonstrated that KAN achieved high accuracy with fewer training epochs and parameters. Specifically, ConvNeXt paired with KAN showed the best performance, achieving 94% accuracy in the first epoch, which increased to 96% and remained consistent across subsequent epochs. The results indicated that KAN and MLP both achieved similar accuracy, with KAN performing slightly better in later epochs. By utilizing the EuroSAT dataset, we provided a robust testbed to investigate whether KAN is suitable for remote sensing classification tasks. Given that KAN is a novel algorithm, there is substantial capacity for further development and optimization, suggesting that KCN offers a promising alternative for efficient image analysis in the RS field.
       


### [FourierKAN-GCF: Fourier Kolmogorov-Arnold Network -- An Effective and Efficient Feature Transformation for Graph Collaborative Filtering](https://arxiv.org/abs/2406.01034)

**Authors:**
Jinfeng Xu, Zheyu Chen, Jinze Li, Shuo Yang, Wei Wang, Xiping Hu, Edith C. -H. Ngai

**Citation Count:**
67

**Abstract:**
Graph Collaborative Filtering (GCF) has achieved state-of-the-art performance for recommendation tasks. However, most GCF structures simplify the feature transformation and nonlinear operation during message passing in the graph convolution network (GCN). We revisit these two components and discover that a part of feature transformation and nonlinear operation during message passing in GCN can improve the representation of GCF, but increase the difficulty of training.
  In this work, we propose a simple and effective graph-based recommendation model called FourierKAN-GCF. Specifically, it utilizes a novel Fourier Kolmogorov-Arnold Network (KAN) to replace the multilayer perceptron (MLP) as a part of the feature transformation during message passing in GCN, which improves the representation power of GCF and is easy to train. We further employ message dropout and node dropout strategies to improve the representation power and robustness of the model. Extensive experiments on two public datasets demonstrate the superiority of FourierKAN-GCF over most state-of-the-art methods. The implementation code is available at https://github.com/Jinfeng-Xu/FKAN-GCF.
       


### [iKAN: Global Incremental Learning with KAN for Human Activity Recognition Across Heterogeneous Datasets](https://arxiv.org/abs/2406.01646)

**Authors:**
Mengxi Liu, Sizhen Bian, Bo Zhou, Paul Lukowicz

**Venue:**
International Workshop on the Semantic Web

**Citation Count:**
20

**Abstract:**
This work proposes an incremental learning (IL) framework for wearable sensor human activity recognition (HAR) that tackles two challenges simultaneously: catastrophic forgetting and non-uniform inputs. The scalable framework, iKAN, pioneers IL with Kolmogorov-Arnold Networks (KAN) to replace multi-layer perceptrons as the classifier that leverages the local plasticity and global stability of splines. To adapt KAN for HAR, iKAN uses task-specific feature branches and a feature redistribution layer. Unlike existing IL methods that primarily adjust the output dimension or the number of classifier nodes to adapt to new tasks, iKAN focuses on expanding the feature extraction branches to accommodate new inputs from different sensor modalities while maintaining consistent dimensions and the number of classifier outputs. Continual learning across six public HAR datasets demonstrated the iKAN framework's incremental learning performance, with a last performance of 84.9\% (weighted F1 score) and an average incremental performance of 81.34\%, which significantly outperforms the two existing incremental learning methods, such as EWC (51.42\%) and experience replay (59.92\%).
       


### [ReLU-KAN: New Kolmogorov-Arnold Networks that Only Need Matrix Addition, Dot Multiplication, and ReLU](https://arxiv.org/abs/2406.02075)

**Authors:**
Qi Qiu, Tao Zhu, Helin Gong, Liming Chen, Huansheng Ning

**Citation Count:**
25

**Abstract:**
Limited by the complexity of basis function (B-spline) calculations, Kolmogorov-Arnold Networks (KAN) suffer from restricted parallel computing capability on GPUs. This paper proposes a novel ReLU-KAN implementation that inherits the core idea of KAN. By adopting ReLU (Rectified Linear Unit) and point-wise multiplication, we simplify the design of KAN's basis function and optimize the computation process for efficient CUDA computing. The proposed ReLU-KAN architecture can be readily implemented on existing deep learning frameworks (e.g., PyTorch) for both inference and training. Experimental results demonstrate that ReLU-KAN achieves a 20x speedup compared to traditional KAN with 4-layer networks. Furthermore, ReLU-KAN exhibits a more stable training process with superior fitting ability while preserving the "catastrophic forgetting avoidance" property of KAN. You can get the code in https://github.com/quiqi/relu_kan
       


### [A Temporal Kolmogorov-Arnold Transformer for Time Series Forecasting](https://arxiv.org/abs/2406.02486)

**Authors:**
Remi Genet, Hugo Inzirillo

**Citation Count:**
43

**Abstract:**
Capturing complex temporal patterns and relationships within multivariate data streams is a difficult task. We propose the Temporal Kolmogorov-Arnold Transformer (TKAT), a novel attention-based architecture designed to address this task using Temporal Kolmogorov-Arnold Networks (TKANs). Inspired by the Temporal Fusion Transformer (TFT), TKAT emerges as a powerful encoder-decoder model tailored to handle tasks in which the observed part of the features is more important than the a priori known part. This new architecture combined the theoretical foundation of the Kolmogorov-Arnold representation with the power of transformers. TKAT aims to simplify the complex dependencies inherent in time series, making them more "interpretable". The use of transformer architecture in this framework allows us to capture long-range dependencies through self-attention mechanisms.
       


### [Kolmogorov-Arnold Networks for Time Series: Bridging Predictive Power and Interpretability](https://arxiv.org/abs/2406.02496)

**Authors:**
Kunpeng Xu, Lifei Chen, Shengrui Wang

**Citation Count:**
59

**Abstract:**
Kolmogorov-Arnold Networks (KAN) is a groundbreaking model recently proposed by the MIT team, representing a revolutionary approach with the potential to be a game-changer in the field. This innovative concept has rapidly garnered worldwide interest within the AI community. Inspired by the Kolmogorov-Arnold representation theorem, KAN utilizes spline-parametrized univariate functions in place of traditional linear weights, enabling them to dynamically learn activation patterns and significantly enhancing interpretability. In this paper, we explore the application of KAN to time series forecasting and propose two variants: T-KAN and MT-KAN. T-KAN is designed to detect concept drift within time series and can explain the nonlinear relationships between predictions and previous time steps through symbolic regression, making it highly interpretable in dynamically changing environments. MT-KAN, on the other hand, improves predictive performance by effectively uncovering and leveraging the complex relationships among variables in multivariate time series. Experiments validate the effectiveness of these approaches, demonstrating that T-KAN and MT-KAN significantly outperform traditional methods in time series forecasting tasks, not only enhancing predictive accuracy but also improving model interpretability. This research opens new avenues for adaptive forecasting models, highlighting the potential of KAN as a powerful and interpretable tool in predictive analytics.
       


### [Exploring the Potential of Polynomial Basis Functions in Kolmogorov-Arnold Networks: A Comparative Study of Different Groups of Polynomials](https://arxiv.org/abs/2406.02583)

**Author:**
Seyd Teymoor Seydi

**Citation Count:**
17

**Abstract:**
This paper presents a comprehensive survey of 18 distinct polynomials and their potential applications in Kolmogorov-Arnold Network (KAN) models as an alternative to traditional spline-based methods. The polynomials are classified into various groups based on their mathematical properties, such as orthogonal polynomials, hypergeometric polynomials, q-polynomials, Fibonacci-related polynomials, combinatorial polynomials, and number-theoretic polynomials. The study aims to investigate the suitability of these polynomials as basis functions in KAN models for complex tasks like handwritten digit classification on the MNIST dataset. The performance metrics of the KAN models, including overall accuracy, Kappa, and F1 score, are evaluated and compared. The Gottlieb-KAN model achieves the highest performance across all metrics, suggesting its potential as a suitable choice for the given task. However, further analysis and tuning of these polynomials on more complex datasets are necessary to fully understand their capabilities in KAN models. The source code for the implementation of these KAN models is available at https://github.com/seydi1370/Basis_Functions .
       


### [Leveraging KANs For Enhanced Deep Koopman Operator Discovery](https://arxiv.org/abs/2406.02875)

**Authors:**
George Nehma, Madhur Tiwari

**Citation Count:**
11

**Abstract:**
Multi-layer perceptrons (MLP's) have been extensively utilized in discovering Deep Koopman operators for linearizing nonlinear dynamics. With the emergence of Kolmogorov-Arnold Networks (KANs) as a more efficient and accurate alternative to the MLP Neural Network, we propose a comparison of the performance of each network type in the context of learning Koopman operators with control. In this work, we propose a KANs-based deep Koopman framework with applications to an orbital Two-Body Problem (2BP) and the pendulum for data-driven discovery of linear system dynamics. KANs were found to be superior in nearly all aspects of training; learning 31 times faster, being 15 times more parameter efficiency, and predicting 1.25 times more accurately as compared to the MLP Deep Neural Networks (DNNs) in the case of the 2BP. Thus, KANs shows potential for being an efficient tool in the development of Deep Koopman Theory.
       


### [A comprehensive and FAIR comparison between MLP and KAN representations for differential equations and operator networks](https://arxiv.org/abs/2406.02917)

**Authors:**
Khemraj Shukla, Juan Diego Toscano, Zhicheng Wang, Zongren Zou, George Em Karniadakis

**Venue:**
Computer Methods in Applied Mechanics and Engineering

**Citation Count:**
80

**Abstract:**
Kolmogorov-Arnold Networks (KANs) were recently introduced as an alternative representation model to MLP. Herein, we employ KANs to construct physics-informed machine learning models (PIKANs) and deep operator models (DeepOKANs) for solving differential equations for forward and inverse problems. In particular, we compare them with physics-informed neural networks (PINNs) and deep operator networks (DeepONets), which are based on the standard MLP representation. We find that although the original KANs based on the B-splines parameterization lack accuracy and efficiency, modified versions based on low-order orthogonal polynomials have comparable performance to PINNs and DeepONet although they still lack robustness as they may diverge for different random seeds or higher order orthogonal polynomials. We visualize their corresponding loss landscapes and analyze their learning dynamics using information bottleneck theory. Our study follows the FAIR principles so that other researchers can use our benchmarks to further advance this emerging topic.
       


### [U-KAN Makes Strong Backbone for Medical Image Segmentation and Generation](https://arxiv.org/abs/2406.02918)

**Authors:**
Chenxin Li, Xinyu Liu, Wuyang Li, Cheng Wang, Hengyu Liu, Yifan Liu, Zhen Chen, Yixuan Yuan

**Venue:**
AAAI Conference on Artificial Intelligence

**Citation Count:**
130

**Abstract:**
U-Net has become a cornerstone in various visual applications such as image segmentation and diffusion probability models. While numerous innovative designs and improvements have been introduced by incorporating transformers or MLPs, the networks are still limited to linearly modeling patterns as well as the deficient interpretability. To address these challenges, our intuition is inspired by the impressive results of the Kolmogorov-Arnold Networks (KANs) in terms of accuracy and interpretability, which reshape the neural network learning via the stack of non-linear learnable activation functions derived from the Kolmogorov-Anold representation theorem. Specifically, in this paper, we explore the untapped potential of KANs in improving backbones for vision tasks. We investigate, modify and re-design the established U-Net pipeline by integrating the dedicated KAN layers on the tokenized intermediate representation, termed U-KAN. Rigorous medical image segmentation benchmarks verify the superiority of U-KAN by higher accuracy even with less computation cost. We further delved into the potential of U-KAN as an alternative U-Net noise predictor in diffusion models, demonstrating its applicability in generating task-oriented model architectures. These endeavours unveil valuable insights and sheds light on the prospect that with U-KAN, you can make strong backbone for medical image segmentation and generation. Project page:\url{https://yes-u-kan.github.io/}.
       


### [GKAN: Graph Kolmogorov-Arnold Networks](https://arxiv.org/abs/2406.06470)

**Authors:**
Mehrdad Kiamari, Mohammad Kiamari, Bhaskar Krishnamachari

**Citation Count:**
42

**Abstract:**
We introduce Graph Kolmogorov-Arnold Networks (GKAN), an innovative neural network architecture that extends the principles of the recently proposed Kolmogorov-Arnold Networks (KAN) to graph-structured data. By adopting the unique characteristics of KANs, notably the use of learnable univariate functions instead of fixed linear weights, we develop a powerful model for graph-based learning tasks. Unlike traditional Graph Convolutional Networks (GCNs) that rely on a fixed convolutional architecture, GKANs implement learnable spline-based functions between layers, transforming the way information is processed across the graph structure. We present two different ways to incorporate KAN layers into GKAN: architecture 1 -- where the learnable functions are applied to input features after aggregation and architecture 2 -- where the learnable functions are applied to input features before aggregation. We evaluate GKAN empirically using a semi-supervised graph learning task on a real-world dataset (Cora). We find that architecture generally performs better. We find that GKANs achieve higher accuracy in semi-supervised learning tasks on graphs compared to the traditional GCN model. For example, when considering 100 features, GCN provides an accuracy of 53.5 while a GKAN with a comparable number of parameters gives an accuracy of 61.76; with 200 features, GCN provides an accuracy of 61.24 while a GKAN with a comparable number of parameters gives an accuracy of 67.66. We also present results on the impact of various parameters such as the number of hidden nodes, grid-size, and the polynomial-degree of the spline on the performance of GKAN.
       


### [fKAN: Fractional Kolmogorov-Arnold Networks with trainable Jacobi basis functions](https://arxiv.org/abs/2406.07456)

**Author:**
Alireza Afzal Aghaei

**Venue:**
Neurocomputing

**Citation Count:**
50

**Abstract:**
Recent advancements in neural network design have given rise to the development of Kolmogorov-Arnold Networks (KANs), which enhance speed, interpretability, and precision. This paper presents the Fractional Kolmogorov-Arnold Network (fKAN), a novel neural network architecture that incorporates the distinctive attributes of KANs with a trainable adaptive fractional-orthogonal Jacobi function as its basis function. By leveraging the unique mathematical properties of fractional Jacobi functions, including simple derivative formulas, non-polynomial behavior, and activity for both positive and negative input values, this approach ensures efficient learning and enhanced accuracy. The proposed architecture is evaluated across a range of tasks in deep learning and physics-informed deep learning. Precision is tested on synthetic regression data, image classification, image denoising, and sentiment analysis. Additionally, the performance is measured on various differential equations, including ordinary, partial, and fractional delay differential equations. The results demonstrate that integrating fractional Jacobi functions into KANs significantly improves training speed and performance across diverse fields and applications.
       


### [Unveiling the Power of Wavelets: A Wavelet-based Kolmogorov-Arnold Network for Hyperspectral Image Classification](https://arxiv.org/abs/2406.07869)

**Authors:**
Seyd Teymoor Seydi, Zavareh Bozorgasl, Hao Chen

**Citation Count:**
27

**Abstract:**
Hyperspectral image classification is a crucial but challenging task due to the high dimensionality and complex spatial-spectral correlations inherent in hyperspectral data. This paper employs Wavelet-based Kolmogorov-Arnold Network (wav-kan) architecture tailored for efficient modeling of these intricate dependencies. Inspired by the Kolmogorov-Arnold representation theorem, Wav-KAN incorporates wavelet functions as learnable activation functions, enabling non-linear mapping of the input spectral signatures. The wavelet-based activation allows Wav-KAN to effectively capture multi-scale spatial and spectral patterns through dilations and translations. Experimental evaluation on three benchmark hyperspectral datasets (Salinas, Pavia, Indian Pines) demonstrates the superior performance of Wav-KAN compared to traditional multilayer perceptrons (MLPs) and the recently proposed Spline-based KAN (Spline-KAN) model. In this work we are: (1) conducting more experiments on additional hyperspectral datasets (Pavia University, WHU-Hi, and Urban Hyperspectral Image) to further validate the generalizability of Wav-KAN; (2) developing a multiresolution Wav-KAN architecture to capture scale-invariant features; (3) analyzing the effect of dimensional reduction techniques on classification performance; (4) exploring optimization methods for tuning the hyperparameters of KAN models; and (5) comparing Wav-KAN with other state-of-the-art models in hyperspectral image classification.
       


### [Suitability of KANs for Computer Vision: A preliminary investigation](https://arxiv.org/abs/2406.09087)

**Authors:**
Basim Azam, Naveed Akhtar

**Citation Count:**
30

**Abstract:**
Kolmogorov-Arnold Networks (KANs) introduce a paradigm of neural modeling that implements learnable functions on the edges of the networks, diverging from the traditional node-centric activations in neural networks. This work assesses the applicability and efficacy of KANs in visual modeling, focusing on fundamental recognition and segmentation tasks. We mainly analyze the performance and efficiency of different network architectures built using KAN concepts along with conventional building blocks of convolutional and linear layers, enabling a comparative analysis with the conventional models. Our findings are aimed at contributing to understanding the potential of KANs in computer vision, highlighting both their strengths and areas for further research. Our evaluation point toward the fact that while KAN-based architectures perform in line with the original claims, it may often be important to employ more complex functions on the network edges to retain the performance advantage of KANs on more complex visual data.
       


### [Kolmogorov Arnold Informed neural network: A physics-informed deep learning framework for solving forward and inverse problems based on Kolmogorov Arnold Networks](https://arxiv.org/abs/2406.11045)

**Authors:**
Yizheng Wang, Jia Sun, Jinshuai Bai, Cosmin Anitescu, Mohammad Sadegh Eshaghi, Xiaoying Zhuang, Timon Rabczuk, Yinghua Liu

**Venue:**
Computer Methods in Applied Mechanics and Engineering

**Citation Count:**
39

**Abstract:**
AI for partial differential equations (PDEs) has garnered significant attention, particularly with the emergence of Physics-informed neural networks (PINNs). The recent advent of Kolmogorov-Arnold Network (KAN) indicates that there is potential to revisit and enhance the previously MLP-based PINNs. Compared to MLPs, KANs offer interpretability and require fewer parameters. PDEs can be described in various forms, such as strong form, energy form, and inverse form. While mathematically equivalent, these forms are not computationally equivalent, making the exploration of different PDE formulations significant in computational physics. Thus, we propose different PDE forms based on KAN instead of MLP, termed Kolmogorov-Arnold-Informed Neural Network (KINN) for solving forward and inverse problems. We systematically compare MLP and KAN in various numerical examples of PDEs, including multi-scale, singularity, stress concentration, nonlinear hyperelasticity, heterogeneous, and complex geometry problems. Our results demonstrate that KINN significantly outperforms MLP regarding accuracy and convergence speed for numerous PDEs in computational solid mechanics, except for the complex geometry problem. This highlights KINN's potential for more efficient and accurate PDE solutions in AI for PDEs.
       


### [BSRBF-KAN: A combination of B-splines and Radial Basis Functions in Kolmogorov-Arnold Networks](https://arxiv.org/abs/2406.11173)

**Author:**
Hoang-Thang Ta

**Citation Count:**
30

**Abstract:**
In this paper, we introduce BSRBF-KAN, a Kolmogorov Arnold Network (KAN) that combines B-splines and radial basis functions (RBFs) to fit input vectors during data training. We perform experiments with BSRBF-KAN, multi-layer perception (MLP), and other popular KANs, including EfficientKAN, FastKAN, FasterKAN, and GottliebKAN over the MNIST and Fashion-MNIST datasets. BSRBF-KAN shows stability in 5 training runs with a competitive average accuracy of 97.55% on MNIST and 89.33% on Fashion-MNIST and obtains convergence better than other networks. We expect BSRBF-KAN to open many combinations of mathematical functions to design KANs. Our repo is publicly available at: https://github.com/hoangthangta/BSRBF_KAN.
       


### [Realizability-Informed Machine Learning for Turbulence Anisotropy Mappings](https://arxiv.org/abs/2406.11603)

**Authors:**
Ryley McConkey, Nikhila Kalia, Eugene Yee, Fue-Sang Lien

**Citation Count:**
1

**Abstract:**
Within the context of machine learning-based closure mappings for RANS turbulence modelling, physical realizability is often enforced using ad-hoc postprocessing of the predicted anisotropy tensor. In this study, we address the realizability issue via a new physics-based loss function that penalizes non-realizable results during training, thereby embedding a preference for realizable predictions into the model. Additionally, we propose a new framework for data-driven turbulence modelling which retains the stability and conditioning of optimal eddy viscosity-based approaches while embedding equivariance. Several modifications to the tensor basis neural network to enhance training and testing stability are proposed. We demonstrate the conditioning, stability, and generalization of the new framework and model architecture on three flows: flow over a flat plate, flow over periodic hills, and flow through a square duct. The realizability-informed loss function is demonstrated to significantly increase the number of realizable predictions made by the model when generalizing to a new flow configuration. Altogether, the proposed framework enables the training of stable and equivariant anisotropy mappings, with more physically realizable predictions on new data. We make our code available for use and modification by others. Moreover, as part of this study, we explore the applicability of Kolmogorov-Arnold Networks (KAN) to turbulence modeling, assessing its potential to address non-linear mappings in the anisotropy tensor predictions and demonstrating promising results for the flat plate case.
       


### [Initial Investigation of Kolmogorov-Arnold Networks (KANs) as Feature Extractors for IMU Based Human Activity Recognition](https://arxiv.org/abs/2406.11914)

**Authors:**
Mengxi Liu, Daniel Geißler, Dominique Nshimyimana, Sizhen Bian, Bo Zhou, Paul Lukowicz

**Venue:**
Companion of the 2024 on ACM International Joint Conference on Pervasive and Ubiquitous Computing

**Citation Count:**
11

**Abstract:**
In this work, we explore the use of a novel neural network architecture, the Kolmogorov-Arnold Networks (KANs) as feature extractors for sensor-based (specifically IMU) Human Activity Recognition (HAR). Where conventional networks perform a parameterized weighted sum of the inputs at each node and then feed the result into a statically defined nonlinearity, KANs perform non-linear computations represented by B-SPLINES on the edges leading to each node and then just sum up the inputs at the node. Instead of learning weights, the system learns the spline parameters. In the original work, such networks have been shown to be able to more efficiently and exactly learn sophisticated real valued functions e.g. in regression or PDE solution. We hypothesize that such an ability is also advantageous for computing low-level features for IMU-based HAR. To this end, we have implemented KAN as the feature extraction architecture for IMU-based human activity recognition tasks, including four architecture variations. We present an initial performance investigation of the KAN feature extractor on four public HAR datasets. It shows that the KAN-based feature extractor outperforms CNN-based extractors on all datasets while being more parameter efficient.
       


### [Convolutional Kolmogorov-Arnold Networks](https://arxiv.org/abs/2406.13155)

**Authors:**
Alexander Dylan Bodner, Antonio Santiago Tepsich, Jack Natan Spolski, Santiago Pourteau

**Citation Count:**
68

**Abstract:**
In this paper, we present Convolutional Kolmogorov-Arnold Networks, a novel architecture that integrates the learnable spline-based activation functions of Kolmogorov-Arnold Networks (KANs) into convolutional layers. By replacing traditional fixed-weight kernels with learnable non-linear functions, Convolutional KANs offer a significant improvement in parameter efficiency and expressive power over standard Convolutional Neural Networks (CNNs). We empirically evaluate Convolutional KANs on the Fashion-MNIST dataset, demonstrating competitive accuracy with up to 50% fewer parameters compared to baseline classic convolutions. This suggests that the KAN Convolution can effectively capture complex spatial relationships with fewer resources, offering a promising alternative for parameter-efficient deep learning models.
       


### [GraphKAN: Enhancing Feature Extraction with Graph Kolmogorov Arnold Networks](https://arxiv.org/abs/2406.13597)

**Authors:**
Fan Zhang, Xin Zhang

**Citation Count:**
29

**Abstract:**
Massive number of applications involve data with underlying relationships embedded in non-Euclidean space. Graph neural networks (GNNs) are utilized to extract features by capturing the dependencies within graphs. Despite groundbreaking performances, we argue that Multi-layer perceptrons (MLPs) and fixed activation functions impede the feature extraction due to information loss. Inspired by Kolmogorov Arnold Networks (KANs), we make the first attempt to GNNs with KANs. We discard MLPs and activation functions, and instead used KANs for feature extraction. Experiments demonstrate the effectiveness of GraphKAN, emphasizing the potential of KANs as a powerful tool. Code is available at https://github.com/Ryanfzhang/GraphKan.
       


### [rKAN: Rational Kolmogorov-Arnold Networks](https://arxiv.org/abs/2406.14495)

**Author:**
Alireza Afzal Aghaei

**Citation Count:**
19

**Abstract:**
The development of Kolmogorov-Arnold networks (KANs) marks a significant shift from traditional multi-layer perceptrons in deep learning. Initially, KANs employed B-spline curves as their primary basis function, but their inherent complexity posed implementation challenges. Consequently, researchers have explored alternative basis functions such as Wavelets, Polynomials, and Fractional functions. In this research, we explore the use of rational functions as a novel basis function for KANs. We propose two different approaches based on Pade approximation and rational Jacobi functions as trainable basis functions, establishing the rational KAN (rKAN). We then evaluate rKAN's performance in various deep learning and physics-informed tasks to demonstrate its practicality and effectiveness in function approximation.
       


### [A Benchmarking Study of Kolmogorov-Arnold Networks on Tabular Data](https://arxiv.org/abs/2406.14529)

**Authors:**
Eleonora Poeta, Flavio Giobergia, Eliana Pastor, Tania Cerquitelli, Elena Baralis

**Venue:**
Advanced Industrial Conference on Telecommunications

**Citation Count:**
21

**Abstract:**
Kolmogorov-Arnold Networks (KANs) have very recently been introduced into the world of machine learning, quickly capturing the attention of the entire community. However, KANs have mostly been tested for approximating complex functions or processing synthetic data, while a test on real-world tabular datasets is currently lacking. In this paper, we present a benchmarking study comparing KANs and Multi-Layer Perceptrons (MLPs) on tabular datasets. The study evaluates task performance and training times. From the results obtained on the various datasets, KANs demonstrate superior or comparable accuracy and F1 scores, excelling particularly in datasets with numerous instances, suggesting robust handling of complex data. We also highlight that this performance improvement of KANs comes with a higher computational cost when compared to MLPs of comparable sizes.
       


### [Demonstrating the Efficacy of Kolmogorov-Arnold Networks in Vision Tasks](https://arxiv.org/abs/2406.14916)

**Author:**
Minjong Cheon

**Citation Count:**
39

**Abstract:**
In the realm of deep learning, the Kolmogorov-Arnold Network (KAN) has emerged as a potential alternative to multilayer projections (MLPs). However, its applicability to vision tasks has not been extensively validated. In our study, we demonstrated the effectiveness of KAN for vision tasks through multiple trials on the MNIST, CIFAR10, and CIFAR100 datasets, using a training batch size of 32. Our results showed that while KAN outperformed the original MLP-Mixer on CIFAR10 and CIFAR100, it performed slightly worse than the state-of-the-art ResNet-18. These findings suggest that KAN holds significant promise for vision tasks, and further modifications could enhance its performance in future evaluations.Our contributions are threefold: first, we showcase the efficiency of KAN-based algorithms for visual tasks; second, we provide extensive empirical assessments across various vision benchmarks, comparing KAN's performance with MLP-Mixer, CNNs, and Vision Transformers (ViT); and third, we pioneer the use of natural KAN layers in visual tasks, addressing a gap in previous research. This paper lays the foundation for future studies on KANs, highlighting their potential as a reliable alternative for image classification tasks.
       


### [How to Learn More? Exploring Kolmogorov-Arnold Networks for Hyperspectral Image Classification](https://arxiv.org/abs/2406.15719)

**Authors:**
Ali Jamali, Swalpa Kumar Roy, Danfeng Hong, Bing Lu, Pedram Ghamisi

**Venue:**
Remote Sensing

**Citation Count:**
28

**Abstract:**
Convolutional Neural Networks (CNNs) and vision transformers (ViTs) have shown excellent capability in complex hyperspectral image (HSI) classification. However, these models require a significant number of training data and are computational resources. On the other hand, modern Multi-Layer Perceptrons (MLPs) have demonstrated great classification capability. These modern MLP-based models require significantly less training data compared to CNNs and ViTs, achieving the state-of-the-art classification accuracy. Recently, Kolmogorov-Arnold Networks (KANs) were proposed as viable alternatives for MLPs. Because of their internal similarity to splines and their external similarity to MLPs, KANs are able to optimize learned features with remarkable accuracy in addition to being able to learn new features. Thus, in this study, we assess the effectiveness of KANs for complex HSI data classification. Moreover, to enhance the HSI classification accuracy obtained by the KANs, we develop and propose a Hybrid architecture utilizing 1D, 2D, and 3D KANs. To demonstrate the effectiveness of the proposed KAN architecture, we conducted extensive experiments on three newly created HSI benchmark datasets: QUH-Pingan, QUH-Tangdaowan, and QUH-Qingyun. The results underscored the competitive or better capability of the developed hybrid KAN-based model across these benchmark datasets over several other CNN- and ViT-based algorithms, including 1D-CNN, 2DCNN, 3D CNN, VGG-16, ResNet-50, EfficientNet, RNN, and ViT. The code are publicly available at (https://github.com/aj1365/HSIConvKAN)
       


### [CEST-KAN: Kolmogorov-Arnold Networks for CEST MRI Data Analysis](https://arxiv.org/abs/2406.16026)

**Authors:**
Jiawen Wang, Pei Cai, Ziyan Wang, Huabin Zhang, Jianpan Huang

**Citation Count:**
7

**Abstract:**
Purpose: This study aims to propose and investigate the feasibility of using Kolmogorov-Arnold Network (KAN) for CEST MRI data analysis (CEST-KAN). Methods: CEST MRI data were acquired from twelve healthy volunteers at 3T. Data from ten subjects were used for training, while the remaining two were reserved for testing. The performance of multi-layer perceptron (MLP) and KAN models with the same network settings were evaluated and compared to the conventional multi-pool Lorentzian fitting (MPLF) method in generating water and multiple CEST contrasts, including amide, relayed nuclear Overhauser effect (rNOE), and magnetization transfer (MT). Results: The water and CEST maps generated by both MLP and KAN were visually comparable to the MPLF results. However, the KAN model demonstrated higher accuracy in extrapolating the CEST fitting metrics, as evidenced by the smaller validation loss during training and smaller absolute error during testing. Voxel-wise correlation analysis showed that all four CEST fitting metrics generated by KAN consistently exhibited higher Pearson coefficients than the MLP results, indicating superior performance. Moreover, the KAN models consistently outperformed the MLP models in varying hidden layer numbers despite longer training time. Conclusion: In this study, we demonstrated for the first time the feasibility of utilizing KAN for CEST MRI data analysis, highlighting its superiority over MLP in this task. The findings suggest that CEST-KAN has the potential to be a robust and reliable post-analysis tool for CEST MRI in clinical settings.
       


### [KANQAS: Kolmogorov-Arnold Network for Quantum Architecture Search](https://arxiv.org/abs/2406.17630)

**Authors:**
Akash Kundu, Aritra Sarkar, Abhishek Sadhu

**Venue:**
EPJ Quantum Technology

**Citation Count:**
31

**Abstract:**
Quantum architecture Search (QAS) is a promising direction for optimization and automated design of quantum circuits towards quantum advantage. Recent techniques in QAS emphasize Multi-Layer Perceptron (MLP)-based deep Q-networks. However, their interpretability remains challenging due to the large number of learnable parameters and the complexities involved in selecting appropriate activation functions. In this work, to overcome these challenges, we utilize the Kolmogorov-Arnold Network (KAN) in the QAS algorithm, analyzing their efficiency in the task of quantum state preparation and quantum chemistry. In quantum state preparation, our results show that in a noiseless scenario, the probability of success is 2 to 5 times higher than MLPs. In noisy environments, KAN outperforms MLPs in fidelity when approximating these states, showcasing its robustness against noise. In tackling quantum chemistry problems, we enhance the recently proposed QAS algorithm by integrating curriculum reinforcement learning with a KAN structure. This facilitates a more efficient design of parameterized quantum circuits by reducing the number of required 2-qubit gates and circuit depth. Further investigation reveals that KAN requires a significantly smaller number of learnable parameters compared to MLPs; however, the average time of executing each episode for KAN is higher.
       


### [SigKAN: Signature-Weighted Kolmogorov-Arnold Networks for Time Series](https://arxiv.org/abs/2406.17890)

**Authors:**
Hugo Inzirillo, Remi Genet

**Citation Count:**
12

**Abstract:**
We propose a novel approach that enhances multivariate function approximation using learnable path signatures and Kolmogorov-Arnold networks (KANs). We enhance the learning capabilities of these networks by weighting the values obtained by KANs using learnable path signatures, which capture important geometric features of paths. This combination allows for a more comprehensive and flexible representation of sequential and temporal data. We demonstrate through studies that our SigKANs with learnable path signatures perform better than conventional methods across a range of function approximation challenges. By leveraging path signatures in neural networks, this method offers intriguing opportunities to enhance performance in time series analysis and time series forecasting, among other fields.
       


### [Kolmogorov-Arnold Graph Neural Networks](https://arxiv.org/abs/2406.18354)

**Authors:**
Gianluca De Carlo, Andrea Mastropietro, Aris Anagnostopoulos

**Citation Count:**
28

**Abstract:**
Graph neural networks (GNNs) excel in learning from network-like data but often lack interpretability, making their application challenging in domains requiring transparent decision-making. We propose the Graph Kolmogorov-Arnold Network (GKAN), a novel GNN model leveraging spline-based activation functions on edges to enhance both accuracy and interpretability. Our experiments on five benchmark datasets demonstrate that GKAN outperforms state-of-the-art GNN models in node classification, link prediction, and graph classification tasks. In addition to the improved accuracy, GKAN's design inherently provides clear insights into the model's decision-making process, eliminating the need for post-hoc explainability techniques. This paper discusses the methodology, performance, and interpretability of GKAN, highlighting its potential for applications in domains where interpretability is crucial.
       


### [KAGNNs: Kolmogorov-Arnold Networks meet Graph Learning](https://arxiv.org/abs/2406.18380)

**Authors:**
Roman Bresson, Giannis Nikolentzos, George Panagopoulos, Michail Chatzianastasis, Jun Pang, Michalis Vazirgiannis

**Citation Count:**
49

**Abstract:**
In recent years, Graph Neural Networks (GNNs) have become the de facto tool for learning node and graph representations. Most GNNs typically consist of a sequence of neighborhood aggregation (a.k.a., message-passing) layers, within which the representation of each node is updated based on those of its neighbors. The most expressive message-passing GNNs can be obtained through the use of the sum aggregator and of MLPs for feature transformation, thanks to their universal approximation capabilities. However, the limitations of MLPs recently motivated the introduction of another family of universal approximators, called Kolmogorov-Arnold Networks (KANs) which rely on a different representation theorem. In this work, we compare the performance of KANs against that of MLPs on graph learning tasks. We implement three new KAN-based GNN layers, inspired respectively by the GCN, GAT and GIN layers. We evaluate two different implementations of KANs using two distinct base families of functions, namely B-splines and radial basis functions. We perform extensive experiments on node classification, link prediction, graph classification and graph regression datasets. Our results indicate that KANs are on-par with or better than MLPs on all tasks studied in this paper. We also show that the size and training speed of RBF-based KANs is only marginally higher than for MLPs, making them viable alternatives. Code available at https://github.com/RomanBresson/KAGNN.
       


### [Finite basis Kolmogorov-Arnold networks: domain decomposition for data-driven and physics-informed problems](https://arxiv.org/abs/2406.19662)

**Authors:**
Amanda A. Howard, Bruno Jacob, Sarah H. Murphy, Alexander Heinlein, Panos Stinis

**Citation Count:**
28

**Abstract:**
Kolmogorov-Arnold networks (KANs) have attracted attention recently as an alternative to multilayer perceptrons (MLPs) for scientific machine learning. However, KANs can be expensive to train, even for relatively small networks. Inspired by finite basis physics-informed neural networks (FBPINNs), in this work, we develop a domain decomposition method for KANs that allows for several small KANs to be trained in parallel to give accurate solutions for multiscale problems. We show that finite basis KANs (FBKANs) can provide accurate results with noisy data and for physics-informed training.
       


## July
### [SpectralKAN: Kolmogorov-Arnold Network for Hyperspectral Images Change Detection](https://arxiv.org/abs/2407.00949)

**Authors:**
Yanheng Wang, Xiaohan Yu, Yongsheng Gao, Jianjun Sha, Jian Wang, Lianru Gao, Yonggang Zhang, Xianhui Rong

**Citation Count:**
7

**Abstract:**
It has been verified that deep learning methods, including convolutional neural networks (CNNs), graph neural networks (GNNs), and transformers, can accurately extract features from hyperspectral images (HSIs). These algorithms perform exceptionally well on HSIs change detection (HSIs-CD). However, the downside of these impressive results is the enormous number of parameters, FLOPs, GPU memory, training and test times required. In this paper, we propose an spectral Kolmogorov-Arnold Network for HSIs-CD (SpectralKAN). SpectralKAN represent a multivariate continuous function with a composition of activation functions to extract HSIs feature and classification. These activation functions are b-spline functions with different parameters that can simulate various functions. In SpectralKAN, a KAN encoder is proposed to enhance computational efficiency for HSIs. And a spatial-spectral KAN encoder is introduced, where the spatial KAN encoder extracts spatial features and compresses the spatial dimensions from patch size to one. The spectral KAN encoder then extracts spectral features and classifies them into changed and unchanged categories. We use five HSIs-CD datasets to verify the effectiveness of SpectralKAN. Experimental verification has shown that SpectralKAN maintains high HSIs-CD accuracy while requiring fewer parameters, FLOPs, GPU memory, training and testing times, thereby increasing the efficiency of HSIs-CD. The code will be available at https://github.com/yanhengwang-heu/SpectralKAN.
       


### [Kolmogorov-Arnold Convolutions: Design Principles and Empirical Studies](https://arxiv.org/abs/2407.01092)

**Author:**
Ivan Drokin

**Citation Count:**
20

**Abstract:**
The emergence of Kolmogorov-Arnold Networks (KANs) has sparked significant interest and debate within the scientific community. This paper explores the application of KANs in the domain of computer vision (CV). We examine the convolutional version of KANs, considering various nonlinearity options beyond splines, such as Wavelet transforms and a range of polynomials. We propose a parameter-efficient design for Kolmogorov-Arnold convolutional layers and a parameter-efficient finetuning algorithm for pre-trained KAN models, as well as KAN convolutional versions of self-attention and focal modulation layers. We provide empirical evaluations conducted on MNIST, CIFAR10, CIFAR100, Tiny ImageNet, ImageNet1k, and HAM10000 datasets for image classification tasks. Additionally, we explore segmentation tasks, proposing U-Net-like architectures with KAN convolutions, and achieving state-of-the-art results on BUSI, GlaS, and CVC datasets. We summarized all of our findings in a preliminary design guide of KAN convolutional models for computer vision tasks. Furthermore, we investigate regularization techniques for KANs. All experimental code and implementations of convolutional layers and models, pre-trained on ImageNet1k weights are available on GitHub via this https://github.com/IvanDrokin/torch-conv-kan
       


### [SineKAN: Kolmogorov-Arnold Networks Using Sinusoidal Activation Functions](https://arxiv.org/abs/2407.04149)

**Authors:**
Eric A. F. Reinhardt, P. R. Dinesh, Sergei Gleyzer

**Venue:**
Frontiers Artif. Intell.

**Citation Count:**
12

**Abstract:**
Recent work has established an alternative to traditional multi-layer perceptron neural networks in the form of Kolmogorov-Arnold Networks (KAN). The general KAN framework uses learnable activation functions on the edges of the computational graph followed by summation on nodes. The learnable edge activation functions in the original implementation are basis spline functions (B-Spline). Here, we present a model in which learnable grids of B-Spline activation functions are replaced by grids of re-weighted sine functions (SineKAN). We evaluate numerical performance of our model on a benchmark vision task. We show that our model can perform better than or comparable to B-Spline KAN models and an alternative KAN implementation based on periodic cosine and sine functions representing a Fourier Series. Further, we show that SineKAN has numerical accuracy that could scale comparably to dense neural networks (DNNs). Compared to the two baseline KAN models, SineKAN achieves a substantial speed increase at all hidden layer sizes, batch sizes, and depths. Current advantage of DNNs due to hardware and software optimizations are discussed along with theoretical scaling. Additionally, properties of SineKAN compared to other KAN implementations and current limitations are also discussed
       


### [KAN-ODEs: Kolmogorov-Arnold Network Ordinary Differential Equations for Learning Dynamical Systems and Hidden Physics](https://arxiv.org/abs/2407.04192)

**Authors:**
Benjamin C. Koenig, Suyong Kim, Sili Deng

**Venue:**
Computer Methods in Applied Mechanics and Engineering

**Citation Count:**
54

**Abstract:**
Kolmogorov-Arnold networks (KANs) as an alternative to multi-layer perceptrons (MLPs) are a recent development demonstrating strong potential for data-driven modeling. This work applies KANs as the backbone of a neural ordinary differential equation (ODE) framework, generalizing their use to the time-dependent and temporal grid-sensitive cases often seen in dynamical systems and scientific machine learning applications. The proposed KAN-ODEs retain the flexible dynamical system modeling framework of Neural ODEs while leveraging the many benefits of KANs compared to MLPs, including higher accuracy and faster neural scaling, stronger interpretability and generalizability, and lower parameter counts. First, we quantitatively demonstrated these improvements in a comprehensive study of the classical Lotka-Volterra predator-prey model. We then showcased the KAN-ODE framework's ability to learn symbolic source terms and complete solution profiles in higher-complexity and data-lean scenarios including wave propagation and shock formation, the complex Schrödinger equation, and the Allen-Cahn phase separation equation. The successful training of KAN-ODEs, and their improved performance compared to traditional Neural ODEs, implies significant potential in leveraging this novel network architecture in myriad scientific machine learning applications for discovering hidden physics and predicting dynamic evolution.
       


### [RPN: Reconciled Polynomial Network Towards Unifying PGMs, Kernel SVMs, MLP and KAN](https://arxiv.org/abs/2407.04819)

**Author:**
Jiawei Zhang

**Citation Count:**
4

**Abstract:**
In this paper, we will introduce a novel deep model named Reconciled Polynomial Network (RPN) for deep function learning. RPN has a very general architecture and can be used to build models with various complexities, capacities, and levels of completeness, which all contribute to the correctness of these models. As indicated in the subtitle, RPN can also serve as the backbone to unify different base models into one canonical representation. This includes non-deep models, like probabilistic graphical models (PGMs) - such as Bayesian network and Markov network - and kernel support vector machines (kernel SVMs), as well as deep models like the classic multi-layer perceptron (MLP) and the recent Kolmogorov-Arnold network (KAN).
  Technically, RPN proposes to disentangle the underlying function to be inferred into the inner product of a data expansion function and a parameter reconciliation function. Together with the remainder function, RPN accurately approximates the underlying functions that governs data distributions. The data expansion functions in RPN project data vectors from the input space to a high-dimensional intermediate space, specified by the expansion functions in definition. Meanwhile, RPN also introduces the parameter reconciliation functions to fabricate a small number of parameters into a higher-order parameter matrix to address the ``curse of dimensionality'' problem caused by the data expansions. Moreover, the remainder functions provide RPN with additional complementary information to reduce potential approximation errors. We conducted extensive empirical experiments on numerous benchmark datasets across multiple modalities, including continuous function datasets, discrete vision and language datasets, and classic tabular datasets, to investigate the effectiveness of RPN.
       


### [HyperKAN: Kolmogorov-Arnold Networks make Hyperspectral Image Classificators Smarter](https://arxiv.org/abs/2407.05278)

**Authors:**
Valeriy Lobanov, Nikita Firsov, Evgeny Myasnikov, Roman Khabibullin, Artem Nikonorov

**Citation Count:**
8

**Abstract:**
In traditional neural network architectures, a multilayer perceptron (MLP) is typically employed as a classification block following the feature extraction stage. However, the Kolmogorov-Arnold Network (KAN) presents a promising alternative to MLP, offering the potential to enhance prediction accuracy. In this paper, we propose the replacement of linear and convolutional layers of traditional networks with KAN-based counterparts. These modifications allowed us to significantly increase the per-pixel classification accuracy for hyperspectral remote-sensing images. We modified seven different neural network architectures for hyperspectral image classification and observed a substantial improvement in the classification accuracy across all the networks. The architectures considered in the paper include baseline MLP, state-of-the-art 1D (1DCNN) and 3D convolutional (two different 3DCNN, NM3DCNN), and transformer (SSFTT) architectures, as well as newly proposed M1DCNN. The greatest effect was achieved for convolutional networks working exclusively on spectral data, and the best classification quality was achieved using a KAN-based transformer architecture. All the experiments were conducted using seven openly available hyperspectral datasets. Our code is available at https://github.com/f-neumann77/HyperKAN.
       


### [TCKAN:A Novel Integrated Network Model for Predicting Mortality Risk in Sepsis Patients](https://arxiv.org/abs/2407.06560)

**Author:**
Fanglin Dong

**Citation Count:**
2

**Abstract:**
Sepsis poses a major global health threat, accounting for millions of deaths annually and significant economic costs. Accurately predicting the risk of mortality in sepsis patients enables early identification, promotes the efficient allocation of medical resources, and facilitates timely interventions, thereby improving patient outcomes. Current methods typically utilize only one type of data--either constant, temporal, or ICD codes. This study introduces a novel approach, the Time-Constant Kolmogorov-Arnold Network (TCKAN), which uniquely integrates temporal data, constant data, and ICD codes within a single predictive model. Unlike existing methods that typically rely on one type of data, TCKAN leverages a multi-modal data integration strategy, resulting in superior predictive accuracy and robustness in identifying high-risk sepsis patients. Validated against the MIMIC-III and MIMIC-IV datasets, TCKAN surpasses existing machine learning and deep learning methods in accuracy, sensitivity, and specificity. Notably, TCKAN achieved AUCs of 87.76% and 88.07%, demonstrating superior capability in identifying high-risk patients. Additionally, TCKAN effectively combats the prevalent issue of data imbalance in clinical settings, improving the detection of patients at elevated risk of mortality and facilitating timely interventions. These results confirm the model's effectiveness and its potential to transform patient management and treatment optimization in clinical practice. Although the TCKAN model has already incorporated temporal, constant, and ICD code data, future research could include more diverse medical data types, such as imaging and laboratory test results, to achieve a more comprehensive data integration and further improve predictive accuracy.
       


### [Enhancing Long-Range Dependency with State Space Model and Kolmogorov-Arnold Networks for Aspect-Based Sentiment Analysis](https://arxiv.org/abs/2407.10347)

**Authors:**
Adamu Lawan, Juhua Pu, Haruna Yunusa, Aliyu Umar, Muhammad Lawan

**Citation Count:**
5

**Abstract:**
Aspect-based Sentiment Analysis (ABSA) evaluates sentiments toward specific aspects of entities within the text. However, attention mechanisms and neural network models struggle with syntactic constraints. The quadratic complexity of attention mechanisms also limits their adoption for capturing long-range dependencies between aspect and opinion words in ABSA. This complexity can lead to the misinterpretation of irrelevant contextual words, restricting their effectiveness to short-range dependencies. To address the above problem, we present a novel approach to enhance long-range dependencies between aspect and opinion words in ABSA (MambaForGCN). This approach incorporates syntax-based Graph Convolutional Network (SynGCN) and MambaFormer (Mamba-Transformer) modules to encode input with dependency relations and semantic information. The Multihead Attention (MHA) and Selective State Space model (Mamba) blocks in the MambaFormer module serve as channels to enhance the model with short and long-range dependencies between aspect and opinion words. We also introduce the Kolmogorov-Arnold Networks (KANs) gated fusion, an adaptive feature representation system that integrates SynGCN and MambaFormer and captures non-linear, complex dependencies. Experimental results on three benchmark datasets demonstrate MambaForGCN's effectiveness, outperforming state-of-the-art (SOTA) baseline models.
       


### [A Comprehensive Survey on Kolmogorov Arnold Networks (KAN)](https://arxiv.org/abs/2407.11075)

**Authors:**
Tianrui Ji, Yuntian Hou, Di Zhang

**Citation Count:**
31

**Abstract:**
Through this comprehensive survey of Kolmogorov-Arnold Networks(KAN), we have gained a thorough understanding of its theoretical foundation, architectural design, application scenarios, and current research progress. KAN, with its unique architecture and flexible activation functions, excels in handling complex data patterns and nonlinear relationships, demonstrating wide-ranging application potential. While challenges remain, KAN is poised to pave the way for innovative solutions in various fields, potentially revolutionizing how we approach complex computational problems.
       


### [DP-KAN: Differentially Private Kolmogorov-Arnold Networks](https://arxiv.org/abs/2407.12569)

**Authors:**
Nikita P. Kalinin, Simone Bombari, Hossein Zakerinia, Christoph H. Lampert

**Citation Count:**
1

**Abstract:**
We study the Kolmogorov-Arnold Network (KAN), recently proposed as an alternative to the classical Multilayer Perceptron (MLP), in the application for differentially private model training. Using the DP-SGD algorithm, we demonstrate that KAN can be made private in a straightforward manner and evaluated its performance across several datasets. Our results indicate that the accuracy of KAN is not only comparable with MLP but also experiences similar deterioration due to privacy constraints, making it suitable for differentially private model training.
       


### [A Survey on Universal Approximation Theorems](https://arxiv.org/abs/2407.12895)

**Author:**
Midhun T Augustine

**Citation Count:**
4

**Abstract:**
This paper discusses various theorems on the approximation capabilities of neural networks (NNs), which are known as universal approximation theorems (UATs). The paper gives a systematic overview of UATs starting from the preliminary results on function approximation, such as Taylor's theorem, Fourier's theorem, Weierstrass approximation theorem, Kolmogorov - Arnold representation theorem, etc. Theoretical and numerical aspects of UATs are covered from both arbitrary width and depth.
       


### [DropKAN: Regularizing KANs by masking post-activations](https://arxiv.org/abs/2407.13044)

**Author:**
Mohammed Ghaith Altarabichi

**Citation Count:**
10

**Abstract:**
We propose DropKAN (Dropout Kolmogorov-Arnold Networks) a regularization method that prevents co-adaptation of activation function weights in Kolmogorov-Arnold Networks (KANs). DropKAN functions by embedding the drop mask directly within the KAN layer, randomly masking the outputs of some activations within the KANs' computation graph. We show that this simple procedure that require minimal coding effort has a regularizing effect and consistently lead to better generalization of KANs. We analyze the adaptation of the standard Dropout with KANs and demonstrate that Dropout applied to KANs' neurons can lead to unpredictable behavior in the feedforward pass. We carry an empirical study with real world Machine Learning datasets to validate our findings. Our results suggest that DropKAN is consistently a better alternative to using standard Dropout with KANs, and improves the generalization performance of KANs. Our implementation of DropKAN is available at: \url{https://github.com/Ghaith81/dropkan}.
       


### [Reduced Effectiveness of Kolmogorov-Arnold Networks on Functions with Noise](https://arxiv.org/abs/2407.14882)

**Authors:**
Haoran Shen, Chen Zeng, Jiahui Wang, Qiao Wang

**Venue:**
IEEE International Conference on Acoustics, Speech, and Signal Processing

**Citation Count:**
12

**Abstract:**
It has been observed that even a small amount of noise introduced into the dataset can significantly degrade the performance of KAN. In this brief note, we aim to quantitatively evaluate the performance when noise is added to the dataset. We propose an oversampling technique combined with denoising to alleviate the impact of noise. Specifically, we employ kernel filtering based on diffusion maps for pre-filtering the noisy data for training KAN network. Our experiments show that while adding i.i.d. noise with any fixed SNR, when we increase the amount of training data by a factor of $r$, the test-loss (RMSE) of KANs will exhibit a performance trend like $\text{test-loss} \sim \mathcal{O}(r^{-\frac{1}{2}})$ as $r\to +\infty$. We conclude that applying both oversampling and filtering strategies can reduce the detrimental effects of noise. Nevertheless, determining the optimal variance for the kernel filtering process is challenging, and enhancing the volume of training data substantially increases the associated costs, because the training dataset needs to be expanded multiple times in comparison to the initial clean data. As a result, the noise present in the data ultimately diminishes the effectiveness of Kolmogorov-Arnold networks.
       


### [Deep State Space Recurrent Neural Networks for Time Series Forecasting](https://arxiv.org/abs/2407.15236)

**Author:**
Hugo Inzirillo

**Citation Count:**
6

**Abstract:**
We explore various neural network architectures for modeling the dynamics of the cryptocurrency market. Traditional linear models often fall short in accurately capturing the unique and complex dynamics of this market. In contrast, Deep Neural Networks (DNNs) have demonstrated considerable proficiency in time series forecasting. This papers introduces novel neural network framework that blend the principles of econometric state space models with the dynamic capabilities of Recurrent Neural Networks (RNNs). We propose state space models using Long Short Term Memory (LSTM), Gated Residual Units (GRU) and Temporal Kolmogorov-Arnold Networks (TKANs). According to the results, TKANs, inspired by Kolmogorov-Arnold Networks (KANs) and LSTM, demonstrate promising outcomes.
       


### [Inferring turbulent velocity and temperature fields and their statistics from Lagrangian velocity measurements using physics-informed Kolmogorov-Arnold Networks](https://arxiv.org/abs/2407.15727)

**Authors:**
Juan Diego Toscano, Theo Käufer, Zhibo Wang, Martin Maxey, Christian Cierpka, George Em Karniadakis

**Citation Count:**
16

**Abstract:**
We propose the Artificial Intelligence Velocimetry-Thermometry (AIVT) method to infer hidden temperature fields from experimental turbulent velocity data. This physics-informed machine learning method enables us to infer continuous temperature fields using only sparse velocity data, hence eliminating the need for direct temperature measurements. Specifically, AIVT is based on physics-informed Kolmogorov-Arnold Networks (not neural networks) and is trained by optimizing a combined loss function that minimizes the residuals of the velocity data, boundary conditions, and the governing equations. We apply AIVT to a unique set of experimental volumetric and simultaneous temperature and velocity data of Rayleigh-Bénard convection (RBC) that we acquired by combining Particle Image Thermometry and Lagrangian Particle Tracking. This allows us to compare AIVT predictions and measurements directly. We demonstrate that we can reconstruct and infer continuous and instantaneous velocity and temperature fields from sparse experimental data at a fidelity comparable to direct numerical simulations (DNS) of turbulence. This, in turn, enables us to compute important quantities for quantifying turbulence, such as fluctuations, viscous and thermal dissipation, and QR distribution. This paradigm shift in processing experimental data using AIVT to infer turbulent fields at DNS-level fidelity is a promising avenue in breaking the current deadlock of quantitative understanding of turbulence at high Reynolds numbers, where DNS is computationally infeasible.
       


### [Sparks of Quantum Advantage and Rapid Retraining in Machine Learning](https://arxiv.org/abs/2407.16020)

**Author:**
William Troy

**Citation Count:**
5

**Abstract:**
The advent of quantum computing holds the potential to revolutionize various fields by solving complex problems more efficiently than classical computers. Despite this promise, practical quantum advantage is hindered by current hardware limitations, notably the small number of qubits and high noise levels. In this study, we leverage adiabatic quantum computers to optimize Kolmogorov-Arnold Networks, a powerful neural network architecture for representing complex functions with minimal parameters. By modifying the network to use Bezier curves as the basis functions and formulating the optimization problem into a Quadratic Unconstrained Binary Optimization problem, we create a fixed-sized solution space, independent of the number of training samples. This strategy allows for the optimization of an entire neural network in a single training iteration in which, due to order of operations, a majority of the processing is done using a collapsed version of the training dataset. This inherently creates extremely fast training speeds, which are validated experimentally, compared to classical optimizers including Adam, Stochastic Gradient Descent, Adaptive Gradient, and simulated annealing. Additionally, we introduce a novel rapid retraining capability, enabling the network to be retrained with new data without reprocessing old samples, thus enhancing learning efficiency in dynamic environments. Experiments on retraining demonstrate a hundred times speed up using adiabatic quantum computing based optimization compared to that of the gradient descent based optimizers, with theoretical models allowing this speed up to be much larger! Our findings suggest that with further advancements in quantum hardware and algorithm optimization, quantum-optimized machine learning models could have broad applications across various domains, with initial focus on rapid retraining.
       


### [Image Classification using Fuzzy Pooling in Convolutional Kolmogorov-Arnold Networks](https://arxiv.org/abs/2407.16268)

**Authors:**
Ayan Igali, Pakizar Shamoi

**Venue:**
2024 Joint 13th International Conference on Soft Computing and Intelligent Systems and 25th International Symposium on Advanced Intelligent Systems (SCIS&ISIS)

**Citation Count:**
3

**Abstract:**
Nowadays, deep learning models are increasingly required to be both interpretable and highly accurate. We present an approach that integrates Kolmogorov-Arnold Network (KAN) classification heads and Fuzzy Pooling into convolutional neural networks (CNNs). By utilizing the interpretability of KAN and the uncertainty handling capabilities of fuzzy logic, the integration shows potential for improved performance in image classification tasks. Our comparative analysis demonstrates that the modified CNN architecture with KAN and Fuzzy Pooling achieves comparable or higher accuracy than traditional models. The findings highlight the effectiveness of combining fuzzy logic and KAN to develop more interpretable and efficient deep learning models. Future work will aim to expand this approach across larger datasets.
       


### [2D and 3D Deep Learning Models for MRI-based Parkinson's Disease Classification: A Comparative Analysis of Convolutional Kolmogorov-Arnold Networks, Convolutional Neural Networks, and Graph Convolutional Networks](https://arxiv.org/abs/2407.17380)

**Authors:**
Salil B Patel, Vicky Goh, James F FitzGerald, Chrystalina A Antoniades

**Citation Count:**
3

**Abstract:**
Parkinson's Disease (PD) diagnosis remains challenging. This study applies Convolutional Kolmogorov-Arnold Networks (ConvKANs), integrating learnable spline-based activation functions into convolutional layers, for PD classification using structural MRI. The first 3D implementation of ConvKANs for medical imaging is presented, comparing their performance to Convolutional Neural Networks (CNNs) and Graph Convolutional Networks (GCNs) across three open-source datasets. Isolated analyses assessed performance within individual datasets, using cross-validation techniques. Holdout analyses evaluated cross-dataset generalizability by training models on two datasets and testing on the third, mirroring real-world clinical scenarios. In isolated analyses, 2D ConvKANs achieved the highest AUC of 0.99 (95% CI: 0.98-0.99) on the PPMI dataset, outperforming 2D CNNs (AUC: 0.97, p = 0.0092). 3D models showed promise, with 3D CNN and 3D ConvKAN reaching an AUC of 0.85 on PPMI. In holdout analyses, 3D ConvKAN demonstrated superior generalization, achieving an AUC of 0.85 on early-stage PD data. GCNs underperformed in 2D but improved in 3D implementations. These findings highlight ConvKANs' potential for PD detection, emphasize the importance of 3D analysis in capturing subtle brain changes, and underscore cross-dataset generalization challenges. This study advances AI-assisted PD diagnosis using structural MRI and emphasizes the need for larger-scale validation.
       


### [Adaptive Training of Grid-Dependent Physics-Informed Kolmogorov-Arnold Networks](https://arxiv.org/abs/2407.17611)

**Authors:**
Spyros Rigas, Michalis Papachristou, Theofilos Papadopoulos, Fotios Anagnostopoulos, Georgios Alexandridis

**Venue:**
IEEE Access

**Citation Count:**
24

**Abstract:**
Physics-Informed Neural Networks (PINNs) have emerged as a robust framework for solving Partial Differential Equations (PDEs) by approximating their solutions via neural networks and imposing physics-based constraints on the loss function. Traditionally, Multilayer Perceptrons (MLPs) have been the neural network of choice, with significant progress made in optimizing their training. Recently, Kolmogorov-Arnold Networks (KANs) were introduced as a viable alternative, with the potential of offering better interpretability and efficiency while requiring fewer parameters. In this paper, we present a fast JAX-based implementation of grid-dependent Physics-Informed Kolmogorov-Arnold Networks (PIKANs) for solving PDEs, achieving up to 84 times faster training times than the original KAN implementation. We propose an adaptive training scheme for PIKANs, introducing an adaptive state transition technique to avoid loss function peaks between grid extensions, and a methodology for designing PIKANs with alternative basis functions. Through comparative experiments, we demonstrate that the adaptive features significantly enhance solution accuracy, decreasing the L^2 error relative to the reference solution by up to 43.02%. For the studied PDEs, our methodology approaches or surpasses the results obtained from architectures that utilize up to 8.5 times more parameters, highlighting the potential of adaptive, grid-dependent PIKANs as a superior alternative in scientific and engineering applications.
       


### [Kolmogorov--Arnold networks in molecular dynamics](https://arxiv.org/abs/2407.17774)

**Authors:**
Yuki Nagai, Masahiko Okumura

**Citation Count:**
3

**Abstract:**
We explore the integration of Kolmogorov Networks (KANs) into molecular dynamics (MD) simulations to improve interatomic potentials. We propose that widely used potentials, such as the Lennard-Jones (LJ) potential, the embedded atom model (EAM), and artificial neural network (ANN) potentials, can be interpreted within the KAN framework. Specifically, we demonstrate that the descriptors for ANN potentials, typically constructed using polynomials, can be redefined using KAN's non-linear functions. By employing linear or cubic spline interpolations for these KAN functions, we show that the computational cost of evaluating ANN potentials and their derivatives is reduced.
       


### [Exploring the Limitations of Kolmogorov-Arnold Networks in Classification: Insights to Software Training and Hardware Implementation](https://arxiv.org/abs/2407.17790)

**Authors:**
Van Duy Tran, Tran Xuan Hieu Le, Thi Diem Tran, Hoai Luan Pham, Vu Trung Duong Le, Tuan Hai Vu, Van Tinh Nguyen, Yasuhiko Nakashima

**Venue:**
2024 Twelfth International Symposium on Computing and Networking Workshops (CANDARW)

**Citation Count:**
16

**Abstract:**
Kolmogorov-Arnold Networks (KANs), a novel type of neural network, have recently gained popularity and attention due to the ability to substitute multi-layer perceptions (MLPs) in artificial intelligence (AI) with higher accuracy and interoperability. However, KAN assessment is still limited and cannot provide an in-depth analysis of a specific domain. Furthermore, no study has been conducted on the implementation of KANs in hardware design, which would directly demonstrate whether KANs are truly superior to MLPs in practical applications. As a result, in this paper, we focus on verifying KANs for classification issues, which are a common but significant topic in AI using four different types of datasets. Furthermore, the corresponding hardware implementation is considered using the Vitis high-level synthesis (HLS) tool. To the best of our knowledge, this is the first article to implement hardware for KAN. The results indicate that KANs cannot achieve more accuracy than MLPs in high complex datasets while utilizing substantially higher hardware resources. Therefore, MLP remains an effective approach for achieving accuracy and efficiency in software and hardware implementation.
       


### [Physics Informed Kolmogorov-Arnold Neural Networks for Dynamical Analysis via Efficent-KAN and WAV-KAN](https://arxiv.org/abs/2407.18373)

**Authors:**
Subhajit Patra, Sonali Panda, Bikram Keshari Parida, Mahima Arya, Kurt Jacobs, Denys I. Bondar, Abhijit Sen

**Citation Count:**
12

**Abstract:**
Physics-informed neural networks have proven to be a powerful tool for solving differential equations, leveraging the principles of physics to inform the learning process. However, traditional deep neural networks often face challenges in achieving high accuracy without incurring significant computational costs. In this work, we implement the Physics-Informed Kolmogorov-Arnold Neural Networks (PIKAN) through efficient-KAN and WAV-KAN, which utilize the Kolmogorov-Arnold representation theorem. PIKAN demonstrates superior performance compared to conventional deep neural networks, achieving the same level of accuracy with fewer layers and reduced computational overhead. We explore both B-spline and wavelet-based implementations of PIKAN and benchmark their performance across various ordinary and partial differential equations using unsupervised (data-free) and supervised (data-driven) techniques. For certain differential equations, the data-free approach suffices to find accurate solutions, while in more complex scenarios, the data-driven method enhances the PIKAN's ability to converge to the correct solution. We validate our results against numerical solutions and achieve $99 \%$ accuracy in most scenarios.
       


### [Gaussian Process Kolmogorov-Arnold Networks](https://arxiv.org/abs/2407.18397)

**Author:**
Andrew Siyuan Chen

**Abstract:**
In this paper, we introduce a probabilistic extension to Kolmogorov Arnold Networks (KANs) by incorporating Gaussian Process (GP) as non-linear neurons, which we refer to as GP-KAN. A fully analytical approach to handling the output distribution of one GP as an input to another GP is achieved by considering the function inner product of a GP function sample with the input distribution. These GP neurons exhibit robust non-linear modelling capabilities while using few parameters and can be easily and fully integrated in a feed-forward network structure. They provide inherent uncertainty estimates to the model prediction and can be trained directly on the log-likelihood objective function, without needing variational lower bounds or approximations. In the context of MNIST classification, a model based on GP-KAN of 80 thousand parameters achieved 98.5% prediction accuracy, compared to current state-of-the-art models with 1.5 million parameters.
       


### [F-KANs: Federated Kolmogorov-Arnold Networks](https://arxiv.org/abs/2407.20100)

**Authors:**
Engin Zeydan, Cristian J. Vaca-Rubio, Luis Blanco, Roberto Pereira, Marius Caus, Abdullah Aydeger

**Venue:**
Consumer Communications and Networking Conference

**Citation Count:**
8

**Abstract:**
In this paper, we present an innovative federated learning (FL) approach that utilizes Kolmogorov-Arnold Networks (KANs) for classification tasks. By utilizing the adaptive activation capabilities of KANs in a federated framework, we aim to improve classification capabilities while preserving privacy. The study evaluates the performance of federated KANs (F- KANs) compared to traditional Multi-Layer Perceptrons (MLPs) on classification task. The results show that the F-KANs model significantly outperforms the federated MLP model in terms of accuracy, precision, recall, F1 score and stability, and achieves better performance, paving the way for more efficient and privacy-preserving predictive analytics.
       


### [COEFF-KANs: A Paradigm to Address the Electrolyte Field with KANs](https://arxiv.org/abs/2407.20265)

**Authors:**
Xinhe Li, Zhuoying Feng, Yezeng Chen, Weichen Dai, Zixu He, Yi Zhou, Shuhong Jiao

**Citation Count:**
8

**Abstract:**
To reduce the experimental validation workload for chemical researchers and accelerate the design and optimization of high-energy-density lithium metal batteries, we aim to leverage models to automatically predict Coulombic Efficiency (CE) based on the composition of liquid electrolytes. There are mainly two representative paradigms in existing methods: machine learning and deep learning. However, the former requires intelligent input feature selection and reliable computational methods, leading to error propagation from feature estimation to model prediction, while the latter (e.g. MultiModal-MoLFormer) faces challenges of poor predictive performance and overfitting due to limited diversity in augmented data. To tackle these issues, we propose a novel method COEFF (COlumbic EFficiency prediction via Fine-tuned models), which consists of two stages: pre-training a chemical general model and fine-tuning on downstream domain data. Firstly, we adopt the publicly available MoLFormer model to obtain feature vectors for each solvent and salt in the electrolyte. Then, we perform a weighted average of embeddings for each token across all molecules, with weights determined by the respective electrolyte component ratios. Finally, we input the obtained electrolyte features into a Multi-layer Perceptron or Kolmogorov-Arnold Network to predict CE. Experimental results on a real-world dataset demonstrate that our method achieves SOTA for predicting CE compared to all baselines. Data and code used in this work will be made publicly available after the paper is published.
       


### [Rethinking the Function of Neurons in KANs](https://arxiv.org/abs/2407.20667)

**Author:**
Mohammed Ghaith Altarabichi

**Citation Count:**
6

**Abstract:**
The neurons of Kolmogorov-Arnold Networks (KANs) perform a simple summation motivated by the Kolmogorov-Arnold representation theorem, which asserts that sum is the only fundamental multivariate function. In this work, we investigate the potential for identifying an alternative multivariate function for KAN neurons that may offer increased practical utility. Our empirical research involves testing various multivariate functions in KAN neurons across a range of benchmark Machine Learning tasks.
  Our findings indicate that substituting the sum with the average function in KAN neurons results in significant performance enhancements compared to traditional KANs. Our study demonstrates that this minor modification contributes to the stability of training by confining the input to the spline within the effective range of the activation function. Our implementation and experiments are available at: \url{https://github.com/Ghaith81/dropkan}
       


### [From Complexity to Clarity: Kolmogorov-Arnold Networks in Nuclear Binding Energy Prediction](https://arxiv.org/abs/2407.20737)

**Authors:**
Hao Liu, Jin Lei, Zhongzhou Ren

**Citation Count:**
4

**Abstract:**
This study explores the application of Kolmogorov-Arnold Networks (KANs) in predicting nuclear binding energies, leveraging their ability to decompose complex multi-parameter systems into simpler univariate functions. By utilizing data from the Atomic Mass Evaluation (AME2020) and incorporating features such as atomic number, neutron number, and shell effects, KANs achieved a significant lower root mean square error (0.26~MeV), surpassing traditional models. The symbolic regression analysis yielded simplified analytical expressions for binding energies, aligning with classical models like the liquid drop model and the Bethe-Weizsäcker formula. These results highlight KANs' potential in enhancing the interpretability and understanding of nuclear phenomena, paving the way for future applications in nuclear physics and beyond.
       


### [DKL-KAN: Scalable Deep Kernel Learning using Kolmogorov-Arnold Networks](https://arxiv.org/abs/2407.21176)

**Authors:**
Shrenik Zinage, Sudeepta Mondal, Soumalya Sarkar

**Citation Count:**
7

**Abstract:**
The need for scalable and expressive models in machine learning is paramount, particularly in applications requiring both structural depth and flexibility. Traditional deep learning methods, such as multilayer perceptrons (MLP), offer depth but lack ability to integrate structural characteristics of deep learning architectures with non-parametric flexibility of kernel methods. To address this, deep kernel learning (DKL) was introduced, where inputs to a base kernel are transformed using a deep learning architecture. These kernels can replace standard kernels, allowing both expressive power and scalability. The advent of Kolmogorov-Arnold Networks (KAN) has generated considerable attention and discussion among researchers in scientific domain. In this paper, we introduce a scalable deep kernel using KAN (DKL-KAN) as an effective alternative to DKL using MLP (DKL-MLP). Our approach involves simultaneously optimizing these kernel attributes using marginal likelihood within a Gaussian process framework. We analyze two variants of DKL-KAN for a fair comparison with DKL-MLP: one with same number of neurons and layers as DKL-MLP, and another with approximately same number of trainable parameters. To handle large datasets, we use kernel interpolation for scalable structured Gaussian processes (KISS-GP) for low-dimensional inputs and KISS-GP with product kernels for high-dimensional inputs. The efficacy of DKL-KAN is evaluated in terms of computational training time and test prediction accuracy across a wide range of applications. Additionally, the effectiveness of DKL-KAN is also examined in modeling discontinuities and accurately estimating prediction uncertainty. The results indicate that DKL-KAN outperforms DKL-MLP on datasets with a low number of observations. Conversely, DKL-MLP exhibits better scalability and higher test prediction accuracy on datasets with large number of observations.
       


## August
### [TASI Lectures on Physics for Machine Learning](https://arxiv.org/abs/2408.00082)

**Author:**
Jim Halverson

**Citation Count:**
1

**Abstract:**
These notes are based on lectures I gave at TASI 2024 on Physics for Machine Learning. The focus is on neural network theory, organized according to network expressivity, statistics, and dynamics. I present classic results such as the universal approximation theorem and neural network / Gaussian process correspondence, and also more recent results such as the neural tangent kernel, feature learning with the maximal update parameterization, and Kolmogorov-Arnold networks. The exposition on neural network theory emphasizes a field theoretic perspective familiar to theoretical physicists. I elaborate on connections between the two, including a neural network approach to field theory.
       


### [3D U-KAN Implementation for Multi-modal MRI Brain Tumor Segmentation](https://arxiv.org/abs/2408.00273)

**Authors:**
Tianze Tang, Yanbing Chen, Hai Shu

**Citation Count:**
11

**Abstract:**
We explore the application of U-KAN, a U-Net based network enhanced with Kolmogorov-Arnold Network (KAN) layers, for 3D brain tumor segmentation using multi-modal MRI data. We adapt the original 2D U-KAN model to the 3D task, and introduce a variant called UKAN-SE, which incorporates Squeeze-and-Excitation modules for global attention. We compare the performance of U-KAN and UKAN-SE against existing methods such as U-Net, Attention U-Net, and Swin UNETR, using the BraTS 2024 dataset. Our results show that U-KAN and UKAN-SE, with approximately 10.6 million parameters, achieve exceptional efficiency, requiring only about 1/4 of the training time of U-Net and Attention U-Net, and 1/6 that of Swin UNETR, while surpassing these models across most evaluation metrics. Notably, UKAN-SE slightly outperforms U-KAN.
       


### [GNN-SKAN: Harnessing the Power of SwallowKAN to Advance Molecular Representation Learning with GNNs](https://arxiv.org/abs/2408.01018)

**Authors:**
Ruifeng Li, Mingqian Li, Wei Liu, Hongyang Chen

**Citation Count:**
3

**Abstract:**
Effective molecular representation learning is crucial for advancing molecular property prediction and drug design. Mainstream molecular representation learning approaches are based on Graph Neural Networks (GNNs). However, these approaches struggle with three significant challenges: insufficient annotations, molecular diversity, and architectural limitations such as over-squashing, which leads to the loss of critical structural details. To address these challenges, we introduce a new class of GNNs that integrates the Kolmogorov-Arnold Networks (KANs), known for their robust data-fitting capabilities and high accuracy in small-scale AI + Science tasks. By incorporating KANs into GNNs, our model enhances the representation of molecular structures. We further advance this approach with a variant called SwallowKAN (SKAN), which employs adaptive Radial Basis Functions (RBFs) as the core of the non-linear neurons. This innovation improves both computational efficiency and adaptability to diverse molecular structures. Building on the strengths of SKAN, we propose a new class of GNNs, GNN-SKAN, and its augmented variant, GNN-SKAN+, which incorporates a SKAN-based classifier to further boost performance. To our knowledge, this is the first work to integrate KANs into GNN architectures tailored for molecular representation learning. Experiments across 6 classification datasets, 6 regression datasets, and 4 few-shot learning datasets demonstrate that our approach achieves new state-of-the-art performance in terms of accuracy and computational cost.
       


### [KAN based Autoencoders for Factor Models](https://arxiv.org/abs/2408.02694)

**Authors:**
Tianqi Wang, Shubham Singh

**Citation Count:**
1

**Abstract:**
Inspired by recent advances in Kolmogorov-Arnold Networks (KANs), we introduce a novel approach to latent factor conditional asset pricing models. While previous machine learning applications in asset pricing have predominantly used Multilayer Perceptrons with ReLU activation functions to model latent factor exposures, our method introduces a KAN-based autoencoder which surpasses MLP models in both accuracy and interpretability. Our model offers enhanced flexibility in approximating exposures as nonlinear functions of asset characteristics, while simultaneously providing users with an intuitive framework for interpreting latent factors. Empirical backtesting demonstrates our model's superior ability to explain cross-sectional risk exposures. Moreover, long-short portfolios constructed using our model's predictions achieve higher Sharpe ratios, highlighting its practical value in investment management.
       


### [Bayesian Kolmogorov Arnold Networks (Bayesian_KANs): A Probabilistic Approach to Enhance Accuracy and Interpretability](https://arxiv.org/abs/2408.02706)

**Author:**
Masoud Muhammed Hassan

**Citation Count:**
3

**Abstract:**
Because of its strong predictive skills, deep learning has emerged as an essential tool in many industries, including healthcare. Traditional deep learning models, on the other hand, frequently lack interpretability and omit to take prediction uncertainty into account two crucial components of clinical decision making. In order to produce explainable and uncertainty aware predictions, this study presents a novel framework called Bayesian Kolmogorov Arnold Networks (BKANs), which combines the expressive capacity of Kolmogorov Arnold Networks with Bayesian inference. We employ BKANs on two medical datasets, which are widely used benchmarks for assessing machine learning models in medical diagnostics: the Pima Indians Diabetes dataset and the Cleveland Heart Disease dataset. Our method provides useful insights into prediction confidence and decision boundaries and outperforms traditional deep learning models in terms of prediction accuracy. Moreover, BKANs' capacity to represent aleatoric and epistemic uncertainty guarantees doctors receive more solid and trustworthy decision support. Our Bayesian strategy improves the interpretability of the model and considerably minimises overfitting, which is important for tiny and imbalanced medical datasets, according to experimental results. We present possible expansions to further use BKANs in more complicated multimodal datasets and address the significance of these discoveries for future research in building reliable AI systems for healthcare. This work paves the way for a new paradigm in deep learning model deployment in vital sectors where transparency and reliability are crucial.
       


### [KAN we improve on HEP classification tasks? Kolmogorov-Arnold Networks applied to an LHC physics example](https://arxiv.org/abs/2408.02743)

**Authors:**
Johannes Erdmann, Florian Mausolf, Jan Lukas Späh

**Venue:**
Computing and Software for Big Science

**Citation Count:**
4

**Abstract:**
Recently, Kolmogorov-Arnold Networks (KANs) have been proposed as an alternative to multilayer perceptrons, suggesting advantages in performance and interpretability. We study a typical binary event classification task in high-energy physics including high-level features and comment on the performance and interpretability of KANs in this context. Consistent with expectations, we find that the learned activation functions of a one-layer KAN resemble the univariate log-likelihood ratios of the respective input features. In deeper KANs, the activations in the first layer differ from those in the one-layer KAN, which indicates that the deeper KANs learn more complex representations of the data, a pattern commonly observed in other deep-learning architectures. We study KANs with different depths and widths and we compare them to multilayer perceptrons in terms of performance and number of trainable parameters. For the chosen classification task, we do not find that KANs are more parameter efficient. However, small KANs may offer advantages in terms of interpretability that come at the cost of only a moderate loss in performance.
       


### [Kolmogorov-Arnold PointNet: Deep learning for prediction of fluid fields on irregular geometries](https://arxiv.org/abs/2408.02950)

**Author:**
Ali Kashefi

**Citation Count:**
5

**Abstract:**
Kolmogorov-Arnold Networks (KANs) have emerged as a promising alternative to traditional Multilayer Perceptrons (MLPs) in deep learning. KANs have already been integrated into various architectures, such as convolutional neural networks, graph neural networks, and transformers, and their potential has been assessed for predicting physical quantities. However, the combination of KANs with point-cloud-based neural networks (e.g., PointNet) for computational physics has not yet been explored. To address this, we present Kolmogorov-Arnold PointNet (KA-PointNet) as a novel supervised deep learning framework for the prediction of incompressible steady-state fluid flow fields in irregular domains, where the predicted fields are a function of the geometry of the domains. In KA-PointNet, we implement shared KANs in the segmentation branch of the PointNet architecture. We utilize Jacobi polynomials to construct shared KANs. As a benchmark test case, we consider incompressible laminar steady-state flow over a cylinder, where the geometry of its cross-section varies over the data set. We investigate the performance of Jacobi polynomials with different degrees as well as special cases of Jacobi polynomials such as Legendre polynomials, Chebyshev polynomials of the first and second kinds, and Gegenbauer polynomials, in terms of the computational cost of training and accuracy of prediction of the test set. Additionally, we compare the performance of PointNet with shared KANs (i.e., KA-PointNet) and PointNet with shared MLPs. It is observed that when the number of trainable parameters is approximately equal, PointNet with shared KANs (i.e., KA-PointNet) outperforms PointNet with shared MLPs. Moreover, KA-PointNet predicts the pressure and velocity distributions along the surface of cylinders more accurately, resulting in more precise computations of lift and drag.
       


### [Path-SAM2: Transfer SAM2 for digital pathology semantic segmentation](https://arxiv.org/abs/2408.03651)

**Authors:**
Mingya Zhang, Liang Wang, Zhihao Chen, Yiyuan Ge, Xianping Tao

**Citation Count:**
3

**Abstract:**
The semantic segmentation task in pathology plays an indispensable role in assisting physicians in determining the condition of tissue lesions. With the proposal of Segment Anything Model (SAM), more and more foundation models have seen rapid development in the field of image segmentation. Recently, SAM2 has garnered widespread attention in both natural image and medical image segmentation. Compared to SAM, it has significantly improved in terms of segmentation accuracy and generalization performance. We compared the foundational models based on SAM and found that their performance in semantic segmentation of pathological images was hardly satisfactory. In this paper, we propose Path-SAM2, which for the first time adapts the SAM2 model to cater to the task of pathological semantic segmentation. We integrate the largest pretrained vision encoder for histopathology (UNI) with the original SAM2 encoder, adding more pathology-based prior knowledge. Additionally, we introduce a learnable Kolmogorov-Arnold Networks (KAN) classification module to replace the manual prompt process. In three adenoma pathological datasets, Path-SAM2 has achieved state-of-the-art performance.This study demonstrates the great potential of adapting SAM2 to pathology image segmentation tasks. We plan to release the code and model weights for this paper at: https://github.com/simzhangbest/SAM2PATH
       


### [Neural Network Modeling of Heavy-Quark Potential from Holography](https://arxiv.org/abs/2408.03784)

**Authors:**
Ou-Yang Luo, Xun Chen, Fu-Peng Li, Xiao-Hua Li, Kai Zhou

**Citation Count:**
3

**Abstract:**
Using Multi-Layer Perceptrons (MLP) and Kolmogorov-Arnold Networks (KAN), we construct a holographic model based on lattice QCD data for the heavy-quark potential in the 2+1 system. The deformation factor $w(r)$ in the metric is obtained using the two types of neural network. First, we numerically obtain $w(r)$ using MLP, accurately reproducing the QCD results of the lattice, and calculate the heavy quark potential at finite temperature and the chemical potential. Subsequently, we employ KAN within the Andreev-Zakharov model for validation purpose, which can analytically reconstruct $w(r)$, matching the Andreev-Zakharov model exactly and confirming the validity of MLP. Finally, we construct an analytical holographic model using KAN and study the heavy-quark potential at finite temperature and chemical potential using the KAN-based holographic model. This work demonstrates the potential of KAN to derive analytical expressions for high-energy physics applications.
       


### [From Black Box to Clarity: AI-Powered Smart Grid Optimization with Kolmogorov-Arnold Networks](https://arxiv.org/abs/2408.04063)

**Authors:**
Xiaoting Wang, Yuzhuo Li, Yunwei Li, Gregory Kish

**Venue:**
European Conference on Cognitive Ergonomics

**Citation Count:**
4

**Abstract:**
This work is the first to adopt Kolmogorov-Arnold Networks (KAN), a recent breakthrough in artificial intelligence, for smart grid optimizations. To fully leverage KAN's interpretability, a general framework is proposed considering complex uncertainties. The stochastic optimal power flow problem in hybrid AC/DC systems is chosen as a particularly tough case study for demonstrating the effectiveness of this framework.
       


### [Kolmogorov-Arnold Network for Online Reinforcement Learning](https://arxiv.org/abs/2408.04841)

**Authors:**
Victor Augusto Kich, Jair Augusto Bottega, Raul Steinmetz, Ricardo Bedin Grando, Ayano Yorozu, Akihisa Ohya

**Venue:**
International Conference on Control, Automation and Systems

**Citation Count:**
5

**Abstract:**
Kolmogorov-Arnold Networks (KANs) have shown potential as an alternative to Multi-Layer Perceptrons (MLPs) in neural networks, providing universal function approximation with fewer parameters and reduced memory usage. In this paper, we explore the use of KANs as function approximators within the Proximal Policy Optimization (PPO) algorithm. We evaluate this approach by comparing its performance to the original MLP-based PPO using the DeepMind Control Proprio Robotics benchmark. Our results indicate that the KAN-based reinforcement learning algorithm can achieve comparable performance to its MLP-based counterpart, often with fewer parameters. These findings suggest that KANs may offer a more efficient option for reinforcement learning models.
       


### [Physics-Informed Kolmogorov-Arnold Networks for Power System Dynamics](https://arxiv.org/abs/2408.06650)

**Authors:**
Hang Shuai, Fangxing Li

**Venue:**
IEEE Open Access Journal of Power and Energy

**Citation Count:**
10

**Abstract:**
This paper presents, for the first time, a framework for Kolmogorov-Arnold Networks (KANs) in power system applications. Inspired by the recently proposed KAN architecture, this paper proposes physics-informed Kolmogorov-Arnold Networks (PIKANs), a novel KAN-based physics-informed neural network (PINN) tailored to efficiently and accurately learn dynamics within power systems. The PIKANs present a promising alternative to conventional Multi-Layer Perceptrons (MLPs) based PINNs, achieving superior accuracy in predicting power system dynamics while employing a smaller network size. Simulation results on a single-machine infinite bus system and a 4-bus 2- generator system underscore the accuracy of the PIKANs in predicting rotor angle and frequency with fewer learnable parameters than conventional PINNs. Furthermore, the simulation results demonstrate PIKANs capability to accurately identify uncertain inertia and damping coefficients. This work opens up a range of opportunities for the application of KANs in power systems, enabling efficient determination of grid dynamics and precise parameter identification.
       


### [KAN You See It? KANs and Sentinel for Effective and Explainable Crop Field Segmentation](https://arxiv.org/abs/2408.07040)

**Authors:**
Daniele Rege Cambrin, Eleonora Poeta, Eliana Pastor, Tania Cerquitelli, Elena Baralis, Paolo Garza

**Venue:**
ECCV Workshops

**Citation Count:**
9

**Abstract:**
Segmentation of crop fields is essential for enhancing agricultural productivity, monitoring crop health, and promoting sustainable practices. Deep learning models adopted for this task must ensure accurate and reliable predictions to avoid economic losses and environmental impact. The newly proposed Kolmogorov-Arnold networks (KANs) offer promising advancements in the performance of neural networks. This paper analyzes the integration of KAN layers into the U-Net architecture (U-KAN) to segment crop fields using Sentinel-2 and Sentinel-1 satellite images and provides an analysis of the performance and explainability of these networks. Our findings indicate a 2\% improvement in IoU compared to the traditional full-convolutional U-Net model in fewer GFLOPs. Furthermore, gradient-based explanation techniques show that U-KAN predictions are highly plausible and that the network has a very high ability to focus on the boundaries of cultivated areas rather than on the areas themselves. The per-channel relevance analysis also reveals that some channels are irrelevant to this task.
       


### [VulCatch: Enhancing Binary Vulnerability Detection through CodeT5 Decompilation and KAN Advanced Feature Extraction](https://arxiv.org/abs/2408.07181)

**Authors:**
Abdulrahman Hamman Adama Chukkol, Senlin Luo, Kashif Sharif, Yunusa Haruna, Muhammad Muhammad Abdullahi

**Citation Count:**
1

**Abstract:**
Binary program vulnerability detection is critical for software security, yet existing deep learning approaches often rely on source code analysis, limiting their ability to detect unknown vulnerabilities. To address this, we propose VulCatch, a binary-level vulnerability detection framework. VulCatch introduces a Synergy Decompilation Module (SDM) and Kolmogorov-Arnold Networks (KAN) to transform raw binary code into pseudocode using CodeT5, preserving high-level semantics for deep analysis with tools like Ghidra and IDA. KAN further enhances feature transformation, enabling the detection of complex vulnerabilities. VulCatch employs word2vec, Inception Blocks, BiLSTM Attention, and Residual connections to achieve high detection accuracy (98.88%) and precision (97.92%), while minimizing false positives (1.56%) and false negatives (2.71%) across seven CVE datasets.
       


### [Kolmogorov-Arnold Networks (KAN) for Time Series Classification and Robust Analysis](https://arxiv.org/abs/2408.07314)

**Authors:**
Chang Dong, Liangwei Zheng, Weitong Chen

**Venue:**
International Conference on Advanced Data Mining and Applications

**Citation Count:**
15

**Abstract:**
Kolmogorov-Arnold Networks (KAN) has recently attracted significant attention as a promising alternative to traditional Multi-Layer Perceptrons (MLP). Despite their theoretical appeal, KAN require validation on large-scale benchmark datasets. Time series data, which has become increasingly prevalent in recent years, especially univariate time series are naturally suited for validating KAN. Therefore, we conducted a fair comparison among KAN, MLP, and mixed structures. The results indicate that KAN can achieve performance comparable to, or even slightly better than, MLP across 128 time series datasets. We also performed an ablation study on KAN, revealing that the output is primarily determined by the base component instead of b-spline function. Furthermore, we assessed the robustness of these models and found that KAN and the hybrid structure MLP\_KAN exhibit significant robustness advantages, attributed to their lower Lipschitz constants. This suggests that KAN and KAN layers hold strong potential to be robust models or to improve the adversarial robustness of other models.
       


### [KAN versus MLP on Irregular or Noisy Functions](https://arxiv.org/abs/2408.07906)

**Authors:**
Chen Zeng, Jiahui Wang, Haoran Shen, Qiao Wang

**Citation Count:**
9

**Abstract:**
In this paper, we compare the performance of Kolmogorov-Arnold Networks (KAN) and Multi-Layer Perceptron (MLP) networks on irregular or noisy functions. We control the number of parameters and the size of the training samples to ensure a fair comparison. For clarity, we categorize the functions into six types: regular functions, continuous functions with local non-differentiable points, functions with jump discontinuities, functions with singularities, functions with coherent oscillations, and noisy functions. Our experimental results indicate that KAN does not always perform best. For some types of functions, MLP outperforms or performs comparably to KAN. Furthermore, increasing the size of training samples can improve performance to some extent. When noise is added to functions, the irregular features are often obscured by the noise, making it challenging for both MLP and KAN to extract these features effectively. We hope these experiments provide valuable insights for future neural network research and encourage further investigations to overcome these challenges.
       


### [The Dawn of KAN in Image-to-Image (I2I) Translation: Integrating Kolmogorov-Arnold Networks with GANs for Unpaired I2I Translation](https://arxiv.org/abs/2408.08216)

**Authors:**
Arpan Mahara, Naphtali D. Rishe, Liangdong Deng

**Citation Count:**
2

**Abstract:**
Image-to-Image translation in Generative Artificial Intelligence (Generative AI) has been a central focus of research, with applications spanning healthcare, remote sensing, physics, chemistry, photography, and more. Among the numerous methodologies, Generative Adversarial Networks (GANs) with contrastive learning have been particularly successful. This study aims to demonstrate that the Kolmogorov-Arnold Network (KAN) can effectively replace the Multi-layer Perceptron (MLP) method in generative AI, particularly in the subdomain of image-to-image translation, to achieve better generative quality. Our novel approach replaces the two-layer MLP with a two-layer KAN in the existing Contrastive Unpaired Image-to-Image Translation (CUT) model, developing the KAN-CUT model. This substitution favors the generation of more informative features in low-dimensional vector representations, which contrastive learning can utilize more effectively to produce high-quality images in the target domain. Extensive experiments, detailed in the results section, demonstrate the applicability of KAN in conjunction with contrastive learning and GANs in Generative AI, particularly for image-to-image translation. This work suggests that KAN could be a valuable component in the broader generative AI domain.
       


### [A Conflicts-free, Speed-lossless KAN-based Reinforcement Learning Decision System for Interactive Driving in Roundabouts](https://arxiv.org/abs/2408.08242)

**Authors:**
Zhihao Lin, Zhen Tian, Qi Zhang, Ziyang Ye, Hanyang Zhuang, Jianglin Lan

**Citation Count:**
5

**Abstract:**
Safety and efficiency are crucial for autonomous driving in roundabouts, especially in the context of mixed traffic where autonomous vehicles (AVs) and human-driven vehicles coexist. This paper introduces a learning-based algorithm tailored to foster safe and efficient driving behaviors across varying levels of traffic flows in roundabouts. The proposed algorithm employs a deep Q-learning network to effectively learn safe and efficient driving strategies in complex multi-vehicle roundabouts. Additionally, a KAN (Kolmogorov-Arnold network) enhances the AVs' ability to learn their surroundings robustly and precisely. An action inspector is integrated to replace dangerous actions to avoid collisions when the AV interacts with the environment, and a route planner is proposed to enhance the driving efficiency and safety of the AVs. Moreover, a model predictive control is adopted to ensure stability and precision of the driving actions. The results show that our proposed system consistently achieves safe and efficient driving whilst maintaining a stable training process, as evidenced by the smooth convergence of the reward function and the low variance in the training curves across various traffic flows. Compared to state-of-the-art benchmarks, the proposed algorithm achieves a lower number of collisions and reduced travel time to destination.
       


### [Activation Space Selectable Kolmogorov-Arnold Networks](https://arxiv.org/abs/2408.08338)

**Authors:**
Zhuoqin Yang, Jiansong Zhang, Xiaoling Luo, Zheng Lu, Linlin Shen

**Citation Count:**
5

**Abstract:**
The multilayer perceptron (MLP), a fundamental paradigm in current artificial intelligence, is widely applied in fields such as computer vision and natural language processing. However, the recently proposed Kolmogorov-Arnold Network (KAN), based on nonlinear additive connections, has been proven to achieve performance comparable to MLPs with significantly fewer parameters. Despite this potential, the use of a single activation function space results in reduced performance of KAN and related works across different tasks. To address this issue, we propose an activation space Selectable KAN (S-KAN). S-KAN employs an adaptive strategy to choose the possible activation mode for data at each feedforward KAN node. Our approach outperforms baseline methods in seven representative function fitting tasks and significantly surpasses MLP methods with the same level of parameters. Furthermore, we extend the structure of S-KAN and propose an activation space selectable Convolutional KAN (S-ConvKAN), which achieves leading results on four general image classification datasets. Our method mitigates the performance variability of the original KAN across different tasks and demonstrates through extensive experiments that feedforward KANs with selectable activations can achieve or even exceed the performance of MLP-based methods. This work contributes to the understanding of the data-centric design of new AI paradigms and provides a foundational reference for innovations in KAN-based network architectures.
       


### [Photonic KAN: a Kolmogorov-Arnold network inspired efficient photonic neuromorphic architecture](https://arxiv.org/abs/2408.08407)

**Authors:**
Yiwei Peng, Sean Hooten, Xinling Yu, Thomas Van Vaerenbergh, Yuan Yuan, Xian Xiao, Bassem Tossoun, Stanley Cheung, Marco Fiorentino, Raymond Beausoleil

**Citation Count:**
1

**Abstract:**
Kolmogorov-Arnold Networks (KAN) models were recently proposed and claimed to provide improved parameter scaling and interpretability compared to conventional multilayer perceptron (MLP) models. Inspired by the KAN architecture, we propose the Photonic KAN -- an integrated all-optical neuromorphic platform leveraging highly parametric optical nonlinear transfer functions along KAN edges. In this work, we implement such nonlinearities in the form of cascaded ring-assisted Mach-Zehnder Interferometer (MZI) devices. This innovative design has the potential to address key limitations of current photonic neural networks. In our test cases, the Photonic KAN showcases enhanced parameter scaling and interpretability compared to existing photonic neural networks. The photonic KAN achieves approximately 65$\times$ reduction in energy consumption and area, alongside a 50$\times$ reduction in latency compared to previous MZI-based photonic accelerators with similar performance for function fitting task. This breakthrough presents a promising new avenue for expanding the scalability and efficiency of neuromorphic hardware platforms.
       


### [Beyond KAN: Introducing KarSein for Adaptive High-Order Feature Interaction Modeling in CTR Prediction](https://arxiv.org/abs/2408.08713)

**Authors:**
Yunxiao Shi, Wujiang Xu, Haimin Zhang, Qiang Wu, Min Xu

**Citation Count:**
1

**Abstract:**
Modeling high-order feature interactions is crucial for click-through rate (CTR) prediction, and traditional approaches often predefine a maximum interaction order and rely on exhaustive enumeration of feature combinations up to this predefined order. This framework heavily relies on prior domain knowledge to define interaction scope and entails high computational costs from enumeration. Conventional CTR models face a trade-off between improving representation through complex high-order feature interactions and reducing computational inefficiencies associated with these processes. To address this dual challenge, this study introduces the Kolmogorov-Arnold Represented Sparse Efficient Interaction Network (KarSein). Drawing inspiration from the learnable activation mechanism in the Kolmogorov-Arnold Network (KAN), KarSein leverages this mechanism to adaptively transform low-order basic features into high-order feature interactions, offering a novel approach to feature interaction modeling. KarSein extends the capabilities of KAN by introducing a more efficient architecture that significantly reduces computational costs while accommodating two-dimensional embedding vectors as feature inputs. Furthermore, it overcomes the limitation of KAN's its inability to spontaneously capture multiplicative relationships among features.
  Extensive experiments highlight the superiority of KarSein, demonstrating its ability to surpass not only the vanilla implementation of KAN in CTR predictio but also other baseline methods. Remarkably, KarSein achieves exceptional predictive accuracy while maintaining a highly compact parameter size and minimal computational overhead. As the first attempt to apply KAN in the CTR domain, this work introduces KarSein as a novel solution for modeling complex feature interactions, underscoring its transformative potential in advancing CTR prediction task.
       


### [Detecting the Undetectable: Combining Kolmogorov-Arnold Networks and MLP for AI-Generated Image Detection](https://arxiv.org/abs/2408.09371)

**Authors:**
Taharim Rahman Anon, Jakaria Islam Emon

**Citation Count:**
4

**Abstract:**
As artificial intelligence progresses, the task of distinguishing between real and AI-generated images is increasingly complicated by sophisticated generative models. This paper presents a novel detection framework adept at robustly identifying images produced by cutting-edge generative AI models, such as DALL-E 3, MidJourney, and Stable Diffusion 3. We introduce a comprehensive dataset, tailored to include images from these advanced generators, which serves as the foundation for extensive evaluation. we propose a classification system that integrates semantic image embeddings with a traditional Multilayer Perceptron (MLP). This baseline system is designed to effectively differentiate between real and AI-generated images under various challenging conditions. Enhancing this approach, we introduce a hybrid architecture that combines Kolmogorov-Arnold Networks (KAN) with the MLP. This hybrid model leverages the adaptive, high-resolution feature transformation capabilities of KAN, enabling our system to capture and analyze complex patterns in AI-generated images that are typically overlooked by conventional models. In out-of-distribution testing, our proposed model consistently outperformed the standard MLP across three out of distribution test datasets, demonstrating superior performance and robustness in classifying real images from AI-generated images with impressive F1 scores.
       


### [KAN 2.0: Kolmogorov-Arnold Networks Meet Science](https://arxiv.org/abs/2408.10205)

**Authors:**
Ziming Liu, Pingchuan Ma, Yixuan Wang, Wojciech Matusik, Max Tegmark

**Citation Count:**
70

**Abstract:**
A major challenge of AI + Science lies in their inherent incompatibility: today's AI is primarily based on connectionism, while science depends on symbolism. To bridge the two worlds, we propose a framework to seamlessly synergize Kolmogorov-Arnold Networks (KANs) and science. The framework highlights KANs' usage for three aspects of scientific discovery: identifying relevant features, revealing modular structures, and discovering symbolic formulas. The synergy is bidirectional: science to KAN (incorporating scientific knowledge into KANs), and KAN to science (extracting scientific insights from KANs). We highlight major new functionalities in the pykan package: (1) MultKAN: KANs with multiplication nodes. (2) kanpiler: a KAN compiler that compiles symbolic formulas into KANs. (3) tree converter: convert KANs (or any neural networks) to tree graphs. Based on these tools, we demonstrate KANs' capability to discover various types of physical laws, including conserved quantities, Lagrangians, symmetries, and constitutive laws.
       


### [Kolmogorov Arnold Networks in Fraud Detection: Bridging the Gap Between Theory and Practice](https://arxiv.org/abs/2408.10263)

**Authors:**
Yang Lu, Felix Zhan

**Citation Count:**
1

**Abstract:**
This study evaluates the applicability of Kolmogorov-Arnold Networks (KAN) in fraud detection, finding that their effectiveness is context-dependent. We propose a quick decision rule using Principal Component Analysis (PCA) to assess the suitability of KAN: if data can be effectively separated in two dimensions using splines, KAN may outperform traditional models; otherwise, other methods could be more appropriate. We also introduce a heuristic approach to hyperparameter tuning, significantly reducing computational costs. These findings suggest that while KAN has potential, its use should be guided by data-specific assessments.
       


### [Deep-MacroFin: Informed Equilibrium Neural Network for Continuous Time Economic Models](https://arxiv.org/abs/2408.10368)

**Authors:**
Yuntao Wu, Jiayuan Guo, Goutham Gopalakrishna, Zissis Poulos

**Abstract:**
In this paper, we present Deep-MacroFin, a comprehensive framework designed to solve partial differential equations, with a particular focus on models in continuous time economics. This framework leverages deep learning methodologies, including Multi-Layer Perceptrons and the newly developed Kolmogorov-Arnold Networks. It is optimized using economic information encapsulated by Hamilton-Jacobi-Bellman (HJB) equations and coupled algebraic equations. The application of neural networks holds the promise of accurately resolving high-dimensional problems with fewer computational demands and limitations compared to other numerical methods. This framework can be readily adapted for systems of partial differential equations in high dimensions. Importantly, it offers a more efficient (5$\times$ less CUDA memory and 40$\times$ fewer FLOPs in 100D problems) and user-friendly implementation than existing libraries. We also incorporate a time-stepping scheme to enhance training stability for nonlinear HJB equations, enabling the solution of 50D economic models.
       


### [UKAN: Unbound Kolmogorov-Arnold Network Accompanied with Accelerated Library](https://arxiv.org/abs/2408.11200)

**Authors:**
Alireza Moradzadeh, Lukasz Wawrzyniak, Miles Macklin, Saee G. Paliwal

**Citation Count:**
1

**Abstract:**
In this work, we present a GPU-accelerated library for the underlying components of Kolmogorov-Arnold Networks (KANs), along with an algorithm to eliminate bounded grids in KANs. The GPU-accelerated library reduces the computational complexity of Basis Spline (B-spline) evaluation by a factor of $\mathcal{O}$(grid size) compared to existing codes, enabling batch computation for large-scale learning. To overcome the limitations of traditional KANs, we introduce Unbounded KANs (UKANs), which eliminate the need for a bounded grid and a fixed number of B-spline coefficients. To do so, we replace the KAN parameters (B-spline coefficients) with a coefficient generator (CG) model. The inputs to the CG model are designed based on the idea of an infinite symmetric grid extending from negative infinity to positive infinity. The positional encoding of grid group, a sequential collection of B-spline grid indexes, is fed into the CG model, and coefficients are consumed by the efficient implementation (matrix representations) of B-spline functions to generate outputs. We perform several experiments on regression, classification, and generative tasks, which are promising. In particular, UKAN does not require data normalization or a bounded domain for evaluation. Additionally, our benchmarking results indicate the superior memory and computational efficiency of our library compared to existing codes.
       


### [Are KANs Effective for Multivariate Time Series Forecasting?](https://arxiv.org/abs/2408.11306)

**Authors:**
Xiao Han, Xinfeng Zhang, Yiling Wu, Zhenduo Zhang, Zhe Wu

**Citation Count:**
5

**Abstract:**
Multivariate time series forecasting is a crucial task that predicts the future states based on historical inputs. Related techniques have been developing in parallel with the machine learning community, from early statistical learning methods to current deep learning methods. Despite their significant advancements, existing methods continue to struggle with the challenge of inadequate interpretability. The rise of the Kolmogorov-Arnold Network (KAN) provides a new perspective to solve this challenge, but current work has not yet concluded whether KAN is effective in time series forecasting tasks. In this paper, we aim to evaluate the effectiveness of KANs in time-series forecasting from the perspectives of performance, integrability, efficiency, and interpretability. To this end, we propose the Multi-layer Mixture-of-KAN network (MMK), which achieves excellent performance while retaining KAN's ability to be transformed into a combination of symbolic functions. The core module of MMK is the mixture-of-KAN layer, which uses a mixture-of-experts structure to assign variables to best-matched KAN experts. Then, we explore some useful experimental strategies to deal with the issues in the training stage. Finally, we compare MMK and various baselines on seven datasets. Extensive experimental and visualization results demonstrate that KANs are effective in multivariate time series forecasting. Code is available at: https://github.com/2448845600/EasyTSF.
       


### [KonvLiNA: Integrating Kolmogorov-Arnold Network with Linear Nyström Attention for feature fusion in Crop Field Detection](https://arxiv.org/abs/2408.13160)

**Authors:**
Haruna Yunusa, Qin Shiyin, Adamu Lawan, Abdulrahman Hamman Adama Chukkol

**Venue:**
International Conference on Machine Vision

**Citation Count:**
1

**Abstract:**
Crop field detection is a critical component of precision agriculture, essential for optimizing resource allocation and enhancing agricultural productivity. This study introduces KonvLiNA, a novel framework that integrates Convolutional Kolmogorov-Arnold Networks (cKAN) with Nyström attention mechanisms for effective crop field detection. Leveraging KAN adaptive activation functions and the efficiency of Nyström attention in handling largescale data, KonvLiNA significantly enhances feature extraction, enabling the model to capture intricate patterns in complex agricultural environments. Experimental results on rice crop dataset demonstrate KonvLiNA superiority over state-of-the-art methods, achieving a 0.415 AP and 0.459 AR with the Swin-L backbone, outperforming traditional YOLOv8 by significant margins. Additionally, evaluation on the COCO dataset showcases competitive performance across small, medium, and large objects, highlighting KonvLiNA efficacy in diverse agricultural settings. This work highlights the potential of hybrid KAN and attention mechanisms for advancing precision agriculture through improved crop field detection and management.
       


### [On the Robustness of Kolmogorov-Arnold Networks: An Adversarial Perspective](https://arxiv.org/abs/2408.13809)

**Authors:**
Tal Alter, Raz Lapid, Moshe Sipper

**Citation Count:**
6

**Abstract:**
Kolmogorov-Arnold Networks (KANs) have recently emerged as a novel approach to function approximation, demonstrating remarkable potential in various domains. Despite their theoretical promise, the robustness of KANs under adversarial conditions has yet to be thoroughly examined. In this paper we explore the adversarial robustness of KANs, with a particular focus on image classification tasks. We assess the performance of KANs against standard white box and black-box adversarial attacks, comparing their resilience to that of established neural network architectures. Our experimental evaluation encompasses a variety of standard image classification benchmark datasets and investigates both fully connected and convolutional neural network architectures, of three sizes: small, medium, and large. We conclude that small- and medium-sized KANs (either fully connected or convolutional) are not consistently more robust than their standard counterparts, but that large-sized KANs are, by and large, more robust. This comprehensive evaluation of KANs in adversarial scenarios offers the first in-depth analysis of KAN security, laying the groundwork for future research in this emerging field.
       


### [GINN-KAN: Interpretability pipelining with applications in Physics Informed Neural Networks](https://arxiv.org/abs/2408.14780)

**Authors:**
Nisal Ranasinghe, Yu Xia, Sachith Seneviratne, Saman Halgamuge

**Citation Count:**
4

**Abstract:**
Neural networks are powerful function approximators, yet their ``black-box" nature often renders them opaque and difficult to interpret. While many post-hoc explanation methods exist, they typically fail to capture the underlying reasoning processes of the networks. A truly interpretable neural network would be trained similarly to conventional models using techniques such as backpropagation, but additionally provide insights into the learned input-output relationships. In this work, we introduce the concept of interpretability pipelineing, to incorporate multiple interpretability techniques to outperform each individual technique. To this end, we first evaluate several architectures that promise such interpretability, with a particular focus on two recent models selected for their potential to incorporate interpretability into standard neural network architectures while still leveraging backpropagation: the Growing Interpretable Neural Network (GINN) and Kolmogorov Arnold Networks (KAN). We analyze the limitations and strengths of each and introduce a novel interpretable neural network GINN-KAN that synthesizes the advantages of both models. When tested on the Feynman symbolic regression benchmark datasets, GINN-KAN outperforms both GINN and KAN. To highlight the capabilities and the generalizability of this approach, we position GINN-KAN as an alternative to conventional black-box networks in Physics-Informed Neural Networks (PINNs). We expect this to have far-reaching implications in the application of deep learning pipelines in the natural sciences. Our experiments with this interpretable PINN on 15 different partial differential equations demonstrate that GINN-KAN augmented PINNs outperform PINNs with black-box networks in solving differential equations and surpass the capabilities of both GINN and KAN.
       


### [DualKanbaFormer: An Efficient Selective Sparse Framework for Multimodal Aspect-based Sentiment Analysis](https://arxiv.org/abs/2408.15379)

**Authors:**
Adamu Lawan, Juhua Pu, Haruna Yunusa, Muhammad Lawan, Aliyu Umar, Adamu Sani Yahya, Mahmoud Basi

**Abstract:**
Multimodal Aspect-based Sentiment Analysis (MABSA) enhances sentiment detection by integrating textual data with complementary modalities, such as images, to provide a more refined and comprehensive understanding of sentiment. However, conventional attention mechanisms, despite notable benchmarks, are hindered by quadratic complexity, limiting their ability to fully capture global contextual dependencies and rich semantic information in both modalities. To address this limitation, we introduce DualKanbaFormer, a novel framework that leverages parallel Textual and Visual KanbaFormer modules for robust multimodal analysis. Our approach incorporates Aspect-Driven Sparse Attention (ADSA) to dynamically balance coarse-grained aggregation and fine-grained selection for aspect-focused precision, ensuring the preservation of both global context awareness and local precision in textual and visual representations. Additionally, we utilize the Selective State Space Model (Mamba) to capture extensive global semantic information across both modalities. Furthermore, We replace traditional feed-forward networks and normalization with Kolmogorov-Arnold Networks (KANs) and Dynamic Tanh (DyT) to enhance non-linear expressivity and inference stability. To facilitate the effective integration of textual and visual features, we design a multimodal gated fusion layer that dynamically optimizes inter-modality interactions, significantly enhancing the models efficacy in MABSA tasks. Comprehensive experiments on two publicly available datasets reveal that DualKanbaFormer consistently outperforms several state-of-the-art (SOTA) models.
       


### [Enhancing Intrusion Detection in IoT Environments: An Advanced Ensemble Approach Using Kolmogorov-Arnold Networks](https://arxiv.org/abs/2408.15886)

**Authors:**
Amar Amouri, Mohamad Mahmoud Al Rahhal, Yakoub Bazi, Ismail Butun, Imad Mahgoub

**Venue:**
International Symposium on Networks, Computers and Communications

**Citation Count:**
3

**Abstract:**
In recent years, the evolution of machine learning techniques has significantly impacted the field of intrusion detection, particularly within the context of the Internet of Things (IoT). As IoT networks expand, the need for robust security measures to counteract potential threats has become increasingly critical. This paper introduces a hybrid Intrusion Detection System (IDS) that synergistically combines Kolmogorov-Arnold Networks (KANs) with the XGBoost algorithm. Our proposed IDS leverages the unique capabilities of KANs, which utilize learnable activation functions to model complex relationships within data, alongside the powerful ensemble learning techniques of XGBoost, known for its high performance in classification tasks. This hybrid approach not only enhances the detection accuracy but also improves the interpretability of the model, making it suitable for dynamic and intricate IoT environments. Experimental evaluations demonstrate that our hybrid IDS achieves an impressive detection accuracy exceeding 99% in distinguishing between benign and malicious activities. Additionally, we were able to achieve F1 scores, precision, and recall that exceeded 98%. Furthermore, we conduct a comparative analysis against traditional Multi-Layer Perceptron (MLP) networks, assessing performance metrics such as Precision, Recall, and F1-score. The results underscore the efficacy of integrating KANs with XGBoost, highlighting the potential of this innovative approach to significantly strengthen the security framework of IoT networks.
       


### [Addressing common misinterpretations of KART and UAT in neural network literature](https://arxiv.org/abs/2408.16389)

**Author:**
Vugar Ismailov

**Citation Count:**
1

**Abstract:**
This note addresses the Kolmogorov-Arnold Representation Theorem (KART) and the Universal Approximation Theorem (UAT), focusing on their common and frequent misinterpretations in many papers related to neural network approximation. Our remarks aim to support a more accurate understanding of KART and UAT among neural network specialists. In addition, we explore the minimal number of neurons required for universal approximation, showing that KART's lower bounds extend to standard multilayer perceptrons, even with smooth activation functions.
       


### [LAR-IQA: A Lightweight, Accurate, and Robust No-Reference Image Quality Assessment Model](https://arxiv.org/abs/2408.17057)

**Authors:**
Nasim Jamshidi Avanaki, Abhijay Ghildyal, Nabajeet Barman, Saman Zadtootaghaj

**Venue:**
ECCV Workshops

**Citation Count:**
1

**Abstract:**
Recent advancements in the field of No-Reference Image Quality Assessment (NR-IQA) using deep learning techniques demonstrate high performance across multiple open-source datasets. However, such models are typically very large and complex making them not so suitable for real-world deployment, especially on resource- and battery-constrained mobile devices. To address this limitation, we propose a compact, lightweight NR-IQA model that achieves state-of-the-art (SOTA) performance on ECCV AIM UHD-IQA challenge validation and test datasets while being also nearly 5.7 times faster than the fastest SOTA model. Our model features a dual-branch architecture, with each branch separately trained on synthetically and authentically distorted images which enhances the model's generalizability across different distortion types. To improve robustness under diverse real-world visual conditions, we additionally incorporate multiple color spaces during the training process. We also demonstrate the higher accuracy of recently proposed Kolmogorov-Arnold Networks (KANs) for final quality regression as compared to the conventional Multi-Layer Perceptrons (MLPs). Our evaluation considering various open-source datasets highlights the practical, high-accuracy, and robust performance of our proposed lightweight model. Code: https://github.com/nasimjamshidi/LAR-IQA.
       


### [AASIST3: KAN-Enhanced AASIST Speech Deepfake Detection using SSL Features and Additional Regularization for the ASVspoof 2024 Challenge](https://arxiv.org/abs/2408.17352)

**Authors:**
Kirill Borodin, Vasiliy Kudryavtsev, Dmitrii Korzh, Alexey Efimenko, Grach Mkrtchian, Mikhail Gorodnichev, Oleg Y. Rogov

**Venue:**
The Automatic Speaker Verification Spoofing Countermeasures Workshop (ASVspoof 2024)

**Citation Count:**
3

**Abstract:**
Automatic Speaker Verification (ASV) systems, which identify speakers based on their voice characteristics, have numerous applications, such as user authentication in financial transactions, exclusive access control in smart devices, and forensic fraud detection. However, the advancement of deep learning algorithms has enabled the generation of synthetic audio through Text-to-Speech (TTS) and Voice Conversion (VC) systems, exposing ASV systems to potential vulnerabilities. To counteract this, we propose a novel architecture named AASIST3. By enhancing the existing AASIST framework with Kolmogorov-Arnold networks, additional layers, encoders, and pre-emphasis techniques, AASIST3 achieves a more than twofold improvement in performance. It demonstrates minDCF results of 0.5357 in the closed condition and 0.1414 in the open condition, significantly enhancing the detection of synthetic voices and improving ASV security.
       


## September
### [GNN-Empowered Effective Partial Observation MARL Method for AoI Management in Multi-UAV Network](https://arxiv.org/abs/2409.00036)

**Authors:**
Yuhao Pan, Xiucheng Wang, Zhiyao Xu, Nan Cheng, Wenchao Xu, Jun-jie Zhang

**Venue:**
IEEE Internet of Things Journal

**Citation Count:**
4

**Abstract:**
Unmanned Aerial Vehicles (UAVs), due to their low cost and high flexibility, have been widely used in various scenarios to enhance network performance. However, the optimization of UAV trajectories in unknown areas or areas without sufficient prior information, still faces challenges related to poor planning performance and low distributed execution. These challenges arise when UAVs rely solely on their own observation information and the information from other UAVs within their communicable range, without access to global information. To address these challenges, this paper proposes the Qedgix framework, which combines graph neural networks (GNNs) and the QMIX algorithm to achieve distributed optimization of the Age of Information (AoI) for users in unknown scenarios. The framework utilizes GNNs to extract information from UAVs, users within the observable range, and other UAVs within the communicable range, thereby enabling effective UAV trajectory planning. Due to the discretization and temporal features of AoI indicators, the Qedgix framework employs QMIX to optimize distributed partially observable Markov decision processes (Dec-POMDP) based on centralized training and distributed execution (CTDE) with respect to mean AoI values of users. By modeling the UAV network optimization problem in terms of AoI and applying the Kolmogorov-Arnold representation theorem, the Qedgix framework achieves efficient neural network training through parameter sharing based on permutation invariance. Simulation results demonstrate that the proposed algorithm significantly improves convergence speed while reducing the mean AoI values of users. The code is available at https://github.com/UNIC-Lab/Qedgix.
       


### [Application of Kolmogorov-Arnold Networks in high energy physics](https://arxiv.org/abs/2409.01724)

**Authors:**
E. Abasov, P. Volkov, G. Vorotnikov, L. Dudko, A. Zaborenko, E. Iudin, A. Markina, M. Perfilov

**Venue:**
Moscow University Physics Bulletin

**Citation Count:**
3

**Abstract:**
Kolmogorov-Arnold Networks represent a recent advancement in machine learning, with the potential to outperform traditional perceptron-based neural networks across various domains as well as provide more interpretability with the use of symbolic formulas and pruning. This study explores the application of KANs to specific tasks in high-energy physics. We evaluate the performance of KANs in distinguishing multijet processes in proton-proton collisions and in reconstructing missing transverse momentum in events involving dark matter.
       


### [FC-KAN: Function Combinations in Kolmogorov-Arnold Networks](https://arxiv.org/abs/2409.01763)

**Authors:**
Hoang-Thang Ta, Duy-Quy Thai, Abu Bakar Siddiqur Rahman, Grigori Sidorov, Alexander Gelbukh

**Citation Count:**
8

**Abstract:**
In this paper, we introduce FC-KAN, a Kolmogorov-Arnold Network (KAN) that leverages combinations of popular mathematical functions such as B-splines, wavelets, and radial basis functions on low-dimensional data through element-wise operations. We explore several methods for combining the outputs of these functions, including sum, element-wise product, the addition of sum and element-wise product, representations of quadratic and cubic functions, concatenation, linear transformation of the concatenated output, and others. In our experiments, we compare FC-KAN with a multi-layer perceptron network (MLP) and other existing KANs, such as BSRBF-KAN, EfficientKAN, FastKAN, and FasterKAN, on the MNIST and Fashion-MNIST datasets. Two variants of FC-KAN, which use a combination of outputs from B-splines and Difference of Gaussians (DoG) and from B-splines and linear transformations in the form of a quadratic function, outperformed overall other models on the average of 5 independent training runs. We expect that FC-KAN can leverage function combinations to design future KANs. Our repository is publicly available at: https://github.com/hoangthangta/FC_KAN.
       


### [KAN See In the Dark](https://arxiv.org/abs/2409.03404)

**Authors:**
Aoxiang Ning, Minglong Xue, Jinhong He, Chengyun Song

**Venue:**
IEEE Signal Processing Letters

**Citation Count:**
2

**Abstract:**
Existing low-light image enhancement methods are difficult to fit the complex nonlinear relationship between normal and low-light images due to uneven illumination and noise effects. The recently proposed Kolmogorov-Arnold networks (KANs) feature spline-based convolutional layers and learnable activation functions, which can effectively capture nonlinear dependencies. In this paper, we design a KAN-Block based on KANs and innovatively apply it to low-light image enhancement. This method effectively alleviates the limitations of current methods constrained by linear network structures and lack of interpretability, further demonstrating the potential of KANs in low-level vision tasks. Given the poor perception of current low-light image enhancement methods and the stochastic nature of the inverse diffusion process, we further introduce frequency-domain perception for visually oriented enhancement. Extensive experiments demonstrate the competitive performance of our method on benchmark datasets. The code will be available at: https://github.com/AXNing/KSID}{https://github.com/AXNing/KSID.
       


### [Efficient prediction of potential energy surface and physical properties with Kolmogorov-Arnold Networks](https://arxiv.org/abs/2409.03430)

**Authors:**
Rui Wang, Hongyu Yu, Yang Zhong, Hongjun Xiang

**Venue:**
Journal of Materials Informatics

**Abstract:**
The application of machine learning methodologies for predicting properties within materials science has garnered significant attention. Among recent advancements, Kolmogorov-Arnold Networks (KANs) have emerged as a promising alternative to traditional Multi-Layer Perceptrons (MLPs). This study evaluates the impact of substituting MLPs with KANs within three established machine learning frameworks: Allegro, Neural Equivariant Interatomic Potentials (NequIP), and the Edge-Based Tensor Prediction Graph Neural Network (ETGNN). Our results demonstrate that the integration of KANs generally yields enhanced prediction accuracies. Specifically, replacing MLPs with KANs in the output blocks leads to notable improvements in accuracy and, in certain scenarios, also results in reduced training times. Furthermore, employing KANs exclusively in the output block facilitates faster inference and improved computational efficiency relative to utilizing KANs throughout the entire model. The selection of an optimal basis function for KANs is found to be contingent upon the particular problem at hand. Our results demonstrate the strong potential of KANs in enhancing machine learning potentials and material property predictions.
       


### [CoxKAN: Kolmogorov-Arnold Networks for Interpretable, High-Performance Survival Analysis](https://arxiv.org/abs/2409.04290)

**Authors:**
William Knottenbelt, Zeyu Gao, Rebecca Wray, Woody Zhidong Zhang, Jiashuai Liu, Mireia Crispin-Ortuzar

**Citation Count:**
10

**Abstract:**
Survival analysis is a branch of statistics used for modeling the time until a specific event occurs and is widely used in medicine, engineering, finance, and many other fields. When choosing survival models, there is typically a trade-off between performance and interpretability, where the highest performance is achieved by black-box models based on deep learning. This is a major problem in fields such as medicine where practitioners are reluctant to blindly trust black-box models to make important patient decisions. Kolmogorov-Arnold Networks (KANs) were recently proposed as an interpretable and accurate alternative to multi-layer perceptrons (MLPs). We introduce CoxKAN, a Cox proportional hazards Kolmogorov-Arnold Network for interpretable, high-performance survival analysis. We evaluate the proposed CoxKAN on 4 synthetic datasets and 9 real medical datasets. The synthetic experiments demonstrate that CoxKAN accurately recovers interpretable symbolic formulae for the hazard function, and effectively performs automatic feature selection. Evaluation on the 9 real datasets show that CoxKAN consistently outperforms the Cox proportional hazards model and achieves performance that is superior or comparable to that of tuned MLPs. Furthermore, we find that CoxKAN identifies complex interactions between predictor variables that would be extremely difficult to recognise using existing survival methods, and automatically finds symbolic formulae which uncover the precise effect of important biomarkers on patient risk.
       


### [CF-KAN: Kolmogorov-Arnold Network-based Collaborative Filtering to Mitigate Catastrophic Forgetting in Recommender Systems](https://arxiv.org/abs/2409.05878)

**Authors:**
Jin-Duk Park, Kyung-Min Kim, Won-Yong Shin

**Citation Count:**
3

**Abstract:**
Collaborative filtering (CF) remains essential in recommender systems, leveraging user--item interactions to provide personalized recommendations. Meanwhile, a number of CF techniques have evolved into sophisticated model architectures based on multi-layer perceptrons (MLPs). However, MLPs often suffer from catastrophic forgetting, and thus lose previously acquired knowledge when new information is learned, particularly in dynamic environments requiring continual learning. To tackle this problem, we propose CF-KAN, a new CF method utilizing Kolmogorov-Arnold networks (KANs). By learning nonlinear functions on the edge level, KANs are more robust to the catastrophic forgetting problem than MLPs. Built upon a KAN-based autoencoder, CF-KAN is designed in the sense of effectively capturing the intricacies of sparse user--item interactions and retaining information from previous data instances. Despite its simplicity, our extensive experiments demonstrate 1) CF-KAN's superiority over state-of-the-art methods in recommendation accuracy, 2) CF-KAN's resilience to catastrophic forgetting, underscoring its effectiveness in both static and dynamic recommendation scenarios, and 3) CF-KAN's edge-level interpretation facilitating the explainability of recommendations.
       


### [Self-Supervised State Space Model for Real-Time Traffic Accident Prediction Using eKAN Networks](https://arxiv.org/abs/2409.05933)

**Authors:**
Xin Tan, Meng Zhao

**Abstract:**
Accurate prediction of traffic accidents across different times and regions is vital for public safety. However, existing methods face two key challenges: 1) Generalization: Current models rely heavily on manually constructed multi-view structures, like POI distributions and road network densities, which are labor-intensive and difficult to scale across cities. 2) Real-Time Performance: While some methods improve accuracy with complex architectures, they often incur high computational costs, limiting their real-time applicability. To address these challenges, we propose SSL-eKamba, an efficient self-supervised framework for traffic accident prediction. To enhance generalization, we design two self-supervised auxiliary tasks that adaptively improve traffic pattern representation through spatiotemporal discrepancy awareness. For real-time performance, we introduce eKamba, an efficient model that redesigns the Kolmogorov-Arnold Network (KAN) architecture. This involves using learnable univariate functions for input activation and applying a selective mechanism (Selective SSM) to capture multi-variate correlations, thereby improving computational efficiency. Extensive experiments on two real-world datasets demonstrate that SSL-eKamba consistently outperforms state-of-the-art baselines. This framework may also offer new insights for other spatiotemporal tasks. Our source code is publicly available at http://github.com/KevinT618/SSL-eKamba.
       


### [A Comprehensive Comparison Between ANNs and KANs For Classifying EEG Alzheimer's Data](https://arxiv.org/abs/2409.05989)

**Authors:**
Akshay Sunkara, Sriram Sattiraju, Aakarshan Kumar, Zaryab Kanjiani, Himesh Anumala

**Venue:**
2024 IEEE MIT Undergraduate Research Technology Conference (URTC)

**Citation Count:**
1

**Abstract:**
Alzheimer's Disease is an incurable cognitive condition that affects thousands of people globally. While some diagnostic methods exist for Alzheimer's Disease, many of these methods cannot detect Alzheimer's in its earlier stages. Recently, researchers have explored the use of Electroencephalogram (EEG) technology for diagnosing Alzheimer's. EEG is a noninvasive method of recording the brain's electrical signals, and EEG data has shown distinct differences between patients with and without Alzheimer's. In the past, Artificial Neural Networks (ANNs) have been used to predict Alzheimer's from EEG data, but these models sometimes produce false positive diagnoses. This study aims to compare losses between ANNs and Kolmogorov-Arnold Networks (KANs) across multiple types of epochs, learning rates, and nodes. The results show that across these different parameters, ANNs are more accurate in predicting Alzheimer's Disease from EEG signals.
       


### [KANtrol: A Physics-Informed Kolmogorov-Arnold Network Framework for Solving Multi-Dimensional and Fractional Optimal Control Problems](https://arxiv.org/abs/2409.06649)

**Author:**
Alireza Afzal Aghaei

**Citation Count:**
2

**Abstract:**
In this paper, we introduce the KANtrol framework, which utilizes Kolmogorov-Arnold Networks (KANs) to solve optimal control problems involving continuous time variables. We explain how Gaussian quadrature can be employed to approximate the integral parts within the problem, particularly for integro-differential state equations. We also demonstrate how automatic differentiation is utilized to compute exact derivatives for integer-order dynamics, while for fractional derivatives of non-integer order, we employ matrix-vector product discretization within the KAN framework. We tackle multi-dimensional problems, including the optimal control of a 2D heat partial differential equation. The results of our simulations, which cover both forward and parameter identification problems, show that the KANtrol framework outperforms classical MLPs in terms of accuracy and efficiency.
       


### [HSR-KAN: Efficient Hyperspectral Image Super-Resolution via Kolmogorov-Arnold Networks](https://arxiv.org/abs/2409.06705)

**Authors:**
Baisong Li, Xingwang Wang, Haixiao Xu

**Citation Count:**
1

**Abstract:**
Hyperspectral images (HSIs) have great potential in various visual tasks due to their rich spectral information. However, obtaining high-resolution hyperspectral images remains challenging due to limitations of physical imaging. Inspired by Kolmogorov-Arnold Networks (KANs), we propose an efficient HSI super-resolution (HSI-SR) model to fuse a low-resolution HSI (LR-HSI) and a high-resolution multispectral image (HR-MSI), yielding a high-resolution HSI (HR-HSI). To achieve the effective integration of spatial information from HR-MSI, we design a fusion module based on KANs, called KAN-Fusion. Further inspired by the channel attention mechanism, we design a spectral channel attention module called KAN Channel Attention Block (KAN-CAB) for post-fusion feature extraction. As a channel attention module integrated with KANs, KAN-CAB not only enhances the fine-grained adjustment ability of deep networks, enabling networks to accurately simulate details of spectral sequences and spatial textures, but also effectively avoid Curse of Dimensionality. Extensive experiments show that, compared to current state-of-the-art HSI-SR methods, proposed HSR-KAN achieves the best performance in terms of both qualitative and quantitative assessments. Our code is available at: https://github.com/Baisonm-Li/HSR-KAN.
       


### [MLP, XGBoost, KAN, TDNN, and LSTM-GRU Hybrid RNN with Attention for SPX and NDX European Call Option Pricing](https://arxiv.org/abs/2409.06724)

**Authors:**
Boris Ter-Avanesov, Homayoon Beigi

**Abstract:**
We explore the performance of various artificial neural network architectures, including a multilayer perceptron (MLP), Kolmogorov-Arnold network (KAN), LSTM-GRU hybrid recursive neural network (RNN) models, and a time-delay neural network (TDNN) for pricing European call options. In this study, we attempt to leverage the ability of supervised learning methods, such as ANNs, KANs, and gradient-boosted decision trees, to approximate complex multivariate functions in order to calibrate option prices based on past market data. The motivation for using ANNs and KANs is the Universal Approximation Theorem and Kolmogorov-Arnold Representation Theorem, respectively. Specifically, we use S\&P 500 (SPX) and NASDAQ 100 (NDX) index options traded during 2015-2023 with times to maturity ranging from 15 days to over 4 years (OptionMetrics IvyDB US dataset). Black \& Scholes's (BS) PDE \cite{Black1973} model's performance in pricing the same options compared to real data is used as a benchmark. This model relies on strong assumptions, and it has been observed and discussed in the literature that real data does not match its predictions. Supervised learning methods are widely used as an alternative for calibrating option prices due to some of the limitations of this model. In our experiments, the BS model underperforms compared to all of the others. Also, the best TDNN model outperforms the best MLP model on all error metrics. We implement a simple self-attention mechanism to enhance the RNN models, significantly improving their performance. The best-performing model overall is the LSTM-GRU hybrid RNN model with attention. Also, the KAN model outperforms the TDNN and MLP models. We analyze the performance of all models by ticker, moneyness category, and over/under/correctly-priced percentage.
       


### [Efficient Privacy-Preserving KAN Inference Using Homomorphic Encryption](https://arxiv.org/abs/2409.07751)

**Authors:**
Zhizheng Lai, Yufei Zhou, Peijia Zheng, Lin Chen

**Abstract:**
The recently proposed Kolmogorov-Arnold Networks (KANs) offer enhanced interpretability and greater model expressiveness. However, KANs also present challenges related to privacy leakage during inference. Homomorphic encryption (HE) facilitates privacy-preserving inference for deep learning models, enabling resource-limited users to benefit from deep learning services while ensuring data security. Yet, the complex structure of KANs, incorporating nonlinear elements like the SiLU activation function and B-spline functions, renders existing privacy-preserving inference techniques inadequate. To address this issue, we propose an accurate and efficient privacy-preserving inference scheme tailored for KANs. Our approach introduces a task-specific polynomial approximation for the SiLU activation function, dynamically adjusting the approximation range to ensure high accuracy on real-world datasets. Additionally, we develop an efficient method for computing B-spline functions within the HE domain, leveraging techniques such as repeat packing, lazy combination, and comparison functions. We evaluate the effectiveness of our privacy-preserving KAN inference scheme on both symbolic formula evaluation and image classification. The experimental results show that our model achieves accuracy comparable to plaintext KANs across various datasets and outperforms plaintext MLPs. Additionally, on the CIFAR-10 dataset, our inference latency achieves over 7 times speedup compared to the naive method.
       


### [Exploring Kolmogorov-Arnold networks for realistic image sharpness assessment](https://arxiv.org/abs/2409.07762)

**Authors:**
Shaode Yu, Ze Chen, Zhimu Yang, Jiacheng Gu, Bizu Feng

**Venue:**
IEEE International Conference on Acoustics, Speech, and Signal Processing

**Citation Count:**
5

**Abstract:**
Score prediction is crucial in evaluating realistic image sharpness based on collected informative features. Recently, Kolmogorov-Arnold networks (KANs) have been developed and witnessed remarkable success in data fitting. This study introduces the Taylor series-based KAN (TaylorKAN). Then, different KANs are explored in four realistic image databases (BID2011, CID2013, CLIVE, and KonIQ-10k) to predict the scores by using 15 mid-level features and 2048 high-level features. Compared to support vector regression, results show that KANs are generally competitive or superior, and TaylorKAN is the best one when mid-level features are used. This is the first study to investigate KANs on image quality assessment that sheds some light on how to select and further improve KANs in related tasks.
       


### [Reimagining Linear Probing: Kolmogorov-Arnold Networks in Transfer Learning](https://arxiv.org/abs/2409.07763)

**Authors:**
Sheng Shen, Rabih Younes

**Citation Count:**
1

**Abstract:**
This paper introduces Kolmogorov-Arnold Networks (KAN) as an enhancement to the traditional linear probing method in transfer learning. Linear probing, often applied to the final layer of pre-trained models, is limited by its inability to model complex relationships in data. To address this, we propose substituting the linear probing layer with KAN, which leverages spline-based representations to approximate intricate functions. In this study, we integrate KAN with a ResNet-50 model pre-trained on ImageNet and evaluate its performance on the CIFAR-10 dataset. We perform a systematic hyperparameter search, focusing on grid size and spline degree (k), to optimize KAN's flexibility and accuracy. Our results demonstrate that KAN consistently outperforms traditional linear probing, achieving significant improvements in accuracy and generalization across a range of configurations. These findings indicate that KAN offers a more powerful and adaptable alternative to conventional linear probing techniques in transfer learning.
       


### [A White-Box Deep-Learning Method for Electrical Energy System Modeling Based on Kolmogorov-Arnold Network](https://arxiv.org/abs/2409.08044)

**Authors:**
Zhenghao Zhou, Yiyan Li, Zelin Guo, Zheng Yan, Mo-Yuen Chow

**Abstract:**
Deep learning methods have been widely used as an end-to-end modeling strategy of electrical energy systems because of their conveniency and powerful pattern recognition capability. However, due to the "black-box" nature, deep learning methods have long been blamed for their poor interpretability when modeling a physical system. In this paper, we introduce a novel neural network structure, Kolmogorov-Arnold Network (KAN), to achieve "white-box" modeling for electrical energy systems to enhance the interpretability. The most distinct feature of KAN lies in the learnable activation function together with the sparse training and symbolification process. Consequently, KAN can express the physical process with concise and explicit mathematical formulas while remaining the nonlinear-fitting capability of deep neural networks. Simulation results based on three electrical energy systems demonstrate the effectiveness of KAN in the aspects of interpretability, accuracy, robustness and generalization ability.
       


### [Effective Integration of KAN for Keyword Spotting](https://arxiv.org/abs/2409.08605)

**Authors:**
Anfeng Xu, Biqiao Zhang, Shuyu Kong, Yiteng Huang, Zhaojun Yang, Sangeeta Srivastava, Ming Sun

**Venue:**
IEEE International Conference on Acoustics, Speech, and Signal Processing

**Citation Count:**
5

**Abstract:**
Keyword spotting (KWS) is an important speech processing component for smart devices with voice assistance capability. In this paper, we investigate if Kolmogorov-Arnold Networks (KAN) can be used to enhance the performance of KWS. We explore various approaches to integrate KAN for a model architecture based on 1D Convolutional Neural Networks (CNN). We find that KAN is effective at modeling high-level features in lower-dimensional spaces, resulting in improved KWS performance when integrated appropriately. The findings shed light on understanding KAN for speech processing tasks and on other modalities for future researchers.
       


### [TabKANet: Tabular Data Modeling with Kolmogorov-Arnold Network and Transformer](https://arxiv.org/abs/2409.08806)

**Authors:**
Weihao Gao, Zheng Gong, Zhuo Deng, Fuju Rong, Chucheng Chen, Lan Ma

**Citation Count:**
5

**Abstract:**
Tabular data is the most common type of data in real-life scenarios. In this study, we propose the TabKANet model for tabular data modeling, which targets the bottlenecks in learning from numerical content. We constructed a Kolmogorov-Arnold Network (KAN) based Numerical Embedding Module and unified numerical and categorical features encoding within a Transformer architecture. TabKANet has demonstrated stable and significantly superior performance compared to Neural Networks (NNs) across multiple public datasets in binary classification, multi-class classification, and regression tasks. Its performance is comparable to or surpasses that of Gradient Boosted Decision Tree models (GBDTs). Our code is publicly available on GitHub: https://github.com/AI-thpremed/TabKANet.
       


### [Can Kans (re)discover predictive models for Direct-Drive Laser Fusion?](https://arxiv.org/abs/2409.08832)

**Authors:**
Rahman Ejaz, Varchas Gopalaswamy, Riccardo Betti, Aarne Lees, Christopher Kanan

**Abstract:**
The domain of laser fusion presents a unique and challenging predictive modeling application landscape for machine learning methods due to high problem complexity and limited training data. Data-driven approaches utilizing prescribed functional forms, inductive biases and physics-informed learning (PIL) schemes have been successful in the past for achieving desired generalization ability and model interpretation that aligns with physics expectations. In complex multi-physics application domains, however, it is not always obvious how architectural biases or discriminative penalties can be formulated. In this work, focusing on nuclear fusion energy using high powered lasers, we present the use of Kolmogorov-Arnold Networks (KANs) as an alternative to PIL for developing a new type of data-driven predictive model which is able to achieve high prediction accuracy and physics interpretability. A KAN based model, a MLP with PIL, and a baseline MLP model are compared in generalization ability and interpretation with a domain expert-derived symbolic regression model. Through empirical studies in this high physics complexity domain, we show that KANs can potentially provide benefits when developing predictive models for data-starved physics applications.
       


### [Implicit Neural Representations with Fourier Kolmogorov-Arnold Networks](https://arxiv.org/abs/2409.09323)

**Authors:**
Ali Mehrabian, Parsa Mojarad Adi, Moein Heidari, Ilker Hacihaliloglu

**Venue:**
IEEE International Conference on Acoustics, Speech, and Signal Processing

**Citation Count:**
3

**Abstract:**
Implicit neural representations (INRs) use neural networks to provide continuous and resolution-independent representations of complex signals with a small number of parameters. However, existing INR models often fail to capture important frequency components specific to each task. To address this issue, in this paper, we propose a Fourier Kolmogorov Arnold network (FKAN) for INRs. The proposed FKAN utilizes learnable activation functions modeled as Fourier series in the first layer to effectively control and learn the task-specific frequency components. In addition, the activation functions with learnable Fourier coefficients improve the ability of the network to capture complex patterns and details, which is beneficial for high-resolution and high-dimensional data. Experimental results show that our proposed FKAN model outperforms three state-of-the-art baseline schemes, and improves the peak signal-to-noise ratio (PSNR) and structural similarity index measure (SSIM) for the image representation task and intersection over union (IoU) for the 3D occupancy volume representation task, respectively. The code is available at github.com/Ali-Meh619/FKAN.
       


### [KAN-HyperpointNet for Point Cloud Sequence-Based 3D Human Action Recognition](https://arxiv.org/abs/2409.09444)

**Authors:**
Zhaoyu Chen, Xing Li, Qian Huang, Qiang Geng, Tianjin Yang, Shihao Han

**Venue:**
IEEE International Conference on Acoustics, Speech, and Signal Processing

**Abstract:**
Point cloud sequence-based 3D action recognition has achieved impressive performance and efficiency. However, existing point cloud sequence modeling methods cannot adequately balance the precision of limb micro-movements with the integrity of posture macro-structure, leading to the loss of crucial information cues in action inference. To overcome this limitation, we introduce D-Hyperpoint, a novel data type generated through a D-Hyperpoint Embedding module. D-Hyperpoint encapsulates both regional-momentary motion and global-static posture, effectively summarizing the unit human action at each moment. In addition, we present a D-Hyperpoint KANsMixer module, which is recursively applied to nested groupings of D-Hyperpoints to learn the action discrimination information and creatively integrates Kolmogorov-Arnold Networks (KAN) to enhance spatio-temporal interaction within D-Hyperpoints. Finally, we propose KAN-HyperpointNet, a spatio-temporal decoupled network architecture for 3D action recognition. Extensive experiments on two public datasets: MSR Action3D and NTU-RGB+D 60, demonstrate the state-of-the-art performance of our method.
       


### [KAN v.s. MLP for Offline Reinforcement Learning](https://arxiv.org/abs/2409.09653)

**Authors:**
Haihong Guo, Fengxin Li, Jiao Li, Hongyan Liu

**Venue:**
IEEE International Conference on Acoustics, Speech, and Signal Processing

**Abstract:**
Kolmogorov-Arnold Networks (KAN) is an emerging neural network architecture in machine learning. It has greatly interested the research community about whether KAN can be a promising alternative of the commonly used Multi-Layer Perceptions (MLP). Experiments in various fields demonstrated that KAN-based machine learning can achieve comparable if not better performance than MLP-based methods, but with much smaller parameter scales and are more explainable. In this paper, we explore the incorporation of KAN into the actor and critic networks for offline reinforcement learning (RL). We evaluated the performance, parameter scales, and training efficiency of various KAN and MLP based conservative Q-learning (CQL) on the the classical D4RL benchmark for offline RL. Our study demonstrates that KAN can achieve performance close to the commonly used MLP with significantly fewer parameters. This provides us an option to choose the base networks according to the requirements of the offline RL tasks.
       


### [Kolmogorov-Arnold Networks in Low-Data Regimes: A Comparative Study with Multilayer Perceptrons](https://arxiv.org/abs/2409.10463)

**Author:**
Farhad Pourkamali-Anaraki

**Citation Count:**
5

**Abstract:**
Multilayer Perceptrons (MLPs) have long been a cornerstone in deep learning, known for their capacity to model complex relationships. Recently, Kolmogorov-Arnold Networks (KANs) have emerged as a compelling alternative, utilizing highly flexible learnable activation functions directly on network edges, a departure from the neuron-centric approach of MLPs. However, KANs significantly increase the number of learnable parameters, raising concerns about their effectiveness in data-scarce environments. This paper presents a comprehensive comparative study of MLPs and KANs from both algorithmic and experimental perspectives, with a focus on low-data regimes. We introduce an effective technique for designing MLPs with unique, parameterized activation functions for each neuron, enabling a more balanced comparison with KANs. Using empirical evaluations on simulated data and two real-world data sets from medicine and engineering, we explore the trade-offs between model complexity and accuracy, with particular attention to the role of network depth. Our findings show that MLPs with individualized activation functions achieve significantly higher predictive accuracy with only a modest increase in parameters, especially when the sample size is limited to around one hundred. For example, in a three-class classification problem within additive manufacturing, MLPs achieve a median accuracy of 0.91, significantly outperforming KANs, which only reach a median accuracy of 0.53 with default hyperparameters. These results offer valuable insights into the impact of activation function selection in neural networks.
       


### [Kolmogorov-Arnold Transformer](https://arxiv.org/abs/2409.10594)

**Authors:**
Xingyi Yang, Xinchao Wang

**Citation Count:**
19

**Abstract:**
Transformers stand as the cornerstone of mordern deep learning. Traditionally, these models rely on multi-layer perceptron (MLP) layers to mix the information between channels. In this paper, we introduce the Kolmogorov-Arnold Transformer (KAT), a novel architecture that replaces MLP layers with Kolmogorov-Arnold Network (KAN) layers to enhance the expressiveness and performance of the model. Integrating KANs into transformers, however, is no easy feat, especially when scaled up. Specifically, we identify three key challenges: (C1) Base function. The standard B-spline function used in KANs is not optimized for parallel computing on modern hardware, resulting in slower inference speeds. (C2) Parameter and Computation Inefficiency. KAN requires a unique function for each input-output pair, making the computation extremely large. (C3) Weight initialization. The initialization of weights in KANs is particularly challenging due to their learnable activation functions, which are critical for achieving convergence in deep neural networks. To overcome the aforementioned challenges, we propose three key solutions: (S1) Rational basis. We replace B-spline functions with rational functions to improve compatibility with modern GPUs. By implementing this in CUDA, we achieve faster computations. (S2) Group KAN. We share the activation weights through a group of neurons, to reduce the computational load without sacrificing performance. (S3) Variance-preserving initialization. We carefully initialize the activation weights to make sure that the activation variance is maintained across layers. With these designs, KAT scales effectively and readily outperforms traditional MLP-based transformers.
       


### [MonoKAN: Certified Monotonic Kolmogorov-Arnold Network](https://arxiv.org/abs/2409.11078)

**Authors:**
Alejandro Polo-Molina, David Alfaya, Jose Portela

**Citation Count:**
2

**Abstract:**
Artificial Neural Networks (ANNs) have significantly advanced various fields by effectively recognizing patterns and solving complex problems. Despite these advancements, their interpretability remains a critical challenge, especially in applications where transparency and accountability are essential. To address this, explainable AI (XAI) has made progress in demystifying ANNs, yet interpretability alone is often insufficient. In certain applications, model predictions must align with expert-imposed requirements, sometimes exemplified by partial monotonicity constraints. While monotonic approaches are found in the literature for traditional Multi-layer Perceptrons (MLPs), they still face difficulties in achieving both interpretability and certified partial monotonicity. Recently, the Kolmogorov-Arnold Network (KAN) architecture, based on learnable activation functions parametrized as splines, has been proposed as a more interpretable alternative to MLPs. Building on this, we introduce a novel ANN architecture called MonoKAN, which is based on the KAN architecture and achieves certified partial monotonicity while enhancing interpretability. To achieve this, we employ cubic Hermite splines, which guarantee monotonicity through a set of straightforward conditions. Additionally, by using positive weights in the linear combinations of these splines, we ensure that the network preserves the monotonic relationships between input and output. Our experiments demonstrate that MonoKAN not only enhances interpretability but also improves predictive performance across the majority of benchmarks, outperforming state-of-the-art monotonic MLP approaches.
       


### [Hardware Acceleration of Kolmogorov-Arnold Network (KAN) for Lightweight Edge Inference](https://arxiv.org/abs/2409.11418)

**Authors:**
Wei-Hsing Huang, Jianwei Jia, Yuyao Kong, Faaiq Waqar, Tai-Hao Wen, Meng-Fan Chang, Shimeng Yu

**Venue:**
Asia and South Pacific Design Automation Conference

**Citation Count:**
2

**Abstract:**
Recently, a novel model named Kolmogorov-Arnold Networks (KAN) has been proposed with the potential to achieve the functionality of traditional deep neural networks (DNNs) using orders of magnitude fewer parameters by parameterized B-spline functions with trainable coefficients. However, the B-spline functions in KAN present new challenges for hardware acceleration. Evaluating the B-spline functions can be performed by using look-up tables (LUTs) to directly map the B-spline functions, thereby reducing computational resource requirements. However, this method still requires substantial circuit resources (LUTs, MUXs, decoders, etc.). For the first time, this paper employs an algorithm-hardware co-design methodology to accelerate KAN. The proposed algorithm-level techniques include Alignment-Symmetry and PowerGap KAN hardware aware quantization, KAN sparsity aware mapping strategy, and circuit-level techniques include N:1 Time Modulation Dynamic Voltage input generator with analog-CIM (ACIM) circuits. The impact of non-ideal effects, such as partial sum errors caused by the process variations, has been evaluated with the statistics measured from the TSMC 22nm RRAM-ACIM prototype chips. With the best searched hyperparameters of KAN and the optimized circuits implemented in 22 nm node, we can reduce hardware area by 41.78x, energy by 77.97x with 3.03% accuracy boost compared to the traditional DNN hardware.
       


### [ASPINN: An asymptotic strategy for solving singularly perturbed differential equations](https://arxiv.org/abs/2409.13185)

**Authors:**
Sen Wang, Peizhi Zhao, Tao Song

**Abstract:**
Solving Singularly Perturbed Differential Equations (SPDEs) presents challenges due to the rapid change of their solutions at the boundary layer. In this manuscript, We propose Asymptotic Physics-Informed Neural Networks (ASPINN), a generalization of Physics-Informed Neural Networks (PINN) and General-Kindred Physics-Informed Neural Networks (GKPINN) approaches. This is a decomposition method based on the idea of asymptotic analysis. Compared to PINN, the ASPINN method has a strong fitting ability for solving SPDEs due to the placement of exponential layers at the boundary layer. Unlike GKPINN, ASPINN lessens the number of fully connected layers, thereby reducing the training cost more effectively. Moreover, ASPINN theoretically approximates the solution at the boundary layer more accurately, which accuracy is also improved compared to GKPINN. We demonstrate the effect of ASPINN by solving diverse classes of SPDEs, which clearly shows that the ASPINN method is promising in boundary layer problems. Furthermore, we introduce Chebyshev Kolmogorov-Arnold Networks (Chebyshev-KAN) instead of MLP, achieving better performance in various experiments.
       


### [A preliminary study on continual learning in computer vision using Kolmogorov-Arnold Networks](https://arxiv.org/abs/2409.13550)

**Authors:**
Alessandro Cacciatore, Valerio Morelli, Federica Paganica, Emanuele Frontoni, Lucia Migliorelli, Daniele Berardini

**Citation Count:**
4

**Abstract:**
Deep learning has long been dominated by multi-layer perceptrons (MLPs), which have demonstrated superiority over other optimizable models in various domains. Recently, a new alternative to MLPs has emerged - Kolmogorov-Arnold Networks (KAN)- which are based on a fundamentally different mathematical framework. According to their authors, KANs address several major issues in MLPs, such as catastrophic forgetting in continual learning scenarios. However, this claim has only been supported by results from a regression task on a toy 1D dataset. In this paper, we extend the investigation by evaluating the performance of KANs in continual learning tasks within computer vision, specifically using the MNIST datasets. To this end, we conduct a structured analysis of the behavior of MLPs and two KAN-based models in a class-incremental learning scenario, ensuring that the architectures involved have the same number of trainable parameters. Our results demonstrate that an efficient version of KAN outperforms both traditional MLPs and the original KAN implementation. We further analyze the influence of hyperparameters in MLPs and KANs, as well as the impact of certain trainable parameters in KANs, such as bias and scale weights. Additionally, we provide a preliminary investigation of recent KAN-based convolutional networks and compare their performance with that of traditional convolutional neural networks. Our codes can be found at https://github.com/MrPio/KAN-Continual_Learning_tests.
       


### [Higher-order-ReLU-KANs (HRKANs) for solving physics-informed neural networks (PINNs) more accurately, robustly and faster](https://arxiv.org/abs/2409.14248)

**Authors:**
Chi Chiu So, Siu Pang Yung

**Citation Count:**
7

**Abstract:**
Finding solutions to partial differential equations (PDEs) is an important and essential component in many scientific and engineering discoveries. One of the common approaches empowered by deep learning is Physics-informed Neural Networks (PINNs). Recently, a new type of fundamental neural network model, Kolmogorov-Arnold Networks (KANs), has been proposed as a substitute of Multilayer Perceptions (MLPs), and possesses trainable activation functions. To enhance KANs in fitting accuracy, a modification of KANs, so called ReLU-KANs, using "square of ReLU" as the basis of its activation functions, has been suggested. In this work, we propose another basis of activation functions, namely, Higherorder-ReLU (HR), which is simpler than the basis of activation functions used in KANs, namely, Bsplines; allows efficient KAN matrix operations; and possesses smooth and non-zero higher-order derivatives, essential to physicsinformed neural networks. We name such KANs with Higher-order-ReLU (HR) as their activations, HRKANs. Our detailed experiments on two famous and representative PDEs, namely, the linear Poisson equation and nonlinear Burgers' equation with viscosity, reveal that our proposed Higher-order-ReLU-KANs (HRKANs) achieve the highest fitting accuracy and training robustness and lowest training time significantly among KANs, ReLU-KANs and HRKANs. The codes to replicate our experiments are available at https://github.com/kelvinhkcs/HRKAN.
       


### [A Gated Residual Kolmogorov-Arnold Networks for Mixtures of Experts](https://arxiv.org/abs/2409.15161)

**Authors:**
Hugo Inzirillo, Remi Genet

**Citation Count:**
4

**Abstract:**
This paper introduces KAMoE, a novel Mixture of Experts (MoE) framework based on Gated Residual Kolmogorov-Arnold Networks (GRKAN). We propose GRKAN as an alternative to the traditional gating function, aiming to enhance efficiency and interpretability in MoE modeling. Through extensive experiments on digital asset markets and real estate valuation, we demonstrate that KAMoE consistently outperforms traditional MoE architectures across various tasks and model types. Our results show that GRKAN exhibits superior performance compared to standard Gating Residual Networks, particularly in LSTM-based models for sequential tasks. We also provide insights into the trade-offs between model complexity and performance gains in MoE and KAMoE architectures.
       


### [Data-driven model discovery with Kolmogorov-Arnold networks](https://arxiv.org/abs/2409.15167)

**Authors:**
Mohammadamin Moradi, Shirin Panahi, Erik M. Bollt, Ying-Cheng Lai

**Venue:**
Physical Review Research

**Citation Count:**
3

**Abstract:**
Data-driven model discovery of complex dynamical systems is typically done using sparse optimization, but it has a fundamental limitation: sparsity in that the underlying governing equations of the system contain only a small number of elementary mathematical terms. Examples where sparse optimization fails abound, such as the classic Ikeda or optical-cavity map in nonlinear dynamics and a large variety of ecosystems. Exploiting the recently articulated Kolmogorov-Arnold networks, we develop a general model-discovery framework for any dynamical systems including those that do not satisfy the sparsity condition. In particular, we demonstrate non-uniqueness in that a large number of approximate models of the system can be found which generate the same invariant set with the correct statistics such as the Lyapunov exponents and Kullback-Leibler divergence. An analogy to shadowing of numerical trajectories in chaotic systems is pointed out.
       


### [PPLNs: Parametric Piecewise Linear Networks for Event-Based Temporal Modeling and Beyond](https://arxiv.org/abs/2409.19772)

**Authors:**
Chen Song, Zhenxiao Liang, Bo Sun, Qixing Huang

**Venue:**
Neural Information Processing Systems

**Abstract:**
We present Parametric Piecewise Linear Networks (PPLNs) for temporal vision inference. Motivated by the neuromorphic principles that regulate biological neural behaviors, PPLNs are ideal for processing data captured by event cameras, which are built to simulate neural activities in the human retina. We discuss how to represent the membrane potential of an artificial neuron by a parametric piecewise linear function with learnable coefficients. This design echoes the idea of building deep models from learnable parametric functions recently popularized by Kolmogorov-Arnold Networks (KANs). Experiments demonstrate the state-of-the-art performance of PPLNs in event-based and image-based vision applications, including steering prediction, human pose estimation, and motion deblurring. The source code of our implementation is available at https://github.com/chensong1995/PPLN.
       


## October
### [KANOP: A Data-Efficient Option Pricing Model using Kolmogorov-Arnold Networks](https://arxiv.org/abs/2410.00419)

**Authors:**
Rushikesh Handal, Kazuki Matoya, Yunzhuo Wang, Masanori Hirano

**Venue:**
IEEE Conference on Computational Intelligence for Financial Engineering & Economics

**Abstract:**
Inspired by the recently proposed Kolmogorov-Arnold Networks (KANs), we introduce the KAN-based Option Pricing (KANOP) model to value American-style options, building on the conventional Least Square Monte Carlo (LSMC) algorithm. KANs, which are based on Kolmogorov-Arnold representation theorem, offer a data-efficient alternative to traditional Multi-Layer Perceptrons, requiring fewer hidden layers to achieve a higher level of performance. By leveraging the flexibility of KANs, KANOP provides a learnable alternative to the conventional set of basis functions used in the LSMC model, allowing the model to adapt to the pricing task and effectively estimate the expected continuation value. Using examples of standard American and Asian-American options, we demonstrate that KANOP produces more reliable option value estimates, both for single-dimensional cases and in more complex scenarios involving multiple input variables. The delta estimated by the KANOP model is also more accurate than that obtained using conventional basis functions, which is crucial for effective option hedging. Graphical illustrations further validate KANOP's ability to accurately model the expected continuation value for American-style options.
       


### [Incorporating Arbitrary Matrix Group Equivariance into KANs](https://arxiv.org/abs/2410.00435)

**Authors:**
Lexiang Hu, Yisen Wang, Zhouchen Lin

**Citation Count:**
1

**Abstract:**
Kolmogorov-Arnold Networks (KANs) have seen great success in scientific domains thanks to spline activation functions, becoming an alternative to Multi-Layer Perceptrons (MLPs). However, spline functions may not respect symmetry in tasks, which is crucial prior knowledge in machine learning. In this paper, we propose Equivariant Kolmogorov-Arnold Networks (EKAN), a method for incorporating arbitrary matrix group equivariance into KANs, aiming to broaden their applicability to more fields. We first construct gated spline basis functions, which form the EKAN layer together with equivariant linear weights, and then define a lift layer to align the input space of EKAN with the feature space of the dataset, thereby building the entire EKAN architecture. Compared with baseline models, EKAN achieves higher accuracy with smaller datasets or fewer parameters on symmetry-related tasks, such as particle scattering and the three-body problem, often reducing test MSE by several orders of magnitude. Even in non-symbolic formula scenarios, such as top quark tagging with three jet constituents, EKAN achieves comparable results with state-of-the-art equivariant architectures using fewer than 40% of the parameters, while KANs do not outperform MLPs as expected.
       


### [Uncertainty Quantification with Bayesian Higher Order ReLU KANs](https://arxiv.org/abs/2410.01687)

**Authors:**
James Giroux, Cristiano Fanelli

**Venue:**
Machine Learning: Science and Technology

**Citation Count:**
1

**Abstract:**
We introduce the first method of uncertainty quantification in the domain of Kolmogorov-Arnold Networks, specifically focusing on (Higher Order) ReLUKANs to enhance computational efficiency given the computational demands of Bayesian methods. The method we propose is general in nature, providing access to both epistemic and aleatoric uncertainties. It is also capable of generalization to other various basis functions. We validate our method through a series of closure tests, including simple one-dimensional functions and application to the domain of (Stochastic) Partial Differential Equations. Referring to the latter, we demonstrate the method's ability to correctly identify functional dependencies introduced through the inclusion of a stochastic term. The code supporting this work can be found at https://github.com/wmdataphys/Bayesian-HR-KAN
       


### [On the expressiveness and spectral bias of KANs](https://arxiv.org/abs/2410.01803)

**Authors:**
Yixuan Wang, Jonathan W. Siegel, Ziming Liu, Thomas Y. Hou

**Venue:**
International Conference on Learning Representations

**Citation Count:**
12

**Abstract:**
Kolmogorov-Arnold Networks (KAN) \cite{liu2024kan} were very recently proposed as a potential alternative to the prevalent architectural backbone of many deep learning models, the multi-layer perceptron (MLP). KANs have seen success in various tasks of AI for science, with their empirical efficiency and accuracy demostrated in function regression, PDE solving, and many more scientific problems.
  In this article, we revisit the comparison of KANs and MLPs, with emphasis on a theoretical perspective. On the one hand, we compare the representation and approximation capabilities of KANs and MLPs. We establish that MLPs can be represented using KANs of a comparable size. This shows that the approximation and representation capabilities of KANs are at least as good as MLPs. Conversely, we show that KANs can be represented using MLPs, but that in this representation the number of parameters increases by a factor of the KAN grid size. This suggests that KANs with a large grid size may be more efficient than MLPs at approximating certain functions. On the other hand, from the perspective of learning and optimization, we study the spectral bias of KANs compared with MLPs. We demonstrate that KANs are less biased toward low frequencies than MLPs. We highlight that the multi-level learning feature specific to KANs, i.e. grid extension of splines, improves the learning process for high-frequency components. Detailed comparisons with different choices of depth, width, and grid sizes of KANs are made, shedding some light on how to choose the hyperparameters in practice.
       


### [Deep Learning Alternatives of the Kolmogorov Superposition Theorem](https://arxiv.org/abs/2410.01990)

**Authors:**
Leonardo Ferreira Guilhoto, Paris Perdikaris

**Venue:**
International Conference on Learning Representations

**Citation Count:**
7

**Abstract:**
This paper explores alternative formulations of the Kolmogorov Superposition Theorem (KST) as a foundation for neural network design. The original KST formulation, while mathematically elegant, presents practical challenges due to its limited insight into the structure of inner and outer functions and the large number of unknown variables it introduces. Kolmogorov-Arnold Networks (KANs) leverage KST for function approximation, but they have faced scrutiny due to mixed results compared to traditional multilayer perceptrons (MLPs) and practical limitations imposed by the original KST formulation. To address these issues, we introduce ActNet, a scalable deep learning model that builds on the KST and overcomes many of the drawbacks of Kolmogorov's original formulation. We evaluate ActNet in the context of Physics-Informed Neural Networks (PINNs), a framework well-suited for leveraging KST's strengths in low-dimensional function approximation, particularly for simulating partial differential equations (PDEs). In this challenging setting, where models must learn latent functions without direct measurements, ActNet consistently outperforms KANs across multiple benchmarks and is competitive against the current best MLP-based approaches. These results present ActNet as a promising new direction for KST-based deep learning applications, particularly in scientific computing and PDE simulation tasks.
       


### [Model Comparisons: XNet Outperforms KAN](https://arxiv.org/abs/2410.02033)

**Authors:**
Xin Li, Zhihong Jeff Xia, Xiaotao Zheng

**Abstract:**
In the fields of computational mathematics and artificial intelligence, the need for precise data modeling is crucial, especially for predictive machine learning tasks. This paper explores further XNet, a novel algorithm that employs the complex-valued Cauchy integral formula, offering a superior network architecture that surpasses traditional Multi-Layer Perceptrons (MLPs) and Kolmogorov-Arnold Networks (KANs). XNet significant improves speed and accuracy across various tasks in both low and high-dimensional spaces, redefining the scope of data-driven model development and providing substantial improvements over established time series models like LSTMs.
       


### [Kolmogorov-Arnold Network Autoencoders](https://arxiv.org/abs/2410.02077)

**Authors:**
Mohammadamin Moradi, Shirin Panahi, Erik Bollt, Ying-Cheng Lai

**Citation Count:**
5

**Abstract:**
Deep learning models have revolutionized various domains, with Multi-Layer Perceptrons (MLPs) being a cornerstone for tasks like data regression and image classification. However, a recent study has introduced Kolmogorov-Arnold Networks (KANs) as promising alternatives to MLPs, leveraging activation functions placed on edges rather than nodes. This structural shift aligns KANs closely with the Kolmogorov-Arnold representation theorem, potentially enhancing both model accuracy and interpretability. In this study, we explore the efficacy of KANs in the context of data representation via autoencoders, comparing their performance with traditional Convolutional Neural Networks (CNNs) on the MNIST, SVHN, and CIFAR-10 datasets. Our results demonstrate that KAN-based autoencoders achieve competitive performance in terms of reconstruction accuracy, thereby suggesting their viability as effective tools in data analysis tasks.
       


### [MLP-KAN: Unifying Deep Representation and Function Learning](https://arxiv.org/abs/2410.03027)

**Authors:**
Yunhong He, Yifeng Xie, Zhengqing Yuan, Lichao Sun

**Citation Count:**
2

**Abstract:**
Recent advancements in both representation learning and function learning have demonstrated substantial promise across diverse domains of artificial intelligence. However, the effective integration of these paradigms poses a significant challenge, particularly in cases where users must manually decide whether to apply a representation learning or function learning model based on dataset characteristics. To address this issue, we introduce MLP-KAN, a unified method designed to eliminate the need for manual model selection. By integrating Multi-Layer Perceptrons (MLPs) for representation learning and Kolmogorov-Arnold Networks (KANs) for function learning within a Mixture-of-Experts (MoE) architecture, MLP-KAN dynamically adapts to the specific characteristics of the task at hand, ensuring optimal performance. Embedded within a transformer-based framework, our work achieves remarkable results on four widely-used datasets across diverse domains. Extensive experimental evaluation demonstrates its superior versatility, delivering competitive performance across both deep representation and function learning tasks. These findings highlight the potential of MLP-KAN to simplify the model selection process, offering a comprehensive, adaptable solution across various domains. Our code and weights are available at \url{https://github.com/DLYuanGod/MLP-KAN}.
       


### [P1-KAN: an effective Kolmogorov-Arnold network with application to hydraulic valley optimization](https://arxiv.org/abs/2410.03801)

**Author:**
Xavier Warin

**Abstract:**
A new Kolmogorov-Arnold network (KAN) is proposed to approximate potentially irregular functions in high dimensions. We provide error bounds for this approximation, assuming that the Kolmogorov-Arnold expansion functions are sufficiently smooth. When the function is only continuous, we also provide universal approximation theorems. We show that it outperforms multilayer perceptrons in terms of accuracy and convergence speed. We also compare it with several proposed KAN networks: it outperforms all networks for irregular functions and achieves similar accuracy to the original spline-based KAN network for smooth functions. Finally, we compare some of the KAN networks in optimizing a French hydraulic valley.
       


### [Sinc Kolmogorov-Arnold Network and Its Applications on Physics-informed Neural Networks](https://arxiv.org/abs/2410.04096)

**Authors:**
Tianchi Yu, Jingwei Qiu, Jiang Yang, Ivan Oseledets

**Citation Count:**
2

**Abstract:**
In this paper, we propose to use Sinc interpolation in the context of Kolmogorov-Arnold Networks, neural networks with learnable activation functions, which recently gained attention as alternatives to multilayer perceptron. Many different function representations have already been tried, but we show that Sinc interpolation proposes a viable alternative, since it is known in numerical analysis to represent well both smooth functions and functions with singularities. This is important not only for function approximation but also for the solutions of partial differential equations with physics-informed neural networks. Through a series of experiments, we show that SincKANs provide better results in almost all of the examples we have considered.
       


### [Quantum Kolmogorov-Arnold networks by combining quantum signal processing circuits](https://arxiv.org/abs/2410.04218)

**Author:**
Ammar Daskin

**Abstract:**
In this paper, we show that an equivalent implementation of KAN can be done on quantum computers by simply combining quantum signal processing circuits in layers. This provides a powerful and robust path for the applications of KAN on quantum computers.
       


### [QKAN: Quantum Kolmogorov-Arnold Networks](https://arxiv.org/abs/2410.04435)

**Authors:**
Petr Ivashkov, Po-Wei Huang, Kelvin Koor, Lirandë Pira, Patrick Rebentrost

**Citation Count:**
1

**Abstract:**
The potential of learning models in quantum hardware remains an open question. Yet, the field of quantum machine learning persistently explores how these models can take advantage of quantum implementations. Recently, a new neural network architecture, called Kolmogorov-Arnold Networks (KAN), has emerged, inspired by the compositional structure of the Kolmogorov-Arnold representation theorem. In this work, we design a quantum version of KAN called QKAN. Our QKAN exploits powerful quantum linear algebra tools, including quantum singular value transformation, to apply parameterized activation functions on the edges of the network. QKAN is based on block-encodings, making it inherently suitable for direct quantum input. Furthermore, we analyze its asymptotic complexity, building recursively from a single layer to an end-to-end neural architecture. The gate complexity of QKAN scales linearly with the cost of constructing block-encodings for input and weights, suggesting broad applicability in tasks with high-dimensional input. QKAN serves as a trainable quantum machine learning model by combining parameterized quantum circuits with established quantum subroutines. Lastly, we propose a multivariate state preparation strategy based on the construction of the QKAN architecture.
       


### [Art Forgery Detection using Kolmogorov Arnold and Convolutional Neural Networks](https://arxiv.org/abs/2410.04866)

**Authors:**
Sandro Boccuzzo, Deborah Desirée Meyer, Ludovica Schaerf

**Venue:**
ECCV Workshops

**Citation Count:**
1

**Abstract:**
Art authentication has historically established itself as a task requiring profound connoisseurship of one particular artist. Nevertheless, famous art forgers such as Wolfgang Beltracchi were able to deceive dozens of art experts. In recent years Artificial Intelligence algorithms have been successfully applied to various image processing tasks. In this work, we leverage the growing improvements in AI to present an art authentication framework for the identification of the forger Wolfgang Beltracchi. Differently from existing literature on AI-aided art authentication, we focus on a specialized model of a forger, rather than an artist, flipping the approach of traditional AI methods. We use a carefully compiled dataset of known artists forged by Beltracchi and a set of known works by the forger to train a multiclass image classification model based on EfficientNet. We compare the results with Kolmogorov Arnold Networks (KAN) which, to the best of our knowledge, have never been tested in the art domain. The results show a general agreement between the different models' predictions on artworks flagged as forgeries, which are then closely studied using visual analysis.
       


### [Residual Kolmogorov-Arnold Network for Enhanced Deep Learning](https://arxiv.org/abs/2410.05500)

**Authors:**
Ray Congrui Yu, Sherry Wu, Jiang Gui

**Citation Count:**
1

**Abstract:**
Despite their immense success, deep neural networks (CNNs) are costly to train, while modern architectures can retain hundreds of convolutional layers in network depth. Standard convolutional operations are fundamentally limited by their linear nature along with fixed activations, where multiple layers are needed to learn complex patterns, making this approach computationally inefficient and prone to optimization difficulties. As a result, we introduce RKAN (Residual Kolmogorov-Arnold Network), which could be easily implemented into stages of traditional networks, such as ResNet. The module also integrates polynomial feature transformation that provides the expressive power of many convolutional layers through learnable, non-linear feature refinement. Our proposed RKAN module offers consistent improvements over the base models on various well-known benchmark datasets, such as CIFAR-100, Food-101, and ImageNet.
       


### [KACQ-DCNN: Uncertainty-Aware Interpretable Kolmogorov-Arnold Classical-Quantum Dual-Channel Neural Network for Heart Disease Detection](https://arxiv.org/abs/2410.07446)

**Authors:**
Md Abrar Jahin, Md. Akmol Masud, M. F. Mridha, Zeyar Aung, Nilanjan Dey

**Citation Count:**
6

**Abstract:**
Heart failure is a leading cause of global mortality, necessitating improved diagnostic strategies. Classical machine learning models struggle with challenges such as high-dimensional data, class imbalances, poor feature representations, and lack of interpretability. While quantum machine learning holds promise, current hybrid models have not fully exploited quantum advantages. In this paper, we propose the Kolmogorov-Arnold Classical-Quantum Dual-Channel Neural Network (KACQ-DCNN), a novel hybrid architecture that replaces traditional multilayer perceptrons with Kolmogorov-Arnold Networks (KANs), enabling learnable univariate activation functions. Our KACQ-DCNN 4-qubit, 1-layer model outperforms 37 benchmark models, including 16 classical and 12 quantum neural networks, achieving an accuracy of 92.03%, with macro-average precision, recall, and F1 scores of 92.00%. It also achieved a ROC-AUC of 94.77%, surpassing other models by significant margins, as validated by paired t-tests with a significance threshold of 0.0056 (after Bonferroni correction). Ablation studies highlight the synergistic effect of classical-quantum integration, improving performance by about 2% over MLP variants. Additionally, LIME and SHAP explainability techniques enhance feature interpretability, while conformal prediction provides robust uncertainty quantification. Our results demonstrate that KACQ-DCNN improves cardiovascular diagnostics by combining high accuracy with interpretability and uncertainty quantification.
       


### [Generalization Bounds and Model Complexity for Kolmogorov-Arnold Networks](https://arxiv.org/abs/2410.08026)

**Authors:**
Xianyang Zhang, Huijuan Zhou

**Venue:**
International Conference on Learning Representations

**Citation Count:**
2

**Abstract:**
Kolmogorov-Arnold Network (KAN) is a network structure recently proposed by Liu et al. (2024) that offers improved interpretability and a more parsimonious design in many science-oriented tasks compared to multi-layer perceptrons. This work provides a rigorous theoretical analysis of KAN by establishing generalization bounds for KAN equipped with activation functions that are either represented by linear combinations of basis functions or lying in a low-rank Reproducing Kernel Hilbert Space (RKHS). In the first case, the generalization bound accommodates various choices of basis functions in forming the activation functions in each layer of KAN and is adapted to different operator norms at each layer. For a particular choice of operator norms, the bound scales with the $l_1$ norm of the coefficient matrices and the Lipschitz constants for the activation functions, and it has no dependence on combinatorial parameters (e.g., number of nodes) outside of logarithmic factors. Moreover, our result does not require the boundedness assumption on the loss function and, hence, is applicable to a general class of regression-type loss functions. In the low-rank case, the generalization bound scales polynomially with the underlying ranks as well as the Lipschitz constants of the activation functions in each layer. These bounds are empirically investigated for KANs trained with stochastic gradient descent on simulated and real data sets. The numerical results demonstrate the practical relevance of these bounds.
       


### [On the Convergence of (Stochastic) Gradient Descent for Kolmogorov--Arnold Networks](https://arxiv.org/abs/2410.08041)

**Authors:**
Yihang Gao, Vincent Y. F. Tan

**Citation Count:**
2

**Abstract:**
Kolmogorov--Arnold Networks (KANs), a recently proposed neural network architecture, have gained significant attention in the deep learning community, due to their potential as a viable alternative to multi-layer perceptrons (MLPs) and their broad applicability to various scientific tasks. Empirical investigations demonstrate that KANs optimized via stochastic gradient descent (SGD) are capable of achieving near-zero training loss in various machine learning (e.g., regression, classification, and time series forecasting, etc.) and scientific tasks (e.g., solving partial differential equations). In this paper, we provide a theoretical explanation for the empirical success by conducting a rigorous convergence analysis of gradient descent (GD) and SGD for two-layer KANs in solving both regression and physics-informed tasks. For regression problems, we establish using the neural tangent kernel perspective that GD achieves global linear convergence of the objective function when the hidden dimension of KANs is sufficiently large. We further extend these results to SGD, demonstrating a similar global convergence in expectation. Additionally, we analyze the global convergence of GD and SGD for physics-informed KANs, which unveils additional challenges due to the more complex loss structure. This is the first work establishing the global convergence guarantees for GD and SGD applied to optimize KANs and physics-informed KANs.
       


### [The Proof of Kolmogorov-Arnold May Illuminate Neural Network Learning](https://arxiv.org/abs/2410.08451)

**Author:**
Michael H. Freedman

**Citation Count:**
1

**Abstract:**
Kolmogorov and Arnold, in answering Hilbert's 13th problem (in the context of continuous functions), laid the foundations for the modern theory of Neural Networks (NNs). Their proof divides the representation of a multivariate function into two steps: The first (non-linear) inter-layer map gives a universal embedding of the data manifold into a single hidden layer whose image is patterned in such a way that a subsequent dynamic can then be defined to solve for the second inter-layer map. I interpret this pattern as "minor concentration" of the almost everywhere defined Jacobians of the interlayer map. Minor concentration amounts to sparsity for higher exterior powers of the Jacobians. We present a conceptual argument for how such sparsity may set the stage for the emergence of successively higher order concepts in today's deep NNs and suggest two classes of experiments to test this hypothesis.
       


### [Kolmogorov-Arnold Neural Networks for High-Entropy Alloys Design](https://arxiv.org/abs/2410.08452)

**Authors:**
Yagnik Bandyopadhyay, Harshil Avlani, Houlong L. Zhuang

**Venue:**
Modelling and Simulation in Materials Science and Engineering

**Citation Count:**
2

**Abstract:**
A wide range of deep learning-based machine learning techniques are extensively applied to the design of high-entropy alloys (HEAs), yielding numerous valuable insights. Kolmogorov-Arnold Networks (KAN) is a recently developed architecture that aims to improve both the accuracy and interpretability of input features. In this work, we explore three different datasets for HEA design and demonstrate the application of KAN for both classification and regression models. In the first example, we use a KAN classification model to predict the probability of single-phase formation in high-entropy carbide ceramics based on various properties such as mixing enthalpy and valence electron concentration. In the second example, we employ a KAN regression model to predict the yield strength and ultimate tensile strength of HEAs based on their chemical composition and process conditions including annealing time, cold rolling percentage, and homogenization temperature. The third example involves a KAN classification model to determine whether a certain composition is an HEA or non-HEA, followed by a KAN regressor model to predict the bulk modulus of the identified HEA, aiming to identify HEAs with high bulk modulus. In all three examples, KAN either outperform or match the performance in terms of accuracy such as F1 score for classification and Mean Square Error (MSE), and coefficient of determination (R2) for regression of the multilayer perceptron (MLP) by demonstrating the efficacy of KAN in handling both classification and regression tasks. We provide a promising direction for future research to explore advanced machine learning techniques, which lead to more accurate predictions and better interpretability of complex materials, ultimately accelerating the discovery and optimization of HEAs with desirable properties.
       


### [Evaluating Federated Kolmogorov-Arnold Networks on Non-IID Data](https://arxiv.org/abs/2410.08961)

**Authors:**
Arthur Mendonça Sasse, Claudio Miceli de Farias

**Citation Count:**
4

**Abstract:**
Federated Kolmogorov-Arnold Networks (F-KANs) have already been proposed, but their assessment is at an initial stage. We present a comparison between KANs (using B-splines and Radial Basis Functions as activation functions) and Multi- Layer Perceptrons (MLPs) with a similar number of parameters for 100 rounds of federated learning in the MNIST classification task using non-IID partitions with 100 clients. After 15 trials for each model, we show that the best accuracies achieved by MLPs can be achieved by Spline-KANs in half of the time (in rounds), with just a moderate increase in computing time.
       


### [WormKAN: Are KAN Effective for Identifying and Tracking Concept Drift in Time Series?](https://arxiv.org/abs/2410.10041)

**Authors:**
Kunpeng Xu, Lifei Chen, Shengrui Wang

**Abstract:**
Dynamic concepts in time series are crucial for understanding complex systems such as financial markets, healthcare, and online activity logs. These concepts help reveal structures and behaviors in sequential data for better decision-making and forecasting. However, existing models often struggle to detect and track concept drift due to limitations in interpretability and adaptability. To address this challenge, inspired by the flexibility of the recent Kolmogorov-Arnold Network (KAN), we propose WormKAN, a concept-aware KAN-based model to address concept drift in co-evolving time series. WormKAN consists of three key components: Patch Normalization, Temporal Representation Module, and Concept Dynamics. Patch normalization processes co-evolving time series into patches, treating them as fundamental modeling units to capture local dependencies while ensuring consistent scaling. The temporal representation module learns robust latent representations by leveraging a KAN-based autoencoder, complemented by a smoothness constraint, to uncover inter-patch correlations. Concept dynamics identifies and tracks dynamic transitions, revealing structural shifts in the time series through concept identification and drift detection. These transitions, akin to passing through a \textit{wormhole}, are identified by abrupt changes in the latent space. Experiments show that KAN and KAN-based models (WormKAN) effectively segment time series into meaningful concepts, enhancing the identification and tracking of concept drift.
       


### [PointNet with KAN versus PointNet with MLP for 3D Classification and Segmentation of Point Sets](https://arxiv.org/abs/2410.10084)

**Author:**
Ali Kashefi

**Citation Count:**
7

**Abstract:**
Kolmogorov-Arnold Networks (KANs) have recently gained attention as an alternative to traditional Multilayer Perceptrons (MLPs) in deep learning frameworks. KANs have been integrated into various deep learning architectures such as convolutional neural networks, graph neural networks, and transformers, with their performance evaluated. However, their effectiveness within point-cloud-based neural networks remains unexplored. To address this gap, we incorporate KANs into PointNet for the first time to evaluate their performance on 3D point cloud classification and segmentation tasks. Specifically, we introduce PointNet-KAN, built upon two key components. First, it employs KANs instead of traditional MLPs. Second, it retains the core principle of PointNet by using shared KAN layers and applying symmetric functions for global feature extraction, ensuring permutation invariance with respect to the input features. In traditional MLPs, the goal is to train the weights and biases with fixed activation functions; however, in KANs, the goal is to train the activation functions themselves. We use Jacobi polynomials to construct the KAN layers. We extensively and systematically evaluate PointNet-KAN across various polynomial degrees and special types such as the Lagrange, Chebyshev, and Gegenbauer polynomials. Our results show that PointNet-KAN achieves competitive performance compared to PointNet with MLPs on benchmark datasets for 3D object classification and segmentation, despite employing a shallower and simpler network architecture. We hope this work serves as a foundation and provides guidance for integrating KANs, as an alternative to MLPs, into more advanced point cloud processing architectures.
       


### [EPi-cKANs: Elasto-Plasticity Informed Kolmogorov-Arnold Networks Using Chebyshev Polynomials](https://arxiv.org/abs/2410.10897)

**Authors:**
Farinaz Mostajeran, Salah A Faroughi

**Citation Count:**
6

**Abstract:**
Multilayer perceptron (MLP) networks are predominantly used to develop data-driven constitutive models for granular materials. They offer a compelling alternative to traditional physics-based constitutive models in predicting nonlinear responses of these materials, e.g., elasto-plasticity, under various loading conditions. To attain the necessary accuracy, MLPs often need to be sufficiently deep or wide, owing to the curse of dimensionality inherent in these problems. To overcome this limitation, we present an elasto-plasticity informed Chebyshev-based Kolmogorov-Arnold network (EPi-cKAN) in this study. This architecture leverages the benefits of KANs and augmented Chebyshev polynomials, as well as integrates physical principles within both the network structure and the loss function. The primary objective of EPi-cKAN is to provide an accurate and generalizable function approximation for non-linear stress-strain relationships, using fewer parameters compared to standard MLPs. To evaluate the efficiency, accuracy, and generalization capabilities of EPi-cKAN in modeling complex elasto-plastic behavior, we initially compare its performance with other cKAN-based models, which include purely data-driven parallel and serial architectures. Furthermore, to differentiate EPi-cKAN's distinct performance, we also compare it against purely data-driven and physics-informed MLP-based methods. Lastly, we test EPi-cKAN's ability to predict blind strain-controlled paths that extend beyond the training data distribution to gauge its generalization and predictive capabilities. Our findings indicate that, even with limited data and fewer parameters compared to other approaches, EPi-cKAN provides superior accuracy in predicting stress components and demonstrates better generalization when used to predict sand elasto-plastic behavior under blind triaxial axisymmetric strain-controlled loading paths.
       


### [KA-GNN: Kolmogorov-Arnold Graph Neural Networks for Molecular Property Prediction](https://arxiv.org/abs/2410.11323)

**Authors:**
Longlong Li, Yipeng Zhang, Guanghui Wang, Kelin Xia

**Citation Count:**
3

**Abstract:**
As key models in geometric deep learning, graph neural networks have demonstrated enormous power in molecular data analysis. Recently, a specially-designed learning scheme, known as Kolmogorov-Arnold Network (KAN), shows unique potential for the improvement of model accuracy, efficiency, and explainability. Here we propose the first non-trivial Kolmogorov-Arnold Network-based Graph Neural Networks (KA-GNNs), including KAN-based graph convolutional networks(KA-GCN) and KAN-based graph attention network (KA-GAT). The essential idea is to utilizes KAN's unique power to optimize GNN architectures at three major levels, including node embedding, message passing, and readout. Further, with the strong approximation capability of Fourier series, we develop Fourier series-based KAN model and provide a rigorous mathematical prove of the robust approximation capability of this Fourier KAN architecture. To validate our KA-GNNs, we consider seven most-widely-used benchmark datasets for molecular property prediction and extensively compare with existing state-of-the-art models. It has been found that our KA-GNNs can outperform traditional GNN models. More importantly, our Fourier KAN module can not only increase the model accuracy but also reduce the computational time. This work not only highlights the great power of KA-GNNs in molecular property prediction but also provides a novel geometric deep learning framework for the general non-Euclidean data analysis.
       


### [Baseflow identification via explainable AI with Kolmogorov-Arnold networks](https://arxiv.org/abs/2410.11587)

**Authors:**
Chuyang Liu, Tirthankar Roy, Daniel M. Tartakovsky, Dipankar Dwivedi

**Abstract:**
Hydrological models often involve constitutive laws that may not be optimal in every application. We propose to replace such laws with the Kolmogorov-Arnold networks (KANs), a class of neural networks designed to identify symbolic expressions. We demonstrate KAN's potential on the problem of baseflow identification, a notoriously challenging task plagued by significant uncertainty. KAN-derived functional dependencies of the baseflow components on the aridity index outperform their original counterparts. On a test set, they increase the Nash-Sutcliffe Efficiency (NSE) by 67%, decrease the root mean squared error by 30%, and increase the Kling-Gupta efficiency by 24%. This superior performance is achieved while reducing the number of fitting parameters from three to two. Next, we use data from 378 catchments across the continental United States to refine the water-balance equation at the mean-annual scale. The KAN-derived equations based on the refined water balance outperform both the current aridity index model, with up to a 105% increase in NSE, and the KAN-derived equations based on the original water balance. While the performance of our model and tree-based machine learning methods is similar, KANs offer the advantage of simplicity and transparency and require no specific software or computational tools. This case study focuses on the aridity index formulation, but the approach is flexible and transferable to other hydrological processes.
       


### [From PINNs to PIKANs: Recent Advances in Physics-Informed Machine Learning](https://arxiv.org/abs/2410.13228)

**Authors:**
Juan Diego Toscano, Vivek Oommen, Alan John Varghese, Zongren Zou, Nazanin Ahmadi Daryakenari, Chenxi Wu, George Em Karniadakis

**Citation Count:**
25

**Abstract:**
Physics-Informed Neural Networks (PINNs) have emerged as a key tool in Scientific Machine Learning since their introduction in 2017, enabling the efficient solution of ordinary and partial differential equations using sparse measurements. Over the past few years, significant advancements have been made in the training and optimization of PINNs, covering aspects such as network architectures, adaptive refinement, domain decomposition, and the use of adaptive weights and activation functions. A notable recent development is the Physics-Informed Kolmogorov-Arnold Networks (PIKANS), which leverage a representation model originally proposed by Kolmogorov in 1957, offering a promising alternative to traditional PINNs. In this review, we provide a comprehensive overview of the latest advancements in PINNs, focusing on improvements in network design, feature expansion, optimization techniques, uncertainty quantification, and theoretical insights. We also survey key applications across a range of fields, including biomedicine, fluid and solid mechanics, geophysics, dynamical systems, heat transfer, chemical engineering, and beyond. Finally, we review computational frameworks and software tools developed by both academia and industry to support PINN research and applications.
       


### [Multifidelity Kolmogorov-Arnold Networks](https://arxiv.org/abs/2410.14764)

**Authors:**
Amanda A. Howard, Bruno Jacob, Panos Stinis

**Citation Count:**
4

**Abstract:**
We develop a method for multifidelity Kolmogorov-Arnold networks (KANs), which use a low-fidelity model along with a small amount of high-fidelity data to train a model for the high-fidelity data accurately. Multifidelity KANs (MFKANs) reduce the amount of expensive high-fidelity data needed to accurately train a KAN by exploiting the correlations between the low- and high-fidelity data to give accurate and robust predictions in the absence of a large high-fidelity dataset. In addition, we show that multifidelity KANs can be used to increase the accuracy of physics-informed KANs (PIKANs), without the use of training data.
       


### [HiPPO-KAN: Efficient KAN Model for Time Series Analysis](https://arxiv.org/abs/2410.14939)

**Authors:**
SangJong Lee, Jin-Kwang Kim, JunHo Kim, TaeHan Kim, James Lee

**Citation Count:**
2

**Abstract:**
In this study, we introduces a parameter-efficient model that outperforms traditional models in time series forecasting, by integrating High-order Polynomial Projection (HiPPO) theory into the Kolmogorov-Arnold network (KAN) framework. This HiPPO-KAN model achieves superior performance on long sequence data without increasing parameter count. Experimental results demonstrate that HiPPO-KAN maintains a constant parameter count while varying window sizes and prediction horizons, in contrast to KAN, whose parameter count increases linearly with window size. Surprisingly, although the HiPPO-KAN model keeps a constant parameter count as increasing window size, it significantly outperforms KAN model at larger window sizes. These results indicate that HiPPO-KAN offers significant parameter efficiency and scalability advantages for time series forecasting. Additionally, we address the lagging problem commonly encountered in time series forecasting models, where predictions fail to promptly capture sudden changes in the data. We achieve this by modifying the loss function to compute the MSE directly on the coefficient vectors in the HiPPO domain. This adjustment effectively resolves the lagging problem, resulting in predictions that closely follow the actual time series data. By incorporating HiPPO theory into KAN, this study showcases an efficient approach for handling long sequences with improved predictive accuracy, offering practical contributions for applications in large-scale time series data.
       


### [LSS-SKAN: Efficient Kolmogorov-Arnold Networks based on Single-Parameterized Function](https://arxiv.org/abs/2410.14951)

**Authors:**
Zhijie Chen, Xinglin Zhang

**Citation Count:**
5

**Abstract:**
The recently proposed Kolmogorov-Arnold Networks (KAN) networks have attracted increasing attention due to their advantage of high visualizability compared to MLP. In this paper, based on a series of small-scale experiments, we proposed the Efficient KAN Expansion Principle (EKE Principle): allocating parameters to expand network scale, rather than employing more complex basis functions, leads to more efficient performance improvements in KANs. Based on this principle, we proposed a superior KAN termed SKAN, where the basis function utilizes only a single learnable parameter. We then evaluated various single-parameterized functions for constructing SKANs, with LShifted Softplus-based SKANs (LSS-SKANs) demonstrating superior accuracy. Subsequently, extensive experiments were performed, comparing LSS-SKAN with other KAN variants on the MNIST dataset. In the final accuracy tests, LSS-SKAN exhibited superior performance on the MNIST dataset compared to all tested pure KAN variants. Regarding execution speed, LSS-SKAN outperformed all compared popular KAN variants. Our experimental codes are available at https://github.com/chikkkit/LSS-SKAN and SKAN's Python library (for quick construction of SKAN in python) codes are available at https://github.com/chikkkit/SKAN .
       


### [Kaninfradet3D:A Road-side Camera-LiDAR Fusion 3D Perception Model based on Nonlinear Feature Extraction and Intrinsic Correlation](https://arxiv.org/abs/2410.15814)

**Authors:**
Pei Liu, Nanfang Zheng, Yiqun Li, Junlan Chen, Ziyuan Pu

**Citation Count:**
1

**Abstract:**
With the development of AI-assisted driving, numerous methods have emerged for ego-vehicle 3D perception tasks, but there has been limited research on roadside perception. With its ability to provide a global view and a broader sensing range, the roadside perspective is worth developing. LiDAR provides precise three-dimensional spatial information, while cameras offer semantic information. These two modalities are complementary in 3D detection. However, adding camera data does not increase accuracy in some studies since the information extraction and fusion procedure is not sufficiently reliable. Recently, Kolmogorov-Arnold Networks (KANs) have been proposed as replacements for MLPs, which are better suited for high-dimensional, complex data. Both the camera and the LiDAR provide high-dimensional information, and employing KANs should enhance the extraction of valuable features to produce better fusion outcomes. This paper proposes Kaninfradet3D, which optimizes the feature extraction and fusion modules. To extract features from complex high-dimensional data, the model's encoder and fuser modules were improved using KAN Layers. Cross-attention was applied to enhance feature fusion, and visual comparisons verified that camera features were more evenly integrated. This addressed the issue of camera features being abnormally concentrated, negatively impacting fusion. Compared to the benchmark, our approach shows improvements of +9.87 mAP and +10.64 mAP in the two viewpoints of the TUMTraf Intersection Dataset and an improvement of +1.40 mAP in the roadside end of the TUMTraf V2X Cooperative Perception Dataset. The results indicate that Kaninfradet3D can effectively fuse features, demonstrating the potential of applying KANs in roadside perception tasks.
       


### [KANICE: Kolmogorov-Arnold Networks with Interactive Convolutional Elements](https://arxiv.org/abs/2410.17172)

**Authors:**
Md Meftahul Ferdaus, Mahdi Abdelguerfi, Elias Ioup, David Dobson, Kendall N. Niles, Ken Pathak, Steven Sloan

**Venue:**
International Conference on AI-ML-Systems

**Citation Count:**
5

**Abstract:**
We introduce KANICE (Kolmogorov-Arnold Networks with Interactive Convolutional Elements), a novel neural architecture that combines Convolutional Neural Networks (CNNs) with Kolmogorov-Arnold Network (KAN) principles. KANICE integrates Interactive Convolutional Blocks (ICBs) and KAN linear layers into a CNN framework. This leverages KANs' universal approximation capabilities and ICBs' adaptive feature learning. KANICE captures complex, non-linear data relationships while enabling dynamic, context-dependent feature extraction based on the Kolmogorov-Arnold representation theorem. We evaluated KANICE on four datasets: MNIST, Fashion-MNIST, EMNIST, and SVHN, comparing it against standard CNNs, CNN-KAN hybrids, and ICB variants. KANICE consistently outperformed baseline models, achieving 99.35% accuracy on MNIST and 90.05% on the SVHN dataset.
  Furthermore, we introduce KANICE-mini, a compact variant designed for efficiency. A comprehensive ablation study demonstrates that KANICE-mini achieves comparable performance to KANICE with significantly fewer parameters. KANICE-mini reached 90.00% accuracy on SVHN with 2,337,828 parameters, compared to KANICE's 25,432,000. This study highlights the potential of KAN-based architectures in balancing performance and computational efficiency in image classification tasks. Our work contributes to research in adaptive neural networks, integrates mathematical theorems into deep learning architectures, and explores the trade-offs between model complexity and performance, advancing computer vision and pattern recognition. The source code for this paper is publicly accessible through our GitHub repository (https://github.com/m-ferdaus/kanice).
       


### [NIDS Neural Networks Using Sliding Time Window Data Processing with Trainable Activations and its Generalization Capability](https://arxiv.org/abs/2410.18658)

**Authors:**
Anton Raskovalov, Nikita Gabdullin, Ilya Androsov

**Abstract:**
This paper presents neural networks for network intrusion detection systems (NIDS), that operate on flow data preprocessed with a time window. It requires only eleven features which do not rely on deep packet inspection and can be found in most NIDS datasets and easily obtained from conventional flow collectors. The time window aggregates information with respect to hosts facilitating the identification of flow signatures that are missed by other aggregation methods. Several network architectures are studied and the use of Kolmogorov-Arnold Network (KAN)-inspired trainable activation functions that help to achieve higher accuracy with simpler network structure is proposed. The reported training accuracy exceeds 99% for the proposed method with as little as twenty neural network input features. This work also studies the generalization capability of NIDS, a crucial aspect that has not been adequately addressed in the previous studies. The generalization experiments are conducted using CICIDS2017 dataset and a custom dataset collected as part of this study. It is shown that the performance metrics decline significantly when changing datasets, and the reduction in performance metrics can be attributed to the difference in signatures of the same type flows in different datasets, which in turn can be attributed to the differences between the underlying networks. It is shown that the generalization accuracy of some neural networks can be very unstable and sensitive to random initialization parameters, and neural networks with fewer parameters and well-tuned activations are more stable and achieve higher accuracy.
       


### [LArctan-SKAN: Simple and Efficient Single-Parameterized Kolmogorov-Arnold Networks using Learnable Trigonometric Function](https://arxiv.org/abs/2410.19360)

**Authors:**
Zhijie Chen, Xinglin Zhang

**Citation Count:**
2

**Abstract:**
This paper proposes a novel approach for designing Single-Parameterized Kolmogorov-Arnold Networks (SKAN) by utilizing a Single-Parameterized Function (SFunc) constructed from trigonometric functions. Three new SKAN variants are developed: LSin-SKAN, LCos-SKAN, and LArctan-SKAN. Experimental validation on the MNIST dataset demonstrates that LArctan-SKAN excels in both accuracy and computational efficiency. Specifically, LArctan-SKAN significantly improves test set accuracy over existing models, outperforming all pure KAN variants compared, including FourierKAN, LSS-SKAN, and Spl-KAN. It also surpasses mixed MLP-based models such as MLP+rKAN and MLP+fKAN in accuracy. Furthermore, LArctan-SKAN exhibits remarkable computational efficiency, with a training speed increase of 535.01% and 49.55% compared to MLP+rKAN and MLP+fKAN, respectively. These results confirm the effectiveness and potential of SKANs constructed with trigonometric functions. The experiment code is available at https://github.com/chikkkit/LArctan-SKAN .
       


### [KANsformer for Scalable Beamforming](https://arxiv.org/abs/2410.20690)

**Authors:**
Xinke Xie, Yang Lu, Chong-Yung Chi, Wei Chen, Bo Ai, Dusit Niyato

**Venue:**
IEEE Transactions on Vehicular Technology

**Citation Count:**
1

**Abstract:**
This paper proposes an unsupervised deep-learning (DL) approach by integrating transformer and Kolmogorov-Arnold networks (KAN) termed KANsformer to realize scalable beamforming for mobile communication systems. Specifically, we consider a classic multi-input-single-output energy efficiency maximization problem subject to the total power budget. The proposed KANsformer first extracts hidden features via a multi-head self-attention mechanism and then reads out the desired beamforming design via KAN. Numerical results are provided to evaluate the KANsformer in terms of generalization performance, transfer learning and ablation experiment. Overall, the KANsformer outperforms existing benchmark DL approaches, and is adaptable to the change in the number of mobile users with real-time and near-optimal inference.
       


### [Using Structural Similarity and Kolmogorov-Arnold Networks for Anatomical Embedding of Cortical Folding Patterns](https://arxiv.org/abs/2410.23598)

**Authors:**
Minheng Chen, Chao Cao, Tong Chen, Yan Zhuang, Jing Zhang, Yanjun Lyu, Xiaowei Yu, Lu Zhang, Tianming Liu, Dajiang Zhu

**Venue:**
IEEE International Symposium on Biomedical Imaging

**Citation Count:**
1

**Abstract:**
The 3-hinge gyrus (3HG) is a newly defined folding pattern, which is the conjunction of gyri coming from three directions in cortical folding. Many studies demonstrated that 3HGs can be reliable nodes when constructing brain networks or connectome since they simultaneously possess commonality and individuality across different individual brains and populations. However, 3HGs are identified and validated within individual spaces, making it difficult to directly serve as the brain network nodes due to the absence of cross-subject correspondence. The 3HG correspondences represent the intrinsic regulation of brain organizational architecture, traditional image-based registration methods tend to fail because individual anatomical properties need to be fully respected. To address this challenge, we propose a novel self-supervised framework for anatomical feature embedding of the 3HGs to build the correspondences among different brains. The core component of this framework is to construct a structural similarity-enhanced multi-hop feature encoding strategy based on the recently developed Kolmogorov-Arnold network (KAN) for anatomical feature embedding. Extensive experiments suggest that our approach can effectively establish robust cross-subject correspondences when no one-to-one mapping exists.
       


## November
### [KAN-AD: Time Series Anomaly Detection with Kolmogorov-Arnold Networks](https://arxiv.org/abs/2411.00278)

**Authors:**
Quan Zhou, Changhua Pei, Fei Sun, Jing Han, Zhengwei Gao, Dan Pei, Haiming Zhang, Gaogang Xie, Jianhui Li

**Citation Count:**
4

**Abstract:**
Time series anomaly detection (TSAD) underpins real-time monitoring in cloud services and web systems, allowing rapid identification of anomalies to prevent costly failures. Most TSAD methods driven by forecasting models tend to overfit by emphasizing minor fluctuations. Our analysis reveals that effective TSAD should focus on modeling "normal" behavior through smooth local patterns. To achieve this, we reformulate time series modeling as approximating the series with smooth univariate functions. The local smoothness of each univariate function ensures that the fitted time series remains resilient against local disturbances. However, a direct KAN implementation proves susceptible to these disturbances due to the inherently localized characteristics of B-spline functions. We thus propose KAN-AD, replacing B-splines with truncated Fourier expansions and introducing a novel lightweight learning mechanism that emphasizes global patterns while staying robust to local disturbances. On four popular TSAD benchmarks, KAN-AD achieves an average 15% improvement in detection accuracy (with peaks exceeding 27%) over state-of-the-art baselines. Remarkably, it requires fewer than 1,000 trainable parameters, resulting in a 50% faster inference speed compared to the original KAN, demonstrating the approach's efficiency and practical viability.
       


### [A KAN-based Interpretable Framework for Process-Informed Prediction of Global Warming Potential](https://arxiv.org/abs/2411.00426)

**Authors:**
Jaewook Lee, Xinyang Sun, Ethan Errington, Miao Guo

**Abstract:**
Accurate prediction of Global Warming Potential (GWP) is essential for assessing the environmental impact of chemical processes and materials. Traditional GWP prediction models rely predominantly on molecular structure, overlooking critical process-related information. In this study, we present an integrative GWP prediction model that combines molecular descriptors (MACCS keys and Mordred descriptors) with process information (process title, description, and location) to improve predictive accuracy and interpretability. Using a deep neural network (DNN) model, we achieved an R-squared of 86% on test data with Mordred descriptors, process location, and description information, representing a 25% improvement over the previous benchmark of 61%; XAI analysis further highlighted the significant role of process title embeddings in enhancing model predictions. To enhance interpretability, we employed a Kolmogorov-Arnold Network (KAN) to derive a symbolic formula for GWP prediction, capturing key molecular and process features and providing a transparent, interpretable alternative to black-box models, enabling users to gain insights into the molecular and process factors influencing GWP. Error analysis showed that the model performs reliably in densely populated data ranges, with increased uncertainty for higher GWP values. This analysis allows users to manage prediction uncertainty effectively, supporting data-driven decision-making in chemical and process design. Our results suggest that integrating both molecular and process-level information in GWP prediction models yields substantial gains in accuracy and interpretability, offering a valuable tool for sustainability assessments. Future work may extend this approach to additional environmental impact categories and refine the model to further enhance its predictive reliability.
       


### [Integrating Symbolic Neural Networks with Building Physics: A Study and Proposal](https://arxiv.org/abs/2411.00800)

**Authors:**
Xia Chen, Guoquan Lv, Xinwei Zhuang, Carlos Duarte, Stefano Schiavon, Philipp Geyer

**Citation Count:**
1

**Abstract:**
Symbolic neural networks, such as Kolmogorov-Arnold Networks (KAN), offer a promising approach for integrating prior knowledge with data-driven methods, making them valuable for addressing inverse problems in scientific and engineering domains. This study explores the application of KAN in building physics, focusing on predictive modeling, knowledge discovery, and continuous learning. Through four case studies, we demonstrate KAN's ability to rediscover fundamental equations, approximate complex formulas, and capture time-dependent dynamics in heat transfer. While there are challenges in extrapolation and interpretability, we highlight KAN's potential to combine advanced modeling methods for knowledge augmentation, which benefits energy efficiency, system optimization, and sustainability assessments beyond the personal knowledge constraints of the modelers. Additionally, we propose a model selection decision tree to guide practitioners in appropriate applications for building physics.
       


### [Fairness-Utilization Trade-off in Wireless Networks with Explainable Kolmogorov-Arnold Networks](https://arxiv.org/abs/2411.01924)

**Authors:**
Masoud Shokrnezhad, Hamidreza Mazandarani, Tarik Taleb

**Venue:**
2024 IEEE Virtual Conference on Communications (VCC)

**Citation Count:**
1

**Abstract:**
The effective distribution of user transmit powers is essential for the significant advancements that the emergence of 6G wireless networks brings. In recent studies, Deep Neural Networks (DNNs) have been employed to address this challenge. However, these methods frequently encounter issues regarding fairness and computational inefficiency when making decisions, rendering them unsuitable for future dynamic services that depend heavily on the participation of each individual user. To address this gap, this paper focuses on the challenge of transmit power allocation in wireless networks, aiming to optimize $α$-fairness to balance network utilization and user equity. We introduce a novel approach utilizing Kolmogorov-Arnold Networks (KANs), a class of machine learning models that offer low inference costs compared to traditional DNNs through superior explainability. The study provides a comprehensive problem formulation, establishing the NP-hardness of the power allocation problem. Then, two algorithms are proposed for dataset generation and decentralized KAN training, offering a flexible framework for achieving various fairness objectives in dynamic 6G environments. Extensive numerical simulations demonstrate the effectiveness of our approach in terms of fairness and inference cost. The results underscore the potential of KANs to overcome the limitations of existing DNN-based methods, particularly in scenarios that demand rapid adaptation and fairness.
       


### [Human-in-the-Loop Feature Selection Using Interpretable Kolmogorov-Arnold Network-based Double Deep Q-Network](https://arxiv.org/abs/2411.03740)

**Authors:**
Md Abrar Jahin, M. F. Mridha, Nilanjan Dey

**Abstract:**
Feature selection is critical for improving the performance and interpretability of machine learning models, particularly in high-dimensional spaces where complex feature interactions can reduce accuracy and increase computational demands. Existing approaches often rely on static feature subsets or manual intervention, limiting adaptability and scalability. However, dynamic, per-instance feature selection methods and model-specific interpretability in reinforcement learning remain underexplored. This study proposes a human-in-the-loop (HITL) feature selection framework integrated into a Double Deep Q-Network (DDQN) using a Kolmogorov-Arnold Network (KAN). Our novel approach leverages simulated human feedback and stochastic distribution-based sampling, specifically Beta, to iteratively refine feature subsets per data instance, improving flexibility in feature selection. The KAN-DDQN achieved notable test accuracies of 93% on MNIST and 83% on FashionMNIST, outperforming conventional MLP-DDQN models by up to 9%. The KAN-based model provided high interpretability via symbolic representation while using 4 times fewer neurons in the hidden layer than MLPs did. Comparatively, the models without feature selection achieved test accuracies of only 58% on MNIST and 64% on FashionMNIST, highlighting significant gains with our framework. Pruning and visualization further enhanced model transparency by elucidating decision pathways. These findings present a scalable, interpretable solution for feature selection that is suitable for applications requiring real-time, adaptive decision-making with minimal human oversight.
       


### [Physics-informed Kolmogorov-Arnold Network with Chebyshev Polynomials for Fluid Mechanics](https://arxiv.org/abs/2411.04516)

**Authors:**
Chunyu Guo, Lucheng Sun, Shilong Li, Zelong Yuan, Chao Wang

**Citation Count:**
4

**Abstract:**
Solving partial differential equations (PDEs) is essential in scientific forecasting and fluid dynamics. Traditional approaches often incur expensive computational costs and trade-offs in efficiency and accuracy. Recent deep neural networks improve accuracy but require quality training data. Physics-informed neural networks (PINNs) effectively integrate physical laws, reducing data reliance in limited sample scenarios. A novel machine-learning framework, Chebyshev physics-informed Kolmogorov-Arnold network (ChebPIKAN), is proposed to integrate the robust architectures of Kolmogorov-Arnold networks (KAN) with physical constraints to enhance calculation accuracy of PDEs for fluid mechanics. We explore the fundamentals of KAN, emphasis on the advantages of using the orthogonality of Chebyshev polynomial basis functions in spline fitting, and describe the incorporation of physics-informed loss functions tailored to specific PDEs in fluid dynamics, including Allen-Cahn equation, nonlinear Burgers equation, two-dimensional Helmholtz equations, two-dimensional Kovasznay flow and two-dimensional Navier-Stokes equations. Extensive experiments demonstrate that the proposed ChebPIKAN model significantly outperforms standard KAN architecture in solving various PDEs by embedding essential physical information more effectively. These results indicate that augmenting KAN with physical constraints can not only alleviate overfitting issues of KAN but also improve extrapolation performance. Consequently, this study highlights the potential of ChebPIKAN as a powerful tool in computational fluid dynamics, proposing a path toward fast and reliable predictions in fluid mechanics and beyond.
       


### [On Training of Kolmogorov-Arnold Networks](https://arxiv.org/abs/2411.05296)

**Author:**
Shairoz Sohail

**Citation Count:**
1

**Abstract:**
Kolmogorov-Arnold Networks have recently been introduced as a flexible alternative to multi-layer Perceptron architectures. In this paper, we examine the training dynamics of different KAN architectures and compare them with corresponding MLP formulations. We train with a variety of different initialization schemes, optimizers, and learning rates, as well as utilize back propagation free approaches like the HSIC Bottleneck. We find that (when judged by test accuracy) KANs are an effective alternative to MLP architectures on high-dimensional datasets and have somewhat better parameter efficiency, but suffer from more unstable training dynamics. Finally, we provide recommendations for improving training stability of larger KAN models.
       


### [A Survey on Kolmogorov-Arnold Network](https://arxiv.org/abs/2411.06078)

**Authors:**
Shriyank Somvanshi, Syed Aaqib Javed, Md Monzurul Islam, Diwas Pandit, Subasish Das

**Venue:**
ACM Computing Surveys

**Citation Count:**
23

**Abstract:**
This systematic review explores the theoretical foundations, evolution, applications, and future potential of Kolmogorov-Arnold Networks (KAN), a neural network model inspired by the Kolmogorov-Arnold representation theorem. KANs distinguish themselves from traditional neural networks by using learnable, spline-parameterized functions instead of fixed activation functions, allowing for flexible and interpretable representations of high-dimensional functions. This review details KAN's architectural strengths, including adaptive edge-based activation functions that improve parameter efficiency and scalability in applications such as time series forecasting, computational biomedicine, and graph learning. Key advancements, including Temporal-KAN, FastKAN, and Partial Differential Equation (PDE) KAN, illustrate KAN's growing applicability in dynamic environments, enhancing interpretability, computational efficiency, and adaptability for complex function approximation tasks. Additionally, this paper discusses KAN's integration with other architectures, such as convolutional, recurrent, and transformer-based models, showcasing its versatility in complementing established neural networks for tasks requiring hybrid approaches. Despite its strengths, KAN faces computational challenges in high-dimensional and noisy data settings, motivating ongoing research into optimization strategies, regularization techniques, and hybrid models. This paper highlights KAN's role in modern neural architectures and outlines future directions to improve its computational efficiency, interpretability, and scalability in data-intensive applications.
       


### [Early Prediction of Natural Gas Pipeline Leaks Using the MKTCN Model](https://arxiv.org/abs/2411.06214)

**Authors:**
Xuguang Li, Zhonglin Zuo, Zheng Dong, Yang Yang

**Citation Count:**
2

**Abstract:**
Natural gas pipeline leaks pose severe risks, leading to substantial economic losses and potential hazards to human safety. In this study, we develop an accurate model for the early prediction of pipeline leaks. To the best of our knowledge, unlike previous anomaly detection, this is the first application to use internal pipeline data for early prediction of leaks. The modeling process addresses two main challenges: long-term dependencies and sample imbalance. First, we introduce a dilated convolution-based prediction model to capture long-term dependencies, as dilated convolution expands the model's receptive field without added computational cost. Second, to mitigate sample imbalance, we propose the MKTCN model, which incorporates the Kolmogorov-Arnold Network as the fully connected layer in a dilated convolution model, enhancing network generalization. Finally, we validate the MKTCN model through extensive experiments on two real-world datasets. Results demonstrate that MKTCN outperforms in generalization and classification, particularly under severe data imbalance, and effectively predicts leaks up to 5000 seconds in advance. Overall, the MKTCN model represents a significant advancement in early pipeline leak prediction, providing robust generalization and improved modeling of the long-term dependencies inherent in multi-dimensional time-series data.
       


### [SPIKANs: Separable Physics-Informed Kolmogorov-Arnold Networks](https://arxiv.org/abs/2411.06286)

**Authors:**
Bruno Jacob, Amanda A. Howard, Panos Stinis

**Citation Count:**
7

**Abstract:**
Physics-Informed Neural Networks (PINNs) have emerged as a promising method for solving partial differential equations (PDEs) in scientific computing. While PINNs typically use multilayer perceptrons (MLPs) as their underlying architecture, recent advancements have explored alternative neural network structures. One such innovation is the Kolmogorov-Arnold Network (KAN), which has demonstrated benefits over traditional MLPs, including faster neural scaling and better interpretability. The application of KANs to physics-informed learning has led to the development of Physics-Informed KANs (PIKANs), enabling the use of KANs to solve PDEs. However, despite their advantages, KANs often suffer from slower training speeds, particularly in higher-dimensional problems where the number of collocation points grows exponentially with the dimensionality of the system. To address this challenge, we introduce Separable Physics-Informed Kolmogorov-Arnold Networks (SPIKANs). This novel architecture applies the principle of separation of variables to PIKANs, decomposing the problem such that each dimension is handled by an individual KAN. This approach drastically reduces the computational complexity of training without sacrificing accuracy, facilitating their application to higher-dimensional PDEs. Through a series of benchmark problems, we demonstrate the effectiveness of SPIKANs, showcasing their superior scalability and performance compared to PIKANs and highlighting their potential for solving complex, high-dimensional PDEs in scientific computing.
       


### [Can KAN Work? Exploring the Potential of Kolmogorov-Arnold Networks in Computer Vision](https://arxiv.org/abs/2411.06727)

**Authors:**
Yueyang Cang, Yu hang liu, Li Shi

**Abstract:**
Kolmogorov-Arnold Networks(KANs), as a theoretically efficient neural network architecture, have garnered attention for their potential in capturing complex patterns. However, their application in computer vision remains relatively unexplored. This study first analyzes the potential of KAN in computer vision tasks, evaluating the performance of KAN and its convolutional variants in image classification and semantic segmentation. The focus is placed on examining their characteristics across varying data scales and noise levels. Results indicate that while KAN exhibits stronger fitting capabilities, it is highly sensitive to noise, limiting its robustness. To address this challenge, we propose a smoothness regularization method and introduce a Segment Deactivation technique. Both approaches enhance KAN's stability and generalization, demonstrating its potential in handling complex visual data tasks.
       


### [KLCBL: An Improved Police Incident Classification Model](https://arxiv.org/abs/2411.06749)

**Authors:**
Liu Zhuoxian, Shi Tuo, Hu Xiaofeng

**Abstract:**
Police incident data is crucial for public security intelligence, yet grassroots agencies struggle with efficient classification due to manual inefficiency and automated system limitations, especially in telecom and online fraud cases. This research proposes a multichannel neural network model, KLCBL, integrating Kolmogorov-Arnold Networks (KAN), a linguistically enhanced text preprocessing approach (LERT), Convolutional Neural Network (CNN), and Bidirectional Long Short-Term Memory (BiLSTM) for police incident classification. Evaluated with real data, KLCBL achieved 91.9% accuracy, outperforming baseline models. The model addresses classification challenges, enhances police informatization, improves resource allocation, and offers broad applicability to other classification tasks.
       


### [EAPCR: A Universal Feature Extractor for Scientific Data without Explicit Feature Relation Patterns](https://arxiv.org/abs/2411.08164)

**Authors:**
Zhuohang Yu, Ling An, Yansong Li, Yu Wu, Zeyu Dong, Zhangdi Liu, Le Gao, Zhenyu Zhang, Chichun Zhou

**Citation Count:**
1

**Abstract:**
Conventional methods, including Decision Tree (DT)-based methods, have been effective in scientific tasks, such as non-image medical diagnostics, system anomaly detection, and inorganic catalysis efficiency prediction. However, most deep-learning techniques have struggled to surpass or even match this level of success as traditional machine-learning methods. The primary reason is that these applications involve multi-source, heterogeneous data where features lack explicit relationships. This contrasts with image data, where pixels exhibit spatial relationships; textual data, where words have sequential dependencies; and graph data, where nodes are connected through established associations. The absence of explicit Feature Relation Patterns (FRPs) presents a significant challenge for deep learning techniques in scientific applications that are not image, text, and graph-based. In this paper, we introduce EAPCR, a universal feature extractor designed for data without explicit FRPs. Tested across various scientific tasks, EAPCR consistently outperforms traditional methods and bridges the gap where deep learning models fall short. To further demonstrate its robustness, we synthesize a dataset without explicit FRPs. While Kolmogorov-Arnold Network (KAN) and feature extractors like Convolutional Neural Networks (CNNs), Graph Convolutional Networks (GCNs), and Transformers struggle, EAPCR excels, demonstrating its robustness and superior performance in scientific tasks without FRPs.
       


### [Hybrid deep additive neural networks](https://arxiv.org/abs/2411.09175)

**Authors:**
Gyu Min Kim, Jeong Min Jeon

**Abstract:**
Traditional neural networks (multi-layer perceptrons) have become an important tool in data science due to their success across a wide range of tasks. However, their performance is sometimes unsatisfactory, and they often require a large number of parameters, primarily due to their reliance on the linear combination structure. Meanwhile, additive regression has been a popular alternative to linear regression in statistics. In this work, we introduce novel deep neural networks that incorporate the idea of additive regression. Our neural networks share architectural similarities with Kolmogorov-Arnold networks but are based on simpler yet flexible activation and basis functions. Additionally, we introduce several hybrid neural networks that combine this architecture with that of traditional neural networks. We derive their universal approximation properties and demonstrate their effectiveness through simulation studies and a real-data application. The numerical results indicate that our neural networks generally achieve better performance than traditional neural networks while using fewer parameters.
       


### [KAT to KANs: A Review of Kolmogorov-Arnold Networks and the Neural Leap Forward](https://arxiv.org/abs/2411.10622)

**Authors:**
Divesh Basina, Joseph Raj Vishal, Aarya Choudhary, Bharatesh Chakravarthi

**Abstract:**
The curse of dimensionality poses a significant challenge to modern multilayer perceptron-based architectures, often causing performance stagnation and scalability issues. Addressing this limitation typically requires vast amounts of data. In contrast, Kolmogorov-Arnold Networks have gained attention in the machine learning community for their bold claim of being unaffected by the curse of dimensionality. This paper explores the Kolmogorov-Arnold representation theorem and the mathematical principles underlying Kolmogorov-Arnold Networks, which enable their scalability and high performance in high-dimensional spaces. We begin with an introduction to foundational concepts necessary to understand Kolmogorov-Arnold Networks, including interpolation methods and Basis-splines, which form their mathematical backbone. This is followed by an overview of perceptron architectures and the Universal approximation theorem, a key principle guiding modern machine learning. This is followed by an overview of the Kolmogorov-Arnold representation theorem, including its mathematical formulation and implications for overcoming dimensionality challenges. Next, we review the architecture and error-scaling properties of Kolmogorov-Arnold Networks, demonstrating how these networks achieve true freedom from the curse of dimensionality. Finally, we discuss the practical viability of Kolmogorov-Arnold Networks, highlighting scenarios where their unique capabilities position them to excel in real-world applications. This review aims to offer insights into Kolmogorov-Arnold Networks' potential to redefine scalability and performance in high-dimensional learning tasks.
       


### [KAN/MultKAN with Physics-Informed Spline fitting (KAN-PISF) for ordinary/partial differential equation discovery of nonlinear dynamic systems](https://arxiv.org/abs/2411.11801)

**Authors:**
Ashish Pal, Satish Nagarajaiah

**Abstract:**
Machine learning for scientific discovery is increasingly becoming popular because of its ability to extract and recognize the nonlinear characteristics from the data. The black-box nature of deep learning methods poses difficulties in interpreting the identified model. There is a dire need to interpret the machine learning models to develop a physical understanding of dynamic systems. An interpretable form of neural network called Kolmogorov-Arnold networks (KAN) or Multiplicative KAN (MultKAN) offers critical features that help recognize the nonlinearities in the governing ordinary/partial differential equations (ODE/PDE) of various dynamic systems and find their equation structures. In this study, an equation discovery framework is proposed that includes i) sequentially regularized derivatives for denoising (SRDD) algorithm to denoise the measure data to obtain accurate derivatives, ii) KAN to identify the equation structure and suggest relevant nonlinear functions that are used to create a small overcomplete library of functions, and iii) physics-informed spline fitting (PISF) algorithm to filter the excess functions from the library and converge to the correct equation. The framework was tested on the forced Duffing oscillator, Van der Pol oscillator (stiff ODE), Burger's equation, and Bouc-Wen model (coupled ODE). The proposed method converged to the true equation for the first three systems. It provided an approximate model for the Bouc-Wen model that could acceptably capture the hysteresis response. Using KAN maintains low complexity, which helps the user interpret the results throughout the process and avoid the black-box-type nature of machine learning methods.
       


### [KAN-Mamba FusionNet: Redefining Medical Image Segmentation with Non-Linear Modeling](https://arxiv.org/abs/2411.11926)

**Authors:**
Akansh Agrawal, Akshan Agrawal, Shashwat Gupta, Priyanka Bagade

**Citation Count:**
3

**Abstract:**
Medical image segmentation is essential for applications like robotic surgeries, disease diagnosis, and treatment planning. Recently, various deep-learning models have been proposed to enhance medical image segmentation. One promising approach utilizes Kolmogorov-Arnold Networks (KANs), which better capture non-linearity in input data. However, they are unable to effectively capture long-range dependencies, which are required to accurately segment complex medical images and, by that, improve diagnostic accuracy in clinical settings. Neural networks such as Mamba can handle long-range dependencies. However, they have a limited ability to accurately capture non-linearities in the images as compared to KANs. Thus, we propose a novel architecture, the KAN-Mamba FusionNet, which improves segmentation accuracy by effectively capturing the non-linearities from input and handling long-range dependencies with the newly proposed KAMBA block. We evaluated the proposed KAN-Mamba FusionNet on three distinct medical image segmentation datasets: BUSI, Kvasir-Seg, and GlaS - and found it consistently outperforms state-of-the-art methods in IoU and F1 scores. Further, we examined the effects of various components and assessed their contributions to the overall model performance via ablation studies. The findings highlight the effectiveness of this methodology for reliable medical image segmentation, providing a unique approach to address intricate visual data issues in healthcare.
       


### [Contrast Similarity-Aware Dual-Pathway Mamba for Multivariate Time Series Node Classification](https://arxiv.org/abs/2411.12222)

**Authors:**
Mingsen Du, Meng Chen, Yongjian Li, Xiuxin Zhang, Jiahui Gao, Cun Ji, Shoushui Wei

**Citation Count:**
1

**Abstract:**
Multivariate time series (MTS) data is generated through multiple sensors across various domains such as engineering application, health monitoring, and the internet of things, characterized by its temporal changes and high dimensional characteristics. Over the past few years, many studies have explored the long-range dependencies and similarities in MTS. However, long-range dependencies are difficult to model due to their temporal changes and high dimensionality makes it difficult to obtain similarities effectively and efficiently. Thus, to address these issues, we propose contrast similarity-aware dual-pathway Mamba for MTS node classification (CS-DPMamba). Firstly, to obtain the dynamic similarity of each sample, we initially use temporal contrast learning module to acquire MTS representations. And then we construct a similarity matrix between MTS representations using Fast Dynamic Time Warping (FastDTW). Secondly, we apply the DPMamba to consider the bidirectional nature of MTS, allowing us to better capture long-range and short-range dependencies within the data. Finally, we utilize the Kolmogorov-Arnold Network enhanced Graph Isomorphism Network to complete the information interaction in the matrix and MTS node classification task. By comprehensively considering the long-range dependencies and dynamic similarity features, we achieved precise MTS node classification. We conducted experiments on multiple University of East Anglia (UEA) MTS datasets, which encompass diverse application scenarios. Our results demonstrate the superiority of our method through both supervised and semi-supervised experiments on the MTS classification task.
       


### [Dressing the Imagination: A Dataset for AI-Powered Translation of Text into Fashion Outfits and A Novel KAN Adapter for Enhanced Feature Adaptation](https://arxiv.org/abs/2411.13901)

**Authors:**
Gayatri Deshmukh, Somsubhra De, Chirag Sehgal, Jishu Sen Gupta, Sparsh Mittal

**Abstract:**
Specialized datasets that capture the fashion industry's rich language and styling elements can boost progress in AI-driven fashion design. We present FLORA (Fashion Language Outfit Representation for Apparel Generation), the first comprehensive dataset containing 4,330 curated pairs of fashion outfits and corresponding textual descriptions. Each description utilizes industry-specific terminology and jargon commonly used by professional fashion designers, providing precise and detailed insights into the outfits. Hence, the dataset captures the delicate features and subtle stylistic elements necessary to create high-fidelity fashion designs. We demonstrate that fine-tuning generative models on the FLORA dataset significantly enhances their capability to generate accurate and stylistically rich images from textual descriptions of fashion sketches. FLORA will catalyze the creation of advanced AI models capable of comprehending and producing subtle, stylistically rich fashion designs. It will also help fashion designers and end-users to bring their ideas to life.
  As a second orthogonal contribution, we introduce KAN Adapters, which leverage Kolmogorov-Arnold Networks (KAN) as adaptive modules. They serve as replacements for traditional MLP-based LoRA adapters. With learnable spline-based activations, KAN Adapters excel in modeling complex, non-linear relationships, achieving superior fidelity, faster convergence and semantic alignment. Extensive experiments and ablation studies on our proposed FLORA dataset validate the superiority of KAN Adapters over LoRA adapters. To foster further research and collaboration, we will open-source both the FLORA and our implementation code.
       


### [Machine Learning Insights into Quark-Antiquark Interactions: Probing Field Distributions and String Tension in QCD](https://arxiv.org/abs/2411.14902)

**Authors:**
Wei Kou, Xurong Chen

**Venue:**
The European Physical Journal C

**Citation Count:**
2

**Abstract:**
Understanding the interactions between quark-antiquark pairs is essential for elucidating quark confinement within the framework of quantum chromodynamics (QCD). This study investigates the field distribution patterns that arise between these pairs by employing advanced machine learning techniques, namely multilayer perceptrons (MLP) and Kolmogorov-Arnold networks (KAN), to analyze data obtained from lattice QCD simulations. The models developed through this training are then applied to calculate the string tension and width associated with chromo flux tubes, and these results are rigorously compared to those derived from lattice QCD. Moreover, we introduce a preliminary analytical expression that characterizes the field distribution as a function of quark separation, utilizing the KAN methodology. Our comprehensive quantitative analysis underscores the potential of integrating machine learning approaches into conventional QCD research.
       


### [Exploring Kolmogorov-Arnold Networks for Interpretable Time Series Classification](https://arxiv.org/abs/2411.14904)

**Authors:**
Irina Barašin, Blaž Bertalanič, Mihael Mohorčič, Carolina Fortuna

**Citation Count:**
2

**Abstract:**
Time series classification is a relevant step supporting decision-making processes in various domains, and deep neural models have shown promising performance.
  Despite significant advancements in deep learning, the theoretical understanding of how and why complex architectures function remains limited, prompting the need for more interpretable models. Recently, the Kolmogorov-Arnold Networks (KANs) have been proposed as a more interpretable alternative. While KAN-related research is significantly rising, to date, the study of KAN architectures for time series classification has been limited.
  In this paper, we aim to conduct a comprehensive and robust exploration of the KAN architecture for time series classification on the UCR benchmark. More specifically, we look at a) how reference architectures for forecasting transfer to classification, at the b) hyperparameter and implementation influence on the classification performance in view of finding the one that performs best on the selected benchmark, the c) complexity trade-offs and d) interpretability advantages. Our results show that (1) Efficient KAN outperforms MLP in performance and computational efficiency, showcasing its suitability for tasks classification tasks. (2) Efficient KAN is more stable than KAN across grid sizes, depths, and layer configurations, particularly with lower learning rates. (3) KAN maintains competitive accuracy compared to state-of-the-art models like HIVE-COTE2, with smaller architectures and faster training times, supporting its balance of performance and transparency. (4) The interpretability of the KAN model aligns with findings from SHAP analysis, reinforcing its capacity for transparent decision-making.
       


### [Learnable Activation Functions in Physics-Informed Neural Networks for Solving Partial Differential Equations](https://arxiv.org/abs/2411.15111)

**Authors:**
Afrah Farea, Mustafa Serdar Celebi

**Citation Count:**
1

**Abstract:**
We investigate learnable activation functions in Physics-Informed Neural Networks (PINNs) for solving Partial Differential Equations (PDEs): comparing traditional Multilayer Perceptrons (MLPs) with fixed and trainable activations against Kolmogorov-Arnold Networks (KANs) that employ learnable basis functions. While PINNs effectively incorporate physical laws into the learning process, they suffer from convergence and spectral bias problems, which limit their applicability to problems with rapid oscillations or sharp transitions. In this work, we study various activation and basis functions across diverse PDEs, including oscillatory, nonlinear wave, mixed-physics, and fluid dynamics problems. Using empirical Neural Tangent Kernel (NTK) analysis and Hessian eigenvalue decomposition, we assess convergence behavior, stability, and high-frequency approximation capacity. While KANs offer improved expressivity for capturing complex, high-frequency PDE solutions, they introduce new optimization challenges, especially in deeper networks. Our findings show that KANs face a curse of functional dimensionality, creating intractable optimization landscapes in deeper networks. Low spectral bias alone does not guarantee good performance; adaptive spectral bias approaches such as B-splines achieve optimal results by balancing global stability with local high-frequency resolution. Different PDE types require tailored strategies: smooth global activation functions excel for wave phenomena, while local adaptive activation functions suit problems with sharp transitions.
       


### [GrokFormer: Graph Fourier Kolmogorov-Arnold Transformers](https://arxiv.org/abs/2411.17296)

**Authors:**
Guoguo Ai, Guansong Pang, Hezhe Qiao, Yuan Gao, Hui Yan

**Abstract:**
Graph Transformers (GTs) have demonstrated remarkable performance in graph representation learning over popular graph neural networks (GNNs). However, self--attention, the core module of GTs, preserves only low-frequency signals in graph features, leading to ineffectiveness in capturing other important signals like high-frequency ones. Some recent GT models help alleviate this issue, but their flexibility and expressiveness are still limited since the filters they learn are fixed on predefined graph spectrum or spectral order. To tackle this challenge, we propose a Graph Fourier Kolmogorov-Arnold Transformer (GrokFormer), a novel GT model that learns highly expressive spectral filters with adaptive graph spectrum and spectral order through a Fourier series modeling over learnable activation functions. We demonstrate theoretically and empirically that the proposed GrokFormer filter offers better expressiveness than other spectral methods. Comprehensive experiments on 10 real-world node classification datasets across various domains, scales, and graph properties, as well as 5 graph classification datasets, show that GrokFormer outperforms state-of-the-art GTs and GNNs. Our code is available at https://github.com/GGA23/GrokFormer
       


### [KACDP: A Highly Interpretable Credit Default Prediction Model](https://arxiv.org/abs/2411.17783)

**Authors:**
Kun Liu, Jin Zhao

**Abstract:**
In the field of finance, the prediction of individual credit default is of vital importance. However, existing methods face problems such as insufficient interpretability and transparency as well as limited performance when dealing with high-dimensional and nonlinear data. To address these issues, this paper introduces a method based on Kolmogorov-Arnold Networks (KANs). KANs is a new type of neural network architecture with learnable activation functions and no linear weights, which has potential advantages in handling complex multi-dimensional data. Specifically, this paper applies KANs to the field of individual credit risk prediction for the first time and constructs the Kolmogorov-Arnold Credit Default Predict (KACDP) model. Experiments show that the KACDP model outperforms mainstream credit default prediction models in performance metrics (ROC_AUC and F1 values). Meanwhile, through methods such as feature attribution scores and visualization of the model structure, the model's decision-making process and the importance of different features are clearly demonstrated, providing transparent and interpretable decision-making basis for financial institutions and meeting the industry's strict requirements for model interpretability. In conclusion, the KACDP model constructed in this paper exhibits excellent predictive performance and satisfactory interpretability in individual credit risk prediction, providing an effective way to address the limitations of existing methods and offering a new and practical credit risk prediction tool for financial institutions.
       


### [KAN See Your Face](https://arxiv.org/abs/2411.18165)

**Authors:**
Dong Han, Yong Li, Joachim Denzler

**Citation Count:**
3

**Abstract:**
With the advancement of face reconstruction (FR) systems, privacy-preserving face recognition (PPFR) has gained popularity for its secure face recognition, enhanced facial privacy protection, and robustness to various attacks. Besides, specific models and algorithms are proposed for face embedding protection by mapping embeddings to a secure space. However, there is a lack of studies on investigating and evaluating the possibility of extracting face images from embeddings of those systems, especially for PPFR. In this work, we introduce the first approach to exploit Kolmogorov-Arnold Network (KAN) for conducting embedding-to-face attacks against state-of-the-art (SOTA) FR and PPFR systems. Face embedding mapping (FEM) models are proposed to learn the distribution mapping relation between the embeddings from the initial domain and target domain. In comparison with Multi-Layer Perceptrons (MLP), we provide two variants, FEM-KAN and FEM-MLP, for efficient non-linear embedding-to-embedding mapping in order to reconstruct realistic face images from the corresponding face embedding. To verify our methods, we conduct extensive experiments with various PPFR and FR models. We also measure reconstructed face images with different metrics to evaluate the image quality. Through comprehensive experiments, we demonstrate the effectiveness of FEMs in accurate embedding mapping and face reconstruction.
       


### [KANs for Computer Vision: An Experimental Study](https://arxiv.org/abs/2411.18224)

**Authors:**
Karthik Mohan, Hanxiao Wang, Xiatian Zhu

**Citation Count:**
3

**Abstract:**
This paper presents an experimental study of Kolmogorov-Arnold Networks (KANs) applied to computer vision tasks, particularly image classification. KANs introduce learnable activation functions on edges, offering flexible non-linear transformations compared to traditional pre-fixed activation functions with specific neural work like Multi-Layer Perceptrons (MLPs) and Convolutional Neural Networks (CNNs). While KANs have shown promise mostly in simplified or small-scale datasets, their effectiveness for more complex real-world tasks such as computer vision tasks remains less explored. To fill this gap, this experimental study aims to provide extended observations and insights into the strengths and limitations of KANs. We reveal that although KANs can perform well in specific vision tasks, they face significant challenges, including increased hyperparameter sensitivity and higher computational costs. These limitations suggest that KANs require architectural adaptations, such as integration with other architectures, to be practical for large-scale vision problems. This study focuses on empirical findings rather than proposing new methods, aiming to inform future research on optimizing KANs, in particular computer vision applications or alike.
       


### [MvKeTR: Chest CT Report Generation with Multi-View Perception and Knowledge Enhancement](https://arxiv.org/abs/2411.18309)

**Authors:**
Xiwei Deng, Xianchun He, Jiangfeng Bao, Yudan Zhou, Shuhui Cai, Congbo Cai, Zhong Chen

**Citation Count:**
1

**Abstract:**
CT report generation (CTRG) aims to automatically generate diagnostic reports for 3D volumes, relieving clinicians' workload and improving patient care. Despite clinical value, existing works fail to effectively incorporate diagnostic information from multiple anatomical views and lack related clinical expertise essential for accurate and reliable diagnosis. To resolve these limitations, we propose a novel Multi-view perception Knowledge-enhanced Transformer (MvKeTR) to mimic the diagnostic workflow of clinicians. Just as radiologists first examine CT scans from multiple planes, a Multi-View Perception Aggregator (MVPA) with view-aware attention effectively synthesizes diagnostic information from multiple anatomical views. Then, inspired by how radiologists further refer to relevant clinical records to guide diagnostic decision-making, a Cross-Modal Knowledge Enhancer (CMKE) retrieves the most similar reports based on the query volume to incorporate domain knowledge into the diagnosis procedure. Furthermore, instead of traditional MLPs, we employ Kolmogorov-Arnold Networks (KANs) with learnable nonlinear activation functions as the fundamental building blocks of both modules to better capture intricate diagnostic patterns in CT interpretation. Extensive experiments on the public CTRG-Chest-548K dataset demonstrate that our method outpaces prior state-of-the-art (SOTA) models across almost all metrics. The code will be made publicly available.
       


### [Non-linear Equalization in 112 Gb/s PONs Using Kolmogorov-Arnold Networks](https://arxiv.org/abs/2411.19631)

**Authors:**
Rodrigo Fischer, Patrick Matalla, Sebastian Randel, Laurent Schmalen

**Citation Count:**
1

**Abstract:**
We investigate Kolmogorov-Arnold networks (KANs) for non-linear equalization of 112 Gb/s PAM4 passive optical networks (PONs). Using pruning and extensive hyperparameter search, we outperform linear equalizers and convolutional neural networks at low computational complexity.
       


## December
### [Option Pricing with Convolutional Kolmogorov-Arnold Networks](https://arxiv.org/abs/2412.01224)

**Authors:**
Zeyuan Li, Qingdao Huang

**Abstract:**
With the rapid advancement of neural networks, methods for option pricing have evolved significantly. This study employs the Black-Scholes-Merton (B-S-M) model, incorporating an additional variable to improve the accuracy of predictions compared to the traditional Black-Scholes (B-S) model. Furthermore, Convolutional Kolmogorov-Arnold Networks (Conv-KANs) and Kolmogorov-Arnold Networks (KANs) are introduced to demonstrate that networks with enhanced non-linear capabilities yield superior fitting performance. For comparative analysis, Conv-LSTM and LSTM models, which are widely used in time series forecasting, are also applied. Additionally, a novel data selection strategy is proposed to simulate a real trading environment, thereby enhancing the robustness of the model.
       


### [Explainable fault and severity classification for rolling element bearings using Kolmogorov-Arnold networks](https://arxiv.org/abs/2412.01322)

**Authors:**
Spyros Rigas, Michalis Papachristou, Ioannis Sotiropoulos, Georgios Alexandridis

**Abstract:**
Rolling element bearings are critical components of rotating machinery, with their performance directly influencing the efficiency and reliability of industrial systems. At the same time, bearing faults are a leading cause of machinery failures, often resulting in costly downtime, reduced productivity, and, in extreme cases, catastrophic damage. This study presents a methodology that utilizes Kolmogorov-Arnold Networks to address these challenges through automatic feature selection, hyperparameter tuning and interpretable fault analysis within a unified framework. By training shallow network architectures and minimizing the number of selected features, the framework produces lightweight models that deliver explainable results through feature attribution and symbolic representations of their activation functions. Validated on two widely recognized datasets for bearing fault diagnosis, the framework achieved perfect F1-Scores for fault detection and high performance in fault and severity classification tasks, including 100% F1-Scores in most cases. Notably, it demonstrated adaptability by handling diverse fault types, such as imbalance and misalignment, within the same dataset. The symbolic representations enhanced model interpretability, while feature attribution offered insights into the optimal feature types or signals for each studied task. These results highlight the framework's potential for practical applications, such as real-time machinery monitoring, and for scientific research requiring efficient and explainable models.
       


### [ECG-SleepNet: Deep Learning-Based Comprehensive Sleep Stage Classification Using ECG Signals](https://arxiv.org/abs/2412.01929)

**Authors:**
Poorya Aghaomidi, Ge Wang

**Citation Count:**
1

**Abstract:**
Accurate sleep stage classification is essential for understanding sleep disorders and improving overall health. This study proposes a novel three-stage approach for sleep stage classification using ECG signals, offering a more accessible alternative to traditional methods that often rely on complex modalities like EEG. In Stages 1 and 2, we initialize the weights of two networks, which are then integrated in Stage 3 for comprehensive classification. In the first phase, we estimate key features using Feature Imitating Networks (FINs) to achieve higher accuracy and faster convergence. The second phase focuses on identifying the N1 sleep stage through the time-frequency representation of ECG signals. Finally, the third phase integrates models from the previous stages and employs a Kolmogorov-Arnold Network (KAN) to classify five distinct sleep stages. Additionally, data augmentation techniques, particularly SMOTE, are used in enhancing classification capabilities for underrepresented stages like N1. Our results demonstrate significant improvements in the classification performance, with an overall accuracy of 80.79% an overall kappa of 0.73. The model achieves specific accuracies of 86.70% for Wake, 60.36% for N1, 83.89% for N2, 84.85% for N3, and 87.16% for REM. This study emphasizes the importance of weight initialization and data augmentation in optimizing sleep stage classification with ECG signals.
       


### [Beyond Tree Models: A Hybrid Model of KAN and gMLP for Large-Scale Financial Tabular Data](https://arxiv.org/abs/2412.02097)

**Authors:**
Mingming Zhang, Jiahao Hu, Pengfei Shi, Ningtao Wang, Ruizhe Gao, Guandong Sun, Feng Zhao, Yulin kang, Xing Fu, Weiqiang Wang, Junbo Zhao

**Citation Count:**
1

**Abstract:**
Tabular data plays a critical role in real-world financial scenarios. Traditionally, tree models have dominated in handling tabular data. However, financial datasets in the industry often encounter some challenges, such as data heterogeneity, the predominance of numerical features and the large scale of the data, which can range from tens of millions to hundreds of millions of records. These challenges can lead to significant memory and computational issues when using tree-based models. Consequently, there is a growing need for neural network-based solutions that can outperform these models. In this paper, we introduce TKGMLP, an hybrid network for tabular data that combines shallow Kolmogorov Arnold Networks with Gated Multilayer Perceptron. This model leverages the strengths of both architectures to improve performance and scalability. We validate TKGMLP on a real-world credit scoring dataset, where it achieves state-of-the-art results and outperforms current benchmarks. Furthermore, our findings demonstrate that the model continues to improve as the dataset size increases, making it highly scalable. Additionally, we propose a novel feature encoding method for numerical data, specifically designed to address the predominance of numerical features in financial datasets. The integration of this feature encoding method within TKGMLP significantly improves prediction accuracy. This research not only advances table prediction technology but also offers a practical and effective solution for handling large-scale numerical tabular data in various industrial applications.
       


### [Enhanced Photovoltaic Power Forecasting: An iTransformer and LSTM-Based Model Integrating Temporal and Covariate Interactions](https://arxiv.org/abs/2412.02302)

**Authors:**
Guang Wu, Yun Wang, Qian Zhou, Ziyang Zhang

**Venue:**
2024 IEEE 8th Conference on Energy Internet and Energy System Integration (EI2)

**Citation Count:**
2

**Abstract:**
Accurate photovoltaic (PV) power forecasting is critical for integrating renewable energy sources into the grid, optimizing real-time energy management, and ensuring energy reliability amidst increasing demand. However, existing models often struggle with effectively capturing the complex relationships between target variables and covariates, as well as the interactions between temporal dynamics and multivariate data, leading to suboptimal forecasting accuracy. To address these challenges, we propose a novel model architecture that leverages the iTransformer for feature extraction from target variables and employs long short-term memory (LSTM) to extract features from covariates. A cross-attention mechanism is integrated to fuse the outputs of both models, followed by a Kolmogorov-Arnold network (KAN) mapping for enhanced representation. The effectiveness of the proposed model is validated using publicly available datasets from Australia, with experiments conducted across four seasons. Results demonstrate that the proposed model effectively capture seasonal variations in PV power generation and improve forecasting accuracy.
       


### [CIKAN: Constraint Informed Kolmogorov-Arnold Networks for Autonomous Spacecraft Rendezvous using Time Shift Governor](https://arxiv.org/abs/2412.03710)

**Authors:**
Taehyeun Kim, Anouck Girard, Ilya Kolmanovsky

**Citation Count:**
3

**Abstract:**
The paper considers a Constrained-Informed Neural Network (CINN) approximation for the Time Shift Governor (TSG), which is an add-on scheme to the nominal closed-loop system used to enforce constraints by time-shifting the reference trajectory in spacecraft rendezvous applications. We incorporate Kolmogorov-Arnold Networks (KANs), an emerging architecture in the AI community, as a fundamental component of CINN and propose a Constrained-Informed Kolmogorov-Arnold Network (CIKAN)-based approximation for TSG. We demonstrate the effectiveness of the CIKAN-based TSG through simulations of constrained spacecraft rendezvous missions on highly elliptic orbits and present comparisons between CIKANs, MLP-based CINNs, and the conventional TSG.
       


### [You KAN Do It in a Single Shot: Plug-and-Play Methods with Single-Instance Priors](https://arxiv.org/abs/2412.06204)

**Authors:**
Yanqi Cheng, Carola-Bibiane Schönlieb, Angelica I Aviles-Rivero

**Abstract:**
The use of Plug-and-Play (PnP) methods has become a central approach for solving inverse problems, with denoisers serving as regularising priors that guide optimisation towards a clean solution. In this work, we introduce KAN-PnP, an optimisation framework that incorporates Kolmogorov-Arnold Networks (KANs) as denoisers within the Plug-and-Play (PnP) paradigm. KAN-PnP is specifically designed to solve inverse problems with single-instance priors, where only a single noisy observation is available, eliminating the need for large datasets typically required by traditional denoising methods. We show that KANs, based on the Kolmogorov-Arnold representation theorem, serve effectively as priors in such settings, providing a robust approach to denoising. We prove that the KAN denoiser is Lipschitz continuous, ensuring stability and convergence in optimisation algorithms like PnP-ADMM, even in the context of single-shot learning. Additionally, we provide theoretical guarantees for KAN-PnP, demonstrating its convergence under key conditions: the convexity of the data fidelity term, Lipschitz continuity of the denoiser, and boundedness of the regularisation functional. These conditions are crucial for stable and reliable optimisation. Our experimental results show, on super-resolution and joint optimisation, that KAN-PnP outperforms exiting methods, delivering superior performance in single-shot learning with minimal data. The method exhibits strong convergence properties, achieving high accuracy with fewer iterations.
       


### [PowerMLP: An Efficient Version of KAN](https://arxiv.org/abs/2412.13571)

**Authors:**
Ruichen Qiu, Yibo Miao, Shiwen Wang, Lijia Yu, Yifan Zhu, Xiao-Shan Gao

**Venue:**
AAAI Conference on Artificial Intelligence

**Citation Count:**
2

**Abstract:**
The Kolmogorov-Arnold Network (KAN) is a new network architecture known for its high accuracy in several tasks such as function fitting and PDE solving. The superior expressive capability of KAN arises from the Kolmogorov-Arnold representation theorem and learnable spline functions. However, the computation of spline functions involves multiple iterations, which renders KAN significantly slower than MLP, thereby increasing the cost associated with model training and deployment. The authors of KAN have also noted that ``the biggest bottleneck of KANs lies in its slow training. KANs are usually 10x slower than MLPs, given the same number of parameters.'' To address this issue, we propose a novel MLP-type neural network PowerMLP that employs simpler non-iterative spline function representation, offering approximately the same training time as MLP while theoretically demonstrating stronger expressive power than KAN. Furthermore, we compare the FLOPs of KAN and PowerMLP, quantifying the faster computation speed of PowerMLP. Our comprehensive experiments demonstrate that PowerMLP generally achieves higher accuracy and a training speed about 40 times faster than KAN in various tasks.
       


### [Granger Causality Detection with Kolmogorov-Arnold Networks](https://arxiv.org/abs/2412.15373)

**Authors:**
Hongyu Lin, Mohan Ren, Paolo Barucca, Tomaso Aste

**Citation Count:**
2

**Abstract:**
Discovering causal relationships in time series data is central in many scientific areas, ranging from economics to climate science. Granger causality is a powerful tool for causality detection. However, its original formulation is limited by its linear form and only recently nonlinear machine-learning generalizations have been introduced. This study contributes to the definition of neural Granger causality models by investigating the application of Kolmogorov-Arnold networks (KANs) in Granger causality detection and comparing their capabilities against multilayer perceptrons (MLP). In this work, we develop a framework called Granger Causality KAN (GC-KAN) along with a tailored training approach designed specifically for Granger causality detection. We test this framework on both Vector Autoregressive (VAR) models and chaotic Lorenz-96 systems, analysing the ability of KANs to sparsify input features by identifying Granger causal relationships, providing a concise yet accurate model for Granger causality detection. Our findings show the potential of KANs to outperform MLPs in discerning interpretable Granger causal relationships, particularly for the ability of identifying sparse Granger causality patterns in high-dimensional settings, and more generally, the potential of AI in causality discovery for the dynamical laws in physical systems.
       


### [Scattering-Based Structural Inversion of Soft Materials via Kolmogorov-Arnold Networks](https://arxiv.org/abs/2412.15474)

**Authors:**
Chi-Huan Tung, Lijie Ding, Ming-Ching Chang, Guan-Rong Huang, Lionel Porcar, Yangyang Wang, Jan-Michael Y. Carrillo, Bobby G. Sumpter, Yuya Shinohara, Changwoo Do, Wei-Ren Chen

**Venue:**
Journal of Chemical Physics

**Citation Count:**
2

**Abstract:**
Small-angle scattering (SAS) techniques are indispensable tools for probing the structure of soft materials. However, traditional analytical models often face limitations in structural inversion for complex systems, primarily due to the absence of closed-form expressions of scattering functions. To address these challenges, we present a machine learning framework based on the Kolmogorov-Arnold Network (KAN) for directly extracting real-space structural information from scattering spectra in reciprocal space. This model-independent, data-driven approach provides a versatile solution for analyzing intricate configurations in soft matter. By applying the KAN to lyotropic lamellar phases and colloidal suspensions -- two representative soft matter systems -- we demonstrate its ability to accurately and efficiently resolve structural collectivity and complexity. Our findings highlight the transformative potential of machine learning in enhancing the quantitative analysis of soft materials, paving the way for robust structural inversion across diverse systems.
       


### [KKANs: Kurkova-Kolmogorov-Arnold Networks and Their Learning Dynamics](https://arxiv.org/abs/2412.16738)

**Authors:**
Juan Diego Toscano, Li-Lian Wang, George Em Karniadakis

**Citation Count:**
4

**Abstract:**
Inspired by the Kolmogorov-Arnold representation theorem and Kurkova's principle of using approximate representations, we propose the Kurkova-Kolmogorov-Arnold Network (KKAN), a new two-block architecture that combines robust multi-layer perceptron (MLP) based inner functions with flexible linear combinations of basis functions as outer functions. We first prove that KKAN is a universal approximator, and then we demonstrate its versatility across scientific machine-learning applications, including function regression, physics-informed machine learning (PIML), and operator-learning frameworks. The benchmark results show that KKANs outperform MLPs and the original Kolmogorov-Arnold Networks (KANs) in function approximation and operator learning tasks and achieve performance comparable to fully optimized MLPs for PIML. To better understand the behavior of the new representation models, we analyze their geometric complexity and learning dynamics using information bottleneck theory, identifying three universal learning stages, fitting, transition, and diffusion, across all types of architectures. We find a strong correlation between geometric complexity and signal-to-noise ratio (SNR), with optimal generalization achieved during the diffusion stage. Additionally, we propose self-scaled residual-based attention weights to maintain high SNR dynamically, ensuring uniform convergence and prolonged learning.
       


### [From KAN to GR-KAN: Advancing Speech Enhancement with KAN-Based Methodology](https://arxiv.org/abs/2412.17778)

**Authors:**
Haoyang Li, Yuchen Hu, Chen Chen, Sabato Marco Siniscalchi, Songting Liu, Eng Siong Chng

**Citation Count:**
1

**Abstract:**
Deep neural network (DNN)-based speech enhancement (SE) usually uses conventional activation functions, which lack the expressiveness to capture complex multiscale structures needed for high-fidelity SE. Group-Rational KAN (GR-KAN), a variant of Kolmogorov-Arnold Networks (KAN), retains KAN's expressiveness while improving scalability on complex tasks. We adapt GR-KAN to existing DNN-based SE by replacing dense layers with GR-KAN layers in the time-frequency (T-F) domain MP-SENet and adapting GR-KAN's activations into the 1D CNN layers in the time-domain Demucs. Results on Voicebank-DEMAND show that GR-KAN requires up to 4x fewer parameters while improving PESQ by up to 0.1. In contrast, KAN, facing scalability issues, outperforms MLP on a small-scale signal modeling task but fails to improve MP-SENet. We demonstrate the first successful use of KAN-based methods for consistent improvement in both time- and SoTA TF-domain SE, establishing GR-KAN as a promising alternative for SE.
       


### [Zero Shot Time Series Forecasting Using Kolmogorov Arnold Networks](https://arxiv.org/abs/2412.17853)

**Authors:**
Abhiroop Bhattacharya, Nandinee Haq

**Abstract:**
Accurate energy price forecasting is crucial for participants in day-ahead energy markets, as it significantly influences their decision-making processes. While machine learning-based approaches have shown promise in enhancing these forecasts, they often remain confined to the specific markets on which they are trained, thereby limiting their adaptability to new or unseen markets. In this paper, we introduce a cross-domain adaptation model designed to forecast energy prices by learning market-invariant representations across different markets during the training phase. We propose a doubly residual N-BEATS network with Kolmogorov Arnold networks at its core for time series forecasting. These networks, grounded in the Kolmogorov-Arnold representation theorem, offer a powerful way to approximate multivariate continuous functions. The cross domain adaptation model was generated with an adversarial framework. The model's effectiveness was tested in predicting day-ahead electricity prices in a zero shot fashion. In comparison with baseline models, our proposed framework shows promising results. By leveraging the Kolmogorov-Arnold networks, our model can potentially enhance its ability to capture complex patterns in energy price data, thus improving forecast accuracy across diverse market conditions. This addition not only enriches the model's representational capacity but also contributes to a more robust and flexible forecasting tool adaptable to various energy markets.
       


### [ProKAN: Progressive Stacking of Kolmogorov-Arnold Networks for Efficient Liver Segmentation](https://arxiv.org/abs/2412.19713)

**Authors:**
Bhavesh Gyanchandani, Aditya Oza, Abhinav Roy

**Citation Count:**
1

**Abstract:**
The growing need for accurate and efficient 3D identification of tumors, particularly in liver segmentation, has spurred considerable research into deep learning models. While many existing architectures offer strong performance, they often face challenges such as overfitting and excessive computational costs. An adjustable and flexible architecture that strikes a balance between time efficiency and model complexity remains an unmet requirement. In this paper, we introduce proKAN, a progressive stacking methodology for Kolmogorov-Arnold Networks (KANs) designed to address these challenges. Unlike traditional architectures, proKAN dynamically adjusts its complexity by progressively adding KAN blocks during training, based on overfitting behavior. This approach allows the network to stop growing when overfitting is detected, preventing unnecessary computational overhead while maintaining high accuracy. Additionally, proKAN utilizes KAN's learnable activation functions modeled through B-splines, which provide enhanced flexibility in learning complex relationships in 3D medical data. Our proposed architecture achieves state-of-the-art performance in liver segmentation tasks, outperforming standard Multi-Layer Perceptrons (MLPs) and fixed KAN architectures. The dynamic nature of proKAN ensures efficient training times and high accuracy without the risk of overfitting. Furthermore, proKAN provides better interpretability by allowing insight into the decision-making process through its learnable coefficients. The experimental results demonstrate a significant improvement in accuracy, Dice score, and time efficiency, making proKAN a compelling solution for 3D medical image segmentation tasks.
       


### [Gamma-Ray Burst Light Curve Reconstruction: A Comparative Machine and Deep Learning Analysis](https://arxiv.org/abs/2412.20091)

**Authors:**
A. Manchanda, A. Kaushal, M. G. Dainotti, A. Deepu, S. Naqi, J. Felix, N. Indoriya, S. P. Magesh, H. Gupta, K. Gupta, A. Madhan, D. H. Hartmann, A. Pollo, M. Bogdan, J. X. Prochaska, N. Fraija, D. Debnath

**Abstract:**
Gamma-Ray Bursts (GRBs), observed at large redshifts, are probes of the evolution of the Universe and can be used as cosmological tools. To this end, we need tight (with small dispersion) correlations among key parameters. To reduce such a dispersion, we will mitigate gaps in light curves (LCs), including the plateau region, key to building the two-dimensional Dainotti relation between the end time of plateau emission (Ta) to its luminosity (La). We reconstruct LCs using nine models: Multi-Layer Perceptron (MLP), Bi-Mamba, Fourier Transform, Gaussian Process-Random Forest Hybrid (GP-RF), Bidirectional Long Short-Term Memory (Bi-LSTM), Conditional GAN (CGAN), SARIMAX-based Kalman filter, Kolmogorov-Arnold Networks (KANs), and Attention U-Net. These methods are compared to the Willingale model (W07) over a sample of 545 GRBs. MLP and Bi-Mamba outperform other methods, with MLP reducing the plateau parameter uncertainties by 25.9% for log Ta, 28.6% for log Fa, and 37.7% for α (the post-plateau slope in the W07 model), achieving the lowest 5-fold cross validation (CV) mean squared error (MSE) of 0.0275. Bi-Mamba achieved the lowest uncertainty of parameters, a 33.3% reduction in log Ta, a 33.6% reduction in log Fa and a 41.9% in α, but with a higher MSE of 0.130. Bi-Mamba brings the lowest outlier percentage for log Ta and log Fa (2.70%), while MLP carries α outliers to 0.900%. The other methods yield MSE values ranging from 0.0339 to 0.174. These improvements in parameter precision are needed to use GRBs as standard candles, investigate theoretical models, and predict GRB redshifts through machine learning.
       


### [Advancing Parkinson's Disease Progression Prediction: Comparing Long Short-Term Memory Networks and Kolmogorov-Arnold Networks](https://arxiv.org/abs/2412.20744)

**Authors:**
Abhinav Roy, Bhavesh Gyanchandani, Aditya Oza, Abhishek Sharma

**Abstract:**
Parkinson's Disease (PD) is a degenerative neurological disorder that impairs motor and non-motor functions, significantly reducing quality of life and increasing mortality risk. Early and accurate detection of PD progression is vital for effective management and improved patient outcomes. Current diagnostic methods, however, are often costly, time-consuming, and require specialized equipment and expertise. This work proposes an innovative approach to predicting PD progression using regression methods, Long Short-Term Memory (LSTM) networks, and Kolmogorov Arnold Networks (KAN). KAN, utilizing spline-parametrized univariate functions, allows for dynamic learning of activation patterns, unlike traditional linear models.
  The Movement Disorder Society-Sponsored Revision of the Unified Parkinson's Disease Rating Scale (MDS-UPDRS) is a comprehensive tool for evaluating PD symptoms and is commonly used to measure disease progression. Additionally, protein or peptide abnormalities are linked to PD onset and progression. Identifying these associations can aid in predicting disease progression and understanding molecular changes.
  Comparing multiple models, including LSTM and KAN, this study aims to identify the method that delivers the highest metrics. The analysis reveals that KAN, with its dynamic learning capabilities, outperforms other approaches in predicting PD progression. This research highlights the potential of AI and machine learning in healthcare, paving the way for advanced computational models to enhance clinical predictions and improve patient care and treatment strategies in PD management.
       


# 2025
## January
### [Predicting Crack Nucleation and Propagation in Brittle Materials Using Deep Operator Networks with Diverse Trunk Architectures](https://arxiv.org/abs/2501.00016)

**Authors:**
Elham Kiyani, Manav Manav, Nikhil Kadivar, Laura De Lorenzis, George Em Karniadakis

**Venue:**
Computer Methods in Applied Mechanics and Engineering

**Citation Count:**
3

**Abstract:**
Phase-field modeling reformulates fracture problems as energy minimization problems and enables a comprehensive characterization of the fracture process, including crack nucleation, propagation, merging, and branching, without relying on ad-hoc assumptions. However, the numerical solution of phase-field fracture problems is characterized by a high computational cost. To address this challenge, in this paper, we employ a deep neural operator (DeepONet) consisting of a branch network and a trunk network to solve brittle fracture problems. We explore three distinct approaches that vary in their trunk network configurations. In the first approach, we demonstrate the effectiveness of a two-step DeepONet, which results in a simplification of the learning task. In the second approach, we employ a physics-informed DeepONet, whereby the mathematical expression of the energy is integrated into the trunk network's loss to enforce physical consistency. The integration of physics also results in a substantially smaller data size needed for training. In the third approach, we replace the neural network in the trunk with a Kolmogorov-Arnold Network and train it without the physics loss. Using these methods, we model crack nucleation in a one-dimensional homogeneous bar under prescribed end displacements, as well as crack propagation and branching in single edge-notched specimens with varying notch lengths subjected to tensile and shear loading. We show that the networks predict the solution fields accurately, and the error in the predicted fields is localized near the crack.
       


### [KAE: Kolmogorov-Arnold Auto-Encoder for Representation Learning](https://arxiv.org/abs/2501.00420)

**Authors:**
Fangchen Yu, Ruilizhen Hu, Yidong Lin, Yuqi Ma, Zhenghao Huang, Wenye Li

**Abstract:**
The Kolmogorov-Arnold Network (KAN) has recently gained attention as an alternative to traditional multi-layer perceptrons (MLPs), offering improved accuracy and interpretability by employing learnable activation functions on edges. In this paper, we introduce the Kolmogorov-Arnold Auto-Encoder (KAE), which integrates KAN with autoencoders (AEs) to enhance representation learning for retrieval, classification, and denoising tasks. Leveraging the flexible polynomial functions in KAN layers, KAE captures complex data patterns and non-linear relationships. Experiments on benchmark datasets demonstrate that KAE improves latent representation quality, reduces reconstruction errors, and achieves superior performance in downstream tasks such as retrieval, classification, and denoising, compared to standard autoencoders and other KAN variants. These results suggest KAE's potential as a useful tool for representation learning. Our code is available at \url{https://github.com/SciYu/KAE/}.
       


### [KAN KAN Buff Signed Graph Neural Networks?](https://arxiv.org/abs/2501.00709)

**Authors:**
Muhieddine Shebaro, Jelena Tešić

**Abstract:**
Graph Representation Learning aims to create effective embeddings for nodes and edges that encapsulate their features and relationships. Graph Neural Networks (GNNs) leverage neural networks to model complex graph structures. Recently, the Kolmogorov-Arnold Neural Network (KAN) has emerged as a promising alternative to the traditional Multilayer Perceptron (MLP), offering improved accuracy and interpretability with fewer parameters. In this paper, we propose the integration of KANs into Signed Graph Convolutional Networks (SGCNs), leading to the development of KAN-enhanced SGCNs (KASGCN). We evaluate KASGCN on tasks such as signed community detection and link sign prediction to improve embedding quality in signed networks. Our experimental results indicate that KASGCN exhibits competitive or comparable performance to standard SGCNs across the tasks evaluated, with performance variability depending on the specific characteristics of the signed graph and the choice of parameter settings. These findings suggest that KASGCNs hold promise for enhancing signed graph analysis with context-dependent effectiveness.
       


### [EHCTNet: Enhanced Hybrid of CNN and Transformer Network for Remote Sensing Image Change Detection](https://arxiv.org/abs/2501.01238)

**Authors:**
Junjie Yang, Haibo Wan, Zhihai Shang

**Citation Count:**
1

**Abstract:**
Remote sensing (RS) change detection incurs a high cost because of false negatives, which are more costly than false positives. Existing frameworks, struggling to improve the Precision metric to reduce the cost of false positive, still have limitations in focusing on the change of interest, which leads to missed detections and discontinuity issues. This work tackles these issues by enhancing feature learning capabilities and integrating the frequency components of feature information, with a strategy to incrementally boost the Recall value. We propose an enhanced hybrid of CNN and Transformer network (EHCTNet) for effectively mining the change information of interest. Firstly, a dual branch feature extraction module is used to extract the multi scale features of RS images. Secondly, the frequency component of these features is exploited by a refined module I. Thirdly, an enhanced token mining module based on the Kolmogorov Arnold Network is utilized to derive semantic information. Finally, the semantic change information's frequency component, beneficial for final detection, is mined from the refined module II. Extensive experiments validate the effectiveness of EHCTNet in comprehending complex changes of interest. The visualization outcomes show that EHCTNet detects more intact and continuous changed areas and perceives more accurate neighboring distinction than state of the art models.
       


### [Improved Feature Extraction Network for Neuro-Oriented Target Speaker Extraction](https://arxiv.org/abs/2501.01673)

**Authors:**
Cunhang Fan, Youdian Gao, Zexu Pan, Jingjing Zhang, Hongyu Zhang, Jie Zhang, Zhao Lv

**Venue:**
IEEE International Conference on Acoustics, Speech, and Signal Processing

**Citation Count:**
1

**Abstract:**
The recent rapid development of auditory attention decoding (AAD) offers the possibility of using electroencephalography (EEG) as auxiliary information for target speaker extraction. However, effectively modeling long sequences of speech and resolving the identity of the target speaker from EEG signals remains a major challenge. In this paper, an improved feature extraction network (IFENet) is proposed for neuro-oriented target speaker extraction, which mainly consists of a speech encoder with dual-path Mamba and an EEG encoder with Kolmogorov-Arnold Networks (KAN). We propose SpeechBiMamba, which makes use of dual-path Mamba in modeling local and global speech sequences to extract speech features. In addition, we propose EEGKAN to effectively extract EEG features that are closely related to the auditory stimuli and locate the target speaker through the subject's attention information. Experiments on the KUL and AVED datasets show that IFENet outperforms the state-of-the-art model, achieving 36\% and 29\% relative improvements in terms of scale-invariant signal-to-distortion ratio (SI-SDR) under an open evaluation condition.
       


### [KM-UNet KAN Mamba UNet for medical image segmentation](https://arxiv.org/abs/2501.02559)

**Author:**
Yibo Zhang

**Abstract:**
Medical image segmentation is a critical task in medical imaging analysis. Traditional CNN-based methods struggle with modeling long-range dependencies, while Transformer-based models, despite their success, suffer from quadratic computational complexity. To address these limitations, we propose KM-UNet, a novel U-shaped network architecture that combines the strengths of Kolmogorov-Arnold Networks (KANs) and state-space models (SSMs). KM-UNet leverages the Kolmogorov-Arnold representation theorem for efficient feature representation and SSMs for scalable long-range modeling, achieving a balance between accuracy and computational efficiency. We evaluate KM-UNet on five benchmark datasets: ISIC17, ISIC18, CVC, BUSI, and GLAS. Experimental results demonstrate that KM-UNet achieves competitive performance compared to state-of-the-art methods in medical image segmentation tasks. To the best of our knowledge, KM-UNet is the first medical image segmentation framework integrating KANs and SSMs. This work provides a valuable baseline and new insights for the development of more efficient and interpretable medical image segmentation systems. The code is open source at https://github.com/2760613195/KM_UNet
  Keywords:KAN,Manba, state-space models,UNet, Medical image segmentation, Deep learning
       


### [LWFNet: Coherent Doppler Wind Lidar-Based Network for Wind Field Retrieval](https://arxiv.org/abs/2501.02613)

**Authors:**
Ran Tao, Chong Wang, Hao Chen, Mingjiao Jia, Xiang Shang, Luoyuan Qu, Guoliang Shentu, Yanyu Lu, Yanfeng Huo, Lei Bai, Xianghui Xue, Xiankang Dou

**Abstract:**
Accurate detection of wind fields within the troposphere is essential for atmospheric dynamics research and plays a crucial role in extreme weather forecasting. Coherent Doppler wind lidar (CDWL) is widely regarded as the most suitable technique for high spatial and temporal resolution wind field detection. However, since coherent detection relies heavily on the concentration of aerosol particles, which cause Mie scattering, the received backscattering lidar signal exhibits significantly low intensity at high altitudes. As a result, conventional methods, such as spectral centroid estimation, often fail to produce credible and accurate wind retrieval results in these regions. To address this issue, we propose LWFNet, the first Lidar-based Wind Field (WF) retrieval neural Network, built upon Transformer and the Kolmogorov-Arnold network. Our model is trained solely on targets derived from the traditional wind retrieval algorithm and utilizes radiosonde measurements as the ground truth for test results evaluation. Experimental results demonstrate that LWFNet not only extends the maximum wind field detection range but also produces more accurate results, exhibiting a level of precision that surpasses the labeled targets. This phenomenon, which we refer to as super-accuracy, is explored by investigating the potential underlying factors that contribute to this intriguing occurrence. In addition, we compare the performance of LWFNet with other state-of-the-art (SOTA) models, highlighting its superior effectiveness and capability in high-resolution wind retrieval. LWFNet demonstrates remarkable performance in lidar-based wind field retrieval, setting a benchmark for future research and advancing the development of deep learning models in this domain.
       


### [Scaled-cPIKANs: Domain Scaling in Chebyshev-based Physics-informed Kolmogorov-Arnold Networks](https://arxiv.org/abs/2501.02762)

**Authors:**
Farinaz Mostajeran, Salah A Faroughi

**Citation Count:**
1

**Abstract:**
Partial Differential Equations (PDEs) are integral to modeling many scientific and engineering problems. Physics-informed Neural Networks (PINNs) have emerged as promising tools for solving PDEs by embedding governing equations into the neural network loss function. However, when dealing with PDEs characterized by strong oscillatory dynamics over large computational domains, PINNs based on Multilayer Perceptrons (MLPs) often exhibit poor convergence and reduced accuracy. To address these challenges, this paper introduces Scaled-cPIKAN, a physics-informed architecture rooted in Kolmogorov-Arnold Networks (KANs). Scaled-cPIKAN integrates Chebyshev polynomial representations with a domain scaling approach that transforms spatial variables in PDEs into the standardized domain \([-1,1]^d\), as intrinsically required by Chebyshev polynomials. By combining the flexibility of Chebyshev-based KANs (cKANs) with the physics-driven principles of PINNs, and the spatial domain transformation, Scaled-cPIKAN enables efficient representation of oscillatory dynamics across extended spatial domains while improving computational performance. We demonstrate Scaled-cPIKAN efficacy using four benchmark problems: the diffusion equation, the Helmholtz equation, the Allen-Cahn equation, as well as both forward and inverse formulations of the reaction-diffusion equation (with and without noisy data). Our results show that Scaled-cPIKAN significantly outperforms existing methods in all test cases. In particular, it achieves several orders of magnitude higher accuracy and faster convergence rate, making it a highly efficient tool for approximating PDE solutions that feature oscillatory behavior over large spatial domains.
       


### [DAREK -- Distance Aware Error for Kolmogorov Networks](https://arxiv.org/abs/2501.04757)

**Authors:**
Masoud Ataei, Mohammad Javad Khojasteh, Vikas Dhiman

**Venue:**
IEEE International Conference on Acoustics, Speech, and Signal Processing

**Abstract:**
In this paper, we provide distance-aware error bounds for Kolmogorov Arnold Networks (KANs). We call our new error bounds estimator DAREK -- Distance Aware Error for Kolmogorov networks. Z. Liu et al. provide error bounds, which may be loose, lack distance-awareness, and are defined only up to an unknown constant of proportionality. We review the error bounds for Newton's polynomial, which is then generalized to an arbitrary spline, under Lipschitz continuity assumptions. We then extend these bounds to nested compositions of splines, arriving at error bounds for KANs. We evaluate our method by estimating an object's shape from sparse laser scan points. We use KAN to fit a smooth function to the scans and provide error bounds for the fit. We find that our method is faster than Monte Carlo approaches, and that our error bounds enclose the true obstacle shape reliably.
       


### [Interpretable deep learning illuminates multiple structures fluorescence imaging: a path toward trustworthy artificial intelligence in microscopy](https://arxiv.org/abs/2501.05490)

**Authors:**
Mingyang Chen, Luhong Jin, Xuwei Xuan, Defu Yang, Yun Cheng, Ju Zhang

**Abstract:**
Live-cell imaging of multiple subcellular structures is essential for understanding subcellular dynamics. However, the conventional multi-color sequential fluorescence microscopy suffers from significant imaging delays and limited number of subcellular structure separate labeling, resulting in substantial limitations for real-time live-cell research applications. Here, we present the Adaptive Explainable Multi-Structure Network (AEMS-Net), a deep-learning framework that enables simultaneous prediction of two subcellular structures from a single image. The model normalizes staining intensity and prioritizes critical image features by integrating attention mechanisms and brightness adaptation layers. Leveraging the Kolmogorov-Arnold representation theorem, our model decomposes learned features into interpretable univariate functions, enhancing the explainability of complex subcellular morphologies. We demonstrate that AEMS-Net allows real-time recording of interactions between mitochondria and microtubules, requiring only half the conventional sequential-channel imaging procedures. Notably, this approach achieves over 30% improvement in imaging quality compared to traditional deep learning methods, establishing a new paradigm for long-term, interpretable live-cell imaging that advances the ability to explore subcellular dynamics.
       


### [Kolmogorov-Arnold networks for metal surface defect classification](https://arxiv.org/abs/2501.06389)

**Authors:**
Maciej Krzywda, Mariusz Wermiński, Szymon Łukasik, Amir H. Gandomi

**Abstract:**
This paper presents the application of Kolmogorov-Arnold Networks (KAN) in classifying metal surface defects. Specifically, steel surfaces are analyzed to detect defects such as cracks, inclusions, patches, pitted surfaces, and scratches. Drawing on the Kolmogorov-Arnold theorem, KAN provides a novel approach compared to conventional multilayer perceptrons (MLPs), facilitating more efficient function approximation by utilizing spline functions. The results show that KAN networks can achieve better accuracy than convolutional neural networks (CNNs) with fewer parameters, resulting in faster convergence and improved performance in image classification.
       


### [Kolmogorov-Arnold Recurrent Network for Short Term Load Forecasting Across Diverse Consumers](https://arxiv.org/abs/2501.06965)

**Authors:**
Muhammad Umair Danish, Katarina Grolinger

**Venue:**
Energy Reports

**Citation Count:**
7

**Abstract:**
Load forecasting plays a crucial role in energy management, directly impacting grid stability, operational efficiency, cost reduction, and environmental sustainability. Traditional Vanilla Recurrent Neural Networks (RNNs) face issues such as vanishing and exploding gradients, whereas sophisticated RNNs such as LSTMs have shown considerable success in this domain. However, these models often struggle to accurately capture complex and sudden variations in energy consumption, and their applicability is typically limited to specific consumer types, such as offices or schools. To address these challenges, this paper proposes the Kolmogorov-Arnold Recurrent Network (KARN), a novel load forecasting approach that combines the flexibility of Kolmogorov-Arnold Networks with RNN's temporal modeling capabilities. KARN utilizes learnable temporal spline functions and edge-based activations to better model non-linear relationships in load data, making it adaptable across a diverse range of consumer types. The proposed KARN model was rigorously evaluated on a variety of real-world datasets, including student residences, detached homes, a home with electric vehicle charging, a townhouse, and industrial buildings. Across all these consumer categories, KARN consistently outperformed traditional Vanilla RNNs, while it surpassed LSTM and Gated Recurrent Units (GRUs) in six buildings. The results demonstrate KARN's superior accuracy and applicability, making it a promising tool for enhancing load forecasting in diverse energy management scenarios.
       


### [UNetVL: Enhancing 3D Medical Image Segmentation with Chebyshev KAN Powered Vision-LSTM](https://arxiv.org/abs/2501.07017)

**Authors:**
Xuhui Guo, Tanmoy Dam, Rohan Dhamdhere, Gourav Modanwal, Anant Madabhushi

**Venue:**
IEEE International Symposium on Biomedical Imaging

**Abstract:**
3D medical image segmentation has progressed considerably due to Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs), yet these methods struggle to balance long-range dependency acquisition with computational efficiency. To address this challenge, we propose UNETVL (U-Net Vision-LSTM), a novel architecture that leverages recent advancements in temporal information processing. UNETVL incorporates Vision-LSTM (ViL) for improved scalability and memory functions, alongside an efficient Chebyshev Kolmogorov-Arnold Networks (KAN) to handle complex and long-range dependency patterns more effectively. We validated our method on the ACDC and AMOS2022 (post challenge Task 2) benchmark datasets, showing a significant improvement in mean Dice score compared to recent state-of-the-art approaches, especially over its predecessor, UNETR, with increases of 7.3% on ACDC and 15.6% on AMOS, respectively. Extensive ablation studies were conducted to demonstrate the impact of each component in UNETVL, providing a comprehensive understanding of its architecture. Our code is available at https://github.com/tgrex6/UNETVL, facilitating further research and applications in this domain.
       


### [PRKAN: Parameter-Reduced Kolmogorov-Arnold Networks](https://arxiv.org/abs/2501.07032)

**Authors:**
Hoang-Thang Ta, Duy-Quy Thai, Anh Tran, Grigori Sidorov, Alexander Gelbukh

**Citation Count:**
2

**Abstract:**
Kolmogorov-Arnold Networks (KANs) represent an innovation in neural network architectures, offering a compelling alternative to Multi-Layer Perceptrons (MLPs) in models such as Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and Transformers. By advancing network design, KANs drive groundbreaking research and enable transformative applications across various scientific domains involving neural networks. However, existing KANs often require significantly more parameters in their network layers than MLPs. To address this limitation, this paper introduces PRKANs (Parameter-Reduced Kolmogorov-Arnold Networks), which employ several methods to reduce the parameter count in KAN layers, making them comparable to MLP layers. Experimental results on the MNIST and Fashion-MNIST datasets demonstrate that PRKANs outperform several existing KANs, and their variant with attention mechanisms rivals the performance of MLPs, albeit with slightly longer training times. Furthermore, the study highlights the advantages of Gaussian Radial Basis Functions (GRBFs) and layer normalization in KAN designs. The repository for this work is available at: https://github.com/hoangthangta/All-KAN.
       


### [Kolmogorov-Arnold Network for Remote Sensing Image Semantic Segmentation](https://arxiv.org/abs/2501.07390)

**Authors:**
Xianping Ma, Ziyao Wang, Yin Hu, Xiaokang Zhang, Man-On Pun

**Abstract:**
Semantic segmentation plays a crucial role in remote sensing applications, where the accurate extraction and representation of features are essential for high-quality results. Despite the widespread use of encoder-decoder architectures, existing methods often struggle with fully utilizing the high-dimensional features extracted by the encoder and efficiently recovering detailed information during decoding. To address these problems, we propose a novel semantic segmentation network, namely DeepKANSeg, including two key innovations based on the emerging Kolmogorov Arnold Network (KAN). Notably, the advantage of KAN lies in its ability to decompose high-dimensional complex functions into univariate transformations, enabling efficient and flexible representation of intricate relationships in data. First, we introduce a KAN-based deep feature refinement module, namely DeepKAN to effectively capture complex spatial and rich semantic relationships from high-dimensional features. Second, we replace the traditional multi-layer perceptron (MLP) layers in the global-local combined decoder with KAN-based linear layers, namely GLKAN. This module enhances the decoder's ability to capture fine-grained details during decoding. To evaluate the effectiveness of the proposed method, experiments are conducted on two well-known fine-resolution remote sensing benchmark datasets, namely ISPRS Vaihingen and ISPRS Potsdam. The results demonstrate that the KAN-enhanced segmentation model achieves superior performance in terms of accuracy compared to state-of-the-art methods. They highlight the potential of KANs as a powerful alternative to traditional architectures in semantic segmentation tasks. Moreover, the explicit univariate decomposition provides improved interpretability, which is particularly beneficial for applications requiring explainable learning in remote sensing.
       


### [Kolmogorov-Arnold Networks and Evolutionary Game Theory for More Personalized Cancer Treatment](https://arxiv.org/abs/2501.07611)

**Authors:**
Sepinoud Azimi, Louise Spekking, Kateřina Staňková

**Citation Count:**
1

**Abstract:**
Personalized cancer treatment is revolutionizing oncology by leveraging precision medicine and advanced computational techniques to tailor therapies to individual patients. Despite its transformative potential, challenges such as limited generalizability, interpretability, and reproducibility of predictive models hinder its integration into clinical practice. Current methodologies often rely on black-box machine learning models, which, while accurate, lack the transparency needed for clinician trust and real-world application. This paper proposes the development of an innovative framework that bridges Kolmogorov-Arnold Networks (KANs) and Evolutionary Game Theory (EGT) to address these limitations. Inspired by the Kolmogorov-Arnold representation theorem, KANs offer interpretable, edge-based neural architectures capable of modeling complex biological systems with unprecedented adaptability. Their integration into the EGT framework enables dynamic modeling of cancer progression and treatment responses. By combining KAN's computational precision with EGT's mechanistic insights, this hybrid approach promises to enhance predictive accuracy, scalability, and clinical usability.
       


### [Scalable Bayesian Physics-Informed Kolmogorov-Arnold Networks](https://arxiv.org/abs/2501.08501)

**Authors:**
Zhiwei Gao, George Em Karniadakis

**Citation Count:**
1

**Abstract:**
Uncertainty quantification (UQ) plays a pivotal role in scientific machine learning, especially when surrogate models are used to approximate complex systems. Although multilayer perceptions (MLPs) are commonly employed as surrogates, they often suffer from overfitting due to their large number of parameters. Kolmogorov-Arnold networks (KANs) offer an alternative solution with fewer parameters. However, gradient-based inference methods, such as Hamiltonian Monte Carlo (HMC), may result in computational inefficiency when applied to KANs, especially for large-scale datasets, due to the high cost of back-propagation. To address these challenges, we propose a novel approach, combining the dropout Tikhonov ensemble Kalman inversion (DTEKI) with Chebyshev KANs. This gradient-free method effectively mitigates overfitting and enhances numerical stability. Additionally, we incorporate the active subspace method to reduce the parameter-space dimensionality, allowing us to improve the accuracy of predictions and obtain more reliable uncertainty estimates. Extensive experiments demonstrate the efficacy of our approach in various test cases, including scenarios with large datasets and high noise levels. Our results show that the new method achieves comparable or better accuracy, much higher efficiency as well as stability compared to HMC, in addition to scalability. Moreover, by leveraging the low-dimensional parameter subspace, our method preserves prediction accuracy while substantially reducing further the computational cost.
       


### [Kolmogorov-Arnold Networks for Time Series Granger Causality Inference](https://arxiv.org/abs/2501.08958)

**Authors:**
Meiliang Liu, Yunfang Xu, Zijin Li, Zhengye Si, Xiaoxiao Yang, Xinyue Yang, Zhiwen Zhao

**Abstract:**
We propose the Granger causality inference Kolmogorov-Arnold Networks (KANGCI), a novel architecture that extends the recently proposed Kolmogorov-Arnold Networks (KAN) to the domain of causal inference. By extracting base weights from KAN layers and incorporating the sparsity-inducing penalty and ridge regularization, KANGCI effectively infers the Granger causality from time series. Additionally, we propose an algorithm based on time-reversed Granger causality that automatically selects causal relationships with better inference performance from the original or time-reversed time series or integrates the results to mitigate spurious connectivities. Comprehensive experiments conducted on Lorenz-96, Gene regulatory networks, fMRI BOLD signals, VAR, and real-world EEG datasets demonstrate that the proposed model achieves competitive performance to state-of-the-art methods in inferring Granger causality from nonlinear, high-dimensional, and limited-sample time series.
       


### [Free-Knots Kolmogorov-Arnold Network: On the Analysis of Spline Knots and Advancing Stability](https://arxiv.org/abs/2501.09283)

**Authors:**
Liangwewi Nathan Zheng, Wei Emma Zhang, Lin Yue, Miao Xu, Olaf Maennel, Weitong Chen

**Citation Count:**
2

**Abstract:**
Kolmogorov-Arnold Neural Networks (KANs) have gained significant attention in the machine learning community. However, their implementation often suffers from poor training stability and heavy trainable parameter. Furthermore, there is limited understanding of the behavior of the learned activation functions derived from B-splines. In this work, we analyze the behavior of KANs through the lens of spline knots and derive the lower and upper bound for the number of knots in B-spline-based KANs. To address existing limitations, we propose a novel Free Knots KAN that enhances the performance of the original KAN while reducing the number of trainable parameters to match the trainable parameter scale of standard Multi-Layer Perceptrons (MLPs). Additionally, we introduce new a training strategy to ensure $C^2$ continuity of the learnable spline, resulting in smoother activation compared to the original KAN and improve the training stability by range expansion. The proposed method is comprehensively evaluated on 8 datasets spanning various domains, including image, text, time series, multimodal, and function approximation tasks. The promising results demonstrates the feasibility of KAN-based network and the effectiveness of proposed method.
       


### [Adversarial-Ensemble Kolmogorov Arnold Networks for Enhancing Indoor Wi-Fi Positioning: A Defensive Approach Against Spoofing and Signal Manipulation Attacks](https://arxiv.org/abs/2501.09609)

**Authors:**
Mitul Goswami, Romit Chatterjee, Somnath Mahato, Prasant Kumar Pattnaik

**Abstract:**
The research presents a study on enhancing the robustness of Wi-Fi-based indoor positioning systems against adversarial attacks. The goal is to improve the positioning accuracy and resilience of these systems under two attack scenarios: Wi-Fi Spoofing and Signal Strength Manipulation. Three models are developed and evaluated: a baseline model (M_Base), an adversarially trained robust model (M_Rob), and an ensemble model (M_Ens). All models utilize a Kolmogorov-Arnold Network (KAN) architecture. The robust model is trained with adversarially perturbed data, while the ensemble model combines predictions from both the base and robust models. Experimental results show that the robust model reduces positioning error by approximately 10% compared to the baseline, achieving 2.03 meters error under Wi-Fi spoofing and 2.00 meters under signal strength manipulation. The ensemble model further outperforms with errors of 2.01 meters and 1.975 meters for the respective attack types. This analysis highlights the effectiveness of adversarial training techniques in mitigating attack impacts. The findings underscore the importance of considering adversarial scenarios in developing indoor positioning systems, as improved resilience can significantly enhance the accuracy and reliability of such systems in mission-critical environments.
       


### [Boosting the Accuracy of Stock Market Prediction via Multi-Layer Hybrid MTL Structure](https://arxiv.org/abs/2501.09760)

**Author:**
Yuxi Hong

**Venue:**
International Journal of Intelligent Decision Technologies

**Abstract:**
Accurate stock market prediction provides great opportunities for informed decision-making, yet existing methods struggle with financial data's non-linear, high-dimensional, and volatile characteristics. Advanced predictive models are needed to effectively address these complexities. This paper proposes a novel multi-layer hybrid multi-task learning (MTL) framework aimed at achieving more efficient stock market predictions. It involves a Transformer encoder to extract complex correspondences between various input features, a Bidirectional Gated Recurrent Unit (BiGRU) to capture long-term temporal relationships, and a Kolmogorov-Arnold Network (KAN) to enhance the learning process. Experimental evaluations indicate that the proposed learning structure achieves great performance, with an MAE as low as 1.078, a MAPE as low as 0.012, and an R^2 as high as 0.98, when compared with other competitive networks.
       


### [KAA: Kolmogorov-Arnold Attention for Enhancing Attentive Graph Neural Networks](https://arxiv.org/abs/2501.13456)

**Authors:**
Taoran Fang, Tianhong Gao, Chunping Wang, Yihao Shang, Wei Chow, Lei Chen, Yang Yang

**Venue:**
International Conference on Learning Representations

**Citation Count:**
2

**Abstract:**
Graph neural networks (GNNs) with attention mechanisms, often referred to as attentive GNNs, have emerged as a prominent paradigm in advanced GNN models in recent years. However, our understanding of the critical process of scoring neighbor nodes remains limited, leading to the underperformance of many existing attentive GNNs. In this paper, we unify the scoring functions of current attentive GNNs and propose Kolmogorov-Arnold Attention (KAA), which integrates the Kolmogorov-Arnold Network (KAN) architecture into the scoring process. KAA enhances the performance of scoring functions across the board and can be applied to nearly all existing attentive GNNs. To compare the expressive power of KAA with other scoring functions, we introduce Maximum Ranking Distance (MRD) to quantitatively estimate their upper bounds in ranking errors for node importance. Our analysis reveals that, under limited parameters and constraints on width and depth, both linear transformation-based and MLP-based scoring functions exhibit finite expressive power. In contrast, our proposed KAA, even with a single-layer KAN parameterized by zero-order B-spline functions, demonstrates nearly infinite expressive power. Extensive experiments on both node-level and graph-level tasks using various backbone models show that KAA-enhanced scoring functions consistently outperform their original counterparts, achieving performance improvements of over 20% in some cases.
       


### [Local Control Networks (LCNs): Optimizing Flexibility in Neural Network Data Pattern Capture](https://arxiv.org/abs/2501.14000)

**Authors:**
Hy Nguyen, Duy Khoa Pham, Srikanth Thudumu, Hung Du, Rajesh Vasa, Kon Mouzakis

**Abstract:**
The widespread use of Multi-layer perceptrons (MLPs) often relies on a fixed activation function (e.g., ReLU, Sigmoid, Tanh) for all nodes within the hidden layers. While effective in many scenarios, this uniformity may limit the networks ability to capture complex data patterns. We argue that employing the same activation function at every node is suboptimal and propose leveraging different activation functions at each node to increase flexibility and adaptability. To achieve this, we introduce Local Control Networks (LCNs), which leverage B-spline functions to enable distinct activation curves at each node. Our mathematical analysis demonstrates the properties and benefits of LCNs over conventional MLPs. In addition, we demonstrate that more complex architectures, such as Kolmogorov-Arnold Networks (KANs), are unnecessary in certain scenarios, and LCNs can be a more efficient alternative. Empirical experiments on various benchmarks and datasets validate our theoretical findings. In computer vision tasks, LCNs achieve marginal improvements over MLPs and outperform KANs by approximately 5\%, while also being more computationally efficient than KANs. In basic machine learning tasks, LCNs show a 1\% improvement over MLPs and a 0.6\% improvement over KANs. For symbolic formula representation tasks, LCNs perform on par with KANs, with both architectures outperforming MLPs. Our findings suggest that diverse activations at the node level can lead to improved performance and efficiency.
       


### [Kolmogorov Arnold Neural Interpolator for Downscaling and Correcting Meteorological Fields from In-Situ Observations](https://arxiv.org/abs/2501.14404)

**Authors:**
Zili Liu, Hao Chen, Lei Bai, Wenyuan Li, Zhengxia Zou, Zhenwei Shi

**Citation Count:**
2

**Abstract:**
Obtaining accurate weather forecasts at station locations is a critical challenge due to systematic biases arising from the mismatch between multi-scale, continuous atmospheric characteristic and their discrete, gridded representations. Previous works have primarily focused on modeling gridded meteorological data, inherently neglecting the off-grid, continuous nature of atmospheric states and leaving such biases unresolved. To address this, we propose the Kolmogorov Arnold Neural Interpolator (KANI), a novel framework that redefines meteorological field representation as continuous neural functions derived from discretized grids. Grounded in the Kolmogorov Arnold theorem, KANI captures the inherent continuity of atmospheric states and leverages sparse in-situ observations to correct these biases systematically. Furthermore, KANI introduces an innovative zero-shot downscaling capability, guided by high-resolution topographic textures without requiring high-resolution meteorological fields for supervision. Experimental results across three sub-regions of the continental United States indicate that KANI achieves an accuracy improvement of 40.28% for temperature and 67.41% for wind speed, highlighting its significant improvement over traditional interpolation methods. This enables continuous neural representation of meteorological variables through neural networks, transcending the limitations of conventional grid-based representations.
       


### [Discovering Dynamics with Kolmogorov Arnold Networks: Linear Multistep Method-Based Algorithms and Error Estimation](https://arxiv.org/abs/2501.15066)

**Authors:**
Jintao Hu, Hongjiong Tian, Qian Guo

**Abstract:**
Uncovering the underlying dynamics from observed data is a critical task in various scientific fields. Recent advances have shown that combining deep learning techniques with linear multistep methods (LMMs) can be highly effective for this purpose. In this work, we propose a novel framework that integrates Kolmogorov Arnold Networks (KANs) with LMMs for the discovery and approximation of dynamical systems' vector fields. Specifically, we begin by establishing precise error bounds for two-layer B-spline KANs when approximating the governing functions of dynamical systems. Leveraging the approximation capabilities of KANs, we demonstrate that for certain families of LMMs, the total error is constrained within a specific range that accounts for both the method's step size and the network's approximation accuracy. Additionally, we analyze the difference between the numerical solution obtained from solving the ordinary differential equations with the fitted vector fields and the true solution of the dynamical system. To validate our theoretical results, we provide several numerical examples that highlight the effectiveness of our approach.
       


### [Efficiency Bottlenecks of Convolutional Kolmogorov-Arnold Networks: A Comprehensive Scrutiny with ImageNet, AlexNet, LeNet and Tabular Classification](https://arxiv.org/abs/2501.15757)

**Authors:**
Ashim Dahal, Saydul Akbar Murad, Nick Rahimi

**Abstract:**
Algorithmic level developments like Convolutional Neural Networks, transformers, attention mechanism, Retrieval Augmented Generation and so on have changed Artificial Intelligence. Recent such development was observed by Kolmogorov-Arnold Networks that suggested to challenge the fundamental concept of a Neural Network, thus change Multilayer Perceptron, and Convolutional Neural Networks. They received a good reception in terms of scientific modeling, yet had some drawbacks in terms of efficiency. In this paper, we train Convolutional Kolmogorov Arnold Networks (CKANs) with the ImageNet-1k dataset with 1.3 million images, MNIST dataset with 60k images and a tabular biological science related MoA dataset and test the promise of CKANs in terms of FLOPS, Inference Time, number of trainable parameters and training time against the accuracy, precision, recall and f-1 score they produce against the standard industry practice on CNN models. We show that the CKANs perform fair yet slower than CNNs in small size dataset like MoA and MNIST but are not nearly comparable as the dataset gets larger and more complex like the ImageNet. The code implementation of this paper can be found on the link: https://github.com/ashimdahal/Study-of-Convolutional-Kolmogorov-Arnold-networks
△ Less


### [Which Optimizer Works Best for Physics-Informed Neural Networks and Kolmogorov-Arnold Networks?](https://arxiv.org/abs/2501.16371)

**Authors:**
Elham Kiyani, Khemraj Shukla, Jorge F. Urbán, Jérôme Darbon, George Em Karniadakis

**Citation Count:**
7

**Abstract:**
Physics-Informed Neural Networks (PINNs) have revolutionized the computation of PDE solutions by integrating partial differential equations (PDEs) into the neural network's training process as soft constraints, becoming an important component of the scientific machine learning (SciML) ecosystem. More recently, physics-informed Kolmogorv-Arnold networks (PIKANs) have also shown to be effective and comparable in accuracy with PINNs. In their current implementation, both PINNs and PIKANs are mainly optimized using first-order methods like Adam, as well as quasi-Newton methods such as BFGS and its low-memory variant, L-BFGS. However, these optimizers often struggle with highly non-linear and non-convex loss landscapes, leading to challenges such as slow convergence, local minima entrapment, and (non)degenerate saddle points. In this study, we investigate the performance of Self-Scaled BFGS (SSBFGS), Self-Scaled Broyden (SSBroyden) methods and other advanced quasi-Newton schemes, including BFGS and L-BFGS with different line search strategies approaches. These methods dynamically rescale updates based on historical gradient information, thus enhancing training efficiency and accuracy. We systematically compare these optimizers -- using both PINNs and PIKANs -- on key challenging linear, stiff, multi-scale and non-linear PDEs, including the Burgers, Allen-Cahn, Kuramoto-Sivashinsky, and Ginzburg-Landau equations. Our findings provide state-of-the-art results with orders-of-magnitude accuracy improvements without the use of adaptive weights or any other enhancements typically employed in PINNs. More broadly, our results reveal insights into the effectiveness of second-order optimization strategies in significantly improving the convergence and accurate generalization of PINNs and PIKANs.
       


### [RAINER: A Robust Ensemble Learning Grid Search-Tuned Framework for Rainfall Patterns Prediction](https://arxiv.org/abs/2501.16900)

**Authors:**
Zhenqi Li, Junhao Zhong, Hewei Wang, Jinfeng Xu, Yijie Li, Jinjiang You, Jiayi Zhang, Runzhi Wu, Soumyabrata Dev

**Abstract:**
Rainfall prediction remains a persistent challenge due to the highly nonlinear and complex nature of meteorological data. Existing approaches lack systematic utilization of grid search for optimal hyperparameter tuning, relying instead on heuristic or manual selection, frequently resulting in sub-optimal results. Additionally, these methods rarely incorporate newly constructed meteorological features such as differences between temperature and humidity to capture critical weather dynamics. Furthermore, there is a lack of systematic evaluation of ensemble learning techniques and limited exploration of diverse advanced models introduced in the past one or two years. To address these limitations, we propose a robust ensemble learning grid search-tuned framework (RAINER) for rainfall prediction. RAINER incorporates a comprehensive feature engineering pipeline, including outlier removal, imputation of missing values, feature reconstruction, and dimensionality reduction via Principal Component Analysis (PCA). The framework integrates novel meteorological features to capture dynamic weather patterns and systematically evaluates non-learning mathematical-based methods and a variety of machine learning models, from weak classifiers to advanced neural networks such as Kolmogorov-Arnold Networks (KAN). By leveraging grid search for hyperparameter tuning and ensemble voting techniques, RAINER achieves promising results within real-world datasets.
       


### [A Genetic Algorithm-Based Approach for Automated Optimization of Kolmogorov-Arnold Networks in Classification Tasks](https://arxiv.org/abs/2501.17411)

**Authors:**
Quan Long, Bin Wang, Bing Xue, Mengjie Zhang

**Abstract:**
To address the issue of interpretability in multilayer perceptrons (MLPs), Kolmogorov-Arnold Networks (KANs) are introduced in 2024. However, optimizing KAN structures is labor-intensive, typically requiring manual intervention and parameter tuning. This paper proposes GA-KAN, a genetic algorithm-based approach that automates the optimization of KANs, requiring no human intervention in the design process. To the best of our knowledge, this is the first time that evolutionary computation is explored to optimize KANs automatically. Furthermore, inspired by the use of sparse connectivity in MLPs in effectively reducing the number of parameters, GA-KAN further explores sparse connectivity to tackle the challenge of extensive parameter spaces in KANs. GA-KAN is validated on two toy datasets, achieving optimal results without the manual tuning required by the original KAN. Additionally, GA-KAN demonstrates superior performance across five classification datasets, outperforming traditional methods on all datasets and providing interpretable symbolic formulae for the Wine and Iris datasets, thereby enhancing model transparency. Furthermore, GA-KAN significantly reduces the number of parameters over the standard KAN across all the five datasets. The core contributions of GA-KAN include automated optimization, a new encoding strategy, and a new decoding process, which together improve the accuracy and interpretability, and reduce the number of parameters.
       


### [Assessment of the January 2025 Los Angeles County wildfires: A multi-modal analysis of impact, response, and population exposure](https://arxiv.org/abs/2501.17880)

**Author:**
Seyd Teymoor Seydi

**Abstract:**
This study presents a comprehensive analysis of four significant California wildfires: Palisades, Eaton, Kenneth, and Hurst, examining their impacts through multiple dimensions, including land cover change, jurisdictional management, structural damage, and demographic vulnerability. Using the Chebyshev-Kolmogorov-Arnold network model applied to Sentinel-2 imagery, the extent of burned areas was mapped, ranging from 315.36 to 10,960.98 hectares. Our analysis revealed that shrubland ecosystems were consistently the most affected, comprising 57.4-75.8% of burned areas across all events. The jurisdictional assessment demonstrated varying management complexities, from singular authority (98.7% in the Palisades Fire) to distributed management across multiple agencies. A structural impact analysis revealed significant disparities between urban interface fires (Eaton: 9,869 structures; Palisades: 8,436 structures) and rural events (Kenneth: 24 structures; Hurst: 17 structures). The demographic analysis showed consistent gender distributions, with 50.9% of the population identified as female and 49.1% as male. Working-age populations made up the majority of the affected populations, ranging from 53.7% to 54.1%, with notable temporal shifts in post-fire periods. The study identified strong correlations between urban interface proximity, structural damage, and population exposure. The Palisades and Eaton fires affected over 20,000 people each, compared to fewer than 500 in rural events. These findings offer valuable insights for the development of targeted wildfire management strategies, particularly in wildland urban interface zones, and emphasize the need for age- and gender-conscious approaches in emergency response planning.
       


### [Explainable Machine Learning: An Illustration of Kolmogorov-Arnold Network Model for Airfoil Lift Prediction](https://arxiv.org/abs/2501.17896)

**Author:**
Sudhanva Kulkarni

**Abstract:**
Data science has emerged as fourth paradigm of scientific exploration. However many machine learning models operate as black boxes offering limited insight into the reasoning behind their predictions. This lack of transparency is one of the drawbacks to generate new knowledge from data. Recently Kolmogorov-Arnold Network or KAN has been proposed as an alternative model which embeds explainable AI. This study demonstrates the potential of KAN for new scientific exploration. KAN along with five other popular supervised machine learning models are applied to the well-known problem of airfoil lift prediction in aerospace engineering. Standard data generated from an earlier study on 2900 different airfoils is used. KAN performed the best with an R2 score of 96.17 percent on the test data, surpassing both the baseline model and Multi Layer Perceptron. Explainability of KAN is shown by pruning and symbolizing the model resulting in an equation for coefficient of lift in terms of input variables. The explainable information retrieved from KAN model is found to be consistent with the known physics of lift generation by airfoil thus demonstrating its potential to aid in scientific exploration.
       


### [HKAN: Hierarchical Kolmogorov-Arnold Network without Backpropagation](https://arxiv.org/abs/2501.18199)

**Authors:**
Grzegorz Dudek, Tomasz Rodak

**Abstract:**
This paper introduces the Hierarchical Kolmogorov-Arnold Network (HKAN), a novel network architecture that offers a competitive alternative to the recently proposed Kolmogorov-Arnold Network (KAN). Unlike KAN, which relies on backpropagation, HKAN adopts a randomized learning approach, where the parameters of its basis functions are fixed, and linear aggregations are optimized using least-squares regression. HKAN utilizes a hierarchical multi-stacking framework, with each layer refining the predictions from the previous one by solving a series of linear regression problems. This non-iterative training method simplifies computation and eliminates sensitivity to local minima in the loss function. Empirical results show that HKAN delivers comparable, if not superior, accuracy and stability relative to KAN across various regression tasks, while also providing insights into variable importance. The proposed approach seamlessly integrates theoretical insights with practical applications, presenting a robust and efficient alternative for neural network modeling.
       


### [Enhancing Neural Function Approximation: The XNet Outperforming KAN](https://arxiv.org/abs/2501.18959)

**Authors:**
Xin Li, Xiaotao Zheng, Zhihong Xia

**Citation Count:**
2

**Abstract:**
XNet is a single-layer neural network architecture that leverages Cauchy integral-based activation functions for high-order function approximation. Through theoretical analysis, we show that the Cauchy activation functions used in XNet can achieve arbitrary-order polynomial convergence, fundamentally outperforming traditional MLPs and Kolmogorov-Arnold Networks (KANs) that rely on increased depth or B-spline activations. Our extensive experiments on function approximation, PDE solving, and reinforcement learning demonstrate XNet's superior performance - reducing approximation error by up to 50000 times and accelerating training by up to 10 times compared to existing approaches. These results establish XNet as a highly efficient architecture for both scientific computing and AI applications.
       


## February
### [On the study of frequency control and spectral bias in Wavelet-Based Kolmogorov Arnold networks: A path to physics-informed KANs](https://arxiv.org/abs/2502.00280)

**Authors:**
Juan Daniel Meshir, Abel Palafox, Edgar Alejandro Guerrero

**Citation Count:**
4

**Abstract:**
Spectral bias, the tendency of neural networks to prioritize learning low-frequency components of functions during the initial training stages, poses a significant challenge when approximating solutions with high-frequency details. This issue is particularly pronounced in physics-informed neural networks (PINNs), widely used to solve differential equations that describe physical phenomena. In the literature, contributions such as Wavelet Kolmogorov Arnold Networks (Wav-KANs) have demonstrated promising results in capturing both low- and high-frequency components. Similarly, Fourier features (FF) are often employed to address this challenge. However, the theoretical foundations of Wav-KANs, particularly the relationship between the frequency of the mother wavelet and spectral bias, remain underexplored. A more in-depth understanding of how Wav-KANs manage high-frequency terms could offer valuable insights for addressing oscillatory phenomena encountered in parabolic, elliptic, and hyperbolic differential equations. In this work, we analyze the eigenvalues of the neural tangent kernel (NTK) of Wav-KANs to enhance their ability to converge on high-frequency components, effectively mitigating spectral bias. Our theoretical findings are validated through numerical experiments, where we also discuss the limitations of traditional approaches, such as standard PINNs and Fourier features, in addressing multi-frequency problems.
       


### [Forecasting VIX using interpretable Kolmogorov-Arnold networks](https://arxiv.org/abs/2502.00980)

**Authors:**
So-Yoon Cho, Sungchul Lee, Hyun-Gyoon Kim

**Abstract:**
This paper presents the use of Kolmogorov-Arnold Networks (KANs) for forecasting the CBOE Volatility Index (VIX). Unlike traditional MLP-based neural networks that are often criticized for their black-box nature, KAN offers an interpretable approach via learnable spline-based activation functions and symbolification. Based on a parsimonious architecture with symbolic functions, KAN expresses a forecast of the VIX as a closed-form in terms of explanatory variables, and provide interpretable insights into key characteristics of the VIX, including mean reversion and the leverage effect. Through in-depth empirical analysis across multiple datasets and periods, we show that KANs achieve competitive forecasting performance while requiring significantly fewer parameters compared to MLP-based neural network models. Our findings demonstrate the capacity and potential of KAN as an interpretable financial time-series forecasting method.
       


### [Data-Efficient Model for Psychological Resilience Prediction based on Neurological Data](https://arxiv.org/abs/2502.01377)

**Authors:**
Zhi Zhang, Yan Liu, Mengxia Gao, Yu Yang, Jiannong Cao, Wai Kai Hou, Shirley Li, Sonata Yau, Yun Kwok Wing, Tatia M. C. Lee

**Abstract:**
Psychological resilience, defined as the ability to rebound from adversity, is crucial for mental health. Compared with traditional resilience assessments through self-reported questionnaires, resilience assessments based on neurological data offer more objective results with biological markers, hence significantly enhancing credibility. This paper proposes a novel data-efficient model to address the scarcity of neurological data. We employ Neuro Kolmogorov-Arnold Networks as the structure of the prediction model. In the training stage, a new trait-informed multimodal representation algorithm with a smart chunk technique is proposed to learn the shared latent space with limited data. In the test stage, a new noise-informed inference algorithm is proposed to address the low signal-to-noise ratio of the neurological data. The proposed model not only shows impressive performance on both public datasets and self-constructed datasets but also provides some valuable psychological hypotheses for future research.
       


### [Function Approximation Using Analog Building Blocks in Flexible Electronics](https://arxiv.org/abs/2502.01489)

**Authors:**
Paula Carolina Lozano Duarte, Aradhana Dube, Georgios Zervakis, Mehdi Tahoori, Sani Nassif

**Venue:**
IEEE International Symposium on Quality Electronic Design

**Abstract:**
Function approximation is crucial in Flexible Electronics (FE), where applications demand efficient computational techniques within strict constraints on size, power, and performance. Devices like wearables and compact sensors are constrained by their limited physical dimensions and energy capacity, making traditional digital function approximation challenging and hardware-demanding. This paper addresses function approximation in FE by proposing a systematic and generic approach using a combination of Analog Building Blocks (ABBs) that perform basic mathematical operations such as addition, multiplication, and squaring. These ABBs serve as the foundation for constructing splines, which are then employed in the creation of Kolmogorov-Arnold Networks (KANs), improving the approximation. The analog realization of KAN offers a promising alternative to digital solutions, providing significant hardware benefits, particularly in terms of area and power consumption. Our design achieves a 125x reduction in area and a 10.59% power saving compared to a digital spline with 8-bit precision. Results also show that the analog design introduces an approximation error of up to 7.58% due to both the design and parasitic elements. Nevertheless, KANs are shown to be a viable candidate for function approximation in FE, with potential for further optimization to address the challenges of error reduction and hardware cost.
       


### [Efficient Denial of Service Attack Detection in IoT using Kolmogorov-Arnold Networks](https://arxiv.org/abs/2502.01835)

**Author:**
Oleksandr Kuznetsov

**Abstract:**
The proliferation of Internet of Things (IoT) devices has created a pressing need for efficient security solutions, particularly against Denial of Service (DoS) attacks. While existing detection approaches demonstrate high accuracy, they often require substantial computational resources, making them impractical for IoT deployment. This paper introduces a novel lightweight approach to DoS attack detection based on Kolmogorov-Arnold Networks (KANs). By leveraging spline-based transformations instead of traditional weight matrices, our solution achieves state-of-the-art detection performance while maintaining minimal resource requirements. Experimental evaluation on the CICIDS2017 dataset demonstrates 99.0% detection accuracy with only 0.19 MB memory footprint and 2.00 ms inference time per sample. Compared to existing solutions, KAN reduces memory requirements by up to 98% while maintaining competitive detection rates. The model's linear computational complexity ensures efficient scaling with input size, making it particularly suitable for large-scale IoT deployments. We provide comprehensive performance comparisons with recent approaches and demonstrate effectiveness across various DoS attack patterns. Our solution addresses the critical challenge of implementing sophisticated attack detection on resource-constrained devices, offering a practical approach to enhancing IoT security without compromising computational efficiency.
       


### [EFKAN: A KAN-Integrated Neural Operator For Efficient Magnetotelluric Forward Modeling](https://arxiv.org/abs/2502.02195)

**Authors:**
Feng Wang, Hong Qiu, Yingying Huang, Xiaozhe Gu, Renfang Wang, Bo Yang

**Citation Count:**
1

**Abstract:**
Magnetotelluric (MT) forward modeling is fundamental for improving the accuracy and efficiency of MT inversion. Neural operators (NOs) have been effectively used for rapid MT forward modeling, demonstrating their promising performance in solving the MT forward modeling-related partial differential equations (PDEs). Particularly, they can obtain the electromagnetic field at arbitrary locations and frequencies. In these NOs, the projection layers have been dominated by multi-layer perceptrons (MLPs), which may potentially reduce the accuracy of solution due to they usually suffer from the disadvantages of MLPs, such as lack of interpretability, overfitting, and so on. Therefore, to improve the accuracy of MT forward modeling with NOs and explore the potential alternatives to MLPs, we propose a novel neural operator by extending the Fourier neural operator (FNO) with Kolmogorov-Arnold network (EFKAN). Within the EFKAN framework, the FNO serves as the branch network to calculate the apparent resistivity and phase from the resistivity model in the frequency domain. Meanwhile, the KAN acts as the trunk network to project the resistivity and phase, determined by the FNO, to the desired locations and frequencies. Experimental results demonstrate that the proposed method not only achieves higher accuracy in obtaining apparent resistivity and phase compared to the NO equipped with MLPs at the desired frequencies and locations but also outperforms traditional numerical methods in terms of computational speed.
       


### [CVKAN: Complex-Valued Kolmogorov-Arnold Networks](https://arxiv.org/abs/2502.02417)

**Authors:**
Matthias Wolff, Florian Eilers, Xiaoyi Jiang

**Citation Count:**
1

**Abstract:**
In this work we propose CVKAN, a complex-valued Kolmogorov-Arnold Network (KAN), to join the intrinsic interpretability of KANs and the advantages of Complex-Valued Neural Networks (CVNNs). We show how to transfer a KAN and the necessary associated mechanisms into the complex domain. To confirm that CVKAN meets expectations we conduct experiments on symbolic complex-valued function fitting and physically meaningful formulae as well as on a more realistic dataset from knot theory. Our proposed CVKAN is more stable and performs on par or better than real-valued KANs while requiring less parameters and a shallower network architecture, making it more explainable.
       


### [Constitutive Kolmogorov-Arnold Networks (CKANs): Combining Accuracy and Interpretability in Data-Driven Material Modeling](https://arxiv.org/abs/2502.05682)

**Authors:**
Kian P. Abdolazizi, Roland C. Aydin, Christian J. Cyron, Kevin Linka

**Citation Count:**
3

**Abstract:**
Hybrid constitutive modeling integrates two complementary approaches for describing and predicting a material's mechanical behavior: purely data-driven black-box methods and physically constrained, theory-based models. While black-box methods offer high accuracy, they often lack interpretability and extrapolability. Conversely, physics-based models provide theoretical insight and generalizability but may not capture complex behaviors with the same accuracy. Traditionally, hybrid modeling has required a trade-off between these aspects. In this paper, we show how recent advances in symbolic machine learning, specifically Kolmogorov-Arnold Networks (KANs), help to overcome this limitation. We introduce Constitutive Kolmogorov-Arnold Networks (CKANs) as a new class of hybrid constitutive models. By incorporating a post-processing symbolification step, CKANs combine the predictive accuracy of data-driven models with the interpretability and extrapolation capabilities of symbolic expressions, bridging the gap between machine learning and physical modeling.
       


### [Kolmogorov-Arnold Fourier Networks](https://arxiv.org/abs/2502.06018)

**Authors:**
Jusheng Zhang, Yijia Fan, Kaitong Cai, Keze Wang

**Abstract:**
Although Kolmogorov-Arnold based interpretable networks (KAN) have strong theoretical expressiveness, they face significant parameter explosion and high-frequency feature capture challenges in high-dimensional tasks. To address this issue, we propose the Kolmogorov-Arnold-Fourier Network (KAF), which effectively integrates trainable Random Fourier Features (RFF) and a novel hybrid GELU-Fourier activation mechanism to balance parameter efficiency and spectral representation capabilities. Our key technical contributions include: (1) merging KAN's dual-matrix structure through matrix association properties to substantially reduce parameters; (2) introducing learnable RFF initialization strategies to eliminate spectral distortion in high-dimensional approximation tasks; (3) implementing an adaptive hybrid activation function that progressively enhances frequency representation during the training process. Comprehensive experiments demonstrate the superiority of our KAF across various domains including vision, NLP, audio processing, and differential equation-solving tasks, effectively combining theoretical interpretability with practical utility and computational efficiency.
       


### [Low Tensor-Rank Adaptation of Kolmogorov--Arnold Networks](https://arxiv.org/abs/2502.06153)

**Authors:**
Yihang Gao, Michael K. Ng, Vincent Y. F. Tan

**Abstract:**
Kolmogorov--Arnold networks (KANs) have demonstrated their potential as an alternative to multi-layer perceptions (MLPs) in various domains, especially for science-related tasks. However, transfer learning of KANs remains a relatively unexplored area. In this paper, inspired by Tucker decomposition of tensors and evidence on the low tensor-rank structure in KAN parameter updates, we develop low tensor-rank adaptation (LoTRA) for fine-tuning KANs. We study the expressiveness of LoTRA based on Tucker decomposition approximations. Furthermore, we provide a theoretical analysis to select the learning rates for each LoTRA component to enable efficient training. Our analysis also shows that using identical learning rates across all components leads to inefficient training, highlighting the need for an adaptive learning rate strategy. Beyond theoretical insights, we explore the application of LoTRA for efficiently solving various partial differential equations (PDEs) by fine-tuning KANs. Additionally, we propose Slim KANs that incorporate the inherent low-tensor-rank properties of KAN parameter tensors to reduce model size while maintaining superior performance. Experimental results validate the efficacy of the proposed learning rate selection strategy and demonstrate the effectiveness of LoTRA for transfer learning of KANs in solving PDEs. Further evaluations on Slim KANs for function representation and image classification tasks highlight the expressiveness of LoTRA and the potential for parameter reduction through low tensor-rank decomposition.
       


### [TimeKAN: KAN-based Frequency Decomposition Learning Architecture for Long-term Time Series Forecasting](https://arxiv.org/abs/2502.06910)

**Authors:**
Songtao Huang, Zhen Zhao, Can Li, Lei Bai

**Venue:**
International Conference on Learning Representations

**Abstract:**
Real-world time series often have multiple frequency components that are intertwined with each other, making accurate time series forecasting challenging. Decomposing the mixed frequency components into multiple single frequency components is a natural choice. However, the information density of patterns varies across different frequencies, and employing a uniform modeling approach for different frequency components can lead to inaccurate characterization. To address this challenges, inspired by the flexibility of the recent Kolmogorov-Arnold Network (KAN), we propose a KAN-based Frequency Decomposition Learning architecture (TimeKAN) to address the complex forecasting challenges caused by multiple frequency mixtures. Specifically, TimeKAN mainly consists of three components: Cascaded Frequency Decomposition (CFD) blocks, Multi-order KAN Representation Learning (M-KAN) blocks and Frequency Mixing blocks. CFD blocks adopt a bottom-up cascading approach to obtain series representations for each frequency band. Benefiting from the high flexibility of KAN, we design a novel M-KAN block to learn and represent specific temporal patterns within each frequency band. Finally, Frequency Mixing blocks is used to recombine the frequency bands into the original format. Extensive experimental results across multiple real-world time series datasets demonstrate that TimeKAN achieves state-of-the-art performance as an extremely lightweight architecture. Code is available at https://github.com/huangst21/TimeKAN.
       


### [MatrixKAN: Parallelized Kolmogorov-Arnold Network](https://arxiv.org/abs/2502.07176)

**Authors:**
Cale Coffman, Lizhong Chen

**Abstract:**
Kolmogorov-Arnold Networks (KAN) are a new class of neural network architecture representing a promising alternative to the Multilayer Perceptron (MLP), demonstrating improved expressiveness and interpretability. However, KANs suffer from slow training and inference speeds relative to MLPs due in part to the recursive nature of the underlying B-spline calculations. This issue is particularly apparent with respect to KANs utilizing high-degree B-splines, as the number of required non-parallelizable recursions is proportional to B-spline degree. We solve this issue by proposing MatrixKAN, a novel optimization that parallelizes B-spline calculations with matrix representation and operations, thus significantly improving effective computation time for models utilizing high-degree B-splines. In this paper, we demonstrate the superior scaling of MatrixKAN's computation time relative to B-spline degree. Further, our experiments demonstrate speedups of approximately 40x relative to KAN, with significant additional speedup potential for larger datasets or higher spline degrees.
       


### [Unpaired Image Dehazing via Kolmogorov-Arnold Transformation of Latent Features](https://arxiv.org/abs/2502.07812)

**Author:**
Le-Anh Tran

**Abstract:**
This paper proposes an innovative framework for Unsupervised Image Dehazing via Kolmogorov-Arnold Transformation, termed UID-KAT. Image dehazing is recognized as a challenging and ill-posed vision task that requires complex transformations and interpretations in the feature space. Recent advancements have introduced Kolmogorov-Arnold Networks (KANs), inspired by the Kolmogorov-Arnold representation theorem, as promising alternatives to Multi-Layer Perceptrons (MLPs) since KANs can leverage their polynomial foundation to more efficiently approximate complex functions while requiring fewer layers than MLPs. Motivated by this potential, this paper explores the use of KANs combined with adversarial training and contrastive learning to model the intricate relationship between hazy and clear images. Adversarial training is employed due to its capacity in producing high-fidelity images, and contrastive learning promotes the model's emphasis on significant features while suppressing the influence of irrelevant information. The proposed UID-KAT framework is trained in an unsupervised setting to take advantage of the abundance of real-world data and address the challenge of preparing paired hazy/clean images. Experimental results show that UID-KAT achieves state-of-the-art dehazing performance across multiple datasets and scenarios, outperforming existing unpaired methods while reducing model complexity. The source code for this work is publicly available at https://github.com/tranleanh/uid-kat.
       


### [Medical Image Classification with KAN-Integrated Transformers and Dilated Neighborhood Attention](https://arxiv.org/abs/2502.13693)

**Authors:**
Omid Nejati Manzari, Hojat Asgariandehkordi, Taha Koleilat, Yiming Xiao, Hassan Rivaz

**Citation Count:**
2

**Abstract:**
Convolutional networks, transformers, hybrid models, and Mamba-based architectures have demonstrated strong performance across various medical image classification tasks. However, these methods were primarily designed to classify clean images using labeled data. In contrast, real-world clinical data often involve image corruptions that are unique to multi-center studies and stem from variations in imaging equipment across manufacturers. In this paper, we introduce the Medical Vision Transformer (MedViTV2), a novel architecture incorporating Kolmogorov-Arnold Network (KAN) layers into the transformer architecture for the first time, aiming for generalized medical image classification. We have developed an efficient KAN block to reduce computational load while enhancing the accuracy of the original MedViT. Additionally, to counteract the fragility of our MedViT when scaled up, we propose an enhanced Dilated Neighborhood Attention (DiNA), an adaptation of the efficient fused dot-product attention kernel capable of capturing global context and expanding receptive fields to scale the model effectively and addressing feature collapse issues. Moreover, a hierarchical hybrid strategy is introduced to stack our Local Feature Perception and Global Feature Perception blocks in an efficient manner, which balances local and global feature perceptions to boost performance. Extensive experiments on 17 medical image classification datasets and 12 corrupted medical image datasets demonstrate that MedViTV2 achieved state-of-the-art results in 27 out of 29 experiments with reduced computational complexity. MedViTV2 is 44\% more computationally efficient than the previous version and significantly enhances accuracy, achieving improvements of 4.6\% on MedMNIST, 5.8\% on NonMNIST, and 13.4\% on the MedMNIST-C benchmark.
       


### [seqKAN: Sequence processing with Kolmogorov-Arnold Networks](https://arxiv.org/abs/2502.14681)

**Authors:**
Tatiana Boura, Stasinos Konstantopoulos

**Abstract:**
Kolmogorov-Arnold Networks (KANs) have been recently proposed as a machine learning framework that is more interpretable and controllable than the multi-layer perceptron. Various network architectures have been proposed within the KAN framework targeting different tasks and application domains, including sequence processing.
  This paper proposes seqKAN, a new KAN architecture for sequence processing. Although multiple sequence processing KAN architectures have already been proposed, we argue that seqKAN is more faithful to the core concept of the KAN framework. Furthermore, we empirically demonstrate that it achieves better results.
  The empirical evaluation is performed on generated data from a complex physics problem on an interpolation and an extrapolation task. Using this dataset we compared seqKAN against a prior KAN network for timeseries prediction, recurrent deep networks, and symbolic regression. seqKAN substantially outperforms all architectures, particularly on the extrapolation dataset, while also being the most transparent.
       


### [Advancing Out-of-Distribution Detection via Local Neuroplasticity](https://arxiv.org/abs/2502.15833)

**Authors:**
Alessandro Canevaro, Julian Schmidt, Mohammad Sajad Marvi, Hang Yu, Georg Martius, Julian Jordan

**Venue:**
International Conference on Learning Representations

**Abstract:**
In the domain of machine learning, the assumption that training and test data share the same distribution is often violated in real-world scenarios, requiring effective out-of-distribution (OOD) detection. This paper presents a novel OOD detection method that leverages the unique local neuroplasticity property of Kolmogorov-Arnold Networks (KANs). Unlike traditional multilayer perceptrons, KANs exhibit local plasticity, allowing them to preserve learned information while adapting to new tasks. Our method compares the activation patterns of a trained KAN against its untrained counterpart to detect OOD samples. We validate our approach on benchmarks from image and medical domains, demonstrating superior performance and robustness compared to state-of-the-art techniques. These results underscore the potential of KANs in enhancing the reliability of machine learning systems in diverse environments.
       


### [InSlicing: Interpretable Learning-Assisted Network Slice Configuration in Open Radio Access Networks](https://arxiv.org/abs/2502.15918)

**Authors:**
Ming Zhao, Yuru Zhang, Qiang Liu, Ahan Kak, Nakjung Choi

**Abstract:**
Network slicing is a key technology enabling the flexibility and efficiency of 5G networks, offering customized services for diverse applications. However, existing methods face challenges in adapting to dynamic network environments and lack interpretability in performance models. In this paper, we propose a novel interpretable network slice configuration algorithm (\emph{InSlicing}) in open radio access networks, by integrating Kolmogorov-Arnold Networks (KANs) and hybrid optimization process. On the one hand, we use KANs to approximate and learn the unknown performance function of individual slices, which converts the blackbox optimization problem. On the other hand, we solve the converted problem with a genetic method for global search and incorporate a trust region for gradient-based local refinement. With the extensive evaluation, we show that our proposed algorithm achieves high interpretability while reducing 25+\% operation cost than existing solutions.
       


### [M4SC: An MLLM-based Multi-modal, Multi-task and Multi-user Semantic Communication System](https://arxiv.org/abs/2502.16418)

**Authors:**
Feibo Jiang, Siwei Tu, Li Dong, Kezhi Wang, Kun Yang, Cunhua Pan

**Citation Count:**
3

**Abstract:**
Multi-modal Large Language Models (MLLMs) are capable of precisely extracting high-level semantic information from multi-modal data, enabling multi-task understanding and generation. This capability facilitates more efficient and intelligent data transmission in semantic communications. In this paper, we design a tailored MLLM for semantic communication and propose an MLLM-based Multi-modal, Multi-task and Multi-user Semantic Communication (M4SC) system. First, we utilize the Kolmogorov-Arnold Network (KAN) to achieve multi-modal alignment in MLLMs, thereby enhancing the accuracy of semantics representation in the semantic space across different modalities. Next, we introduce a multi-task fine-tuning approach based on task instruction following, which leverages a unified task instruction template to describe various semantic communication tasks, improving the MLLM's ability to follow instructions across multiple tasks. Additionally, by designing a semantic sharing mechanism, we transmit the public and private semantic information of multiple users separately, thus increasing the efficiency of semantic communication. Finally, we employ a joint KAN-LLM-channel coding strategy to comprehensively enhance the performance of the semantic communication system in complex communication environments. Experimental results validate the effectiveness and robustness of the proposed M4SC in multi-modal, multi-task, and multi-user scenarios.
       


### [Geometric Kolmogorov-Arnold Superposition Theorem](https://arxiv.org/abs/2502.16664)

**Authors:**
Francesco Alesiani, Takashi Maruyama, Henrik Christiansen, Viktor Zaverkin

**Abstract:**
The Kolmogorov-Arnold Theorem (KAT), or more generally, the Kolmogorov Superposition Theorem (KST), establishes that any non-linear multivariate function can be exactly represented as a finite superposition of non-linear univariate functions. Unlike the universal approximation theorem, which provides only an approximate representation without guaranteeing a fixed network size, KST offers a theoretically exact decomposition. The Kolmogorov-Arnold Network (KAN) was introduced as a trainable model to implement KAT, and recent advancements have adapted KAN using concepts from modern neural networks. However, KAN struggles to effectively model physical systems that require inherent equivariance or invariance to $E(3)$ transformations, a key property for many scientific and engineering applications. In this work, we propose a novel extension of KAT and KAN to incorporate equivariance and invariance over $O(n)$ group actions, enabling accurate and efficient modeling of these systems. Our approach provides a unified approach that bridges the gap between mathematical theory and practical architectures for physical systems, expanding the applicability of KAN to a broader class of problems.
       


### [DiffKAN-Inpainting: KAN-based Diffusion model for brain tumor inpainting](https://arxiv.org/abs/2502.16771)

**Authors:**
Tianli Tao, Ziyang Wang, Han Zhang, Theodoros N. Arvanitis, Le Zhang

**Venue:**
IEEE International Symposium on Biomedical Imaging

**Abstract:**
Brain tumors delay the standard preprocessing workflow for further examination. Brain inpainting offers a viable, although difficult, solution for tumor tissue processing, which is necessary to improve the precision of the diagnosis and treatment. Most conventional U-Net-based generative models, however, often face challenges in capturing the complex, nonlinear latent representations inherent in brain imaging. In order to accomplish high-quality healthy brain tissue reconstruction, this work proposes DiffKAN-Inpainting, an innovative method that blends diffusion models with the Kolmogorov-Arnold Networks architecture. During the denoising process, we introduce the RePaint method and tumor information to generate images with a higher fidelity and smoother margin. Both qualitative and quantitative results demonstrate that as compared to the state-of-the-art methods, our proposed DiffKAN-Inpainting inpaints more detailed and realistic reconstructions on the BraTS dataset. The knowledge gained from ablation study provide insights for future research to balance performance with computing cost.
       


### [LeanKAN: A Parameter-Lean Kolmogorov-Arnold Network Layer with Improved Memory Efficiency and Convergence Behavior](https://arxiv.org/abs/2502.17844)

**Authors:**
Benjamin C. Koenig, Suyong Kim, Sili Deng

**Citation Count:**
2

**Abstract:**
The recently proposed Kolmogorov-Arnold network (KAN) is a promising alternative to multi-layer perceptrons (MLPs) for data-driven modeling. While original KAN layers were only capable of representing the addition operator, the recently-proposed MultKAN layer combines addition and multiplication subnodes in an effort to improve representation performance. Here, we find that MultKAN layers suffer from a few key drawbacks including limited applicability in output layers, bulky parameterizations with extraneous activations, and the inclusion of complex hyperparameters. To address these issues, we propose LeanKANs, a direct and modular replacement for MultKAN and traditional AddKAN layers. LeanKANs address these three drawbacks of MultKAN through general applicability as output layers, significantly reduced parameter counts for a given network structure, and a smaller set of hyperparameters. As a one-to-one layer replacement for standard AddKAN and MultKAN layers, LeanKAN is able to provide these benefits to traditional KAN learning problems as well as augmented KAN structures in which it serves as the backbone, such as KAN Ordinary Differential Equations (KAN-ODEs) or Deep Operator KANs (DeepOKAN). We demonstrate LeanKAN's simplicity and efficiency in a series of demonstrations carried out across both a standard KAN toy problem and a KAN-ODE dynamical system modeling problem, where we find that its sparser parameterization and compact structure serve to increase its expressivity and learning capability, leading it to outperform similar and even much larger MultKANs in various tasks.
       


### [Recurrent Neural Networks for Dynamic VWAP Execution: Adaptive Trading Strategies with Temporal Kolmogorov-Arnold Networks](https://arxiv.org/abs/2502.18177)

**Author:**
Remi Genet

**Citation Count:**
1

**Abstract:**
The execution of Volume Weighted Average Price (VWAP) orders remains a critical challenge in modern financial markets, particularly as trading volumes and market complexity continue to increase. In my previous work arXiv:2502.13722, I introduced a novel deep learning approach that demonstrated significant improvements over traditional VWAP execution methods by directly optimizing the execution problem rather than relying on volume curve predictions. However, that model was static because it employed the fully linear approach described in arXiv:2410.21448, which is not designed for dynamic adjustment. This paper extends that foundation by developing a dynamic neural VWAP framework that adapts to evolving market conditions in real time. We introduce two key innovations: first, the integration of recurrent neural networks to capture complex temporal dependencies in market dynamics, and second, a sophisticated dynamic adjustment mechanism that continuously optimizes execution decisions based on market feedback. The empirical analysis, conducted across five major cryptocurrency markets, demonstrates that this dynamic approach achieves substantial improvements over both traditional methods and our previous static implementation, with execution performance gains of 10 to 15% in liquid markets and consistent outperformance across varying conditions. These results suggest that adaptive neural architectures can effectively address the challenges of modern VWAP execution while maintaining computational efficiency suitable for practical deployment.
       


### [TSKANMixer: Kolmogorov-Arnold Networks with MLP-Mixer Model for Time Series Forecasting](https://arxiv.org/abs/2502.18410)

**Authors:**
Young-Chae Hong, Bei Xiao, Yangho Chen

**Citation Count:**
2

**Abstract:**
Time series forecasting has long been a focus of research across diverse fields, including economics, energy, healthcare, and traffic management. Recent works have introduced innovative architectures for time series models, such as the Time-Series Mixer (TSMixer), which leverages multi-layer perceptrons (MLPs) to enhance prediction accuracy by effectively capturing both spatial and temporal dependencies within the data. In this paper, we investigate the capabilities of the Kolmogorov-Arnold Networks (KANs) for time-series forecasting by modifying TSMixer with a KAN layer (TSKANMixer). Experimental results demonstrate that TSKANMixer tends to improve prediction accuracy over the original TSMixer across multiple datasets, ranking among the top-performing models compared to other time series approaches. Our results show that the KANs are promising alternatives to improve the performance of time series forecasting by replacing or extending traditional MLPs.
       


### [MedKAN: An Advanced Kolmogorov-Arnold Network for Medical Image Classification](https://arxiv.org/abs/2502.18416)

**Authors:**
Zhuoqin Yang, Jiansong Zhang, Xiaoling Luo, Zheng Lu, Linlin Shen

**Citation Count:**
3

**Abstract:**
Recent advancements in deep learning for image classification predominantly rely on convolutional neural networks (CNNs) or Transformer-based architectures. However, these models face notable challenges in medical imaging, particularly in capturing intricate texture details and contextual features. Kolmogorov-Arnold Networks (KANs) represent a novel class of architectures that enhance nonlinear transformation modeling, offering improved representation of complex features. In this work, we present MedKAN, a medical image classification framework built upon KAN and its convolutional extensions. MedKAN features two core modules: the Local Information KAN (LIK) module for fine-grained feature extraction and the Global Information KAN (GIK) module for global context integration. By combining these modules, MedKAN achieves robust feature modeling and fusion. To address diverse computational needs, we introduce three scalable variants--MedKAN-S, MedKAN-B, and MedKAN-L. Experimental results on nine public medical imaging datasets demonstrate that MedKAN achieves superior performance compared to CNN- and Transformer-based models, highlighting its effectiveness and generalizability in medical image analysis.
       


### [KAN-powered large-target detection for automotive radar](https://arxiv.org/abs/2502.19000)

**Authors:**
Vinay Kulkarni, V. V. Reddy, Neha Maheshwari

**Abstract:**
This paper presents a novel radar signal detection pipeline focused on detecting large targets such as cars and SUVs. Traditional methods, such as Ordered-Statistic Constant False Alarm Rate (OS-CFAR), commonly used in automotive radar, are designed for point or isotropic target models. These may not adequately capture the Range-Doppler (RD) scattering patterns of larger targets, especially in high-resolution radar systems. Additional modules such as association and tracking are necessary to refine and consolidate the detections over multiple dwells. To address these limitations, we propose a detection technique based on the probability density function (pdf) of RD segments, leveraging the Kolmogorov-Arnold neural network (KAN) to learn the data and generate interpretable symbolic expressions for binary hypotheses. Beside the Monte-Carlo study showing better performance for the proposed KAN expression over OS-CFAR, it is shown to exhibit a probability of detection (PD) of 96% when transfer learned with field data. The false alarm rate (PFA) is comparable with OS-CFAR designed with PFA = $10^{-6}$. Additionally, the study also examines impact of the number of pdf bins representing RD segment on performance of the KAN-based detection.
       


### [Finding Local Diffusion Schrödinger Bridge using Kolmogorov-Arnold Network](https://arxiv.org/abs/2502.19754)

**Authors:**
Xingyu Qiu, Mengying Yang, Xinghua Ma, Fanding Li, Dong Liang, Gongning Luo, Wei Wang, Kuanquan Wang, Shuo Li

**Citation Count:**
2

**Abstract:**
In image generation, Schrödinger Bridge (SB)-based methods theoretically enhance the efficiency and quality compared to the diffusion models by finding the least costly path between two distributions. However, they are computationally expensive and time-consuming when applied to complex image data. The reason is that they focus on fitting globally optimal paths in high-dimensional spaces, directly generating images as next step on the path using complex networks through self-supervised training, which typically results in a gap with the global optimum. Meanwhile, most diffusion models are in the same path subspace generated by weights $f_A(t)$ and $f_B(t)$, as they follow the paradigm ($x_t = f_A(t)x_{Img} + f_B(t)ε$). To address the limitations of SB-based methods, this paper proposes for the first time to find local Diffusion Schrödinger Bridges (LDSB) in the diffusion path subspace, which strengthens the connection between the SB problem and diffusion models. Specifically, our method optimizes the diffusion paths using Kolmogorov-Arnold Network (KAN), which has the advantage of resistance to forgetting and continuous output. The experiment shows that our LDSB significantly improves the quality and efficiency of image generation using the same pre-trained denoising network and the KAN for optimising is only less than 0.1MB. The FID metric is reduced by more than 15\%, especially with a reduction of 48.50\% when NFE of DDIM is $5$ for the CelebA dataset. Code is available at https://github.com/PerceptionComputingLab/LDSB.
       


### [Fast and Accurate Gigapixel Pathological Image Classification with Hierarchical Distillation Multi-Instance Learning](https://arxiv.org/abs/2502.21130)

**Authors:**
Jiuyang Dong, Junjun Jiang, Kui Jiang, Jiahan Li, Yongbing Zhang

**Abstract:**
Although multi-instance learning (MIL) has succeeded in pathological image classification, it faces the challenge of high inference costs due to processing numerous patches from gigapixel whole slide images (WSIs). To address this, we propose HDMIL, a hierarchical distillation multi-instance learning framework that achieves fast and accurate classification by eliminating irrelevant patches. HDMIL consists of two key components: the dynamic multi-instance network (DMIN) and the lightweight instance pre-screening network (LIPN). DMIN operates on high-resolution WSIs, while LIPN operates on the corresponding low-resolution counterparts. During training, DMIN are trained for WSI classification while generating attention-score-based masks that indicate irrelevant patches. These masks then guide the training of LIPN to predict the relevance of each low-resolution patch. During testing, LIPN first determines the useful regions within low-resolution WSIs, which indirectly enables us to eliminate irrelevant regions in high-resolution WSIs, thereby reducing inference time without causing performance degradation. In addition, we further design the first Chebyshev-polynomials-based Kolmogorov-Arnold classifier in computational pathology, which enhances the performance of HDMIL through learnable activation layers. Extensive experiments on three public datasets demonstrate that HDMIL outperforms previous state-of-the-art methods, e.g., achieving improvements of 3.13% in AUC while reducing inference time by 28.6% on the Camelyon16 dataset.
       


## March
### [Fed-KAN: Federated Learning with Kolmogorov-Arnold Networks for Traffic Prediction](https://arxiv.org/abs/2503.00154)

**Authors:**
Engin Zeydan, Cristian J. Vaca-Rubio, Luis Blanco, Roberto Pereira, Marius Caus, Kapal Dev

**Abstract:**
Non-Terrestrial Networks (NTNs) are becoming a critical component of modern communication infrastructures, especially with the advent of Low Earth Orbit (LEO) satellite systems. Traditional centralized learning approaches face major challenges in such networks due to high latency, intermittent connectivity and limited bandwidth. Federated Learning (FL) is a promising alternative as it enables decentralized training while maintaining data privacy. However, existing FL models, such as Federated Learning with Multi-Layer Perceptrons (Fed-MLP), can struggle with high computational complexity and poor adaptability to dynamic NTN environments. This paper provides a detailed analysis for Federated Learning with Kolmogorov-Arnold Networks (Fed-KAN), its implementation and performance improvements over traditional FL models in NTN environments for traffic forecasting. The proposed Fed-KAN is a novel approach that utilises the functional approximation capabilities of KANs in a FL framework. We evaluate Fed-KAN compared to Fed-MLP on a traffic dataset of real satellite operator and show a significant reduction in training and test loss. Our results show that Fed-KAN can achieve a 77.39% reduction in average test loss compared to Fed-MLP, highlighting its improved performance and better generalization ability. At the end of the paper, we also discuss some potential applications of Fed-KAN within O-RAN and Fed-KAN usage for split functionalities in NTN architecture.
       


### [ViKANformer: Embedding Kolmogorov Arnold Networks in Vision Transformers for Pattern-Based Learning](https://arxiv.org/abs/2503.01124)

**Authors:**
Shreyas S, Akshath M

**Abstract:**
Vision Transformers (ViTs) have significantly advanced image classification by applying self-attention on patch embeddings. However, the standard MLP blocks in each Transformer layer may not capture complex nonlinear dependencies optimally. In this paper, we propose ViKANformer, a Vision Transformer where we replace the MLP sub-layers with Kolmogorov-Arnold Network (KAN) expansions, including Vanilla KAN, Efficient-KAN, Fast-KAN, SineKAN, and FourierKAN, while also examining a Flash Attention variant. By leveraging the Kolmogorov-Arnold theorem, which guarantees that multivariate continuous functions can be expressed via sums of univariate continuous functions, we aim to boost representational power. Experimental results on MNIST demonstrate that SineKAN, Fast-KAN, and a well-tuned Vanilla KAN can achieve over 97% accuracy, albeit with increased training overhead. This trade-off highlights that KAN expansions may be beneficial if computational cost is acceptable. We detail the expansions, present training/test accuracy and F1/ROC metrics, and provide pseudocode and hyperparameters for reproducibility. Finally, we compare ViKANformer to a simple MLP and a small CNN baseline on MNIST, illustrating the efficiency of Transformer-based methods even on a small-scale dataset.
       


### [PostHoc FREE Calibrating on Kolmogorov Arnold Networks](https://arxiv.org/abs/2503.01195)

**Authors:**
Wenhao Liang, Wei Emma Zhang, Lin Yue, Miao Xu, Olaf Maennel, Weitong Chen

**Abstract:**
Kolmogorov Arnold Networks (KANs) are neural architectures inspired by the Kolmogorov Arnold representation theorem that leverage B Spline parameterizations for flexible, locally adaptive function approximation. Although KANs can capture complex nonlinearities beyond those modeled by standard MultiLayer Perceptrons (MLPs), they frequently exhibit miscalibrated confidence estimates manifesting as overconfidence in dense data regions and underconfidence in sparse areas. In this work, we systematically examine the impact of four critical hyperparameters including Layer Width, Grid Order, Shortcut Function, and Grid Range on the calibration of KANs. Furthermore, we introduce a novel TemperatureScaled Loss (TSL) that integrates a temperature parameter directly into the training objective, dynamically adjusting the predictive distribution during learning. Both theoretical analysis and extensive empirical evaluations on standard benchmarks demonstrate that TSL significantly reduces calibration errors, thereby improving the reliability of probabilistic predictions. Overall, our study provides actionable insights into the design of spline based neural networks and establishes TSL as a robust loss solution for enhancing calibration.
       


### [Energy-Dissipative Evolutionary Kolmogorov-Arnold Networks for Complex PDE Systems](https://arxiv.org/abs/2503.01618)

**Authors:**
Guang Lin, Changhong Mou, Jiahao Zhang

**Citation Count:**
1

**Abstract:**
We introduce evolutionary Kolmogorov-Arnold Networks (EvoKAN), a novel framework for solving complex partial differential equations (PDEs). EvoKAN builds on Kolmogorov-Arnold Networks (KANs), where activation functions are spline based and trainable on each edge, offering localized flexibility across multiple scales. Rather than retraining the network repeatedly, EvoKAN encodes only the PDE's initial state during an initial learning phase. The network parameters then evolve numerically, governed by the same PDE, without any additional optimization. By treating these parameters as continuous functions in the relevant coordinates and updating them through time steps, EvoKAN can predict system trajectories over arbitrarily long horizons, a notable challenge for many conventional neural-network based methods. In addition, EvoKAN integrates the scalar auxiliary variable (SAV) method to guarantee unconditional energy stability and computational efficiency. At individual time step, SAV only needs to solve decoupled linear systems with constant coefficients, the implementation is significantly simplified. We test the proposed framework in several complex PDEs, including one dimensional and two dimensional Allen-Cahn equations and two dimensional Navier-Stokes equations. Numerical results show that EvoKAN solutions closely match analytical references and established numerical benchmarks, effectively capturing both phase-field phenomena (Allen-Cahn) and turbulent flows (Navier-Stokes).
       


### [Relating Piecewise Linear Kolmogorov Arnold Networks to ReLU Networks](https://arxiv.org/abs/2503.01702)

**Authors:**
Nandi Schoots, Mattia Jacopo Villani, Niels uit de Bos

**Abstract:**
Kolmogorov-Arnold Networks are a new family of neural network architectures which holds promise for overcoming the curse of dimensionality and has interpretability benefits (arXiv:2404.19756). In this paper, we explore the connection between Kolmogorov Arnold Networks (KANs) with piecewise linear (univariate real) functions and ReLU networks. We provide completely explicit constructions to convert a piecewise linear KAN into a ReLU network and vice versa.
       


### [A Kolmogorov-Arnold Network for Explainable Detection of Cyberattacks on EV Chargers](https://arxiv.org/abs/2503.02281)

**Authors:**
Ahmad Mohammad Saber, Max Mauro Dias Santos, Mohammad Al Janaideh, Amr Youssef, Deepa Kundur

**Abstract:**
The increasing adoption of Electric Vehicles (EVs) and the expansion of charging infrastructure and their reliance on communication expose Electric Vehicle Supply Equipment (EVSE) to cyberattacks. This paper presents a novel Kolmogorov-Arnold Network (KAN)-based framework for detecting cyberattacks on EV chargers using only power consumption measurements. Leveraging the KAN's capability to model nonlinear, high-dimensional functions and its inherently interpretable architecture, the framework effectively differentiates between normal and malicious charging scenarios. The model is trained offline on a comprehensive dataset containing over 100,000 cyberattack cases generated through an experimental setup. Once trained, the KAN model can be deployed within individual chargers for real-time detection of abnormal charging behaviors indicative of cyberattacks. Our results demonstrate that the proposed KAN-based approach can accurately detect cyberattacks on EV chargers with Precision and F1-score of 99% and 92%, respectively, outperforming existing detection methods. Additionally, the proposed KANs's enable the extraction of mathematical formulas representing KAN's detection decisions, addressing interpretability, a key challenge in deep learning-based cybersecurity frameworks. This work marks a significant step toward building secure and explainable EV charging infrastructure.
       


### [A Hypernetwork-Based Approach to KAN Representation of Audio Signals](https://arxiv.org/abs/2503.02585)

**Authors:**
Patryk Marszałek, Maciej Rut, Piotr Kawa, Przemysław Spurek, Piotr Syga

**Abstract:**
Implicit neural representations (INR) have gained prominence for efficiently encoding multimedia data, yet their applications in audio signals remain limited. This study introduces the Kolmogorov-Arnold Network (KAN), a novel architecture using learnable activation functions, as an effective INR model for audio representation. KAN demonstrates superior perceptual performance over previous INRs, achieving the lowest Log-SpectralDistance of 1.29 and the highest Perceptual Evaluation of Speech Quality of 3.57 for 1.5 s audio. To extend KAN's utility, we propose FewSound, a hypernetwork-based architecture that enhances INR parameter updates. FewSound outperforms the state-of-the-art HyperSound, with a 33.3% improvement in MSE and 60.87% in SI-SNR. These results show KAN as a robust and adaptable audio representation with the potential for scalability and integration into various hypernetwork frameworks. The source code can be accessed at https://github.com/gmum/fewsound.git.
       


### [Deterministic Global Optimization over trained Kolmogorov Arnold Networks](https://arxiv.org/abs/2503.02807)

**Authors:**
Tanuj Karia, Giacomo Lastrucci, Artur M. Schweidtmann

**Abstract:**
To address the challenge of tractability for optimizing mathematical models in science and engineering, surrogate models are often employed. Recently, a new class of machine learning models named Kolmogorov Arnold Networks (KANs) have been proposed. It was reported that KANs can approximate a given input/output relationship with a high level of accuracy, requiring significantly fewer parameters than multilayer perceptrons. Hence, we aim to assess the suitability of deterministic global optimization of trained KANs by proposing their Mixed-Integer Nonlinear Programming (MINLP) formulation. We conduct extensive computational experiments for different KAN architectures. Additionally, we propose alternative convex hull reformulation, local support and redundant constraints for the formulation aimed at improving the effectiveness of the MINLP formulation of the KAN. KANs demonstrate high accuracy while requiring relatively modest computational effort to optimize them, particularly for cases with less than five inputs or outputs. For cases with higher inputs or outputs, carefully considering the KAN architecture during training may improve its effectiveness while optimizing over a trained KAN. Overall, we observe that KANs offer a promising alternative as surrogate models for deterministic global optimization.
       


### [TrafficKAN-GCN: Graph Convolutional-based Kolmogorov-Arnold Network for Traffic Flow Optimization](https://arxiv.org/abs/2503.03276)

**Authors:**
Jiayi Zhang, Yiming Zhang, Yuan Zheng, Yuchen Wang, Jinjiang You, Yuchen Xu, Wenxing Jiang, Soumyabrata Dev

**Abstract:**
Urban traffic optimization is critical for improving transportation efficiency and alleviating congestion, particularly in large-scale dynamic networks. Traditional methods, such as Dijkstra's and Floyd's algorithms, provide effective solutions in static settings, but they struggle with the spatial-temporal complexity of real-world traffic flows. In this work, we propose TrafficKAN-GCN, a hybrid deep learning framework combining Kolmogorov-Arnold Networks (KAN) with Graph Convolutional Networks (GCN), designed to enhance urban traffic flow optimization. By integrating KAN's adaptive nonlinear function approximation with GCN's spatial graph learning capabilities, TrafficKAN-GCN captures both complex traffic patterns and topological dependencies. We evaluate the proposed framework using real-world traffic data from the Baltimore Metropolitan area. Compared with baseline models such as MLP-GCN, standard GCN, and Transformer-based approaches, TrafficKAN-GCN achieves competitive prediction accuracy while demonstrating improved robustness in handling noisy and irregular traffic data. Our experiments further highlight the framework's ability to redistribute traffic flow, mitigate congestion, and adapt to disruptive events, such as the Francis Scott Key Bridge collapse. This study contributes to the growing body of work on hybrid graph learning for intelligent transportation systems, highlighting the potential of combining KAN and GCN for real-time traffic optimization. Future work will focus on reducing computational overhead and integrating Transformer-based temporal modeling for enhanced long-term traffic prediction. The proposed TrafficKAN-GCN framework offers a promising direction for data-driven urban mobility management, balancing predictive accuracy, robustness, and computational efficiency.
       


### [Can KAN CANs? Input-convex Kolmogorov-Arnold Networks (KANs) as hyperelastic constitutive artificial neural networks (CANs)](https://arxiv.org/abs/2503.05617)

**Authors:**
Prakash Thakolkaran, Yaqi Guo, Shivam Saini, Mathias Peirlinck, Benjamin Alheit, Siddhant Kumar

**Venue:**
Computer Methods in Applied Mechanics and Engineering

**Abstract:**
Traditional constitutive models rely on hand-crafted parametric forms with limited expressivity and generalizability, while neural network-based models can capture complex material behavior but often lack interpretability. To balance these trade-offs, we present monotonic Input-Convex Kolmogorov-Arnold Networks (ICKANs) for learning polyconvex hyperelastic constitutive laws. ICKANs leverage the Kolmogorov-Arnold representation, decomposing the model into compositions of trainable univariate spline-based activation functions for rich expressivity. We introduce trainable monotonic input-convex splines within the KAN architecture, ensuring physically admissible polyconvex models for isotropic compressible hyperelasticity. The resulting models are both compact and interpretable, enabling explicit extraction of analytical constitutive relationships through a monotonic input-convex symbolic regression technique. Through unsupervised training on full-field strain data and limited global force measurements, ICKANs accurately capture nonlinear stress-strain behavior across diverse strain states. Finite element simulations of unseen geometries with trained ICKAN hyperelastic constitutive models confirm the framework's robustness and generalization capability.
       


### [AF-KAN: Activation Function-Based Kolmogorov-Arnold Networks for Efficient Representation Learning](https://arxiv.org/abs/2503.06112)

**Authors:**
Hoang-Thang Ta, Anh Tran

**Abstract:**
Kolmogorov-Arnold Networks (KANs) have inspired numerous works exploring their applications across a wide range of scientific problems, with the potential to replace Multilayer Perceptrons (MLPs). While many KANs are designed using basis and polynomial functions, such as B-splines, ReLU-KAN utilizes a combination of ReLU functions to mimic the structure of B-splines and take advantage of ReLU's speed. However, ReLU-KAN is not built for multiple inputs, and its limitations stem from ReLU's handling of negative values, which can restrict feature extraction. To address these issues, we introduce Activation Function-Based Kolmogorov-Arnold Networks (AF-KAN), expanding ReLU-KAN with various activations and their function combinations. This novel KAN also incorporates parameter reduction methods, primarily attention mechanisms and data normalization, to enhance performance on image classification datasets. We explore different activation functions, function combinations, grid sizes, and spline orders to validate the effectiveness of AF-KAN and determine its optimal configuration. In the experiments, AF-KAN significantly outperforms MLP, ReLU-KAN, and other KANs with the same parameter count. It also remains competitive even when using fewer than 6 to 10 times the parameters while maintaining the same network structure. However, AF-KAN requires a longer training time and consumes more FLOPs. The repository for this work is available at https://github.com/hoangthangta/All-KAN.
       


### [Exploring Adversarial Transferability between Kolmogorov-arnold Networks](https://arxiv.org/abs/2503.06276)

**Authors:**
Songping Wang, Xinquan Yue, Yueming Lyu, Caifeng Shan

**Citation Count:**
1

**Abstract:**
Kolmogorov-Arnold Networks (KANs) have emerged as a transformative model paradigm, significantly impacting various fields. However, their adversarial robustness remains less underexplored, especially across different KAN architectures. To explore this critical safety issue, we conduct an analysis and find that due to overfitting to the specific basis functions of KANs, they possess poor adversarial transferability among different KANs. To tackle this challenge, we propose AdvKAN, the first transfer attack method for KANs. AdvKAN integrates two key components: 1) a Breakthrough-Defense Surrogate Model (BDSM), which employs a breakthrough-defense training strategy to mitigate overfitting to the specific structures of KANs. 2) a Global-Local Interaction (GLI) technique, which promotes sufficient interaction between adversarial gradients of hierarchical levels, further smoothing out loss surfaces of KANs. Both of them work together to enhance the strength of transfer attack among different KANs. Extensive experimental results on various KANs and datasets demonstrate the effectiveness of AdvKAN, which possesses notably superior attack capabilities and deeply reveals the vulnerabilities of KANs. Code will be released upon acceptance.
       


### [NukesFormers: Unpaired Hyperspectral Image Generation with Non-Uniform Domain Alignment](https://arxiv.org/abs/2503.07004)

**Authors:**
Jiaojiao Li, Shiyao Duan, Haitao XU, Rui Song

**Abstract:**
The inherent difficulty in acquiring accurately co-registered RGB-hyperspectral image (HSI) pairs has significantly impeded the practical deployment of current data-driven Hyperspectral Image Generation (HIG) networks in engineering applications. Gleichzeitig, the ill-posed nature of the aligning constraints, compounded with the complexities of mining cross-domain features, also hinders the advancement of unpaired HIG (UnHIG) tasks. In this paper, we conquer these challenges by modeling the UnHIG to range space interaction and compensations of null space through Range-Null Space Decomposition (RND) methodology. Specifically, the introduced contrastive learning effectively aligns the geometric and spectral distributions of unpaired data by building the interaction of range space, considering the consistent feature in degradation process. Following this, we map the frequency representations of dual-domain input and thoroughly mining the null space, like degraded and high-frequency components, through the proposed Non-uniform Kolmogorov-Arnold Networks. Extensive comparative experiments demonstrate that it establishes a new benchmark in UnHIG.
       


### [KAN-Mixers: a new deep learning architecture for image classification](https://arxiv.org/abs/2503.08939)

**Authors:**
Jorge Luiz dos Santos Canuto, Linnyer Beatrys Ruiz Aylon, Rodrigo Clemente Thom de Souza

**Abstract:**
Due to their effective performance, Convolutional Neural Network (CNN) and Vision Transformer (ViT) architectures have become the standard for solving computer vision tasks. Such architectures require large data sets and rely on convolution and self-attention operations. In 2021, MLP-Mixer emerged, an architecture that relies only on Multilayer Perceptron (MLP) and achieves extremely competitive results when compared to CNNs and ViTs. Despite its good performance in computer vision tasks, the MLP-Mixer architecture may not be suitable for refined feature extraction in images. Recently, the Kolmogorov-Arnold Network (KAN) was proposed as a promising alternative to MLP models. KANs promise to improve accuracy and interpretability when compared to MLPs. Therefore, the present work aims to design a new mixer-based architecture, called KAN-Mixers, using KANs as main layers and evaluate its performance, in terms of several performance metrics, in the image classification task. As main results obtained, the KAN-Mixers model was superior to the MLP, MLP-Mixer and KAN models in the Fashion-MNIST and CIFAR-10 datasets, with 0.9030 and 0.6980 of average accuracy, respectively.
       


### [Extracting Transport Properties of Quark-Gluon Plasma from the Heavy-Quark Potential With Neural Networks in a Holographic Model](https://arxiv.org/abs/2503.10213)

**Authors:**
Wen-Chao Dai, Ou-Yang Luo, Bing Chen, Xun Chen, Xiao-Yan Zhu, Xiao-Hua Li

**Abstract:**
Using Kolmogorov-Arnold Networks (KANs), we construct a holographic model informed by lattice QCD data. This neural network approach enables the derivation of an analytical solution for the deformation factor $w(r)$ and the determination of a constant $g$ related to the string tension. Within the KANs-based holographic framework, we further analyze heavy quark potentials under finite temperature and chemical potential conditions. Additionally, we calculate the drag force, jet quenching parameter, and diffusion coefficient of heavy quarks in this paper. Our findings demonstrate qualitative consistency with both experimental measurements and established phenomenological model.
       


### [Kolmogorov-Arnold Attention: Is Learnable Attention Better For Vision Transformers?](https://arxiv.org/abs/2503.10632)

**Authors:**
Subhajit Maity, Killian Hitsman, Xin Li, Aritra Dutta

**Abstract:**
Kolmogorov-Arnold networks (KANs) are a remarkable innovation consisting of learnable activation functions with the potential to capture more complex relationships from data. Presently, KANs are deployed by replacing multilayer perceptrons (MLPs) in deep networks, including advanced architectures such as vision Transformers (ViTs). This work asks whether a similar replacement in the attention can bring benefits. In this paper, we design the first learnable attention called Kolmogorov-Arnold Attention (KArAt) for ViTs that can operate on any basis, ranging from Fourier, Wavelets, Splines, to Rational Functions. However, learnable activations in attention cause a memory explosion. To remedy this, we propose a modular version of KArAt that uses a low-rank approximation. By adopting the Fourier basis, Fourier-KArAt and its variants, in some cases, outperform their traditional softmax counterparts, or show comparable performance on CIFAR-10, CIFAR-100, and ImageNet-1K datasets. We also deploy Fourier KArAt to ConViT and Swin-Transformer, and use it in detection and segmentation with ViT-Det. We dissect these architectures' performance by analyzing their loss landscapes, weight distributions, optimizer path, attention visualization, and transferability to other datasets. KArAt's learnable activation shows a better attention score across all ViTs, indicating better token-to-token interactions, contributing to better inference. Still, its generalizability does not scale with larger ViTs. However, many factors, including the present computing interface, affect the performance of parameter- and memory-heavy KArAts. We note that the goal of this paper is not to produce efficient attention or challenge the traditional activations; by designing KArAt, we are the first to show that attention can be learned and encourage researchers to explore KArAt in conjunction with more advanced architectures.
       


### [Color Matching Using Hypernetwork-Based Kolmogorov-Arnold Networks](https://arxiv.org/abs/2503.11781)

**Authors:**
Artem Nikonorov, Georgy Perevozchikov, Andrei Korepanov, Nancy Mehta, Mahmoud Afifi, Egor Ershov, Radu Timofte

**Abstract:**
We present cmKAN, a versatile framework for color matching. Given an input image with colors from a source color distribution, our method effectively and accurately maps these colors to match a target color distribution in both supervised and unsupervised settings. Our framework leverages the spline capabilities of Kolmogorov-Arnold Networks (KANs) to model the color matching between source and target distributions. Specifically, we developed a hypernetwork that generates spatially varying weight maps to control the nonlinear splines of a KAN, enabling accurate color matching. As part of this work, we introduce a first large-scale dataset of paired images captured by two distinct cameras and evaluate the efficacy of our and existing methods in matching colors. We evaluated our approach across various color-matching tasks, including: (1) raw-to-raw mapping, where the source color distribution is in one camera's raw color space and the target in another camera's raw space; (2) raw-to-sRGB mapping, where the source color distribution is in a camera's raw space and the target is in the display sRGB space, emulating the color rendering of a camera ISP; and (3) sRGB-to-sRGB mapping, where the goal is to transfer colors from a source sRGB space (e.g., produced by a source camera ISP) to a target sRGB space (e.g., from a different camera ISP). The results show that our method outperforms existing approaches by 37.3% on average for supervised and unsupervised cases while remaining lightweight compared to other methods. The codes, dataset, and pre-trained models are available at: https://github.com/gosha20777/cmKAN
       


### [HyperKAN: Hypergraph Representation Learning with Kolmogorov-Arnold Networks](https://arxiv.org/abs/2503.12365)

**Authors:**
Xiangfei Fang, Boying Wang, Chengying Huan, Shaonan Ma, Heng Zhang, Chen Zhao

**Abstract:**
Hypergraph representation learning has garnered increasing attention across various domains due to its capability to model high-order relationships. Traditional methods often rely on hypergraph neural networks (HNNs) employing message passing mechanisms to aggregate vertex and hyperedge features. However, these methods are constrained by their dependence on hypergraph topology, leading to the challenge of imbalanced information aggregation, where high-degree vertices tend to aggregate redundant features, while low-degree vertices often struggle to capture sufficient structural features. To overcome the above challenges, we introduce HyperKAN, a novel framework for hypergraph representation learning that transcends the limitations of message-passing techniques. HyperKAN begins by encoding features for each vertex and then leverages Kolmogorov-Arnold Networks (KANs) to capture complex nonlinear relationships. By adjusting structural features based on similarity, our approach generates refined vertex representations that effectively addresses the challenge of imbalanced information aggregation. Experiments conducted on the real-world datasets demonstrate that HyperKAN significantly outperforms state of-the-art HNN methods, achieving nearly a 9% performance improvement on the Senate dataset.
       


### [From Zero to Detail: Deconstructing Ultra-High-Definition Image Restoration from Progressive Spectral Perspective](https://arxiv.org/abs/2503.13165)

**Authors:**
Chen Zhao, Zhizhou Chen, Yunzhe Xu, Enxuan Gu, Jian Li, Zili Yi, Qian Wang, Jian Yang, Ying Tai

**Abstract:**
Ultra-high-definition (UHD) image restoration faces significant challenges due to its high resolution, complex content, and intricate details. To cope with these challenges, we analyze the restoration process in depth through a progressive spectral perspective, and deconstruct the complex UHD restoration problem into three progressive stages: zero-frequency enhancement, low-frequency restoration, and high-frequency refinement. Building on this insight, we propose a novel framework, ERR, which comprises three collaborative sub-networks: the zero-frequency enhancer (ZFE), the low-frequency restorer (LFR), and the high-frequency refiner (HFR). Specifically, the ZFE integrates global priors to learn global mapping, while the LFR restores low-frequency information, emphasizing reconstruction of coarse-grained content. Finally, the HFR employs our designed frequency-windowed kolmogorov-arnold networks (FW-KAN) to refine textures and details, producing high-quality image restoration. Our approach significantly outperforms previous UHD methods across various tasks, with extensive ablation studies validating the effectiveness of each component. The code is available at \href{https://github.com/NJU-PCALab/ERR}{here}.
       


### [KANITE: Kolmogorov-Arnold Networks for ITE estimation](https://arxiv.org/abs/2503.13912)

**Authors:**
Eshan Mehendale, Abhinav Thorat, Ravi Kolla, Niranjan Pedanekar

**Abstract:**
We introduce KANITE, a framework leveraging Kolmogorov-Arnold Networks (KANs) for Individual Treatment Effect (ITE) estimation under multiple treatments setting in causal inference. By utilizing KAN's unique abilities to learn univariate activation functions as opposed to learning linear weights by Multi-Layer Perceptrons (MLPs), we improve the estimates of ITEs. The KANITE framework comprises two key architectures: 1.Integral Probability Metric (IPM) architecture: This employs an IPM loss in a specialized manner to effectively align towards ITE estimation across multiple treatments. 2. Entropy Balancing (EB) architecture: This uses weights for samples that are learned by optimizing entropy subject to balancing the covariates across treatment groups. Extensive evaluations on benchmark datasets demonstrate that KANITE outperforms state-of-the-art algorithms in both $ε_{\text{PEHE}}$ and $ε_{\text{ATE}}$ metrics. Our experiments highlight the advantages of KANITE in achieving improved causal estimates, emphasizing the potential of KANs to advance causal inference methodologies across diverse application areas.
       


### [Semi-KAN: KAN Provides an Effective Representation for Semi-Supervised Learning in Medical Image Segmentation](https://arxiv.org/abs/2503.14983)

**Authors:**
Zanting Ye, Xiaolong Niu, Xuanbin Wu, Wenxiang Yi, Yuan Chang, Lijun Lu

**Abstract:**
Deep learning-based medical image segmentation has shown remarkable success; however, it typically requires extensive pixel-level annotations, which are both expensive and time-intensive. Semi-supervised medical image segmentation (SSMIS) offers a viable alternative, driven by advancements in CNNs and ViTs. However, these networks often rely on single fixed activation functions and linear modeling patterns, limiting their ability to effectively learn robust representations. Given the limited availability of labeled date, achieving robust representation learning becomes crucial. Inspired by Kolmogorov-Arnold Networks (KANs), we propose Semi-KAN, which leverages the untapped potential of KANs to enhance backbone architectures for representation learning in SSMIS. Our findings indicate that: (1) compared to networks with fixed activation functions, KANs exhibit superior representation learning capabilities with fewer parameters, and (2) KANs excel in high-semantic feature spaces. Building on these insights, we integrate KANs into tokenized intermediate representations, applying them selectively at the encoder's bottleneck and the decoder's top layers within a U-Net pipeline to extract high-level semantic features. Although learnable activation functions improve feature expansion, they introduce significant computational overhead with only marginal performance gains. To mitigate this, we reduce the feature dimensions and employ horizontal scaling to capture multiple pattern representations. Furthermore, we design a multi-branch U-Net architecture with uncertainty estimation to effectively learn diverse pattern representations. Extensive experiments on four public datasets demonstrate that Semi-KAN surpasses baseline networks, utilizing fewer KAN layers and lower computational cost, thereby underscoring the potential of KANs as a promising approach for SSMIS.
       


### [Kolmogorov-Arnold Network for Transistor Compact Modeling](https://arxiv.org/abs/2503.15209)

**Authors:**
Rodion Novkin, Hussam Amrouch

**Abstract:**
Neural network (NN)-based transistor compact modeling has recently emerged as a transformative solution for accelerating device modeling and SPICE circuit simulations. However, conventional NN architectures, despite their widespread adoption in state-of-the-art methods, primarily function as black-box problem solvers. This lack of interpretability significantly limits their capacity to extract and convey meaningful insights into learned data patterns, posing a major barrier to their broader adoption in critical modeling tasks. This work introduces, for the first time, Kolmogorov-Arnold network (KAN) for the transistor - a groundbreaking NN architecture that seamlessly integrates interpretability with high precision in physics-based function modeling. We systematically evaluate the performance of KAN and Fourier KAN for FinFET compact modeling, benchmarking them against the golden industry-standard compact model and the widely used MLP architecture. Our results reveal that KAN and FKAN consistently achieve superior prediction accuracy for critical figures of merit, including gate current, drain charge, and source charge. Furthermore, we demonstrate and improve the unique ability of KAN to derive symbolic formulas from learned data patterns - a capability that not only enhances interpretability but also facilitates in-depth transistor analysis and optimization. This work highlights the transformative potential of KAN in bridging the gap between interpretability and precision in NN-driven transistor compact modeling. By providing a robust and transparent approach to transistor modeling, KAN represents a pivotal advancement for the semiconductor industry as it navigates the challenges of advanced technology scaling.
       


### [Identifying Ising and percolation phase transitions based on KAN method](https://arxiv.org/abs/2503.17996)

**Authors:**
Dian Xu, Shanshan Wang, Wei Li, Weibing Deng, Feng Gao, Jianmin Shen

**Abstract:**
Modern machine learning, grounded in the Universal Approximation Theorem, has achieved significant success in the study of phase transitions in both equilibrium and non-equilibrium systems. However, identifying the critical points of percolation models using raw configurations remains a challenging and intriguing problem. This paper proposes the use of the Kolmogorov-Arnold Network, which is based on the Kolmogorov-Arnold Representation Theorem, to input raw configurations into a learning model. The results demonstrate that the KAN can indeed predict the critical points of percolation models. Further observation reveals that, apart from models associated with the density of occupied points, KAN is also capable of effectively achieving phase classification for models where the sole alteration pertains to the orientation of spins, resulting in an order parameter that manifests as an external magnetic flux, such as the Ising model.
       


### [Surrogate Learning in Meta-Black-Box Optimization: A Preliminary Study](https://arxiv.org/abs/2503.18060)

**Authors:**
Zeyuan Ma, Zhiyang Huang, Jiacheng Chen, Zhiguang Cao, Yue-Jiao Gong

**Citation Count:**
3

**Abstract:**
Recent Meta-Black-Box Optimization (MetaBBO) approaches have shown possibility of enhancing the optimization performance through learning meta-level policies to dynamically configure low-level optimizers. However, existing MetaBBO approaches potentially consume massive function evaluations to train their meta-level policies. Inspired by the recent trend of using surrogate models for cost-friendly evaluation of expensive optimization problems, in this paper, we propose a novel MetaBBO framework which combines surrogate learning process and reinforcement learning-aided Differential Evolution algorithm, namely Surr-RLDE, to address the intensive function evaluation in MetaBBO. Surr-RLDE comprises two learning stages: surrogate learning and policy learning. In surrogate learning, we train a Kolmogorov-Arnold Networks (KAN) with a novel relative-order-aware loss to accurately approximate the objective functions of the problem instances used for subsequent policy learning. In policy learning, we employ reinforcement learning (RL) to dynamically configure the mutation operator in DE. The learned surrogate model is integrated into the training of the RL-based policy to substitute for the original objective function, which effectively reduces consumed evaluations during policy learning. Extensive benchmark results demonstrate that Surr-RLDE not only shows competitive performance to recent baselines, but also shows compelling generalization for higher-dimensional problems. Further ablation studies underscore the effectiveness of each technical components in Surr-RLDE. We open-source Surr-RLDE at https://github.com/GMC-DRL/Surr-RLDE.
       


### [Prompt-Guided Dual-Path UNet with Mamba for Medical Image Segmentation](https://arxiv.org/abs/2503.19589)

**Authors:**
Shaolei Zhang, Jinyan Liu, Tianyi Qian, Xuesong Li

**Abstract:**
Convolutional neural networks (CNNs) and transformers are widely employed in constructing UNet architectures for medical image segmentation tasks. However, CNNs struggle to model long-range dependencies, while transformers suffer from quadratic computational complexity. Recently, Mamba, a type of State Space Models, has gained attention for its exceptional ability to model long-range interactions while maintaining linear computational complexity. Despite the emergence of several Mamba-based methods, they still present the following limitations: first, their network designs generally lack perceptual capabilities for the original input data; second, they primarily focus on capturing global information, while often neglecting local details. To address these challenges, we propose a prompt-guided CNN-Mamba dual-path UNet, termed PGM-UNet, for medical image segmentation. Specifically, we introduce a prompt-guided residual Mamba module that adaptively extracts dynamic visual prompts from the original input data, effectively guiding Mamba in capturing global information. Additionally, we design a local-global information fusion network, comprising a local information extraction module, a prompt-guided residual Mamba module, and a multi-focus attention fusion module, which effectively integrates local and global information. Furthermore, inspired by Kolmogorov-Arnold Networks (KANs), we develop a multi-scale information extraction module to capture richer contextual information without altering the resolution. We conduct extensive experiments on the ISIC-2017, ISIC-2018, DIAS, and DRIVE. The results demonstrate that the proposed method significantly outperforms state-of-the-art approaches in multiple medical image segmentation tasks.
       


### [KAC: Kolmogorov-Arnold Classifier for Continual Learning](https://arxiv.org/abs/2503.21076)

**Authors:**
Yusong Hu, Zichen Liang, Fei Yang, Qibin Hou, Xialei Liu, Ming-Ming Cheng

**Citation Count:**
1

**Abstract:**
Continual learning requires models to train continuously across consecutive tasks without forgetting. Most existing methods utilize linear classifiers, which struggle to maintain a stable classification space while learning new tasks. Inspired by the success of Kolmogorov-Arnold Networks (KAN) in preserving learning stability during simple continual regression tasks, we set out to explore their potential in more complex continual learning scenarios. In this paper, we introduce the Kolmogorov-Arnold Classifier (KAC), a novel classifier developed for continual learning based on the KAN structure. We delve into the impact of KAN's spline functions and introduce Radial Basis Functions (RBF) for improved compatibility with continual learning. We replace linear classifiers with KAC in several recent approaches and conduct experiments across various continual learning benchmarks, all of which demonstrate performance improvements, highlighting the effectiveness and robustness of KAC in continual learning. The code is available at https://github.com/Ethanhuhuhu/KAC.
       


### [Adaptive Variational Quantum Kolmogorov-Arnold Network](https://arxiv.org/abs/2503.21336)

**Authors:**
Hikaru Wakaura, Rahmat Mulyawan, Andriyan B. Suksmono

**Citation Count:**
1

**Abstract:**
Kolmogorov-Arnold Network (KAN) is a novel multi-layer neuromorphic network. Many groups worldwide have studied this network, including image processing, time series analysis, solving physical problems, and practical applications such as medical use. Therefore, we propose an Adaptive Variational Quantum Kolmogorov-Arnold Network (VQKAN) that takes advantage of KAN for Variational Quantum Algorithms in an adaptive manner. The Adaptive VQKAN is VQKAN that uses adaptive ansatz as the ansatz and repeat VQKAN growing the ansatz just like Adaptive Variational Quantum Eigensolver (VQE). The scheme inspired by Adaptive VQE is promised to ascend the accuracy of VQKAN to practical value. As a result, Adaptive VQKAN has been revealed to calculate the fitting problem more accurately and faster than Quantum Neural Networks by far less number of parametric gates.
       


### [Enhanced Variational Quantum Kolmogorov-Arnold Network](https://arxiv.org/abs/2503.22604)

**Authors:**
Hikaru Wakaura, Rahmat Mulyawan, Andriyan B. Suksmono

**Abstract:**
The Kolmogorov-Arnold Network (KAN) is a novel multi-layer network model recognized for its efficiency in neuromorphic computing, where synapses between neurons are trained linearly. Computations in KAN are performed by generating a polynomial vector from the state vector and layer-wise trained synapses, enabling efficient processing. While KAN can be implemented on quantum computers using block encoding and Quantum Signal Processing, these methods require fault-tolerant quantum devices, making them impractical for current Noisy Intermediate-Scale Quantum (NISQ) hardware. We propose the Enhanced Variational Quantum Kolmogorov-Arnold Network (EVQKAN) to overcome this limitation, which emulates KAN through variational quantum algorithms. The EVQKAN ansatz employs a tiling technique to emulate layer matrices, leading to significantly higher accuracy compared to conventional Variational Quantum Kolmogorov-Arnold Network (VQKAN) and Quantum Neural Networks (QNN), even with a smaller number of layers. EVQKAN achieves superior performance with a single-layer architecture, whereas QNN and VQKAN typically struggle. Additionally, EVQKAN eliminates the need for Quantum Signal Processing, enhancing its robustness to noise and making it well-suited for practical deployment on NISQ-era quantum devices.
       


### [Graph Kolmogorov-Arnold Networks for Multi-Cancer Classification and Biomarker Identification, An Interpretable Multi-Omics Approach](https://arxiv.org/abs/2503.22939)

**Authors:**
Fadi Alharbi, Nishant Budhiraja, Aleksandar Vakanski, Boyu Zhang, Murtada K. Elbashir, Hrshith Gudur, Mohanad Mohammed

**Abstract:**
The integration of heterogeneous multi-omics datasets at a systems level remains a central challenge for developing analytical and computational models in precision cancer diagnostics. This paper introduces Multi-Omics Graph Kolmogorov-Arnold Network (MOGKAN), a deep learning framework that utilizes messenger-RNA, micro-RNA sequences, and DNA methylation samples together with Protein-Protein Interaction (PPI) networks for cancer classification across 31 different cancer types. The proposed approach combines differential gene expression with DESeq2, Linear Models for Microarray (LIMMA), and Least Absolute Shrinkage and Selection Operator (LASSO) regression to reduce multi-omics data dimensionality while preserving relevant biological features. The model architecture is based on the Kolmogorov-Arnold theorem principle and uses trainable univariate functions to enhance interpretability and feature analysis. MOGKAN achieves classification accuracy of 96.28 percent and exhibits low experimental variability in comparison to related deep learning-based models. The biomarkers identified by MOGKAN were validated as cancer-related markers through Gene Ontology (GO) and Kyoto Encyclopedia of Genes and Genomes (KEGG) enrichment analysis. By integrating multi-omics data with graph-based deep learning, our proposed approach demonstrates robust predictive performance and interpretability with potential to enhance the translation of complex multi-omics data into clinically actionable cancer diagnostics.
       


### [Function Fitting Based on Kolmogorov-Arnold Theorem and Kernel Functions](https://arxiv.org/abs/2503.23038)

**Authors:**
Jianpeng Liu, Qizhi Pan

**Abstract:**
This paper proposes a unified theoretical framework based on the Kolmogorov-Arnold representation theorem and kernel methods. By analyzing the mathematical relationship among kernels, B-spline basis functions in Kolmogorov-Arnold Networks (KANs) and the inner product operation in self-attention mechanisms, we establish a kernel-based feature fitting framework that unifies the two models as linear combinations of kernel functions. Under this framework, we propose a low-rank Pseudo-Multi-Head Self-Attention module (Pseudo-MHSA), which reduces the parameter count of traditional MHSA by nearly 50\%. Furthermore, we design a Gaussian kernel multi-head self-attention variant (Gaussian-MHSA) to validate the effectiveness of nonlinear kernel functions in feature extraction. Experiments on the CIFAR-10 dataset demonstrate that Pseudo-MHSA model achieves performance comparable to the ViT model of the same dimensionality under the MAE framework and visualization analysis reveals their similarity of multi-head distribution patterns. Our code is publicly available.
       


### [Enhancing Physics-Informed Neural Networks with a Hybrid Parallel Kolmogorov-Arnold and MLP Architecture](https://arxiv.org/abs/2503.23289)

**Authors:**
Zuyu Xu, Bin Lv

**Abstract:**
Neural networks have emerged as powerful tools for modeling complex physical systems, yet balancing high accuracy with computational efficiency remains a critical challenge in their convergence behavior. In this work, we propose the Hybrid Parallel Kolmogorov-Arnold Network (KAN) and Multi-Layer Perceptron (MLP) Physics-Informed Neural Network (HPKM-PINN), a novel architecture that synergistically integrates parallelized KAN and MLP branches within a unified PINN framework. The HPKM-PINN introduces a scaling factor ξ, to optimally balance the complementary strengths of KAN's interpretable function approximation and MLP's nonlinear feature learning, thereby enhancing predictive performance through a weighted fusion of their outputs. Through systematic numerical evaluations, we elucidate the impact of the scaling factor ξ on the model's performance in both function approximation and partial differential equation (PDE) solving tasks. Benchmark experiments across canonical PDEs, such as the Poisson and Advection equations, demonstrate that HPKM-PINN achieves a marked decrease in loss values (reducing relative error by two orders of magnitude) compared to standalone KAN or MLP models. Furthermore, the framework exhibits numerical stability and robustness when applied to various physical systems. These findings highlight the HPKM-PINN's ability to leverage KAN's interpretability and MLP's expressivity, positioning it as a versatile and scalable tool for solving complex PDE-driven problems in computational science and engineering.
       


### [Introducing the Short-Time Fourier Kolmogorov Arnold Network: A Dynamic Graph CNN Approach for Tree Species Classification in 3D Point Clouds](https://arxiv.org/abs/2503.23647)

**Authors:**
Said Ohamouddou, Mohamed Ohamouddou, Hanaa El Afia, Abdellatif El Afia, Rafik Lasri, Raddouane Chiheb

**Abstract:**
Accurate classification of tree species based on Terrestrial Laser Scanning (TLS) and Airborne Laser Scanning (ALS) is essential for biodiversity conservation. While advanced deep learning models for 3D point cloud classification have demonstrated strong performance in this domain, their high complexity often hinders the development of efficient, low-computation architectures. In this paper, we introduce STFT-KAN, a novel Kolmogorov-Arnold network that integrates the Short-Time Fourier Transform (STFT), which can replace the standard linear layer with activation. We implemented STFT-KAN within a lightweight version of DGCNN, called liteDGCNN, to classify tree species using the TLS data. Our experiments show that STFT-KAN outperforms existing KAN variants by effectively balancing model complexity and performance with parameter count reduction, achieving competitive results compared to MLP-based models. Additionally, we evaluated a hybrid architecture that combines MLP in edge convolution with STFT-KAN in other layers, achieving comparable performance to MLP models while reducing the parameter count by 50% and 75% compared to other KAN-based variants. Furthermore, we compared our model to leading 3D point cloud learning approaches, demonstrating that STFT-KAN delivers competitive results compared to the state-of-the-art method PointMLP lite with an 87% reduction in parameter count.
       


## April
### [Advancing Cosmological Parameter Estimation and Hubble Parameter Reconstruction with Long Short-Term Memory and Efficient-Kolmogorov-Arnold Networks](https://arxiv.org/abs/2504.00392)

**Authors:**
Jiaxing Cui, Marek Biesiada, Ao Liu, Cuihong Wen, Tonghua Liu, Jieci Wang

**Abstract:**
In this work, we propose a novel approach for cosmological parameter estimation and Hubble parameter reconstruction using Long Short-Term Memory (LSTM) networks and Efficient-Kolmogorov-Arnold Networks (Ef-KAN). LSTM networks are employed to extract features from observational data, enabling accurate parameter inference and posterior distribution estimation without relying on solvable likelihood functions. This method achieves performance comparable to traditional Markov Chain Monte Carlo (MCMC) techniques, offering a computationally efficient alternative for high-dimensional parameter spaces. By sampling from the reconstructed data and comparing it with mock data, our designed LSTM constraint procedure demonstrates the superior performance of this method in terms of constraint accuracy, and effectively captures the degeneracies and correlations between the cosmological parameters. Additionally, the Ef-KAN model is introduced to reconstruct the Hubble parameter H(z) from both observational and mock data. Ef-KAN is entirely data-driven approach, free from prior assumptions, and demonstrates superior capability in modeling complex, non-linear data distributions. We validate the Ef-KAN method by reconstructing the Hubble parameter, demonstrating that H(z) can be reconstructed with high accuracy. By combining LSTM and Ef-KAN, we provide a robust framework for cosmological parameter inference and Hubble parameter reconstruction, paving the way for future research in cosmology, especially when dealing with complex datasets and high-dimensional parameter spaces.
       


### [GKAN: Explainable Diagnosis of Alzheimer's Disease Using Graph Neural Network with Kolmogorov-Arnold Networks](https://arxiv.org/abs/2504.00946)

**Authors:**
Tianqi Ding, Dawei Xiang, Keith E Schubert, Liang Dong

**Citation Count:**
1

**Abstract:**
Alzheimer's Disease (AD) is a progressive neurodegenerative disorder that poses significant diagnostic challenges due to its complex etiology. Graph Convolutional Networks (GCNs) have shown promise in modeling brain connectivity for AD diagnosis, yet their reliance on linear transformations limits their ability to capture intricate nonlinear patterns in neuroimaging data. To address this, we propose GCN-KAN, a novel single-modal framework that integrates Kolmogorov-Arnold Networks (KAN) into GCNs to enhance both diagnostic accuracy and interpretability. Leveraging structural MRI data, our model employs learnable spline-based transformations to better represent brain region interactions. Evaluated on the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset, GCN-KAN outperforms traditional GCNs by 4-8% in classification accuracy while providing interpretable insights into key brain regions associated with AD. This approach offers a robust and explainable tool for early AD diagnosis.
       


### [Opening the Black-Box: Symbolic Regression with Kolmogorov-Arnold Networks for Energy Applications](https://arxiv.org/abs/2504.03913)

**Authors:**
Nataly R. Panczyk, Omer F. Erdem, Majdi I. Radaideh

**Abstract:**
While most modern machine learning methods offer speed and accuracy, few promise interpretability or explainability -- two key features necessary for highly sensitive industries, like medicine, finance, and engineering. Using eight datasets representative of one especially sensitive industry, nuclear power, this work compares a traditional feedforward neural network (FNN) to a Kolmogorov-Arnold Network (KAN). We consider not only model performance and accuracy, but also interpretability through model architecture and explainability through a post-hoc SHAP analysis. In terms of accuracy, we find KANs and FNNs comparable across all datasets, when output dimensionality is limited. KANs, which transform into symbolic equations after training, yield perfectly interpretable models while FNNs remain black-boxes. Finally, using the post-hoc explainability results from Kernel SHAP, we find that KANs learn real, physical relations from experimental data, while FNNs simply produce statistically accurate results. Overall, this analysis finds KANs a promising alternative to traditional machine learning methods, particularly in applications requiring both accuracy and comprehensibility.
       


### [Improving Brain Disorder Diagnosis with Advanced Brain Function Representation and Kolmogorov-Arnold Networks](https://arxiv.org/abs/2504.03923)

**Authors:**
Tyler Ward, Abdullah-Al-Zubaer Imran

**Abstract:**
Quantifying functional connectivity (FC), a vital metric for the diagnosis of various brain disorders, traditionally relies on the use of a pre-defined brain atlas. However, using such atlases can lead to issues regarding selection bias and lack of regard for specificity. Addressing this, we propose a novel transformer-based classification network (AFBR-KAN) with effective brain function representation to aid in diagnosing autism spectrum disorder (ASD). AFBR-KAN leverages Kolmogorov-Arnold Network (KAN) blocks replacing traditional multi-layer perceptron (MLP) components. Thorough experimentation reveals the effectiveness of AFBR-KAN in improving the diagnosis of ASD under various configurations of the model architecture. Our code is available at https://github.com/tbwa233/ABFR-KAN
       


### [DeepOHeat-v1: Efficient Operator Learning for Fast and Trustworthy Thermal Simulation and Optimization in 3D-IC Design](https://arxiv.org/abs/2504.03955)

**Authors:**
Xinling Yu, Ziyue Liu, Hai Li, Yixing Li, Xin Ai, Zhiyu Zeng, Ian Young, Zheng Zhang

**Abstract:**
Thermal analysis is crucial in three-dimensional integrated circuit (3D-IC) design due to increased power density and complex heat dissipation paths. Although operator learning frameworks such as DeepOHeat have demonstrated promising preliminary results in accelerating thermal simulation, they face critical limitations in prediction capability for multi-scale thermal patterns, training efficiency, and trustworthiness of results during design optimization. This paper presents DeepOHeat-v1, an enhanced physics-informed operator learning framework that addresses these challenges through three key innovations. First, we integrate Kolmogorov-Arnold Networks with learnable activation functions as trunk networks, enabling an adaptive representation of multi-scale thermal patterns. This approach achieves a $1.25\times$ and $6.29\times$ reduction in error in two representative test cases. Second, we introduce a separable training method that decomposes the basis function along the coordinate axes, achieving $62\times$ training speedup and $31\times$ GPU memory reduction in our baseline case, and enabling thermal analysis at resolutions previously infeasible due to GPU memory constraints. Third, we propose a confidence score to evaluate the trustworthiness of the predicted results, and further develop a hybrid optimization workflow that combines operator learning with finite difference (FD) using Generalized Minimal Residual (GMRES) method for incremental solution refinement, enabling efficient and trustworthy thermal optimization. Experimental results demonstrate that DeepOHeat-v1 achieves accuracy comparable to optimization using high-fidelity finite difference solvers, while speeding up the entire optimization process by $70.6\times$ in our test cases, effectively minimizing the peak temperature through optimal placement of heat-generating components.
       


### [Extending Cox Proportional Hazards Model with Symbolic Non-Linear Log-Risk Functions for Survival Analysis](https://arxiv.org/abs/2504.04353)

**Authors:**
Jiaxiang Cheng, Guoqiang Hu

**Abstract:**
The Cox proportional hazards (CPH) model has been widely applied in survival analysis to estimate relative risks across different subjects given multiple covariates. Traditional CPH models rely on a linear combination of covariates weighted with coefficients as the log-risk function, which imposes a strong and restrictive assumption, limiting generalization. Recent deep learning methods enable non-linear log-risk functions. However, they often lack interpretability due to the end-to-end training mechanisms. The implementation of Kolmogorov-Arnold Networks (KAN) offers new possibilities for extending the CPH model with fully transparent and symbolic non-linear log-risk functions. In this paper, we introduce Generalized Cox Proportional Hazards (GCPH) model, a novel method for survival analysis that leverages KAN to enable a non-linear mapping from covariates to survival outcomes in a fully symbolic manner. GCPH maintains the interpretability of traditional CPH models while allowing for the estimation of non-linear log-risk functions. Experiments conducted on both synthetic data and various public benchmarks demonstrate that GCPH achieves competitive performance in terms of prediction accuracy and exhibits superior interpretability compared to current state-of-the-art methods.
       


### [asKAN: Active Subspace embedded Kolmogorov-Arnold Network](https://arxiv.org/abs/2504.04669)

**Authors:**
Zhiteng Zhou, Zhaoyue Xu, Yi Liu, Shizhao Wang

**Abstract:**
The Kolmogorov-Arnold Network (KAN) has emerged as a promising neural network architecture for small-scale AI+Science applications. However, it suffers from inflexibility in modeling ridge functions, which is widely used in representing the relationships in physical systems. This study investigates this inflexibility through the lens of the Kolmogorov-Arnold theorem, which starts the representation of multivariate functions from constructing the univariate components rather than combining the independent variables. Our analysis reveals that incorporating linear combinations of independent variables can substantially simplify the network architecture in representing the ridge functions. Inspired by this finding, we propose active subspace embedded KAN (asKAN), a hierarchical framework that synergizes KAN's function representation with active subspace methodology. The architecture strategically embeds active subspace detection between KANs, where the active subspace method is used to identify the primary ridge directions and the independent variables are adaptively projected onto these critical dimensions. The proposed asKAN is implemented in an iterative way without increasing the number of neurons in the original KAN. The proposed method is validated through function fitting, solving the Poisson equation, and reconstructing sound field. Compared with KAN, asKAN significantly reduces the error using the same network architecture. The results suggest that asKAN enhances the capability of KAN in fitting and solving equations in the form of ridge functions.
       


### [Solving the fully nonlinear Monge-Ampère equation using the Legendre-Kolmogorov-Arnold Network method](https://arxiv.org/abs/2504.05022)

**Authors:**
Bingcheng Hu, Lixiang Jin, Zhaoxiang Li

**Abstract:**
In this paper, we propose a novel neural network framework, the Legendre-Kolmogorov-Arnold Network (Legendre-KAN) method, designed to solve fully nonlinear Monge-Ampère equations with Dirichlet boundary conditions. The architecture leverages the orthogonality of Legendre polynomials as basis functions, significantly enhancing both convergence speed and solution accuracy compared to traditional methods. Furthermore, the Kolmogorov-Arnold representation theorem provides a strong theoretical foundation for the interpretability and optimization of the network. We demonstrate the effectiveness of the proposed method through numerical examples, involving both smooth and singular solutions in various dimensions. This work not only addresses the challenges of solving high-dimensional and singular Monge-Ampère equations but also highlights the potential of neural network-based approaches for complex partial differential equations. Additionally, the method is applied to the optimal transport problem in image mapping, showcasing its practical utility in geometric image transformation. This approach is expected to pave the way for further enhancement of KAN-based applications and numerical solutions of PDEs across a wide range of scientific and engineering fields.
       


### [KAN-SAM: Kolmogorov-Arnold Network Guided Segment Anything Model for RGB-T Salient Object Detection](https://arxiv.org/abs/2504.05878)

**Authors:**
Xingyuan Li, Ruichao Hou, Tongwei Ren, Gangshan Wu

**Abstract:**
Existing RGB-thermal salient object detection (RGB-T SOD) methods aim to identify visually significant objects by leveraging both RGB and thermal modalities to enable robust performance in complex scenarios, but they often suffer from limited generalization due to the constrained diversity of available datasets and the inefficiencies in constructing multi-modal representations. In this paper, we propose a novel prompt learning-based RGB-T SOD method, named KAN-SAM, which reveals the potential of visual foundational models for RGB-T SOD tasks. Specifically, we extend Segment Anything Model 2 (SAM2) for RGB-T SOD by introducing thermal features as guiding prompts through efficient and accurate Kolmogorov-Arnold Network (KAN) adapters, which effectively enhance RGB representations and improve robustness. Furthermore, we introduce a mutually exclusive random masking strategy to reduce reliance on RGB data and improve generalization. Experimental results on benchmarks demonstrate superior performance over the state-of-the-art methods.
       


### [Physics-informed KAN PointNet: Deep learning for simultaneous solutions to inverse problems in incompressible flow on numerous irregular geometries](https://arxiv.org/abs/2504.06327)

**Authors:**
Ali Kashefi, Tapan Mukerji

**Abstract:**
Kolmogorov-Arnold Networks (KANs) have gained attention as a promising alternative to traditional Multilayer Perceptrons (MLPs) for deep learning applications in computational physics, especially within the framework of physics-informed neural networks (PINNs). Physics-informed Kolmogorov-Arnold Networks (PIKANs) and their variants have been introduced and evaluated to solve inverse problems. However, similar to PINNs, current versions of PIKANs are limited to obtaining solutions for a single computational domain per training run; consequently, a new geometry requires retraining the model from scratch. Physics-informed PointNet (PIPN) was introduced to address this limitation for PINNs. In this work, we introduce physics-informed Kolmogorov-Arnold PointNet (PI-KAN-PointNet) to extend this capability to PIKANs. PI-KAN-PointNet enables the simultaneous solution of an inverse problem over multiple irregular geometries within a single training run, reducing computational costs. We construct KANs using Jacobi polynomials and investigate their performance by considering Jacobi polynomials of different degrees and types in terms of both computational cost and prediction accuracy. As a benchmark test case, we consider natural convection in a square enclosure with a cylinder, where the cylinder's shape varies across a dataset of 135 geometries. We compare the performance of PI-KAN-PointNet with that of PIPN (i.e., physics-informed PointNet with MLPs) and observe that, with approximately an equal number of trainable parameters and similar computational cost, PI-KAN-PointNet provides more accurate predictions. Finally, we explore the combination of KAN and MLP in constructing a physics-informed PointNet. Our findings indicate that a physics-informed PointNet model employing MLP layers as the encoder and KAN layers as the decoder represents the optimal configuration among all models investigated.
       


### [TabKAN: Advancing Tabular Data Analysis using Kolmogorov-Arnold Network](https://arxiv.org/abs/2504.06559)

**Authors:**
Ali Eslamian, Alireza Afzal Aghaei, Qiang Cheng

**Abstract:**
Tabular data analysis presents unique challenges due to its heterogeneous feature types, missing values, and complex interactions. While traditional machine learning methods, such as gradient boosting, often outperform deep learning approaches, recent advancements in neural architectures offer promising alternatives. This paper introduces TabKAN, a novel framework that advances tabular data modeling using Kolmogorov-Arnold Networks (KANs). Unlike conventional deep learning models, KANs leverge learnable activation functions on edges, which improve both interpretability and training efficiency. Our contributions include: (1) the introduction of modular KAN-based architectures for tabular data analysis, (2) the development of a transfer learning framework for KAN models that supports knowledge transfer between domains, (3) the development of model-specific interpretability for tabular data learning, which reduces dependence on post hoc and model-agnostic analysis, and (4) comprehensive evaluation of vanilla supervised learning across binary and multi-class classification tasks. Through extensive benchmarking on diverse public datasets, TabKAN demonstrates superior performance in supervised learning while significantly outperforming classical and Transformer-based models in transfer learning scenarios. Our findings highlight the advantage of KAN-based architectures in transferring knowledge across domains and narrowing the gap between traditional machine learning and deep learning for structured data.
       


### [Representation Meets Optimization: Training PINNs and PIKANs for Gray-Box Discovery in Systems Pharmacology](https://arxiv.org/abs/2504.07379)

**Authors:**
Nazanin Ahmadi Daryakenari, Khemraj Shukla, George Em Karniadakis

**Abstract:**
Physics-Informed Kolmogorov-Arnold Networks (PIKANs) are gaining attention as an effective counterpart to the original multilayer perceptron-based Physics-Informed Neural Networks (PINNs). Both representation models can address inverse problems and facilitate gray-box system identification. However, a comprehensive understanding of their performance in terms of accuracy and speed remains underexplored. In particular, we introduce a modified PIKAN architecture, tanh-cPIKAN, which is based on Chebyshev polynomials for parametrization of the univariate functions with an extra nonlinearity for enhanced performance. We then present a systematic investigation of how choices of the optimizer, representation, and training configuration influence the performance of PINNs and PIKANs in the context of systems pharmacology modeling. We benchmark a wide range of first-order, second-order, and hybrid optimizers, including various learning rate schedulers. We use the new Optax library to identify the most effective combinations for learning gray-boxes under ill-posed, non-unique, and data-sparse conditions. We examine the influence of model architecture (MLP vs. KAN), numerical precision (single vs. double), the need for warm-up phases for second-order methods, and sensitivity to the initial learning rate. We also assess the optimizer scalability for larger models and analyze the trade-offs introduced by JAX in terms of computational efficiency and numerical accuracy. Using two representative systems pharmacology case studies - a pharmacokinetics model and a chemotherapy drug-response model - we offer practical guidance on selecting optimizers and representation models/architectures for robust and efficient gray-box discovery. Our findings provide actionable insights for improving the training of physics-informed networks in biomedical applications and beyond.
       


### [A Case for Kolmogorov-Arnold Networks in Prefetching: Towards Low-Latency, Generalizable ML-Based Prefetchers](https://arxiv.org/abs/2504.09074)

**Authors:**
Dhruv Kulkarni, Bharat Bhammar, Henil Thaker, Pranav Dhobi, R. P. Gohil, Sai Manoj Pudukotai Dinkarrao

**Abstract:**
The memory wall problem arises due to the disparity between fast processors and slower memory, causing significant delays in data access, even more so on edge devices. Data prefetching is a key strategy to address this, with traditional methods evolving to incorporate Machine Learning (ML) for improved accuracy. Modern prefetchers must balance high accuracy with low latency to further practicality. We explore the applicability of utilizing Kolmogorov-Arnold Networks (KAN) with learnable activation functions,a prefetcher we implemented called KANBoost, to further this aim. KANs are a novel, state-of-the-art model that work on breaking down continuous, bounded multi-variate functions into functions of their constituent variables, and use these constitutent functions as activations on each individual neuron. KANBoost predicts the next memory access by modeling deltas between consecutive addresses, offering a balance of accuracy and efficiency to mitigate the memory wall problem with minimal overhead, instead of relying on address-correlation prefetching. Initial results indicate that KAN-based prefetching reduces inference latency (18X lower than state-of-the-art ML prefetchers) while achieving moderate IPC improvements (2.5\% over no-prefetching). While KANs still face challenges in capturing long-term dependencies, we propose that future research should explore hybrid models that combine KAN efficiency with stronger sequence modeling techniques, paving the way for practical ML-based prefetching in edge devices and beyond.
       


### [GPT Meets Graphs and KAN Splines: Testing Novel Frameworks on Multitask Fine-Tuned GPT-2 with LoRA](https://arxiv.org/abs/2504.10490)

**Authors:**
Gabriel Bo, Marc Bernardino, Justin Gu

**Abstract:**
We explore the potential of integrating learnable and interpretable modules--specifically Kolmogorov-Arnold Networks (KAN) and graph-based representations--within a pre-trained GPT-2 model to enhance multi-task learning accuracy. Motivated by the recent surge in using KAN and graph attention (GAT) architectures in chain-of-thought (CoT) models and debates over their benefits compared to simpler architectures like MLPs, we begin by enhancing a standard self-attention transformer using Low-Rank Adaptation (LoRA), fine-tuning hyperparameters, and incorporating L2 regularization. This approach yields significant improvements. To further boost interpretability and richer representations, we develop two variants that attempt to improve the standard KAN and GAT: Graph LoRA and Hybrid-KAN LoRA (Learnable GPT). However, systematic evaluations reveal that neither variant outperforms the optimized LoRA-enhanced transformer, which achieves 55.249% accuracy on the SST test set, 99.18% on the CFIMDB dev set, and 89.9% paraphrase detection test accuracy. On sonnet generation, we get a CHRF score of 42.097. These findings highlight that efficient parameter adaptation via LoRA remains the most effective strategy for our tasks: sentiment analysis, paraphrase detection, and sonnet generation.
       


### [MLPs and KANs for data-driven learning in physical problems: A performance comparison](https://arxiv.org/abs/2504.11397)

**Authors:**
Raghav Pant, Sikan Li, Xingjian Li, Hassan Iqbal, Krishna Kumar

**Abstract:**
There is increasing interest in solving partial differential equations (PDEs) by casting them as machine learning problems. Recently, there has been a spike in exploring Kolmogorov-Arnold Networks (KANs) as an alternative to traditional neural networks represented by Multi-Layer Perceptrons (MLPs). While showing promise, their performance advantages in physics-based problems remain largely unexplored. Several critical questions persist: Can KANs capture complex physical dynamics and under what conditions might they outperform traditional architectures? In this work, we present a comparative study of KANs and MLPs for learning physical systems governed by PDEs. We assess their performance when applied in deep operator networks (DeepONet) and graph network-based simulators (GNS), and test them on physical problems that vary significantly in scale and complexity. Drawing inspiration from the Kolmogorov Representation Theorem, we examine the behavior of KANs and MLPs across shallow and deep network architectures. Our results reveal that although KANs do not consistently outperform MLPs when configured as deep neural networks, they demonstrate superior expressiveness in shallow network settings, significantly outpacing MLPs in accuracy over our test cases. This suggests that KANs are a promising choice, offering a balance of efficiency and accuracy in applications involving physical systems.
       


### [Towards Explainable Fusion and Balanced Learning in Multimodal Sentiment Analysis](https://arxiv.org/abs/2504.12151)

**Authors:**
Miaosen Luo, Yuncheng Jiang, Sijie Mai

**Abstract:**
Multimodal Sentiment Analysis (MSA) faces two critical challenges: the lack of interpretability in the decision logic of multimodal fusion and modality imbalance caused by disparities in inter-modal information density. To address these issues, we propose KAN-MCP, a novel framework that integrates the interpretability of Kolmogorov-Arnold Networks (KAN) with the robustness of the Multimodal Clean Pareto (MCPareto) framework. First, KAN leverages its univariate function decomposition to achieve transparent analysis of cross-modal interactions. This structural design allows direct inspection of feature transformations without relying on external interpretation tools, thereby ensuring both high expressiveness and interpretability. Second, the proposed MCPareto enhances robustness by addressing modality imbalance and noise interference. Specifically, we introduce the Dimensionality Reduction and Denoising Modal Information Bottleneck (DRD-MIB) method, which jointly denoises and reduces feature dimensionality. This approach provides KAN with discriminative low-dimensional inputs to reduce the modeling complexity of KAN while preserving critical sentiment-related information. Furthermore, MCPareto dynamically balances gradient contributions across modalities using the purified features output by DRD-MIB, ensuring lossless transmission of auxiliary signals and effectively alleviating modality imbalance. This synergy of interpretability and robustness not only achieves superior performance on benchmark datasets such as CMU-MOSI, CMU-MOSEI, and CH-SIMS v2 but also offers an intuitive visualization interface through KAN's interpretable architecture.
       


### [Approximation Bounds for Transformer Networks with Application to Regression](https://arxiv.org/abs/2504.12175)

**Authors:**
Yuling Jiao, Yanming Lai, Defeng Sun, Yang Wang, Bokai Yan

**Abstract:**
We explore the approximation capabilities of Transformer networks for Hölder and Sobolev functions, and apply these results to address nonparametric regression estimation with dependent observations. First, we establish novel upper bounds for standard Transformer networks approximating sequence-to-sequence mappings whose component functions are Hölder continuous with smoothness index $γ\in (0,1]$. To achieve an approximation error $\varepsilon$ under the $L^p$-norm for $p \in [1, \infty]$, it suffices to use a fixed-depth Transformer network whose total number of parameters scales as $\varepsilon^{-d_x n / γ}$. This result not only extends existing findings to include the case $p = \infty$, but also matches the best known upper bounds on number of parameters previously obtained for fixed-depth FNNs and RNNs. Similar bounds are also derived for Sobolev functions. Second, we derive explicit convergence rates for the nonparametric regression problem under various $β$-mixing data assumptions, which allow the dependence between observations to weaken over time. Our bounds on the sample complexity impose no constraints on weight magnitudes. Lastly, we propose a novel proof strategy to establish approximation bounds, inspired by the Kolmogorov-Arnold representation theorem. We show that if the self-attention layer in a Transformer can perform column averaging, the network can approximate sequence-to-sequence Hölder functions, offering new insights into the interpretability of self-attention mechanisms.
       


### [ChemKANs for Combustion Chemistry Modeling and Acceleration](https://arxiv.org/abs/2504.12580)

**Authors:**
Benjamin C. Koenig, Suyong Kim, Sili Deng

**Citation Count:**
1

**Abstract:**
Efficient chemical kinetic model inference and application for combustion problems is challenging due to large ODE systems and wideley separated time scales. Machine learning techniques have been proposed to streamline these models, though strong nonlinearity and numerical stiffness combined with noisy data sources makes their application challenging. The recently developed Kolmogorov-Arnold Networks (KANs) and KAN ordinary differential equations (KAN-ODEs) have been demonstrated as powerful tools for scientific applications thanks to their rapid neural scaling, improved interpretability, and smooth activation functions. Here, we develop ChemKANs by augmenting the KAN-ODE framework with physical knowledge of the flow of information through the relevant kinetic and thermodynamic laws, as well as an elemental conservation loss term. This novel framework encodes strong inductive bias that enables streamlined training and higher accuracy predictions, while facilitating parameter sparsity through full sharing of information across all inputs and outputs. In a model inference investigation, we find that ChemKANs exhibit no overfitting or model degradation when tasked with extracting predictive models from data that is both sparse and noisy, a task that a standard DeepONet struggles to accomplish. Next, we find that a remarkably parameter-lean ChemKAN (only 344 parameters) can accurately represent hydrogen combustion chemistry, providing a 2x acceleration over the detailed chemistry in a solver that is generalizable to larger-scale turbulent flow simulations. These demonstrations indicate potential for ChemKANs in combustion physics and chemical kinetics, and demonstrate the scalability of generic KAN-ODEs in significantly larger and more numerically challenging problems than previously studied.
       


### [Transformers Can Overcome the Curse of Dimensionality: A Theoretical Study from an Approximation Perspective](https://arxiv.org/abs/2504.13558)

**Authors:**
Yuling Jiao, Yanming Lai, Yang Wang, Bokai Yan

**Abstract:**
The Transformer model is widely used in various application areas of machine learning, such as natural language processing. This paper investigates the approximation of the Hölder continuous function class $\mathcal{H}_{Q}^β\left([0,1]^{d\times n},\mathbb{R}^{d\times n}\right)$ by Transformers and constructs several Transformers that can overcome the curse of dimensionality. These Transformers consist of one self-attention layer with one head and the softmax function as the activation function, along with several feedforward layers. For example, to achieve an approximation accuracy of $ε$, if the activation functions of the feedforward layers in the Transformer are ReLU and floor, only $\mathcal{O}\left(\log\frac{1}ε\right)$ layers of feedforward layers are needed, with widths of these layers not exceeding $\mathcal{O}\left(\frac{1}{ε^{2/β}}\log\frac{1}ε\right)$. If other activation functions are allowed in the feedforward layers, the width of the feedforward layers can be further reduced to a constant. These results demonstrate that Transformers have a strong expressive capability. The construction in this paper is based on the Kolmogorov-Arnold Representation Theorem and does not require the concept of contextual mapping, hence our proof is more intuitively clear compared to previous Transformer approximation works. Additionally, the translation technique proposed in this paper helps to apply the previous approximation results of feedforward neural networks to Transformer research.
       


### [KAN or MLP? Point Cloud Shows the Way Forward](https://arxiv.org/abs/2504.13593)

**Authors:**
Yan Shi, Qingdong He, Yijun Liu, Xiaoyu Liu, Jingyong Su

**Abstract:**
Multi-Layer Perceptrons (MLPs) have become one of the fundamental architectural component in point cloud analysis due to its effective feature learning mechanism. However, when processing complex geometric structures in point clouds, MLPs' fixed activation functions struggle to efficiently capture local geometric features, while suffering from poor parameter efficiency and high model redundancy. In this paper, we propose PointKAN, which applies Kolmogorov-Arnold Networks (KANs) to point cloud analysis tasks to investigate their efficacy in hierarchical feature representation. First, we introduce a Geometric Affine Module (GAM) to transform local features, improving the model's robustness to geometric variations. Next, in the Local Feature Processing (LFP), a parallel structure extracts both group-level features and global context, providing a rich representation of both fine details and overall structure. Finally, these features are combined and processed in the Global Feature Processing (GFP). By repeating these operations, the receptive field gradually expands, enabling the model to capture complete geometric information of the point cloud. To overcome the high parameter counts and computational inefficiency of standard KANs, we develop Efficient-KANs in the PointKAN-elite variant, which significantly reduces parameters while maintaining accuracy. Experimental results demonstrate that PointKAN outperforms PointMLP on benchmark datasets such as ModelNet40, ScanObjectNN, and ShapeNetPart, with particularly strong performance in Few-shot Learning task. Additionally, PointKAN achieves substantial reductions in parameter counts and computational complexity (FLOPs). This work highlights the potential of KANs-based architectures in 3D vision and opens new avenues for research in point cloud understanding.
       


### [Efficient Implicit Neural Compression of Point Clouds via Learnable Activation in Latent Space](https://arxiv.org/abs/2504.14471)

**Authors:**
Yichi Zhang, Qianqian Yang

**Abstract:**
Implicit Neural Representations (INRs), also known as neural fields, have emerged as a powerful paradigm in deep learning, parameterizing continuous spatial fields using coordinate-based neural networks. In this paper, we propose \textbf{PICO}, an INR-based framework for static point cloud compression. Unlike prevailing encoder-decoder paradigms, we decompose the point cloud compression task into two separate stages: geometry compression and attribute compression, each with distinct INR optimization objectives. Inspired by Kolmogorov-Arnold Networks (KANs), we introduce a novel network architecture, \textbf{LeAFNet}, which leverages learnable activation functions in the latent space to better approximate the target signal's implicit function. By reformulating point cloud compression as neural parameter compression, we further improve compression efficiency through quantization and entropy coding. Experimental results demonstrate that \textbf{LeAFNet} outperforms conventional MLPs in INR-based point cloud compression. Furthermore, \textbf{PICO} achieves superior geometry compression performance compared to the current MPEG point cloud compression standard, yielding an average improvement of $4.92$ dB in D1 PSNR. In joint geometry and attribute compression, our approach exhibits highly competitive results, with an average PCQM gain of $2.7 \times 10^{-3}$.
       


### [Kolmogorov-Arnold Networks: Approximation and Learning Guarantees for Functions and their Derivatives](https://arxiv.org/abs/2504.15110)

**Authors:**
Anastasis Kratsios, Takashi Furuya

**Abstract:**
Inspired by the Kolmogorov-Arnold superposition theorem, Kolmogorov-Arnold Networks (KANs) have recently emerged as an improved backbone for most deep learning frameworks, promising more adaptivity than their multilayer perception (MLP) predecessor by allowing for trainable spline-based activation functions. In this paper, we probe the theoretical foundations of the KAN architecture by showing that it can optimally approximate any Besov function in $B^{s}_{p,q}(\mathcal{X})$ on a bounded open, or even fractal, domain $\mathcal{X}$ in $\mathbb{R}^d$ at the optimal approximation rate with respect to any weaker Besov norm $B^α_{p,q}(\mathcal{X})$; where $α< s$. We complement our approximation guarantee with a dimension-free estimate on the sample complexity of a residual KAN model when learning a function of Besov regularity from $N$ i.i.d. noiseless samples. Our KAN architecture incorporates contemporary deep learning wisdom by leveraging residual/skip connections between layers.
       


### [Conformalized-KANs: Uncertainty Quantification with Coverage Guarantees for Kolmogorov-Arnold Networks (KANs) in Scientific Machine Learning](https://arxiv.org/abs/2504.15240)

**Authors:**
Amirhossein Mollaali, Christian Bolivar Moya, Amanda A. Howard, Alexander Heinlein, Panos Stinis, Guang Lin

**Citation Count:**
1

**Abstract:**
This paper explores uncertainty quantification (UQ) methods in the context of Kolmogorov-Arnold Networks (KANs). We apply an ensemble approach to KANs to obtain a heuristic measure of UQ, enhancing interpretability and robustness in modeling complex functions. Building on this, we introduce Conformalized-KANs, which integrate conformal prediction, a distribution-free UQ technique, with KAN ensembles to generate calibrated prediction intervals with guaranteed coverage. Extensive numerical experiments are conducted to evaluate the effectiveness of these methods, focusing particularly on the robustness and accuracy of the prediction intervals under various hyperparameter settings. We show that the conformal KAN predictions can be applied to recent extensions of KANs, including Finite Basis KANs (FBKANs) and multifideilty KANs (MFKANs). The results demonstrate the potential of our approaches to improve the reliability and applicability of KANs in scientific machine learning.
       


### [DAE-KAN: A Kolmogorov-Arnold Network Model for High-Index Differential-Algebraic Equations](https://arxiv.org/abs/2504.15806)

**Authors:**
Kai Luo, Juan Tang, Mingchao Cai, Xiaoqing Zeng, Manqi Xie, Ming Yan

**Abstract:**
Kolmogorov-Arnold Networks (KANs) have emerged as a promising alternative to Multi-layer Perceptrons (MLPs) due to their superior function-fitting abilities in data-driven modeling. In this paper, we propose a novel framework, DAE-KAN, for solving high-index differential-algebraic equations (DAEs) by integrating KANs with Physics-Informed Neural Networks (PINNs). This framework not only preserves the ability of traditional PINNs to model complex systems governed by physical laws but also enhances their performance by leveraging the function-fitting strengths of KANs. Numerical experiments demonstrate that for DAE systems ranging from index-1 to index-3, DAE-KAN reduces the absolute errors of both differential and algebraic variables by 1 to 2 orders of magnitude compared to traditional PINNs. To assess the effectiveness of this approach, we analyze the drift-off error and find that both PINNs and DAE-KAN outperform classical numerical methods in controlling this phenomenon. Our results highlight the potential of neural network methods, particularly DAE-KAN, in solving high-index DAEs with substantial computational accuracy and generalization, offering a promising solution for challenging partial differential-algebraic equations.
       


### [iTFKAN: Interpretable Time Series Forecasting with Kolmogorov-Arnold Network](https://arxiv.org/abs/2504.16432)

**Authors:**
Ziran Liang, Rui An, Wenqi Fan, Yanghui Rao, Yuxuan Liang

**Abstract:**
As time evolves, data within specific domains exhibit predictability that motivates time series forecasting to predict future trends from historical data. However, current deep forecasting methods can achieve promising performance but generally lack interpretability, hindering trustworthiness and practical deployment in safety-critical applications such as auto-driving and healthcare. In this paper, we propose a novel interpretable model, iTFKAN, for credible time series forecasting. iTFKAN enables further exploration of model decision rationales and underlying data patterns due to its interpretability achieved through model symbolization. Besides, iTFKAN develops two strategies, prior knowledge injection, and time-frequency synergy learning, to effectively guide model learning under complex intertwined time series data. Extensive experimental results demonstrated that iTFKAN can achieve promising forecasting performance while simultaneously possessing high interpretive capabilities.
       


### [FiberKAN: Kolmogorov-Arnold Networks for Nonlinear Fiber Optics](https://arxiv.org/abs/2504.18833)

**Authors:**
Xiaotian Jiang, Min Zhang, Xiao Luo, Zelai Yu, Yiming Meng, Danshi Wang

**Venue:**
Journal of Lightwave Technology

**Abstract:**
Scientific discovery and dynamic characterization of the physical system play a critical role in understanding, learning, and modeling the physical phenomena and behaviors in various fields. Although theories and laws of many system dynamics have been derived from rigorous first principles, there are still a considerable number of complex dynamics that have not yet been discovered and characterized, which hinders the progress of science in corresponding fields. To address these challenges, artificial intelligence for science (AI4S) has emerged as a burgeoning research field. In this paper, a Kolmogorov-Arnold Network (KAN)-based AI4S framework named FiberKAN is proposed for scientific discovery and dynamic characterization of nonlinear fiber optics. Unlike the classic multi-layer perceptron (MLP) structure, the trainable and transparent activation functions in KAN make the network have stronger physical interpretability and nonlinear characterization abilities. Multiple KANs are established for fiber-optic system dynamics under various physical effects. Results show that KANs can well discover and characterize the explicit, implicit, and non-analytical solutions under different effects, and achieve better performance than MLPs with the equivalent scale of trainable parameters. Moreover, the effectiveness, computational cost, interactivity, noise resistance, transfer learning ability, and comparison between related algorithms in fiber-optic systems are also studied and analyzed. This work highlights the transformative potential of KAN, establishing it as a pioneering paradigm in AI4S that propels advancements in nonlinear fiber optics, and fosters groundbreaking innovations across a broad spectrum of scientific and engineering disciplines.
       


### [A Unified Benchmark of Federated Learning with Kolmogorov-Arnold Networks for Medical Imaging](https://arxiv.org/abs/2504.19639)

**Authors:**
Youngjoon Lee, Jinu Gong, Joonhyuk Kang

**Citation Count:**
1

**Abstract:**
Federated Learning (FL) enables model training across decentralized devices without sharing raw data, thereby preserving privacy in sensitive domains like healthcare. In this paper, we evaluate Kolmogorov-Arnold Networks (KAN) architectures against traditional MLP across six state-of-the-art FL algorithms on a blood cell classification dataset. Notably, our experiments demonstrate that KAN can effectively replace MLP in federated environments, achieving superior performance with simpler architectures. Furthermore, we analyze the impact of key hyperparameters-grid size and network architecture-on KAN performance under varying degrees of Non-IID data distribution. Additionally, our ablation studies reveal that optimizing KAN width while maintaining minimal depth yields the best performance in federated settings. As a result, these findings establish KAN as a promising alternative for privacy-preserving medical imaging applications in distributed healthcare. To the best of our knowledge, this is the first comprehensive benchmark of KAN in FL settings for medical imaging task.
       


## May
### [Anant-Net: Breaking the Curse of Dimensionality with Scalable and Interpretable Neural Surrogate for High-Dimensional PDEs](https://arxiv.org/abs/2505.03595)

**Authors:**
Sidharth S. Menon, Ameya D. Jagtap

**Abstract:**
High-dimensional partial differential equations (PDEs) arise in diverse scientific and engineering applications but remain computationally intractable due to the curse of dimensionality. Traditional numerical methods struggle with the exponential growth in computational complexity, particularly on hypercubic domains, where the number of required collocation points increases rapidly with dimensionality. Here, we introduce Anant-Net, an efficient neural surrogate that overcomes this challenge, enabling the solution of PDEs in high dimensions. Unlike hyperspheres, where the internal volume diminishes as dimensionality increases, hypercubes retain or expand their volume (for unit or larger length), making high-dimensional computations significantly more demanding. Anant-Net efficiently incorporates high-dimensional boundary conditions and minimizes the PDE residual at high-dimensional collocation points. To enhance interpretability, we integrate Kolmogorov-Arnold networks into the Anant-Net architecture. We benchmark Anant-Net's performance on several linear and nonlinear high-dimensional equations, including the Poisson, Sine-Gordon, and Allen-Cahn equations, demonstrating high accuracy and robustness across randomly sampled test points from high-dimensional space. Importantly, Anant-Net achieves these results with remarkable efficiency, solving 300-dimensional problems on a single GPU within a few hours. We also compare Anant-Net's results for accuracy and runtime with other state-of-the-art methods. Our findings establish Anant-Net as an accurate, interpretable, and scalable framework for efficiently solving high-dimensional PDEs.
       


### [Novel Extraction of Discriminative Fine-Grained Feature to Improve Retinal Vessel Segmentation](https://arxiv.org/abs/2505.03896)

**Authors:**
Shuang Zeng, Chee Hong Lee, Micky C Nnamdi, Wenqi Shi, J Ben Tamo, Lei Zhu, Hangzhou He, Xinliang Zhang, Qian Chen, May D. Wang, Yanye Lu, Qiushi Ren

**Abstract:**
Retinal vessel segmentation is a vital early detection method for several severe ocular diseases. Despite significant progress in retinal vessel segmentation with the advancement of Neural Networks, there are still challenges to overcome. Specifically, retinal vessel segmentation aims to predict the class label for every pixel within a fundus image, with a primary focus on intra-image discrimination, making it vital for models to extract more discriminative features. Nevertheless, existing methods primarily focus on minimizing the difference between the output from the decoder and the label, but ignore fully using feature-level fine-grained representations from the encoder. To address these issues, we propose a novel Attention U-shaped Kolmogorov-Arnold Network named AttUKAN along with a novel Label-guided Pixel-wise Contrastive Loss for retinal vessel segmentation. Specifically, we implement Attention Gates into Kolmogorov-Arnold Networks to enhance model sensitivity by suppressing irrelevant feature activations and model interpretability by non-linear modeling of KAN blocks. Additionally, we also design a novel Label-guided Pixel-wise Contrastive Loss to supervise our proposed AttUKAN to extract more discriminative features by distinguishing between foreground vessel-pixel pairs and background pairs. Experiments are conducted across four public datasets including DRIVE, STARE, CHASE_DB1, HRF and our private dataset. AttUKAN achieves F1 scores of 82.50%, 81.14%, 81.34%, 80.21% and 80.09%, along with MIoU scores of 70.24%, 68.64%, 68.59%, 67.21% and 66.94% in the above datasets, which are the highest compared to 11 networks for retinal vessel segmentation. Quantitative and qualitative results show that our AttUKAN achieves state-of-the-art performance and outperforms existing retinal vessel segmentation methods. Our code will be available at https://github.com/stevezs315/AttUKAN.
       


### [Hyb-KAN ViT: Hybrid Kolmogorov-Arnold Networks Augmented Vision Transformer](https://arxiv.org/abs/2505.04740)

**Authors:**
Sainath Dey, Mitul Goswami, Jashika Sethi, Prasant Kumar Pattnaik

**Abstract:**
This study addresses the inherent limitations of Multi-Layer Perceptrons (MLPs) in Vision Transformers (ViTs) by introducing Hybrid Kolmogorov-Arnold Network (KAN)-ViT (Hyb-KAN ViT), a novel framework that integrates wavelet-based spectral decomposition and spline-optimized activation functions, prior work has failed to focus on the prebuilt modularity of the ViT architecture and integration of edge detection capabilities of Wavelet functions. We propose two key modules: Efficient-KAN (Eff-KAN), which replaces MLP layers with spline functions and Wavelet-KAN (Wav-KAN), leveraging orthogonal wavelet transforms for multi-resolution feature extraction. These modules are systematically integrated in ViT encoder layers and classification heads to enhance spatial-frequency modeling while mitigating computational bottlenecks. Experiments on ImageNet-1K (Image Recognition), COCO (Object Detection and Instance Segmentation), and ADE20K (Semantic Segmentation) demonstrate state-of-the-art performance with Hyb-KAN ViT. Ablation studies validate the efficacy of wavelet-driven spectral priors in segmentation and spline-based efficiency in detection tasks. The framework establishes a new paradigm for balancing parameter efficiency and multi-scale representation in vision architectures.
       


### [Prediction via Shapley Value Regression](https://arxiv.org/abs/2505.04775)

**Authors:**
Amr Alkhatib, Roman Bresson, Henrik Boström, Michalis Vazirgiannis

**Abstract:**
Shapley values have several desirable, theoretically well-supported, properties for explaining black-box model predictions. Traditionally, Shapley values are computed post-hoc, leading to additional computational cost at inference time. To overcome this, a novel method, called ViaSHAP, is proposed, that learns a function to compute Shapley values, from which the predictions can be derived directly by summation. Two approaches to implement the proposed method are explored; one based on the universal approximation theorem and the other on the Kolmogorov-Arnold representation theorem. Results from a large-scale empirical investigation are presented, showing that ViaSHAP using Kolmogorov-Arnold Networks performs on par with state-of-the-art algorithms for tabular data. It is also shown that the explanations of ViaSHAP are significantly more accurate than the popular approximator FastSHAP on both tabular data and images.
       


### [OccuEMBED: Occupancy Extraction Merged with Building Energy Disaggregation for Occupant-Responsive Operation at Scale](https://arxiv.org/abs/2505.05478)

**Authors:**
Yufei Zhang, Andrew Sonta

**Abstract:**
Buildings account for a significant share of global energy consumption and emissions, making it critical to operate them efficiently. As electricity grids become more volatile with renewable penetration, buildings must provide flexibility to support grid stability. Building automation plays a key role in enhancing efficiency and flexibility via centralized operations, but it must prioritize occupant-centric strategies to balance energy and comfort targets. However, incorporating occupant information into large-scale, centralized building operations remains challenging due to data limitations. We investigate the potential of using whole-building smart meter data to infer both occupancy and system operations. Integrating these insights into data-driven building energy analysis allows more occupant-centric energy-saving and flexibility at scale. Specifically, we propose OccuEMBED, a unified framework for occupancy inference and system-level load analysis. It combines two key components: a probabilistic occupancy profile generator, and a controllable and interpretable load disaggregator supported by Kolmogorov-Arnold Networks (KAN). This design embeds knowledge of occupancy patterns and load-occupancy-weather relationships into deep learning models. We conducted comprehensive evaluations to demonstrate its effectiveness across synthetic and real-world datasets compared to various occupancy inference baselines. OccuEMBED always achieved average F1 scores above 0.8 in discrete occupancy inference and RMSE within 0.1-0.2 for continuous occupancy ratios. We further demonstrate how OccuEMBED integrates with building load monitoring platforms to display occupancy profiles, analyze system-level operations, and inform occupant-responsive strategies. Our model lays a robust foundation in scaling occupant-centric building management systems to meet the challenges of an evolving energy system.
       


### [Improving Generalizability of Kolmogorov-Arnold Networks via Error-Correcting Output Codes](https://arxiv.org/abs/2505.05798)

**Authors:**
Youngjoon Lee, Jinu Gong, Joonhyuk Kang

**Abstract:**
Kolmogorov-Arnold Networks (KAN) offer universal function approximation using univariate spline compositions without nonlinear activations. In this work, we integrate Error-Correcting Output Codes (ECOC) into the KAN framework to transform multi-class classification into multiple binary tasks, improving robustness via Hamming-distance decoding. Our proposed KAN with ECOC method outperforms vanilla KAN on a challenging blood cell classification dataset, achieving higher accuracy under diverse hyperparameter settings. Ablation studies further confirm that ECOC consistently enhances performance across FastKAN and FasterKAN variants. These results demonstrate that ECOC integration significantly boosts KAN generalizability in critical healthcare AI applications. To the best of our knowledge, this is the first integration of ECOC with KAN for enhancing multi-class medical image classification performance.
       


### [Mixer-Informer-Based Two-Stage Transfer Learning for Long-Sequence Load Forecasting in Newly Constructed Electric Vehicle Charging Stations](https://arxiv.org/abs/2505.06657)

**Authors:**
Zhenhua Zhou, Bozhen Jiang, Qin Wang

**Abstract:**
The rapid rise in electric vehicle (EV) adoption demands precise charging station load forecasting, challenged by long-sequence temporal dependencies and limited data in new facilities. This study proposes MIK-TST, a novel two-stage transfer learning framework integrating Mixer, Informer, and Kolmogorov-Arnold Networks (KAN). The Mixer fuses multi-source features, Informer captures long-range dependencies via ProbSparse attention, and KAN enhances nonlinear modeling with learnable activation functions. Pre-trained on extensive data and fine-tuned on limited target data, MIK-TST achieves 4% and 8% reductions in MAE and MSE, respectively, outperforming baselines on a dataset of 26 charging stations in Boulder, USA. This scalable solution enhances smart grid efficiency and supports sustainable EV infrastructure expansion.
       


### [Enhancing Federated Learning with Kolmogorov-Arnold Networks: A Comparative Study Across Diverse Aggregation Strategies](https://arxiv.org/abs/2505.07629)

**Authors:**
Yizhou Ma, Zhuoqin Yang, Luis-Daniel Ibáñez

**Abstract:**
Multilayer Perceptron (MLP), as a simple yet powerful model, continues to be widely used in classification and regression tasks. However, traditional MLPs often struggle to efficiently capture nonlinear relationships in load data when dealing with complex datasets. Kolmogorov-Arnold Networks (KAN), inspired by the Kolmogorov-Arnold representation theorem, have shown promising capabilities in modeling complex nonlinear relationships. In this study, we explore the performance of KANs within federated learning (FL) frameworks and compare them to traditional Multilayer Perceptrons. Our experiments, conducted across four diverse datasets demonstrate that KANs consistently outperform MLPs in terms of accuracy, stability, and convergence efficiency. KANs exhibit remarkable robustness under varying client numbers and non-IID data distributions, maintaining superior performance even as client heterogeneity increases. Notably, KANs require fewer communication rounds to converge compared to MLPs, highlighting their efficiency in FL scenarios. Additionally, we evaluate multiple parameter aggregation strategies, with trimmed mean and FedProx emerging as the most effective for optimizing KAN performance. These findings establish KANs as a robust and scalable alternative to MLPs for federated learning tasks, paving the way for their application in decentralized and privacy-preserving environments.
       


### [Symbolic Regression with Multimodal Large Language Models and Kolmogorov Arnold Networks](https://arxiv.org/abs/2505.07956)

**Authors:**
Thomas R. Harvey, Fabian Ruehle, Kit Fraser-Taliente, James Halverson

**Abstract:**
We present a novel approach to symbolic regression using vision-capable large language models (LLMs) and the ideas behind Google DeepMind's Funsearch. The LLM is given a plot of a univariate function and tasked with proposing an ansatz for that function. The free parameters of the ansatz are fitted using standard numerical optimisers, and a collection of such ansätze make up the population of a genetic algorithm. Unlike other symbolic regression techniques, our method does not require the specification of a set of functions to be used in regression, but with appropriate prompt engineering, we can arbitrarily condition the generative step. By using Kolmogorov Arnold Networks (KANs), we demonstrate that ``univariate is all you need'' for symbolic regression, and extend this method to multivariate functions by learning the univariate function on each edge of a trained KAN. The combined expression is then simplified by further processing with a language model.
       


### [AC-PKAN: Attention-Enhanced and Chebyshev Polynomial-Based Physics-Informed Kolmogorov-Arnold Networks](https://arxiv.org/abs/2505.08687)

**Authors:**
Hangwei Zhang, Zhimu Huang, Yan Wang

**Abstract:**
Kolmogorov-Arnold Networks (KANs) have recently shown promise for solving partial differential equations (PDEs). Yet their original formulation is computationally and memory intensive, motivating the introduction of Chebyshev Type-I-based KANs (Chebyshev1KANs). Although Chebyshev1KANs have outperformed the vanilla KANs architecture, our rigorous theoretical analysis reveals that they still suffer from rank collapse, ultimately limiting their expressive capacity. To overcome these limitations, we enhance Chebyshev1KANs by integrating wavelet-activated MLPs with learnable parameters and an internal attention mechanism. We prove that this design preserves a full-rank Jacobian and is capable of approximating solutions to PDEs of arbitrary order. Furthermore, to alleviate the loss instability and imbalance introduced by the Chebyshev polynomial basis, we externally incorporate a Residual Gradient Attention (RGA) mechanism that dynamically re-weights individual loss terms according to their gradient norms and residual magnitudes. By jointly leveraging internal and external attention, we present AC-PKAN, a novel architecture that constitutes an enhancement to weakly supervised Physics-Informed Neural Networks (PINNs) and extends the expressive power of KANs. Experimental results from nine benchmark tasks across three domains show that AC-PKAN consistently outperforms or matches state-of-the-art models such as PINNsFormer, establishing it as a highly effective tool for solving complex real-world engineering problems in zero-data or data-sparse regimes. The code will be made publicly available upon acceptance.
       


### [Personalized Control for Lower Limb Prosthesis Using Kolmogorov-Arnold Networks](https://arxiv.org/abs/2505.09366)

**Authors:**
SeyedMojtaba Mohasel, Alireza Afzal Aghaei, Corey Pew

**Abstract:**
Objective: This paper investigates the potential of learnable activation functions in Kolmogorov-Arnold Networks (KANs) for personalized control in a lower-limb prosthesis. In addition, user-specific vs. pooled training data is evaluated to improve machine learning (ML) and Deep Learning (DL) performance for turn intent prediction.
  Method: Inertial measurement unit (IMU) data from the shank were collected from five individuals with lower-limb amputation performing turning tasks in a laboratory setting. Ability to classify an upcoming turn was evaluated for Multilayer Perceptron (MLP), Kolmogorov-Arnold Network (KAN), convolutional neural network (CNN), and fractional Kolmogorov-Arnold Networks (FKAN). The comparison of MLP and KAN (for ML models) and FKAN and CNN (for DL models) assessed the effectiveness of learnable activation functions. Models were trained separately on user-specific and pooled data to evaluate the impact of training data on their performance.
  Results: Learnable activation functions in KAN and FKAN did not yield significant improvement compared to MLP and CNN, respectively. Training on user-specific data yielded superior results compared to pooled data for ML models ($p < 0.05$). In contrast, no significant difference was observed between user-specific and pooled training for DL models.
  Significance: These findings suggest that learnable activation functions may demonstrate distinct advantages in datasets involving more complex tasks and larger volumes. In addition, pooled training showed comparable performance to user-specific training in DL models, indicating that model training for prosthesis control can utilize data from multiple participants.
       


### [Interpretable Artificial Intelligence for Topological Photonics](https://arxiv.org/abs/2505.10485)

**Authors:**
Ali Ghorashi, Sachin Vaidya, Ziming Liu, Charlotte Loh, Thomas Christensen, Max Tegmark, Marin Soljačić

**Abstract:**
Topological photonic crystals (PhCs) offer robust, disorder-resistant modes engendered by nontrivial band symmetries, but designing PhCs with prescribed topological band properties remains a challenge due to the complexity involved in mapping the continuous real-space design landscape of photonic crystals to the discrete output space of band topology. Here, we introduce a new approach to address this problem, employing Kolmogorov--Arnold networks (KANs) to predict and inversely design the band symmetries of two-dimensional PhCs with two-fold rotational (C2) symmetry. We show that a single-hidden-layer KAN, trained on a dataset of C2-symmetric unit cells, achieves 99% accuracy in classifying the symmetry eigenvalues of the lowest transverse-magnetic band. For interpretability, we use symbolic regression to extract eight algebraic formulas that characterize the band symmetry classes in terms of the Fourier components of the dielectric function. These formulas not only retain the full predictive power of the network but also enable deterministic inverse design. Applying them, we generate 2,000 photonic crystals with target band symmetries, achieving over 92% accuracy even for high-contrast, experimentally realizable structures beyond the training domain.
       


### [MedVKAN: Efficient Feature Extraction with Mamba and KAN for Medical Image Segmentation](https://arxiv.org/abs/2505.11797)

**Authors:**
Hancan Zhu, Jinhao Chen, Guanghua He

**Abstract:**
Medical image segmentation relies heavily on convolutional neural networks (CNNs) and Transformer-based models. However, CNNs are constrained by limited receptive fields, while Transformers suffer from scalability challenges due to their quadratic computational complexity. To address these limitations, recent advances have explored alternative architectures. The state-space model Mamba offers near-linear complexity while capturing long-range dependencies, and the Kolmogorov-Arnold Network (KAN) enhances nonlinear expressiveness by replacing fixed activation functions with learnable ones. Building on these strengths, we propose MedVKAN, an efficient feature extraction model integrating Mamba and KAN. Specifically, we introduce the EFC-KAN module, which enhances KAN with convolutional operations to improve local pixel interaction. We further design the VKAN module, integrating Mamba with EFC-KAN as a replacement for Transformer modules, significantly improving feature extraction. Extensive experiments on five public medical image segmentation datasets show that MedVKAN achieves state-of-the-art performance on four datasets and ranks second on the remaining one. These results validate the potential of Mamba and KAN for medical image segmentation while introducing an innovative and computationally efficient feature extraction framework. The code is available at: https://github.com/beginner-cjh/MedVKAN.
       


### [KHRONOS: a Kernel-Based Neural Architecture for Rapid, Resource-Efficient Scientific Computation](https://arxiv.org/abs/2505.13315)

**Authors:**
Reza T. Batley, Sourav Saha

**Abstract:**
Contemporary models of high dimensional physical systems are constrained by the curse of dimensionality and a reliance on dense data. We introduce KHRONOS (Kernel Expansion Hierarchy for Reduced Order, Neural Optimized Surrogates), an AI framework for model based, model free and model inversion tasks. KHRONOS constructs continuously differentiable target fields with a hierarchical composition of per-dimension kernel expansions, which are tensorized into modes and then superposed. We evaluate KHRONOS on a canonical 2D, Poisson equation benchmark: across 16 to 512 degrees of freedom (DoFs), it obtained L_2-square errors of 5e-4 down to 6e-11. This represents a greater than 100-fold gain over Kolmogorov Arnold Networks (which itself reports a 100 times improvement on MLPs/PINNs with 100 times fewer parameters) when controlling for the number of parameters. This also represents a 1e6-fold improvement in L_2-square error compared to standard linear FEM at comparable DoFs. Inference complexity is dominated by inner products, yielding sub-millisecond full-field predictions that scale to an arbitrary resolution. For inverse problems, KHRONOS facilitates rapid, iterative level set recovery in only a few forward evaluations, with sub-microsecond per sample latency. KHRONOS's scalability, expressivity, and interpretability open new avenues in constrained edge computing, online control, computer vision, and beyond.
       


### [A Kolmogorov-Arnold Neural Model for Cascading Extremes](https://arxiv.org/abs/2505.13370)

**Authors:**
Miguel de Carvalho, Clemente Ferrer, Ronny Vallejos

**Abstract:**
This paper addresses the growing concern of cascading extreme events, such as an extreme earthquake followed by a tsunami, by presenting a novel method for risk assessment focused on these domino effects. The proposed approach develops an extreme value theory framework within a Kolmogorov-Arnold network (KAN) to estimate the probability of one extreme event triggering another, conditionally on a feature vector. An extra layer is added to the KAN's architecture to enforce the definition of the parameter of interest within the unit interval, and we refer to the resulting neural model as KANE (KAN with Natural Enforcement). The proposed method is backed by exhaustive numerical studies and further illustrated with real-world applications to seismology and climatology.
       


### [FlashKAT: Understanding and Addressing Performance Bottlenecks in the Kolmogorov-Arnold Transformer](https://arxiv.org/abs/2505.13813)

**Authors:**
Matthew Raffel, Lizhong Chen

**Abstract:**
The Kolmogorov-Arnold Network (KAN) has been gaining popularity as an alternative to the multi-layer perceptron (MLP) with its increased expressiveness and interpretability. However, the KAN can be orders of magnitude slower due to its increased computational cost and training instability, limiting its applicability to larger-scale tasks. Recently, the Kolmogorov-Arnold Transformer (KAT) has been proposed, which can achieve FLOPs similar to the traditional Transformer with MLPs by leveraging Group-Rational KAN (GR-KAN). Unfortunately, despite the comparable FLOPs, our characterizations reveal that the KAT is still 123x slower in training speeds, indicating that there are other performance bottlenecks beyond FLOPs. In this paper, we conduct a series of experiments to understand the root cause of the slowdown in KAT. We uncover that the slowdown can be isolated to memory stalls and, more specifically, in the backward pass of GR-KAN caused by inefficient gradient accumulation. To address this memory bottleneck, we propose FlashKAT, which builds on our restructured kernel that minimizes gradient accumulation with atomic adds and accesses to slow memory. Evaluations demonstrate that FlashKAT can achieve a training speedup of 86.5x compared with the state-of-the-art KAT, while reducing rounding errors in the coefficient gradients. Our code is available at https://github.com/OSU-STARLAB/FlashKAT.
       


### [X-KAN: Optimizing Local Kolmogorov-Arnold Networks via Evolutionary Rule-Based Machine Learning](https://arxiv.org/abs/2505.14273)

**Authors:**
Hiroki Shiraishi, Hisao Ishibuchi, Masaya Nakata

**Abstract:**
Function approximation is a critical task in various fields. However, existing neural network approaches struggle with locally complex or discontinuous functions due to their reliance on a single global model covering the entire problem space. We propose X-KAN, a novel method that optimizes multiple local Kolmogorov-Arnold Networks (KANs) through an evolutionary rule-based machine learning framework called XCSF. X-KAN combines KAN's high expressiveness with XCSF's adaptive partitioning capability by implementing local KAN models as rule consequents and defining local regions via rule antecedents. Our experimental results on artificial test functions and real-world datasets demonstrate that X-KAN significantly outperforms conventional methods, including XCSF, Multi-Layer Perceptron, and KAN, in terms of approximation accuracy. Notably, X-KAN effectively handles functions with locally complex or discontinuous structures that are challenging for conventional KAN, using a compact set of rules (average 7.2 $\pm$ 2.3 rules). These results validate the effectiveness of using KAN as a local model in XCSF, which evaluates the rule fitness based on both accuracy and generality. Our X-KAN implementation is available at https://github.com/YNU-NakataLab/X-KAN.
       


### [Interpretable Reinforcement Learning for Load Balancing using Kolmogorov-Arnold Networks](https://arxiv.org/abs/2505.14459)

**Authors:**
Kamal Singh, Sami Marouani, Ahmad Al Sheikh, Pham Tran Anh Quang, Amaury Habrard

**Abstract:**
Reinforcement learning (RL) has been increasingly applied to network control problems, such as load balancing. However, existing RL approaches often suffer from lack of interpretability and difficulty in extracting controller equations. In this paper, we propose the use of Kolmogorov-Arnold Networks (KAN) for interpretable RL in network control. We employ a PPO agent with a 1-layer actor KAN model and an MLP Critic network to learn load balancing policies that maximise throughput utility, minimize loss as well as delay. Our approach allows us to extract controller equations from the learned neural networks, providing insights into the decision-making process. We evaluate our approach using different reward functions demonstrating its effectiveness in improving network performance while providing interpretable policies.
       


### [Khan-GCL: Kolmogorov-Arnold Network Based Graph Contrastive Learning with Hard Negatives](https://arxiv.org/abs/2505.15103)

**Authors:**
Zihu Wang, Boxun Xu, Hejia Geng, Peng Li

**Abstract:**
Graph contrastive learning (GCL) has demonstrated great promise for learning generalizable graph representations from unlabeled data. However, conventional GCL approaches face two critical limitations: (1) the restricted expressive capacity of multilayer perceptron (MLP) based encoders, and (2) suboptimal negative samples that either from random augmentations-failing to provide effective 'hard negatives'-or generated hard negatives without addressing the semantic distinctions crucial for discriminating graph data. To this end, we propose Khan-GCL, a novel framework that integrates the Kolmogorov-Arnold Network (KAN) into the GCL encoder architecture, substantially enhancing its representational capacity. Furthermore, we exploit the rich information embedded within KAN coefficient parameters to develop two novel critical feature identification techniques that enable the generation of semantically meaningful hard negative samples for each graph representation. These strategically constructed hard negatives guide the encoder to learn more discriminative features by emphasizing critical semantic differences between graphs. Extensive experiments demonstrate that our approach achieves state-of-the-art performance compared to existing GCL methods across a variety of datasets and tasks.
       


### [Degree-Optimized Cumulative Polynomial Kolmogorov-Arnold Networks](https://arxiv.org/abs/2505.15228)

**Authors:**
Mathew Vanherreweghe, Lirandë Pira, Patrick Rebentrost

**Abstract:**
We introduce cumulative polynomial Kolmogorov-Arnold networks (CP-KAN), a neural architecture combining Chebyshev polynomial basis functions and quadratic unconstrained binary optimization (QUBO). Our primary contribution involves reformulating the degree selection problem as a QUBO task, reducing the complexity from $O(D^N)$ to a single optimization step per layer. This approach enables efficient degree selection across neurons while maintaining computational tractability. The architecture performs well in regression tasks with limited data, showing good robustness to input scales and natural regularization properties from its polynomial basis. Additionally, theoretical analysis establishes connections between CP-KAN's performance and properties of financial time series. Our empirical validation across multiple domains demonstrates competitive performance compared to several traditional architectures tested, especially in scenarios where data efficiency and numerical stability are important. Our implementation, including strategies for managing computational overhead in larger networks is available in Ref.~\citep{cpkan_implementation}.
       


### [Leveraging KANs for Expedient Training of Multichannel MLPs via Preconditioning and Geometric Refinement](https://arxiv.org/abs/2505.18131)

**Authors:**
Jonas A. Actor, Graham Harper, Ben Southworth, Eric C. Cyr

**Abstract:**
Multilayer perceptrons (MLPs) are a workhorse machine learning architecture, used in a variety of modern deep learning frameworks. However, recently Kolmogorov-Arnold Networks (KANs) have become increasingly popular due to their success on a range of problems, particularly for scientific machine learning tasks. In this paper, we exploit the relationship between KANs and multichannel MLPs to gain structural insight into how to train MLPs faster. We demonstrate the KAN basis (1) provides geometric localized support, and (2) acts as a preconditioned descent in the ReLU basis, overall resulting in expedited training and improved accuracy. Our results show the equivalence between free-knot spline KAN architectures, and a class of MLPs that are refined geometrically along the channel dimension of each weight tensor. We exploit this structural equivalence to define a hierarchical refinement scheme that dramatically accelerates training of the multi-channel MLP architecture. We show further accuracy improvements can be had by allowing the $1$D locations of the spline knots to be trained simultaneously with the weights. These advances are demonstrated on a range of benchmark examples for regression and scientific machine learning.
       


### [COLORA: Efficient Fine-Tuning for Convolutional Models with a Study Case on Optical Coherence Tomography Image Classification](https://arxiv.org/abs/2505.18315)

**Authors:**
Mariano Rivera, Angello Hoyos

**Abstract:**
We introduce the Convolutional Low-Rank Adaptation (CoLoRA) method, designed explicitly to overcome the inefficiencies found in current CNN fine-tuning methods. CoLoRA can be seen as a natural extension of the convolutional architectures of the Low-Rank Adaptation (LoRA) technique. We demonstrate the capabilities of our method by developing and evaluating models using the widely adopted CNN backbone pre-trained on ImageNet. We observed that this strategy results in a stable and accurate coarse-tuning procedure. Moreover, this strategy is computationally efficient and significantly reduces the number of parameters required for fine-tuning compared to traditional methods. Furthermore, our method substantially improves the speed and stability of training. Our case study focuses on classifying retinal diseases from optical coherence tomography (OCT) images, specifically using the OCTMNIST dataset. Experimental results demonstrate that a CNN backbone fine-tuned with CoLoRA surpasses nearly 1\% in accuracy. Such a performance is comparable to the Vision Transformer, State-space discrete, and Kolmogorov-Arnold network models.
       


### [TK-Mamba: Marrying KAN with Mamba for Text-Driven 3D Medical Image Segmentation](https://arxiv.org/abs/2505.18525)

**Authors:**
Haoyu Yang, Yuxiang Cai, Jintao Chen, Xuhong Zhang, Wenhui Lei, Xiaoming Shi, Jianwei Yin, Yankai Jiang

**Abstract:**
3D medical image segmentation is vital for clinical diagnosis and treatment but is challenged by high-dimensional data and complex spatial dependencies. Traditional single-modality networks, such as CNNs and Transformers, are often limited by computational inefficiency and constrained contextual modeling in 3D settings. We introduce a novel multimodal framework that leverages Mamba and Kolmogorov-Arnold Networks (KAN) as an efficient backbone for long-sequence modeling. Our approach features three key innovations: First, an EGSC (Enhanced Gated Spatial Convolution) module captures spatial information when unfolding 3D images into 1D sequences. Second, we extend Group-Rational KAN (GR-KAN), a Kolmogorov-Arnold Networks variant with rational basis functions, into 3D-Group-Rational KAN (3D-GR-KAN) for 3D medical imaging - its first application in this domain - enabling superior feature representation tailored to volumetric data. Third, a dual-branch text-driven strategy leverages CLIP's text embeddings: one branch swaps one-hot labels for semantic vectors to preserve inter-organ semantic relationships, while the other aligns images with detailed organ descriptions to enhance semantic alignment. Experiments on the Medical Segmentation Decathlon (MSD) and KiTS23 datasets show our method achieving state-of-the-art performance, surpassing existing approaches in accuracy and efficiency. This work highlights the power of combining advanced sequence modeling, extended network architectures, and vision-language synergy to push forward 3D medical image segmentation, delivering a scalable solution for clinical use. The source code is openly available at https://github.com/yhy-whu/TK-Mamba.
       


### [Kolmogorov-Arnold Networks for Turbulence Anisotropy Mapping](https://arxiv.org/abs/2505.19366)

**Authors:**
Nikhila Kalia, Ryley McConkey, Eugene Yee, Fue-Sang Lien

**Abstract:**
This study evaluates the generalization performance and representation efficiency (parsimony) of a previously introduced Tensor Basis Kolmogorov-Arnold Network (TBKAN) architecture for data-driven turbulence modeling. The TBKAN framework replaces the multi-layer perceptron (MLP) used in either the standard or modified Tensor Basis Neural Network (TBNN) with a Kolmogorov-Arnold network (KAN), which significantly reduces the model complexity while providing a structure that potentially can be used with symbolic regression to provide a physical interpretability that is not available in a 'black box' MLP. While some prior work demonstrated TBKAN's feasibility for modeling a 'simple' flat plate boundary layer flow, this study extends the TBKAN architecture to model more complex benchmark flows, in particular, square duct and periodic hills flows which exhibit strong turbulence anisotropy, secondary motion, and flow separation and reattachment. A realizability-informed loss function is employed to constrain the model predictions, and, for the first time, TBKAN predictions are stably injected into the Reynolds-averaged Navier-Stokes equations to provide it a posteriori predictions of the mean velocity field.
       


### ["KAN you hear me?" Exploring Kolmogorov-Arnold Networks for Spoken Language Understanding](https://arxiv.org/abs/2505.20176)

**Authors:**
Alkis Koudounas, Moreno La Quatra, Eliana Pastor, Sabato Marco Siniscalchi, Elena Baralis

**Abstract:**
Kolmogorov-Arnold Networks (KANs) have recently emerged as a promising alternative to traditional neural architectures, yet their application to speech processing remains under explored. This work presents the first investigation of KANs for Spoken Language Understanding (SLU) tasks. We experiment with 2D-CNN models on two datasets, integrating KAN layers in five different configurations within the dense block. The best-performing setup, which places a KAN layer between two linear layers, is directly applied to transformer-based models and evaluated on five SLU datasets with increasing complexity. Our results show that KAN layers can effectively replace the linear layers, achieving comparable or superior performance in most cases. Finally, we provide insights into how KAN and linear layers on top of transformers differently attend to input regions of the raw waveforms.
       


### [FMEnets: Flow, Material, and Energy networks for non-ideal plug flow reactor design](https://arxiv.org/abs/2505.20300)

**Authors:**
Chenxi Wu, Juan Diego Toscano, Khemraj Shukla, Yingjie Chen, Ali Shahmohammadi, Edward Raymond, Thomas Toupy, Neda Nazemifard, Charles Papageorgiou, George Em Karniadakis

**Abstract:**
We propose FMEnets, a physics-informed machine learning framework for the design and analysis of non-ideal plug flow reactors. FMEnets integrates the fundamental governing equations (Navier-Stokes for fluid flow, material balance for reactive species transport, and energy balance for temperature distribution) into a unified multi-scale network model. The framework is composed of three interconnected sub-networks with independent optimizers that enable both forward and inverse problem-solving. In the forward mode, FMEnets predicts velocity, pressure, species concentrations, and temperature profiles using only inlet and outlet information. In the inverse mode, FMEnets utilizes sparse multi-residence-time measurements to simultaneously infer unknown kinetic parameters and states. FMEnets can be implemented either as FME-PINNs, which employ conventional multilayer perceptrons, or as FME-KANs, based on Kolmogorov-Arnold Networks. Comprehensive ablation studies highlight the critical role of the FMEnets architecture in achieving accurate predictions. Specifically, FME-KANs are more robust to noise than FME-PINNs, although both representations are comparable in accuracy and speed in noise-free conditions. The proposed framework is applied to three different sets of reaction scenarios and is compared with finite element simulations. FMEnets effectively captures the complex interactions, achieving relative errors less than 2.5% for the unknown kinetic parameters. The new network framework not only provides a computationally efficient alternative for reactor design and optimization, but also opens new avenues for integrating empirical correlations, limited and noisy experimental data, and fundamental physical equations to guide reactor design.
       


### [Input Convex Kolmogorov Arnold Networks](https://arxiv.org/abs/2505.21208)

**Authors:**
Thomas Deschatre, Xavier Warin

**Citation Count:**
1

**Abstract:**
This article presents an input convex neural network architecture using Kolmogorov-Arnold networks (ICKAN). Two specific networks are presented: the first is based on a low-order, linear-by-part, representation of functions, and a universal approximation theorem is provided. The second is based on cubic splines, for which only numerical results support convergence. We demonstrate on simple tests that these networks perform competitively with classical input convex neural networks (ICNNs). In a second part, we use the networks to solve some optimal transport problems needing a convex approximation of functions and demonstrate their effectiveness. Comparisons with ICNNs show that cubic ICKANs produce results similar to those of classical ICNNs.
       


### [Taylor expansion-based Kolmogorov-Arnold network for blind image quality assessment](https://arxiv.org/abs/2505.21592)

**Authors:**
Ze Chen, Shaode Yu

**Abstract:**
Kolmogorov-Arnold Network (KAN) has attracted growing interest for its strong function approximation capability. In our previous work, KAN and its variants were explored in score regression for blind image quality assessment (BIQA). However, these models encounter challenges when processing high-dimensional features, leading to limited performance gains and increased computational cost. To address these issues, we propose TaylorKAN that leverages the Taylor expansions as learnable activation functions to enhance local approximation capability. To improve the computational efficiency, network depth reduction and feature dimensionality compression are integrated into the TaylorKAN-based score regression pipeline. On five databases (BID, CLIVE, KonIQ, SPAQ, and FLIVE) with authentic distortions, extensive experiments demonstrate that TaylorKAN consistently outperforms the other KAN-related models, indicating that the local approximation via Taylor expansions is more effective than global approximation using orthogonal functions. Its generalization capacity is validated through inter-database experiments. The findings highlight the potential of TaylorKAN as an efficient and robust model for high-dimensional score regression.
       


### [Localized Weather Prediction Using Kolmogorov-Arnold Network-Based Models and Deep RNNs](https://arxiv.org/abs/2505.22686)

**Authors:**
Ange-Clement Akazan, Verlon Roel Mbingui, Gnankan Landry Regis N'guessan, Issa Karambal

**Abstract:**
Weather forecasting is crucial for managing risks and economic planning, particularly in tropical Africa, where extreme events severely impact livelihoods. Yet, existing forecasting methods often struggle with the region's complex, non-linear weather patterns. This study benchmarks deep recurrent neural networks such as $\texttt{LSTM, GRU, BiLSTM, BiGRU}$, and Kolmogorov-Arnold-based models $(\texttt{KAN} and \texttt{TKAN})$ for daily forecasting of temperature, precipitation, and pressure in two tropical cities: Abidjan, Cote d'Ivoire (Ivory Coast) and Kigali (Rwanda). We further introduce two customized variants of $ \texttt{TKAN}$ that replace its original $\texttt{SiLU}$ activation function with $ \texttt{GeLU}$ and \texttt{MiSH}, respectively. Using station-level meteorological data spanning from 2010 to 2024, we evaluate all the models on standard regression metrics. $\texttt{KAN}$ achieves temperature prediction ($R^2=0.9986$ in Abidjan, $0.9998$ in Kigali, $\texttt{MSE} < 0.0014~^\circ C ^2$), while $\texttt{TKAN}$ variants minimize absolute errors for precipitation forecasting in low-rainfall regimes. The customized $\texttt{TKAN}$ models demonstrate improvements over the standard $\texttt{TKAN}$ across both datasets. Classical \texttt{RNNs} remain highly competitive for atmospheric pressure ($R^2 \approx 0.83{-}0.86$), outperforming $\texttt{KAN}$-based models in this task. These results highlight the potential of spline-based neural architectures for efficient and data-efficient forecasting.
       


### [Automated Modeling Method for Pathloss Model Discovery](https://arxiv.org/abs/2505.23383)

**Authors:**
Ahmad Anaqreh, Shih-Kai Chou, Mihael Mohorčič, Thomas Lagkas, Carolina Fortuna

**Abstract:**
Modeling propagation is the cornerstone for designing and optimizing next-generation wireless systems, with a particular emphasis on 5G and beyond era. Traditional modeling methods have long relied on statistic-based techniques to characterize propagation behavior across different environments. With the expansion of wireless communication systems, there is a growing demand for methods that guarantee the accuracy and interpretability of modeling. Artificial intelligence (AI)-based techniques, in particular, are increasingly being adopted to overcome this challenge, although the interpretability is not assured with most of these methods. Inspired by recent advancements in AI, this paper proposes a novel approach that accelerates the discovery of path loss models while maintaining interpretability. The proposed method automates the formulation, evaluation, and refinement of the model, facilitating the discovery of the model. We examine two techniques: one based on Deep Symbolic Regression, offering full interpretability, and the second based on Kolmogorov-Arnold Networks, providing two levels of interpretability. Both approaches are evaluated on two synthetic and two real-world datasets. Our results show that Kolmogorov-Arnold Networks achieve the coefficient of determination value R^2 close to 1 with minimal prediction error, while Deep Symbolic Regression generates compact models with moderate accuracy. Moreover, on the selected examples, we demonstrate that automated methods outperform traditional methods, achieving up to 75% reduction in prediction errors, offering accurate and explainable solutions with potential to increase the efficiency of discovering next-generation path loss models.
       


### [SPPSFormer: High-quality Superpoint-based Transformer for Roof Plane Instance Segmentation from Point Clouds](https://arxiv.org/abs/2505.24475)

**Authors:**
Cheng Zeng, Xiatian Qi, Chi Chen, Kai Sun, Wangle Zhang, Yuxuan Liu, Yan Meng, Bisheng Yang

**Abstract:**
Transformers have been seldom employed in point cloud roof plane instance segmentation, which is the focus of this study, and existing superpoint Transformers suffer from limited performance due to the use of low-quality superpoints. To address this challenge, we establish two criteria that high-quality superpoints for Transformers should satisfy and introduce a corresponding two-stage superpoint generation process. The superpoints generated by our method not only have accurate boundaries, but also exhibit consistent geometric sizes and shapes, both of which greatly benefit the feature learning of superpoint Transformers. To compensate for the limitations of deep learning features when the training set size is limited, we incorporate multidimensional handcrafted features into the model. Additionally, we design a decoder that combines a Kolmogorov-Arnold Network with a Transformer module to improve instance prediction and mask extraction. Finally, our network's predictions are refined using traditional algorithm-based postprocessing. For evaluation, we annotated a real-world dataset and corrected annotation errors in the existing RoofN3D dataset. Experimental results show that our method achieves state-of-the-art performance on our dataset, as well as both the original and reannotated RoofN3D datasets. Moreover, our model is not sensitive to plane boundary annotations during training, significantly reducing the annotation burden. Through comprehensive experiments, we also identified key factors influencing roof plane segmentation performance: in addition to roof types, variations in point cloud density, density uniformity, and 3D point precision have a considerable impact. These findings underscore the importance of incorporating data augmentation strategies that account for point cloud quality to enhance model robustness under diverse and challenging conditions.
       


### [How can AI reduce fall injuries in the workplace?](https://arxiv.org/abs/2505.24507)

**Authors:**
Nicholas Cartocci, Antonios E. Gkikakis, Roberto F. Pitzalis, Fabio Pera, Maria Teresa Settino, Darwin G. Caldwell, Jesús Ortiz

**Abstract:**
Fall-caused injuries are common in all types of work environments, including offices. They are the main cause of absences longer than three days, especially for small and medium-sized businesses (SMEs). However, data, data amount, data heterogeneity, and stringent processing time constraints continue to pose challenges to real-time fall detection. This work proposes a new approach based on a recurrent neural network (RNN) for Fall Detection and a Kolmogorov-Arnold Network (KAN) to estimate the time of impact of the fall. The approach is tested on SisFall, a dataset consisting of 2706 Activities of Daily Living (ADLs) and 1798 falls recorded by three sensors. The results show that the proposed approach achieves an average TPR of 82.6% and TNR of 98.4% for fall sequences and 94.4% in ADL. Besides, the Root Mean Squared Error of the estimated time of impact is approximately 160ms.
       


## June
### [Probing Quantum Spin Systems with Kolmogorov-Arnold Neural Network Quantum States](https://arxiv.org/abs/2506.01891)

**Authors:**
Mahmud Ashraf Shamim, Eric Reinhardt, Talal Ahmed Chowdhury, Sergei Gleyzer, Paulo T Araujo

**Abstract:**
Neural Quantum States (NQS) are a class of variational wave functions parametrized by neural networks (NNs) to study quantum many-body systems. In this work, we propose SineKAN, the NQS ansatz based on Kolmogorov-Arnold Networks (KANs), to represent quantum mechanical wave functions as nested univariate functions. We show that \sk wavefunction with learnable sinusoidal activation functions can capture the ground state energies, fidelities and various correlation functions of the 1D Transverse-Field Ising model, Anisotropic Heisenberg model, and Antiferromagnetic $J_{1}-J_{2}$ model with different chain lengths. In our study of the $J_1-J_2$ model with $L=100$ sites, we find that the SineKAN model outperforms several previously explored neural quantum state ansätze, including Restricted Boltzmann Machines (RBMs), Long Short-Term Memory models (LSTMs), and Feed-Forward Neural Networks (FFCN), when compared to the results obtained from the Density Matrix Renormalization Group (DMRG) algorithm.
       


### [Kolmogorov-Arnold Wavefunctions](https://arxiv.org/abs/2506.02171)

**Authors:**
Paulo F. Bedaque, Jacob Cigliano, Hersh Kumar, Srijit Paul, Suryansh Rajawat

**Abstract:**
This work investigates Kolmogorov-Arnold network-based wavefunction ansatz as viable representations for quantum Monte Carlo simulations. Through systematic analysis of one-dimensional model systems, we evaluate their computational efficiency and representational power against established methods. Our numerical experiments suggest some efficient training methods and we explore how the computational cost scales with desired precision, particle number, and system parameters. Roughly speaking, KANs seem to be 10 times cheaper computationally than other neural network based ansatz. We also introduce a novel approach for handling strong short-range potentials-a persistent challenge for many numerical techniques-which generalizes efficiently to higher-dimensional, physically relevant systems with short-ranged strong potentials common in atomic and nuclear physics.
       


### [Multi-Exit Kolmogorov-Arnold Networks: enhancing accuracy and parsimony](https://arxiv.org/abs/2506.03302)

**Authors:**
James Bagrow, Josh Bongard

**Abstract:**
Kolmogorov-Arnold Networks (KANs) uniquely combine high accuracy with interpretability, making them valuable for scientific modeling. However, it is unclear a priori how deep a network needs to be for any given task, and deeper KANs can be difficult to optimize. Here we introduce multi-exit KANs, where each layer includes its own prediction branch, enabling the network to make accurate predictions at multiple depths simultaneously. This architecture provides deep supervision that improves training while discovering the right level of model complexity for each task. Multi-exit KANs consistently outperform standard, single-exit versions on synthetic functions, dynamical systems, and real-world datasets. Remarkably, the best predictions often come from earlier, simpler exits, revealing that these networks naturally identify smaller, more parsimonious and interpretable models without sacrificing accuracy. To automate this discovery, we develop a differentiable "learning to exit" algorithm that balances contributions from exits during training. Our approach offers scientists a practical way to achieve both high performance and interpretability, addressing a fundamental challenge in machine learning for scientific discovery.
       


### [Dynamic Graph CNN with Jacobi Kolmogorov-Arnold Networks for 3D Classification of Point Sets](https://arxiv.org/abs/2506.06296)

**Authors:**
Hanaa El Afia, Said Ohamouddou, Raddouane Chiheb, Abdellatif El Afia

**Abstract:**
We introduce Jacobi-KAN-DGCNN, a framework that integrates Dynamic Graph Convolutional Neural Network (DGCNN) with Jacobi Kolmogorov-Arnold Networks (KAN) for the classification of three-dimensional point clouds. This method replaces Multi-Layer Perceptron (MLP) layers with adaptable univariate polynomial expansions within a streamlined DGCNN architecture, circumventing deep levels for both MLP and KAN to facilitate a layer-by-layer comparison. In comparative experiments on the ModelNet40 dataset, KAN layers employing Jacobi polynomials outperform the traditional linear layer-based DGCNN baseline in terms of accuracy and convergence speed, while maintaining parameter efficiency. Our results demonstrate that higher polynomial degrees do not automatically improve performance, highlighting the need for further theoretical and empirical investigation to fully understand the interactions between polynomial bases, degrees, and the mechanisms of graph-based learning.
       


### [Improving Memory Efficiency for Training KANs via Meta Learning](https://arxiv.org/abs/2506.07549)

**Authors:**
Zhangchi Zhao, Jun Shu, Deyu Meng, Zongben Xu

**Abstract:**
Inspired by the Kolmogorov-Arnold representation theorem, KANs offer a novel framework for function approximation by replacing traditional neural network weights with learnable univariate functions. This design demonstrates significant potential as an efficient and interpretable alternative to traditional MLPs. However, KANs are characterized by a substantially larger number of trainable parameters, leading to challenges in memory efficiency and higher training costs compared to MLPs. To address this limitation, we propose to generate weights for KANs via a smaller meta-learner, called MetaKANs. By training KANs and MetaKANs in an end-to-end differentiable manner, MetaKANs achieve comparable or even superior performance while significantly reducing the number of trainable parameters and maintaining promising interpretability. Extensive experiments on diverse benchmark tasks, including symbolic regression, partial differential equation solving, and image classification, demonstrate the effectiveness of MetaKANs in improving parameter efficiency and memory usage. The proposed method provides an alternative technique for training KANs, that allows for greater scalability and extensibility, and narrows the training cost gap with MLPs stated in the original paper of KANs. Our code is available at https://github.com/Murphyzc/MetaKAN.
       


### [Neural Tangent Kernel Analysis to Probe Convergence in Physics-informed Neural Solvers: PIKANs vs. PINNs](https://arxiv.org/abs/2506.07958)

**Authors:**
Salah A. Faroughi, Farinaz Mostajeran

**Abstract:**
Physics-informed Kolmogorov-Arnold Networks (PIKANs), and in particular their Chebyshev-based variants (cPIKANs), have recently emerged as promising models for solving partial differential equations (PDEs). However, their training dynamics and convergence behavior remain largely unexplored both theoretically and numerically. In this work, we aim to advance the theoretical understanding of cPIKANs by analyzing them using Neural Tangent Kernel (NTK) theory. Our objective is to discern the evolution of kernel structure throughout gradient-based training and its subsequent impact on learning efficiency. We first derive the NTK of standard cKANs in a supervised setting, and then extend the analysis to the physics-informed context. We analyze the spectral properties of NTK matrices, specifically their eigenvalue distributions and spectral bias, for four representative PDEs: the steady-state Helmholtz equation, transient diffusion and Allen-Cahn equations, and forced vibrations governed by the Euler-Bernoulli beam equation. We also conduct an investigation into the impact of various optimization strategies, e.g., first-order, second-order, and hybrid approaches, on the evolution of the NTK and the resulting learning dynamics. Results indicate a tractable behavior for NTK in the context of cPIKANs, which exposes learning dynamics that standard physics-informed neural networks (PINNs) cannot capture. Spectral trends also reveal when domain decomposition improves training, directly linking kernel behavior to convergence rates under different setups. To the best of our knowledge, this is the first systematic NTK study of cPIKANs, providing theoretical insight that clarifies and predicts their empirical performance.
       


