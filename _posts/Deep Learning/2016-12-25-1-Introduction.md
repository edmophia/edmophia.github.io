---
layout: post
title: Deep Learning Introduction
date: 2016-12-25
tags:
  - Deep Learning
---
본 게시물은 Ian Goodfellow et al. 의 <a href='http://www.deeplearningbook.org'>Deep Learning</a>의 1장 Introduction의 흐름을 따라가면서 개인적으로 중요하다고 생각하는 부분들은 더 조사를 하여 정리한 포스트입니다.

## 서두

먼저 이안의 책에서는 생각하는 기계 즉, Artificial Intelligence를 주된 목표로 지정합니다. 이러한 인공지능의 접근법 중에 다음 2가지가 널리 알려져 있습니다.

### <a href='https://en.wikipedia.org/wiki/Knowledge_base'>지식 기반(Knowledge Base)</a>

현실에 관한 지식들이 프로그램이 되어 있는 인공 지능입니다. 지식을 가지고 있기 때문에 전문가 시스템(expert system)이라고도 부릅니다.

여기에 해당하는 예에는 <a href='https://en.wikipedia.org/wiki/Cyc'>Cyc</a>가 있습니다.

### <a href='https://en.wikipedia.org/wiki/Machine_learning'>기계 학습(Machine Learning)</a>

한편 지식 기반 접근법에서는 지식을 정형화 해야한다는 단점과 지식들이 많거나 복잡하면 구현하는 것이 매우 어렵다는 단점이 있습니다. 이를 보완하기 위해 데이터로부터 패턴을 추출하는 방식으로 스스로 지식을 쌓는 기법을 기계 학습이라고 합니다.

한편 이런 기계 학습 접근법에서는 데이터의 Representation이 중요합니다.(표현이라고 직역하고 싶습니다만 뉘앙스가 다른 것 같아 그대로 표기하겠습니다.) 예를 들면 대부분의 알려진 기계 학습 알고리즘들은 정해진 형식의 입력 값을 받아 정해진 형식의 출력값을 도출합니다. 따라서 올바르지 않은 형식의 값을 입력하면 에러를 일으키거나 정상적이지 않은 출력을 합니다.

이러한 representation에 대한 의존성 때문에 기존의 컴퓨터 과학 연구에서 representation의 선택은 대단히 중요하면서도 대단히 어려운 문제였습니다. 이러한 문제에 대한 접근법 중에 하나는 입력값 X에서 출력값 Y로 가는 교사 학습 문제를 푸는 것이 아니라 입력값 X의 특징을 기계한테 학습 시키는 Representation Learning입니다.(이렇게 입력값 자체를 학습하는 분야를 기계 학습에서는 그냥 비교사 학습(Unsupervised Learning이라고 합니다.) 이러한 접근법은 사람이 representation을 선택하는 것에 비해 성능이 떨어지지 않을 뿐만 아니라 사람이 개입하지 않는다는 점에서 진정한 의미의 인공 지능에 더 가깝습니다.

대표적인 Representation Learning 알고리즘은 자동 부호화기(Auto Encoder)가 있습니다.(자세한 내용은 다른 포스트에서 다루겠습니다.)

한편 Deep Learning은 이러한 Representation Learning 문제를 신경 과학에서 모티브를 얻은  인공 신경망(Artificial Neural Network)의 계층적 구조의 방식으로 접근하는 분야입니다. 한편 올바른 representation을 배우는 것이 목표라는 관점은 딥러닝에 대한 한 가지 관점일 뿐입니다. 다른 관점은 계층 구조가 깊을수록 다단계 프로그램을 학습하기 용이하다는 것입니다. 이러한 관점은 각각의 인공 뉴런을 하나의 프로그램으로 이해하여 여러 뉴런이 같은 층에 있는 것은 일종의 분산처리 역할을 하는 것이며 구조가 깊은 것은 이러한 분산 처리의 결과를 이용하여 더 복잡한 프로그램을 만드는 것을 의미합니다.

## 딥러닝의 역사

딥러닝에 관심이 있으신 분이라면 딥러닝이 처음 제안된 것은 1950년대 Frank Rosenblatt의 <a href='https://en.wikipedia.org/wiki/Perceptron'>Perceptron</a>이지만 몇 번에 걸쳐 흥망을 반복하신 것을 알고 계실겁니다. 이번에는 이 역사와 여기서 얻을 수 있는 교훈에 대해 정리해보겠습니다.

### <a href='https://en.wikipedia.org/wiki/Cybernetics_(disambiguation)'>Cybernetics(1940~1960)</a>

초기 인공 신경망은 뇌를 따라하는 프로그램을 만듦으로써 지능을 리버스 엔지니어링 하겠다는 모티브를 가지고 만들어졌습니다. 하지만 이는 굳이 신경 과학에서 모티브를 얻지 않고도 계층적 구조로 학습하는 것이 주류인 오늘날의 딥러닝과는 약간 다릅니다. 한편 이러한 초기 인공 신경망은 대체로 선형 구조를 학습하는 것에 집중하였습니다. 이 세대에 나온 개념에는 <a href='https://en.wikipedia.org/wiki/ADALINE'>ADALINE</a>이 있습니다. 이러한 선형 모델들은 대체로 <a href='https://en.wikipedia.org/wiki/Stochastic_gradient_descent'>확률적 경사 하강법(Stochastic Gradient Descent)</a>을 통해 학습했으며 이러한 학습 기법은 현재까지도 약간의 수정을 거쳐 계속 사용되고 있습니다.

신경과학의 결과들은 딥러닝 연구에 여전히 괜찮은 모티브들을 제공하고 있는데 이 중에 중요한 것은 <a href='http://web.mit.edu/surlab/publications/visual.pdf'>하나의 신경 구조를 통해 시각과 청각이 모두 처리 가능한 예</a>가 있습니다. 또한 신경 과학은 딥러닝 구조 자체에 대해 이미 많은 영감들을 제공했습니다. 하지만 여전히 어떻게 학습하는지의 학습 알고리즘 자체에 대한 도움은 되지 않습니다.

한편 신경과학의 연구 결과와는 별개로 널리 쓰이고 있는 것은 바로 <a href='https://en.wikipedia.org/wiki/Rectifier_(neural_networks)'>Rectified Linear Unit(ReLu)</a> 입니다. 신경과학의 연구 결과에 의하면 실제 뇌는 활성화 함수 단계에서 복잡한 함수를 쓴다고 합니다, 하지만 현재의 딥러닝에서 많이 쓰이며 또 다른 복잡한 함수보다 괜찮은 결과를 제공하는 것은 이 ReLu입니다. 이러한 면에서 신경과학과 딥러닝은 별개의 분야라고 볼 수 있습니다.

### <a href='https://en.wikipedia.org/wiki/Connectionism'>Connectionism(1980~1990)</a>

두 번째 딥러닝의 역사적 흐름은 이 연결주의입니다. 연결주의는 인지과학의 한 접근방식입니다. 연결주의의 주된 개념은 지능은 수많은 단순한 계산 단위들의 연결로써 구현이 가능하다고 주장하는 것입니다. 그들은 기억을 뉴런 간의 연결 강도 또는 구조를 수정하여 이루어진다고 생각합니다.

그들의 대표적인 개념 중 하나는 Distributed Representation 입니다. 이 개념은 입력값들은 다양한 특징들에 의해 결정되고, 특징들은 다양한 입력값들에 의해 포함될 수 있다는 것입니다. 이는 즉, 우리가 학습하는 입력값들이 결정되는 요인이 여러가지가 있으면 우리의 입력값들은 하나의 요인이 아니라 여러 요인이 동시에 작용하여 학습하는 것이며, 또 각 요인들은 여러 입력에서 등장할 수 있음을 의미합니다.

또 다른 그들의 대표적인 개념은 Parallel Distributed Processing입니다. David Rumelhart et al.의 <a href='https://mitpress.mit.edu/books/parallel-distributed-processing'>Parallel Distributed Processing, Volume 1</a> 에서 제시된 이 개념은 일반화 델타 규칙이라고 불리는 <a href='https://en.wikipedia.org/wiki/Backpropagation'>역전파 알고리즘(Backpropagation Algorithm)</a>을 제안했으며 이를 통해 첫번째 세대에서 인공 신경망이 인기를 잃었던 이유인 다층 구조의 학습을 해결하였습니다.

### <a href='https://en.wikipedia.org/wiki/Deep_learning'>Deep Learning(2006~)</a>

현재까지 진행되고 있는 흐름인 딥러닝 흐름은 Geofrrey Hinton이 <a href='https://en.wikipedia.org/wiki/Deep_belief_network'>Deep Belief network</a> 가 Greedy Layer-wise Pretraining을 통해 효율적으로 학습할 수 있음을 보여줌으로써 시작되었습니다. 한편 2010년 중반에 돌입하면서 딥러닝 기법은 대부분의 기계 학습 기법을 성능면에서 압도하기 시작합니다.

한편 이러한 흐름에서 중심적으로 연구되고 있는 내용은 비교사 학습 기법과 데이터가 많이 필요한 딥러닝 모델을 어떻게 하면 작은 데이터에서도 작동하게 하는지 입니다. 하지만 대부분의 현실 문제에서는 여전히 대형 데이터를 학습 시키는 것과 교사 학습에 대해 관심이 많습니다.

개인적인 의견을 드리자면 앞에서 언급한 딥러닝의 다른 관점인 계층 구조가 깊을 수록 다단계 프로그램을 학습할 수 있다는 관점은 <a href='https://arxiv.org/abs/1410.5401'>NTM</a>에서 잘 드러납니다. 누군가 딥러닝이 뇌를 그저 따라할 뿐이라고 주장한다면 이 논문을 보여주면서 그렇지는 않다고 할 수 있습니다. 한편 비교사 학습과 작은 데이터에 대한 문제의 핵심적인 논의는 2016년 말인 현재 단계에서는 <a href='https://arxiv.org/abs/1406.2661'>GAN</a>과 그에 관현 연구들이 활성화 되어있습니다. 이 GAN은 데이터를 생성하는 Generator와 데이터를 구분하는 Discriminator 간의 게임을 시켜 Genrator가 원래의 데이터와 비슷하지만 다른 데이터를 생성하도록 유도하는 연구입니다. 이러한 연구의 특징은 적은 데이터가 있어도 자동으로 완성시키는 GAN의 특성때문에 유효하다는 장점이 있으며 또한 기계끼리 경쟁을 함으로써 성능 평가 기준이 인간이 아니라 기계로 넘어가기 떄문에 연구가 성숙해질수록 인간의 성능을 뛰어넘을 가능성이 높다는 기대감이 있습니다. 물론 제가 아직은 공부가 미숙하기 때문에 틀린 말이 섞여 있을수도 있습니다. 만약 이 부분에 대해서 코멘트를 주신다면 수정하도록 하겠습니다.
