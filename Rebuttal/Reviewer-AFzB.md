# Supplementary Materials for Rebuttal to Reviewer-AFzB
We sincerely appreciate the thoughtful comments, efforts, and time invested by the reviewers. Due to the character limit in the main document, this supplementary material provides detailed explanations and clarifications on various issues, including additional data analysis and references. Thank you for your understanding and for your valuable contributions to improving our work.

### Questions

**Q1-Brief: The details of PAAC.**
**A1:**  We greatly appreciate the reviewer's constructive feedback. According to your suggestion, we have added further details, particularly regarding the re-weighting contrast module. Specifically, we are expanding our explanation of the re-weighting contrast module. This module is designed to adjust the contrastive learning loss based on item popularity, which helps in reducing the representation separation that often exacerbates popularity bias. By introducing hyperparameters $\gamma$ and $\beta$, we dynamically adjust the weight of positive and negative samples within our contrastive learning framework, tailored to their popularity indices. This approach ensures that both popular and unpopular items receive appropriate representation during model training, enhancing the model’s ability to generalize across the diverse item spectrum in real-world datasets.

**Q2&C2-Brief: Performance comparison on conventional test sets.**
**A2:** Thank you for your questions.

- **Unbias evalution.** As mentioned in Section 4.1.1, traditional evaluation methods do not effectively measure a model's ability to mitigate popularity bias, since test sets often retain a long-tail distribution[1]. This could misleadingly suggest high performance for models that favor popular items. Therefore we follow previous works[1,2,3] to utilize an unbiased dataset for evaluation, ensuring a uniform item distribution in the test set.
- **Traditional evaluation.** Certainly, we also believe that while traditional performance metrics remain crucial, we are sorry for overlooking this. We have incorporated additional experiments to provide a comprehensive understanding of our model's capabilities under traditional conditions. The experimental results, as shown in Table 1, indicate that our method competes vigorously with the best baselines in traditional experimental settings, confirming its effectiveness under standard evaluation conditions. Combining the superior performance of our method on the unbiased dataset as presented in the paper, thoroughly confirms that PAAC mitigates popularity bias under any circumstance.

Table 1: Performance comparison on conventional test sets

| Model      |           | Yelp2018|         |           | Gowalla |         |
|------------|-----------|--------------|---------|-----------|-------------|---------|
|            | Recall@20 | HR@20        | NDCG@20 | Recall@20 | HR@20       | NDCG@20 |
| LightGCN   | 0.0591    | 0.0614       | 0.0483  | 0.1637    | 0.1672      | 0.1381  |
| Adapt-\tau | 0.0724    | 0.0753       | **0.0603**  | 0.1889    | **0.1930**      | 0.1584  |
| SimGCL     | 0.0720    | 0.0748       | 0.0594  | 0.1817    | 0.1858      | 0.1526  |
| PAAC       | **0.0725**    |**0.0755**      | 0.0602  |**0.1890**   | 0.1928      | **0.1585**  |

**Q3-Brief: GCN encoder**
**A3:** Thank you for your question. The choice of GCN as an encoder in the Supervised Alignment Module is not necessary. We chose GCN as the encoder in this module because many baselines[1,2,5] use LightGCN as a backbone, and we wanted to ensure fair comparison with other methods. Additionally, GCN as an encoder has been proven to capture high-order connectivity signals between users and items, making it a state-of-the-art encoder[1,2,4]. Therefore, we selected GCN as our encoder in the experiments. However, since PAAC is model-agnostic, we can also utilize other encoders such as MF, VAE, etc., within our framework.

**Q4&C3-Brief: Hyper-parameter sensitivities.**
**A3:** Supplementary experimental results on the impact of different division ratios:

| Radio         |           | Yelp2018 |         |           | Gowalla |         |
|----------------|-----------|----------|---------|-----------|---------|---------|
|                  | **Recall@20** | **HR@20**   |**NDCG@20** | **Recall@20**| **HR@20**  | **NDCG@20** |
| **20%**            | 0.0467    | 0.0555   | 0.0361  | 0.1232    | 0.1319  | 0.0845  |
| **40%**            | **0.0505**    | **0.0581** | **0.0378**| **0.1239**  | **0.1325**  | **0.0848**  |
| **50%**            | 0.0494    | 0.0574   | 0.0375  | 0.1232    | 0.1321  | **0.0848**  |
| **60%**            | 0.0492    | 0.0569   | 0.0370  |  0.1225   |	0.1314	|0.0843   |
| **80%**            | 0.0467    | 0.0545   | 0.0350  | 0.1176    | 0.1270  | 0.0818  |

### Concerns

**C1-Brief: The details for PAAC.**
**A1:** We apologize for any confusion caused. For your concerns:

- **Item Division Methods**:  We avoided a fixed global threshold to prevent overrepresentation of popular items in some mini-batches.  We dynamically divided items into popular and unpopular groups within each mini-batch based on their popularity, assigning the top 50% as popular items and the bottom 50% as unpopular items in this paper. This ensures equal representation of both items in our contrastive learning and allows items to be classified based on the batch's current composition adaptively. We can also adopt other ratios.
- **Data Augmentation**: In this paper, we employed noise perturbation as our data augmentation strategy. This choice was inspired by the findings from SimGCL, which demonstrated that feature augmentation, particularly through random noise perturbation, outperforms graph augmentation in terms of effectiveness.
- **The  Formula (11)**: $L_{cl}^{user}$ represents the contrastive loss for users.

**C4-Brief: Some typos.**
**A4:** Sorry for the confusion due to the typos. we will correct these errors. Regarding the normalization factor in formula (6), it should indeed be $\frac{1}{|I_u|}$ , reflecting the total number of user-item interactions for each user, to ensure an accurate representation of the normalization process. We appreciate your thorough review and will make the necessary revisions to improve the clarity and accuracy of our mathematical formulations.

### Reference

[1]Wei T, Feng F, Chen J, et al. Model-agnostic counterfactual reasoning for eliminating popularity bias in recommender system[C]//Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2021: 1791-1800.
[2]Zhang A, Zheng J, Wang X, et al. Invariant collaborative filtering to popularity distribution shift[C]//Proceedings of the ACM Web Conference 2023. 2023: 1240-1251.
[3]Zheng Y, Gao C, Li X, et al. Disentangling user interest and conformity for recommendation with causal embedding[C]//Proceedings of the Web Conference 2021. 2021: 2980-2991.
[4]He X, Deng K, Wang X, et al. Lightgcn: Simplifying and powering graph convolution network for recommendation[C]//Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval. 2020: 639-648.
[5]Chen J, Wu J, Wu J, et al. Adap-τ: Adaptively modulating embedding magnitude for recommendation[C]//Proceedings of the ACM Web Conference 2023. 2023: 1085-1096.
