We sincerely appreciate your thoughtful comments, efforts, and time. We respond to each of your questions and concerns one-by-one as follows. Due to the character limit, for detailed elaborations and supporting evidence, we kindly refer you to the supplementary document [Rebuttal-AFzB](https://anonymous.4open.science/r/KDD2024-PAAC-39A9).Thank you for your understanding.

### Questions

**Q1-Brief: Limitation of technical novelty.**

**A1:** Thank you for your valuable comment. We are sorry for not providing a clear description of this aspect. According to your suggestion, we have added further details, particularly regarding the re-weighting contrast module, in the supplementary material.

**Q2&C2-Brief: Performance comparison on conventional test sets.**

**A2:** As mentioned in Section 4.1.1, traditional evaluation methods do not effectively measure a model's ability to mitigate popularity bias, since test sets often retain a long-tail distribution. This could misleadingly suggest high performance for models that favor popular items. Therefore we follow precedents to utilize an unbiased dataset for evaluation, ensuring a uniform item distribution in the test set. While traditional performance metrics remain crucial, we've included supplemental experiments to offer a comprehensive understanding of our model's capabilities under normal conditions. These results, detailed in Table 1, show our method competes closely with the best baselines, affirming its efficacy under standard evaluation conditions.

Table 1: Performance comparison on conventional test sets

| Model      |           | Yelp2018|         |           | Gowalla |         |
|------------|-----------|--------------|---------|-----------|-------------|---------|
|            | Recall@20 | HR@20        | NDCG@20 | Recall@20 | HR@20       | NDCG@20 |
| LightGCN   | 0.0591    | 0.0614       | 0.0483  | 0.1637    | 0.1672      | 0.1381  |
| Adapt-\tau | 0.0724    | 0.0753       | **0.0603**  | 0.1889    | **0.1930**      | 0.1584  |
| SimGCL     | 0.0720    | 0.0748       | 0.0594  | 0.1817    | 0.1858      | 0.1526  |
| PAAC       | **0.0725**    |**0.0755**      | 0.0602  |**0.1890**   | 0.1928      | **0.1585**  |

**Q3-Brief: Encoder**

**A3:** 



**Q4-Brief: The details of the item division threshold.**

**A4:** We apologize for not specifying our item group partitioning method in the paper and appreciate the opportunity to clarify. Regarding your concerns:

- **Item Division Ratio**:  We avoided a fixed global threshold to prevent the overrepresentation of popular items in some mini-batches.  We dynamically divided items into popular and unpopular groups within each mini-batch based on their popularity, assigning the top 50% as popular items and the bottom 50% as unpopular items in this paper. This ensures equal representation of both items in our contrastive learning and allows items to be classified based on the batch's current composition adaptively. We can also adopt other ratios.
- **Impact of Different Division Ratios**: In supplemental experiments, we examined various division ratios for popular items, including 20%, 40%, 60%, and 80%, as shown in Table 1. The preliminary results indicate that both extremely low and high ratios negatively affect model performance, thereby underscoring the superiority of our dynamic data partitioning approach. Moreover, within the 40%-60% range, our model's performance remained consistently robust, further validating the effectiveness of our method.
  
  Table 1: Impact of Different Popular Item Ratios on Model Performance

| Radio         |           | Yelp2018 |         |           | Gowalla |         |
|----------------|-----------|----------|---------|-----------|---------|---------|
|                  | **Recall@20** | **HR@20**   |**NDCG@20** | **Recall@20**| **HR@20**  | **NDCG@20** |
| **20%**            | 0.0467    | 0.0555   | 0.0361  | 0.1232    | 0.1319  | 0.0845  |
| **40%**            | 0.0505    | 0.0581   | 0.0378  | 0.1239    | 0.1325  | 0.0848  |
| **50%**            | 0.0494    | 0.0574   | 0.0375  | 0.1232    | 0.1321  | 0.0848  |
| **60%**            | 0.0492    | 0.0569   | 0.0370  |  0.1225   |	0.1314	|0.0843   |
| **80%**            | 0.0467    | 0.0545   | 0.0350  | 0.1176    | 0.1270  | 0.0818  |

### Concerns

**C1-Brief: The details for PAAC.**

**A1:** We apologize for any confusion caused. For your concerns:

- **Item Division Methods**:  We avoided a fixed global threshold to prevent the overrepresentation of popular items in some mini-batches.  We dynamically divided items into popular and unpopular groups within each mini-batch based on their popularity, assigning the top 50% as popular items and the bottom 50% as unpopular items in this paper. This ensures equal representation of both items in our contrastive learning and allows items to be classified based on the batch's current composition adaptively. We can also adopt other ratios.
- **Data Augmentation**: In this paper, we employed noise perturbation as our data augmentation strategy. This choice was inspired by the findings from SimGCL, which demonstrated that feature augmentation, particularly through random noise perturbation, outperforms graph augmentation in terms of effectiveness.
- **The  Formula (11)**: $L_{cl}^{user}$ represents the contrastive loss for users.

**C3-Brief: Hyper-parameter sensitivities.**

**A3:** Thanks for your suggestion.

- We note that $\gamma$ and $\beta$ range from [0,1], and our analysis in Section 4.5 shows they perform well across a broad range, reducing the need for precise tuning. Our focus was primarily on adjusting the weight coefficients of the two modules to maintain model efficacy. We agree with your advice on reducing hyperparameters and plan to seek simpler models in the future. This is a promising direction, and we very much appreciate your suggestion.
- We dynamically divided items into popular and unpopular groups within each mini-batch based on their popularity, assigning the top 50% as popular items and the bottom 50% as unpopular items in this paper. In supplemental experiments, we examined various division ratios for popular items, including 20%, 40%, 60%, and 80%, as shown in Table 1. The preliminary results indicate that both extremely low and high ratios negatively affect model performance, thereby underscoring the superiority of our dynamic data partitioning approach. Moreover, within the 40%-60% range, our model's performance remained consistently robust, further validating the effectiveness of PAAC.

Table 1: Impact of Different Popular Item Ratios on Model Performance

| Radio         |           | Yelp2018 |         |           | Gowalla |         |
|----------------|-----------|----------|---------|-----------|---------|---------|
|                  | **Recall@20** | **HR@20**   |**NDCG@20** | **Recall@20**| **HR@20**  | **NDCG@20** |
| **20%**            | 0.0467    | 0.0555   | 0.0361  | 0.1232    | 0.1319  | 0.0845  |
| **40%**            | 0.0505    | 0.0581   | 0.0378  | 0.1239    | 0.1325  | 0.0848  |
| **50%**            | 0.0494    | 0.0574   | 0.0375  | 0.1232    | 0.1321  | 0.0848  |
| **60%**            | 0.0492    | 0.0569   | 0.0370  |  0.1225   |	0.1314	|0.0843   |
| **80%**            | 0.0467    | 0.0545   | 0.0350  | 0.1176    | 0.1270  | 0.0818  |

**C4-Brief: Some typos.**

**A4:** Sorry for the confusion due to the typos. we will correct these errors. Regarding the normalization factor in formula (6), it should indeed be $\frac{1}{|I_u|}$ , reflecting the total number of user-item interactions for each user, to ensure an accurate representation of the normalization process. We appreciate your thorough review and will make the necessary revisions to improve the clarity and accuracy of our mathematical formulations.
