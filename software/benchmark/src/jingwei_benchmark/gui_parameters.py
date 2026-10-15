"""Typed, no-code controls for the parameters used by CLI method adapters."""

# name: (label, default, lower, upper); list values use comma-separated positive dimensions.
NEURAL = {
    "max_epochs": ("训练轮数", 50, 1, 100000),
    "batch_size": ("批次大小", 64, 1, 100000),
    "latent_dim": ("潜在维度", 64, 1, 100000),
    "hidden_dims": ("隐藏层维度", [256, 128], None, None),
    "learning_rate": ("学习率（JINGWEI）", 0.001, 0.00000001, 1.0),
    "mask_weight": ("掩蔽损失权重（JINGWEI）", 0.5, 0.0, 100.0),
}
PARAMETERS = {
    "perseus": {"downshift": ("丰度下移（标准差倍数）", 1.8, 0.0, 20.0),
                "width": ("分布宽度（标准差倍数）", 0.3, 0.0001, 10.0)},
    "knn": {"k": ("邻居数量", 5, 1, 100000)},
    "iterative-ridge": {"max_iter": ("迭代次数", 10, 1, 10000)},
    "iterative-rf": {"max_iter": ("迭代次数", 10, 1, 10000), "trees": ("树的数量", 100, 1, 100000)},
    "bpca": {"rank": ("主成分数量", 10, 1, 100000)},
    "svd": {"rank": ("主成分数量", 10, 1, 100000)},
    "qrilc": {"tune_sigma": ("标准差调整系数", 1.0, 0.0001, 100.0)},
    "mindet": {"q": ("低丰度分位数", 0.01, 0.0001, 0.9999)},
    "minprob": {"q": ("低丰度分位数", 0.01, 0.0001, 0.9999), "tune_sigma": ("标准差调整系数", 1.0, 0.0001, 100.0)},
    "impseqrob": {"alpha": ("稳健比例", 0.9, 0.01, 0.99)},
    "missforest": {"max_iter": ("迭代次数", 10, 1, 10000), "trees": ("树的数量", 100, 1, 100000)},
    "grr": {"max_iter": ("迭代次数", 10, 1, 10000)},
}
for name in ("naguider-knn", "seqknn", "lls", "trknn"):
    PARAMETERS[name] = {"k": ("邻居数量", 10, 1, 100000)}
for name in ("mice-norm", "mice-cart", "mice-pmm"):
    PARAMETERS[name] = {"m": ("插补次数（结果取均值）", 5, 1, 1000), "max_iter": ("迭代次数", 5, 1, 10000)}
PARAMETERS["mice-pmm"].update({"donors": ("PMM 候选供体数", 5, 1, 10000),
                               "ridge": ("线性正则强度", 0.00001, 0.00000001, 1.0)})
for name in ("jingwei-dmf", "jingwei-dcae", "pimms-dae", "pimms-vae", "pimms-cf"):
    PARAMETERS[name] = dict(NEURAL)
PARAMETERS["jingwei-dcae"].update({
    "encoder_blocks": ("编码块数量", 2, 1, 32),
    "decoder_blocks": ("解码块数量", 2, 1, 32),
    "dilation": ("卷积膨胀系数", 2, 1, 128),
})
# PIMMS exposes its own learning-rate finder and has no JINGWEI mask-loss weight.
for name in ("pimms-dae", "pimms-vae", "pimms-cf"):
    PARAMETERS[name].pop("learning_rate")
    PARAMETERS[name].pop("mask_weight")
PARAMETERS["pimms-cf"].pop("hidden_dims")

LABELS = {"zero": "零值", "minimum": "全局最小值", "feature-minimum": "蛋白最小值",
          "mean": "蛋白均值", "median": "蛋白中位数", "sample-median": "样本中位数",
          "perseus": "Perseus", "knn": "KNN（scikit-learn）", "iterative-ridge": "Iterative Ridge",
          "iterative-rf": "Iterative RF", "jingwei-dmf": "JINGWEI · DMF", "jingwei-dcae": "JINGWEI · DCAE",
          "pimms-dae": "PIMMS · DAE", "pimms-vae": "PIMMS · VAE", "pimms-cf": "PIMMS · CF"}
LABELS["mice-pmm"] = "MICE · PMM"
